import logging
import os
import re
import warnings
from typing import List

import requests
import streamlit as st
from bs4 import BeautifulSoup
from langchain_chroma import Chroma
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_groq.chat_models import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field
from tavily import TavilyClient
from typing_extensions import TypedDict

# Set up environment variables
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["USER_AGENT"] = "AgenticRAG/1.0"
os.environ["TAVILY_API_KEY"] = st.secrets["TAVILY_API_KEY"]
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
# os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]


# Resolve or suppress warnings
# Set global logging level to ERROR
logging.basicConfig(level=logging.ERROR, force=True)
# Suppress all SageMaker logs
logging.getLogger("sagemaker").setLevel(logging.CRITICAL)
logging.getLogger("sagemaker.config").setLevel(logging.CRITICAL)

# Ignore the specific FutureWarning from Hugging Face Transformers
warnings.filterwarnings(
    "ignore",
    message="`clean_up_tokenization_spaces` was not set.*",
    category=FutureWarning
)
# General suppression for other warnings (optional)
warnings.filterwarnings("ignore")
# Configure logging
logging.basicConfig(level=logging.INFO)
###################################################

# Define paths and parameters
data_file_path = 'Becoming an entrepreneur in Finland.md'
DATA_FOLDER = 'data'
persist_directory_openai = 'data/chroma_db_llamaparse-openai'
persist_directory_huggingface = 'data/chroma_db_llamaparse-huggincface'
collection_name = 'rag'
CHUNK_SIZE = 3000
CHUNK_OVERLAP = 200

finland_rag_prompt = PromptTemplate(
    template=r"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a precise, accurate assistant specializing in Finland business information. Follow these rules:

1. **Language Matching**: Always respond in the same language as the question (English, Finnish, Russian, Estonian, etc.)

2. **Context Adherence**: Base your answer solely on the provided context. If the context contains 'Internet search results', incorporate them.

3. **Answer Style**:
   - "Concise": Brief, direct responses
   - "Moderate": Balanced detail with some explanation
   - "Explanatory": Comprehensive answers with examples

4. **Citations**:
   - For Smart guide results: Include citations as [document_name, page xx]
   - For Internet search results: Include hyperlinked URLs with website names
   - Never invent citations or URLs

5. **Hybrid Content Handling**:
   If and only if the 'page_content' field in the context contains both "Smart guide results: " and "Internet search results: ", create two clearly separated sections, regardless of answer style.
   - **Smart guide results**: Information with proper citations
   - **Internet search results**: Web information with linked sources

6. **Error Handling**:
   - If context explicitly states "I apologize, but I'm designed to answer questions specifically related to business and entrepreneurship in Finland," repeat this verbatim
   - If context states "No information from the documents found.," repeat this verbatim

7. **Formatting**:
   - Use bullet points for lists
   - Bold important information
   - Create tables when appropriate
   - Use line breaks between sections

8. **Conversational Style**:
   - Maintain a helpful, guiding tone
   - Reference conversation history when relevant
   - Use clear, accessible language

Never add information beyond the context, except for clarifying explanations in Moderate or Explanatory styles that enhance the context without adding new facts.

<|eot_id|><|start_header_id|>user<|end_header_id|>
Question: {question} 
Context: {context} 
Answer style: {answer_style}
Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["question", "context", "answer_style"]
)

estonia_rag_prompt = PromptTemplate(
    template=r"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are a helpful, highly accurate and trustworthy assistant specialized in answering questions related to business, entrepreneurship, and the related matters in Estonia.
            Your responses must strictly adhere to the provided context, answer style, and question's language using the following rules:

            1. **Question and answer language**: 
            - Detect the question's main language (e.g., English, Finnish, Russian, Estonian, Arabic, or other) and always answer in the same language. If a question has English words and words from some other language which doesn't have the same letters as English, the answer should be in other language.
            - **very important**: Make sure that your response is in the same language as the question's. 
            2. **Context-Only Answers with a given answer style**:
            - Always base your answers on the provided context and answer style.
            - If the context explicitly states 'I apologize, but I'm designed to answer questions specifically related to business and entrepreneurship in Finland,' output this context verbatim.

            3. **Response style**:
            - Address the query directly without unnecessary or speculative information.
            - Do not draw from your knowledge base; strictly use the given context. However, take some liberty to provide more explanations and illustrations for better clarity and demonstration from your knowledge and experience only if the answer style is "Moderate" or "Explanatory". 
            4. **Answer style**
            - If answer style = "Concise", generate a concise answer.
            - If answer style = "Moderate", use a moderate approach to generate an answer where you can provide a little bit more explanation and elaborate the answer to improve clarity, integrating your own experience. 
            - If answer style = "Explanatory", provide a detailed and elaborated answer in the question' language by providing more explanations with examples and illustrations to improve clarity in the best possible way, integrating your own experience. However, the explanations, examples, and illustrations should be strictly based on the context. 
            5. **Conversational tone**
             - Maintain a conversational and helping style which should tend to guide the user and provide him help, hints and offers to further help and information. 
             - Use simple language. Explain difficult concepts or terms wherever needed. Present the information in the best readable form.

            6. **Formatting Guidelines**:
            - Use bullet points for lists.
            - Include line breaks between sections for clarity.
            - Highlight important numbers, dates, and terms using **bold** formatting.
            - Create tables wherever appropriate to present data clearly.
            - If there are discrepancies in the context, clearly explain them.

            7. **Citation Rules**:
            - **very important**: Include citations in the answer at all relevant places if they are present in the context. Under no circumstances ignore them. 
            -  include all the URLs in hyperlink form returned by the web search. **very important**: The URLs should be labelled with the website name.
            - Do not invent any citation or URL. Only use the citation or URL in the context.
            8. **Integrity and Trustworthiness**:
            - Ensure every part of your response complies with these rules.

            <|eot_id|><|start_header_id|>user<|end_header_id|>
            Question: {question} 
            Context: {context} 
            Answer style: {answer_style}
            Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["question", "context", "answer_style"]
)


def remove_tags(soup):
    # Remove unwanted tags
    for element in soup(["script", "style", "header", "footer", "nav", "aside", "noscript"]):
        element.decompose()

    # Extract text while preserving structure
    content = ""
    for element in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'li']):
        text = element.get_text(strip=True)
        if element.name.startswith('h'):
            level = int(element.name[1])
            content += '#' * level + ' ' + text + '\n\n'  # Markdown-style headings
        elif element.name == 'p':
            content += text + '\n\n'
        elif element.name == 'li':
            content += '- ' + text + '\n'
    return content

# @st.cache_data
def get_info(URLs):
    """
    Fetch and return contact information from predefined URLs.
    """
    combined_info = ""
    for url in URLs:
        try:
            response = requests.get(url)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "html.parser")
                combined_info += "URL: " + url + \
                    ": " + remove_tags(soup) + "\n\n"
            else:
                combined_info += f"Failed to retrieve information from {url}\n\n"
        except Exception as e:
            combined_info += f"Error fetching URL {url}: {e}\n\n"
    return combined_info

# @st.cache_data
def staticChunker(folder_path):
    docs = []
    print(
        f"Creating chunks. CHUNK_SIZE: {CHUNK_SIZE}, CHUNK_OVERLAP: {CHUNK_OVERLAP}")

    # Loop through all .md files in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".md"):
            file_path = os.path.join(folder_path, file_name)
            print(f"Processing file: {file_path}")

            # Load documents from the Markdown file
            loader = UnstructuredMarkdownLoader(file_path)
            documents = loader.load()

            # Add file-specific metadata (optional)
            for doc in documents:
                doc.metadata["source_file"] = file_name

            # Split loaded documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
            chunked_docs = text_splitter.split_documents(documents)
            docs.extend(chunked_docs)
    return docs

# @st.cache_resource
def load_or_create_vs(persist_directory):
    # Check if the vector store directory exists
    if os.path.exists(persist_directory):
        print("Loading existing vector store...")
        # Load the existing vector store
        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=st.session_state.embed_model,
            collection_name=collection_name
        )
    else:
        print("Vector store not found. Creating a new one...\n")
        docs = staticChunker(DATA_FOLDER)
        print("Computing embeddings...")
        # Create and persist a new Chroma vector store
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=st.session_state.embed_model,
            persist_directory=persist_directory,
            collection_name=collection_name
        )
        print('Vector store created and persisted successfully!')

    return vectorstore


def initialize_app(model_name, selected_embedding_model, selected_routing_model, selected_grading_model, hybrid_search, internet_search, answer_style):
    """
    Initialize embeddings, vectorstore, retriever, and LLM for the RAG workflow.
    Reinitialize components only if the selection has changed.
    """
    # Track current state to prevent redundant initialization
    if "current_model_state" not in st.session_state:
        st.session_state.current_model_state = {
            "answering_model": None,
            "embedding_model": None,
            "routing_model": None,
            "grading_model": None,
        }

    # Check if models or settings have changed
    state_changed = (
        st.session_state.current_model_state["answering_model"] != model_name or
        st.session_state.current_model_state["embedding_model"] != selected_embedding_model or
        st.session_state.current_model_state["routing_model"] != selected_routing_model or
        st.session_state.current_model_state["grading_model"] != selected_grading_model
    )

    # Reinitialize components only if settings have changed
    if state_changed:
        try:
            st.session_state.embed_model = initialize_embedding_model(
                selected_embedding_model)

            # Update vectorstore (only for Finland, not needed for Estonia)
            if st.session_state.selected_country == "Finland":
                persist_directory = persist_directory_openai if "text-" in selected_embedding_model else persist_directory_huggingface
                st.session_state.vectorstore = load_or_create_vs(persist_directory)
                st.session_state.retriever = st.session_state.vectorstore.as_retriever(
                    search_kwargs={"k": 5})
            else:
                # For Estonia, set dummy vectors or empty retriever as we'll only use web search
                st.session_state.retriever = None

            st.session_state.llm = initialize_llm(model_name, answer_style)
            st.session_state.router_llm = initialize_router_llm(
                selected_routing_model)
            st.session_state.grader_llm = initialize_grading_llm(
                selected_grading_model)
            st.session_state.doc_grader = initialize_grader_chain()

            # Set the appropriate RAG prompt based on selected country
            if st.session_state.selected_country == "Estonia":
                st.session_state.rag_prompt = estonia_rag_prompt
            else:
                st.session_state.rag_prompt = finland_rag_prompt

            # Save updated state
            st.session_state.current_model_state.update({
                "answering_model": model_name,
                "embedding_model": selected_embedding_model,
                "routing_model": selected_routing_model,
                "grading_model": selected_grading_model,
            })
        except Exception as e:
            st.error(f"Error during initialization: {e}")
            # Restore previous state if available
            if st.session_state.current_model_state["answering_model"]:
                st.warning(f"Continuing with previous configuration")
            else:
                # Fallback to OpenAI if no previous state
                st.session_state.llm = ChatOpenAI(
                    model="gpt-4o-mini", temperature=0.0, streaming=True)
                st.session_state.router_llm = ChatOpenAI(
                    model="gpt-4o-mini", temperature=0.0)
                st.session_state.grader_llm = ChatOpenAI(
                    model="gpt-4o-mini", temperature=0.0)
                
                # Set a default RAG prompt based on country
                if st.session_state.selected_country == "Estonia":
                    st.session_state.rag_prompt = estonia_rag_prompt
                else:
                    st.session_state.rag_prompt = finland_rag_prompt

    print(f"Using LLM: {model_name}, Router LLM: {selected_routing_model}, Grader LLM:{selected_grading_model}, embedding model: {selected_embedding_model}")

    try:
        return workflow.compile()
    except Exception as e:
        st.error(f"Error compiling workflow: {e}")
        # Return a simple dummy workflow that just echoes the input
        return lambda x: {"generation": "Error in workflow. Please try a different model.", "question": x.get("question", "")}
# @st.cache_resource


def initialize_llm(model_name, answer_style):
    if "llm" not in st.session_state or st.session_state.llm.model_name != model_name:
        if answer_style == "Concise":
            temperature = 0.0
        elif answer_style == "Moderate":
            temperature = 0.0
        elif answer_style == "Explanatory":
            temperature = 0.0

        if "gpt-" in model_name:
            st.session_state.llm = ChatOpenAI(
                model=model_name, temperature=temperature, streaming=True)
        elif "deepseek-" in model_name:
            # Deepseek models need "hidden" reasoning_format to prevent <think> tags that otherwise cause issues
            st.session_state.llm = ChatGroq(
                model=model_name,
                temperature=temperature,
                streaming=True,
                # model_kwargs={"reasoning_format": "hidden"}
            )
        else:
            st.session_state.llm = ChatGroq(
                model=model_name, temperature=temperature, streaming=True)

    return st.session_state.llm


def initialize_embedding_model(selected_embedding_model):
    # Check if the embed_model exists in session_state
    if "embed_model" not in st.session_state:
        st.session_state.embed_model = None

    # Check if the current model matches the selected one
    current_model_name = None
    if st.session_state.embed_model:
        if hasattr(st.session_state.embed_model, "model"):
            current_model_name = st.session_state.embed_model.model
        elif hasattr(st.session_state.embed_model, "model_name"):
            current_model_name = st.session_state.embed_model.model_name

    # Initialize a new model if it doesn't match the selected one
    if current_model_name != selected_embedding_model:
        if "text-" in selected_embedding_model:
            st.session_state.embed_model = OpenAIEmbeddings(
                model=selected_embedding_model)
        else:
            st.session_state.embed_model = HuggingFaceEmbeddings(
                model_name=selected_embedding_model)

    return st.session_state.embed_model

# @st.cache_resource

# FIX: mixtral model won't work with ChatGroq idk why. Maybe add gpt-4o-mini as fallback


def initialize_router_llm(selected_routing_model):
    if "router_llm" not in st.session_state or st.session_state.router_llm.model_name != selected_routing_model:
        if "gpt-" in selected_routing_model:
            st.session_state.router_llm = ChatOpenAI(
                model=selected_routing_model, temperature=0.0)
        elif "deepseek-" in selected_routing_model:
            st.session_state.router_llm = ChatGroq(
                model=selected_routing_model,
                temperature=0.0,
                model_kwargs={"reasoning_format": "hidden"}
            )
        # Uncomment this block to use gpt-4o-mini as a fallback for mixtral models. Because 20.2.2025 mixtral model won't in router_llm
        # elif "mixtral" in selected_routing_model.lower():
        #     st.session_state.router_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
        else:
            st.session_state.router_llm = ChatGroq(
                model=selected_routing_model, temperature=0.0)

    return st.session_state.router_llm

# @st.cache_resource


def initialize_grading_llm(selected_grading_model):
    if "grader_llm" not in st.session_state or st.session_state.grader_llm.model_name != selected_grading_model:
        if "gpt-" in selected_grading_model:
            st.session_state.grader_llm = ChatOpenAI(
                model=selected_grading_model, temperature=0.0, max_tokens=8000)
        elif "deepseek-" in selected_grading_model:
            # Deepseek-models need "hidden" reasoning_format to prevent <think> tags from leaking
            st.session_state.grader_llm = ChatGroq(
                model=selected_grading_model,
                temperature=0.0,
                model_kwargs={"reasoning_format": "hidden"}
            )
        else:
            st.session_state.grader_llm = ChatGroq(
                model=selected_grading_model, temperature=0.0)

    return st.session_state.grader_llm


model_list = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "llama3-70b-8192",
    "llama3-8b-8192",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
    "gpt-4o-mini",
    "gpt-4o",
    "deepseek-r1-distill-llama-70b"
]


def initialize_grader_chain():
    # Data model for LLM output format
    class GradeDocuments(BaseModel):
        """Binary score for relevance check on retrieved documents."""
        binary_score: str = Field(
            description="Documents are relevant to the question, 'yes' or 'no'"
        )

    # LLM for grading
    structured_llm_grader = st.session_state.grader_llm.with_structured_output(
        GradeDocuments)

    # Prompt template for grading
    SYS_PROMPT = """You are an expert grader assessing relevance of a retrieved document to a user question.

    Follow these instructions for grading:
    - If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant.
    - Your grade should be either 'Yes' or 'No' to indicate whether the document is relevant to the question or not."""

    grade_prompt = ChatPromptTemplate.from_messages([
        ("system", SYS_PROMPT),
        ("human", """Retrieved document:
    {documents}
    User question:
    {question}
    """),
    ])

    # Build grader chain
    return grade_prompt | structured_llm_grader


def grade_documents(state):
    question = state["question"]
    documents = state.get("documents", [])
    filtered_docs = []

    if not documents:
        print("No documents retrieved for grading.")
        return {"documents": [], "question": question, "web_search_needed": "Yes"}

    print(
        f"Grading retrieved documents with {st.session_state.grader_llm.model_name}")

    for count, doc in enumerate(documents):
        try:
            # Evaluate document relevance
            score = st.session_state.doc_grader.invoke(
                {"documents": [doc], "question": question})
            print(f"Chunk {count} relevance: {score}")
            if score.binary_score == "Yes":
                filtered_docs.append(doc)
        except Exception as e:
            print(f"Error grading document chunk {count}: {e}")

    if not filtered_docs:
        # Create a proper Document object for the error message
        error_doc = Document(page_content="No information from the documents found.")
        filtered_docs = [error_doc]

    web_search_needed = "No"        
    return {"documents": filtered_docs, "question": question, "web_search_needed": web_search_needed}


def route_after_grading(state):
    web_search_needed = state.get("web_search_needed", "No")
    print(f"Routing decision based on web_search_needed={web_search_needed}")
    if web_search_needed == "Yes":
        return "websearch"
    else:
        return "generate"

# Define graph state class


class GraphState(TypedDict):
    question: str
    generation: str
    web_search_needed: str
    documents: List[Document]
    answer_style: str


def retrieve(state):
    print("Retrieving documents")
    question = state["question"]
    documents = st.session_state.retriever.invoke(question)
    return {"documents": documents, "question": question}


def format_documents(documents):
    """Format documents into a single string for context."""
    return "\n\n".join(doc.page_content for doc in documents)


def generate(state):
    question = state["question"]
    documents = state.get("documents", [])
    answer_style = state.get("answer_style", "Concise")

    if "llm" not in st.session_state:
        st.session_state.llm = initialize_llm(
            st.session_state.selected_model, answer_style)

    # Use country-specific RAG prompt from session state
    rag_chain = st.session_state.rag_prompt | st.session_state.llm | StrOutputParser()

    if not documents:
        print("No documents available for generation.")
        return {"generation": "No relevant documents found.", "documents": documents, "question": question}

    tried_models = set()
    original_model = st.session_state.selected_model
    current_model = original_model

    while len(tried_models) < len(model_list):
        try:
            tried_models.add(current_model)
            st.session_state.llm = initialize_llm(current_model, answer_style)
            rag_chain = st.session_state.rag_prompt | st.session_state.llm | StrOutputParser()

            # context = format_documents(documents)
            context = documents
            generation = rag_chain.invoke(
                {"context": context, "question": question, "answer_style": answer_style})

            print(f"Generating a {answer_style} length response.")
            # print(f"Response generated with {st.session_state.llm.model_name} model.")
            print("Done.")

            if current_model != original_model:
                print(f"Reverting to original model: {original_model}")
                st.session_state.llm = initialize_llm(
                    original_model, answer_style)

            return {"documents": documents, "question": question, "generation": generation}

        except Exception as e:
            error_message = str(e)
            if "rate_limit_exceeded" in error_message or "Request too large" in error_message or "Please reduce the length of the messages or completion" in error_message:
                print(f"Model's rate limit exceeded or request too large.")
                current_model = model_list[(model_list.index(
                    current_model) + 1) % len(model_list)]
                print(f"Switching to model: {current_model}")
            else:
                return {
                    "generation": f"Error during generation: {error_message}",
                    "documents": documents,
                    "question": question,
                }

    return {
        "generation": "Unable to process the request due to limitations across all models.",
        "documents": documents,
        "question": question,
    }


def handle_unrelated(state):
    question = state["question"]
    documents = state.get("documents", [])
    
    # Country-specific unrelated response
    response = f"I apologize, but I'm designed to answer questions specifically related to business and entrepreneurship in {st.session_state.selected_country}."
    
    documents.append(Document(page_content=response))
    return {"generation": response, "documents": documents, "question": question}




def grade_retriever_hybrid(vector_docs, question):
    """
    Grade only vector documents during hybrid search.
    
    Parameters:
    - vector_docs: Document list from retriever
    - question: User question
    
    Returns:
    - filtered_vector_docs: List of relevant documents (or error message document)
    """
    filtered_docs = []

    if not vector_docs:
        print("No vector documents available for grading in hybrid search.")
        # Create error document specifically for Smart guide results
        error_doc = Document(page_content="No information from the documents found.")
        return [error_doc]

    print(f"Grading vector documents with {st.session_state.grader_llm.model_name} for hybrid search")

    for count, doc in enumerate(vector_docs):
        try:
            # Evaluate document relevance
            score = st.session_state.doc_grader.invoke(
                {"documents": [doc], "question": question})
            print(f"Vector chunk {count} relevance: {score}")
            if score.binary_score == "Yes":
                filtered_docs.append(doc)
        except Exception as e:
            print(f"Error grading vector document chunk {count}: {e}")
            
    if not filtered_docs:
        # Create a proper Document object for the error message
        error_doc = Document(page_content="No information from the documents found.")
        filtered_docs = [error_doc]
        print("No relevant vector documents found in hybrid search.")
    else:
        print(f"Found {len(filtered_docs)} relevant vector documents in hybrid search.")
        
    return filtered_docs

    
def hybrid_search(state):
    question = state["question"]
    print("Invoking hybrid search...")
    
    # For Finland, do hybrid search
    vector_docs = st.session_state.retriever.invoke(question)
    
    # Grade the vector documents
    filtered_vector_docs = grade_retriever_hybrid(vector_docs, question)
    
    # Add headings to distinguish between vector and web search results
    vector_results = [Document(
        page_content="Smart guide results: " + doc.page_content) for doc in filtered_vector_docs]

    
    # Proceed with web search
    web_docs = web_search({"question": question})["documents"]
    
    # Check if any web_docs already contain "Internet search results:"
    web_results_contain_header = any(
        "Internet search results:" in doc.page_content for doc in web_docs)

    # Add "Internet search results:" only if not already present in any web doc
    if not web_results_contain_header:
        web_results = [
            Document(page_content="Internet search results:" + doc.page_content) for doc in web_docs
        ]
    else:
        web_results = web_docs  # Keep web_docs unchanged if they already contain the header

    # Combine the filtered vector results with web results
    combined_docs = vector_results + web_results
    return {"documents": combined_docs, "question": question}


def web_search(state):
    if "tavily_client" not in st.session_state:
        st.session_state.tavily_client = TavilyClient()
    question = state["question"]
    # question = re.sub(r'\b\w+\\|Internet search\b', '', question).strip()
    
    # Add country-specific suffix to the question
    if st.session_state.selected_country == "Estonia":
        if "estonia" not in question.lower() and "estonian" not in question.lower():
            question += " in Estonia"
    else:  # Default to Finland
        if "finland" not in question.lower() and "finnish" not in question.lower():
            question += " in Finland"

        
    documents = state.get("documents", [])
    try:
        print(f"Invoking internet search for {st.session_state.selected_country}...")
        
        # Select domain list based on country
        if st.session_state.selected_country == "Estonia":
            include_domains = [
                "eesti.ee",
                "e-resident.gov.ee",
                "investinestonia.com",
                "mkm.ee",
                "tallinn.ee",
                "ebs.ee",
                "emta.ee",
                "learn.e-resident.gov.ee",
                "fi.ee",
                "riigiteataja.ee",
                "ttja.ee",
                "stat.ee",
                "ariregister.rik.ee",
                "tradewithestonia.com",
                "kul.ee",
                "pta.agri.ee",
                "terviseamet.ee"
            ]
        else:  # Default to Finland
            include_domains = [
                "migri.fi",
                "enterfinland.fi",
                "businessfinland.fi",
                "kela.fi",
                "vero.fi",
                "suomi.fi",
                "valvira.fi",
                "finlex.fi",
                "hus.fi",
                "lvm.fi",
                "thefinlandbusinesspress.fi",
                "infofinland.fi",
                "ely-keskus.fi",
                "yritystulkki.fi",
                "tem.fi",
                "prh.fi",
                "avi.fi",
                "ruokavirasto.fi",
                "traficom.fi",
                "trade.gov",
                "finlex.fi",
                "te-palvelut.fi",
                "tilastokeskus.fi",
                "veronmaksajat.fi",
                "hel.fi",
                "ukko.fi",
                "yrityssalo.fi",
                "stm.fi",
                "eurofound.europa.eu",
                "oph.fi",
                "oikeusrekisterikeskus.fi"

            ]

        search_result = st.session_state.tavily_client.get_search_context(
            query=question,
            search_depth="basic",
            max_tokens=4000,
            max_results=10,
            include_domains=include_domains,
        )
        
        # Handle different types of results
        if isinstance(search_result, str):
            web_results = search_result
        elif isinstance(search_result, dict) and "documents" in search_result:
            web_results = "Internet search results: ".join(
                [doc.get("content", "") for doc in search_result["documents"]])
        else:
            web_results = "No valid results returned by TavilyClient."
        web_results_doc = Document(page_content=web_results)
        documents.append(web_results_doc)
    except Exception as e:
        print(f"Error during web search: {e}")
        # Ensure workflow can continue gracefully
        documents.append(Document(page_content=f"Web search failed: {e}"))
    return {"documents": documents, "question": question}

    
# # Router function
# def route_question(state):
#     question = state["question"]    
#     hybrid_search_enabled = state.get("hybrid_search", False)
#     internet_search_enabled = state.get("internet_search", False)
    
#     # Define the business topics for relevance checking
#     country = st.session_state.selected_country

#     business_topics = (
#     # Tax and finance related
#     f"tax rate, taxation rules, taxable incomes, tax exemptions, tax filing process, VAT, "
#     f"corporate tax, personal income tax, dividend taxation, capital gains tax, "
#     f"accounting requirements, financial reporting, bookkeeping, invoicing, "
#     f"business bank accounts, merchant accounts, payment processing, "
    
#     # Immigration and legal status
#     f"immigration process, visa requirements, residence permits, work permits, "
#     f"immigration authority, citizenship, permanent residency, "
#     f"e-residency, digital nomad visas, family reunification, "
    
#     # Business formation and structure
#     f"company registration, business registration, legal entity types, "
#     f"sole proprietorship, partnership, limited liability company, corporation, "
#     f"business name registration, articles of association, "
#     f"shareholders agreement, ownership structure, share capital, "
#     f"business licensing, permits, authorizations required for business operation, "
    
#     # Business operations
#     f"business planning, business strategy, market analysis, "
#     f"business model canvas, revenue models, pricing strategies, "
#     f"supply chain management, procurement, inventory management, "
#     f"logistics, import and export procedures, customs regulations, "
#     f"international trade, trade agreements, sanctions, "
    
#     # Entrepreneurship and funding
#     f"startups, entrepreneurship, business incubators, accelerators, "
#     f"venture capital, angel investing, seed funding, business loans, "
#     f"crowdfunding, grants, government subsidies, business incentives, "
#     f"business pitching, valuation, exit strategies, "
    
#     # Employment and HR
#     f"employment law, hiring employees, recruiting, job contracts, "
#     f"labor regulations, minimum wage, working hours, "
#     f"employee benefits, health insurance, sick leave, "
#     f"parental leave, annual leave, vacation policies, "
#     f"unemployment benefits, pensions, retirement, "
#     f"remote work regulations, hybrid work, "
#     f"employee stock options, profit sharing, "
    
#     # Intellectual property and data
#     f"intellectual property, patents, trademarks, copyrights, "
#     f"data protection, GDPR compliance, privacy regulations, "
#     f"cybersecurity requirements, digital signatures, "
    
#     # Business compliance
#     f"regulatory compliance, industry-specific regulations, "
#     f"health and safety requirements, environmental regulations, "
#     f"consumer protection laws, competition law, "
#     f"anti-corruption laws, KYC requirements, AML regulations, "
    
#     # Business services and infrastructure
#     f"business premises, commercial property, office space, "
#     f"coworking spaces, business addresses, virtual offices, "
#     f"utilities, telecommunications, internet services, "
#     f"business insurance, liability insurance, property insurance, "
#     f"Recommendations and tips from local advisers, “how to” (business acumen)"
    
#     # Business support and networking
#     f"chambers of commerce, business associations, industry groups, "
#     f"business networking, mentorship programs, business advisors, "
#     f"business coaching, consultant services, business consultancy"
    
#     # Business closure and restructuring
#     f"business dissolution, bankruptcy process, insolvency, "
#     f"business restructuring, mergers, acquisitions, "
#     f"business succession planning, business valuation, "
#     f"Market information, Internationalization of business e.g. international markets, "
#     f"investment opportunities, how to invest?"
# )

#     tool_selection = {
#         "retrieve": (
#             f"Questions related to business, startups, and practical aspects of operating in {country}, including but not limited to: "
#             f"• ANY aspect of starting, running, managing, or closing businesses "
#             f"• Questions that COULD be asked by someone interested in entrepreneurship "
#             f"• Topics that entrepreneurs or business people commonly need to know "
#             f"• Anything involving economic activities, work, income, or finances "
#             f"• Practical aspects of living, working, or operating in {country} "
#             f"• Topics with implicit (not just explicit) connections to business "
#             f"• Business planning, strategy, and market analysis "
#             f"• Business opportunities, potential ventures, and market gaps "
#             f"• Tax systems, taxation rules, and filing procedures "
#             f"• Immigration processes, visa requirements, and residency options "
#             f"• Company registration, business structures, and legal entities "
#             f"• Licensing, permits, and regulatory compliance requirements "
#             f"• Employment laws, hiring practices, and workplace regulations "
#             f"• Social benefits, insurance systems, and safety nets "
#             f"• Industry insights, sector analysis, and market trends "
#             f"• Import/export procedures, international trade, and customs "
#             f"• Business financing, investment options, and funding sources "
#             f"• Business infrastructure, services, and support networks "
#             f"• Business culture, practices, and communication norms "
#             f"• Questions about specific business opportunities for particular expertise, areas, or sectors "
#             f"• Requests for suggestions, advice, or guidance related to business activities "
#             f"• Any question that could reasonably have a business or entrepreneurship angle, even if uncertain"
#             f"• Questions related to startups, leading startups, successful startups, and related topics."
#         ),
#         "unrelated": (
#             f"Questions not related to business, entrepreneurship, startups, employment, unemployment, pensions, insurance, social benefits, or similar topics, "
#             f"or those related to other countries or cities instead of Finland, or those related to other countries or cities instead of {country}."
#         )
#     }

#     business_relevance_prompt = ChatPromptTemplate.from_messages([
#         ("system", f"""You are evaluating whether questions are broadly related to business, entrepreneurship, or economic activities.

#         Your task is to determine if a question has ANY connection to business or entrepreneurship topics, even if indirect or implied.

#         Use an INCLUSIVE approach:
#         - Say 'yes' if the question relates to ANY aspect of starting, running, managing, or closing businesses or startups
#         - Say 'yes' if the question COULD be asked by someone interested in entrepreneurship
#         - Say 'yes' if the question relates to topics that entrepreneurs or business people commonly need to know
#         - Say 'yes' if the question involves economic activities, work, income, or finances
#         - Say 'yes' if the question is about living, working, or operating in a country from a practical standpoint
#         - Say 'yes' even if the business connection is implicit rather than explicit
#         - Say 'yes' if you're uncertain but the question could reasonably have a business angle

#         Only say 'no' if the question is CLEARLY unrelated to business, entrepreneurship, economics, work, or practical aspects of living in a country.

#         Here are example topics that should be considered business-related (this is NOT an exhaustive list):
#         {business_topics}

#         Answer ONLY with 'yes' or 'no'.
#         """),
#             ("human", "Question: {question}")
#         ])
    
#     country_relevance_prompt = ChatPromptTemplate.from_messages([
#         ("system", f"Determine if the question is explicitly about a country OTHER THAN {country} or a city that is NOT in {country}. If no country is explicitly mentioned, the question is about {country}. Answer only 'yes' or 'no'."),
#         ("human", "Question: {question}")
#     ])
    
#     # Function to check business topic relevance
#     def is_business_related(q):
#         try:
#             result = (business_relevance_prompt | st.session_state.router_llm | StrOutputParser()).invoke({"question": q})
#             return "yes" in result.lower()
#         except Exception as e:
#             print(f"Error in business relevance check: {e}")
#             # Default to True in case of error
#             return True
    
#     # Function to check if question is about a different country/city
#     def is_wrong_country(q):
#         try:
#             result = (country_relevance_prompt | st.session_state.router_llm | StrOutputParser()).invoke({"question": q})
#             return "yes" in result.lower()
#         except Exception as e:
#             print(f"Error in country relevance check: {e}")
#             # Default to False in case of error
#             return False
    
#     # Check both conditions separately
#     business_related = is_business_related(question)
#     different_country = is_wrong_country(question)
    
#     # If question is about a different country, mark as unrelated
#     if different_country:
#         print(f"Question is about a different country than {country}, marking as unrelated")
#         return "unrelated"
    
#     # If question is not business-related, mark as unrelated
#     if not business_related:
#         print(f"Question is not related to business topics, marking as unrelated")
#         return "unrelated"
    
#     # Now we know the question is business-related and not about a different country
#     # For Estonia, always use web search for relevant questions
#     if st.session_state.selected_country == "Estonia":
#         return "websearch"

#     # For Finland with hybrid or internet search enabled
#     if st.session_state.selected_country == "Finland":
#         if hybrid_search_enabled:
#             return "hybrid_search"
        
#         if internet_search_enabled:
#             return "websearch"
    
#     # For Finland with document search only, use more detailed routing
#     SYS_PROMPT = f"""Act as a router to select specific tools or functions based on user's question. 
#                  - Analyze the given question and use the given tool selection dictionary to output the name of the relevant tool based on its description and relevancy with the question. 
#                    The dictionary has tool names as keys and their descriptions as values. 
#                  - Output only and only tool name, i.e., the exact key and nothing else with no explanations at all. 
#                  - For questions mentioning any other country except {country}, or any other city except a city in {country}, output 'unrelated'.
#                 """

#     # Define the ChatPromptTemplate for detailed routing
#     prompt = ChatPromptTemplate.from_messages(
#         [
#             ("system", SYS_PROMPT),
#             ("human", """Here is the question:
#                         {question}
#                         Here is the tool selection dictionary:
#                         {tool_selection}
#                         Output the required tool.
#                     """),
#         ]
#     )

#     # Pass the inputs to the prompt
#     inputs = {
#         "question": question,
#         "tool_selection": tool_selection
#     }

#     # Invoke the chain
#     tool = (prompt | st.session_state.router_llm | StrOutputParser()).invoke(inputs)
#     # Remove backslashes and extra spaces
#     tool = re.sub(r"[\\'\"`]", "", tool.strip())
    
#     if "unrelated" not in tool:
#         print(f"Invoking {tool} tool through {st.session_state.router_llm.model_name}")
#     if "websearch" in tool:
#         print("I need to get recent information from this query.")
    
#     return tool


# Router function
def route_question(state):
    question = state["question"]    
    hybrid_search_enabled = state.get("hybrid_search", False)
    internet_search_enabled = state.get("internet_search", False)
    
    country = st.session_state.selected_country

    # Merged business topics and tool selection criteria
    business_topics = (
        f"Questions related to business, startups, and practical aspects of operating in {country}, including but not limited to: "
        # Tax and finance related
        f"\n• Tax rate, taxation rules, taxable incomes, tax exemptions, tax filing process, VAT, "
        f"corporate tax, personal income tax, dividend taxation, capital gains tax, "
        f"accounting requirements, financial reporting, bookkeeping, invoicing, "
        f"business bank accounts, merchant accounts, payment processing"
        
        # Immigration and legal status
        f"\n• Immigration process, visa requirements, residence permits, work permits, "
        f"immigration authority, citizenship, permanent residency, "
        f"e-residency, digital nomad visas, family reunification"
        
        # Business formation and structure
        f"\n• Company registration, business registration, legal entity types, "
        f"sole proprietorship, partnership, limited liability company, corporation, "
        f"business name registration, articles of association, "
        f"shareholders agreement, ownership structure, share capital, "
        f"business licensing, permits, authorizations required for business operation"
        
        # Business operations
        f"\n• Business planning, business strategy, market analysis, "
        f"business model canvas, revenue models, pricing strategies, "
        f"supply chain management, procurement, inventory management, "
        f"logistics, import and export procedures, customs regulations, "
        f"international trade, trade agreements, sanctions"
        
        # Entrepreneurship and funding
        f"\n• Startups, entrepreneurship, business incubators, accelerators, "
        f"venture capital, angel investing, seed funding, business loans, "
        f"crowdfunding, grants, government subsidies, business incentives, "
        f"business pitching, valuation, exit strategies"
        
        # Employment and HR
        f"\n• Employment law, hiring employees, recruiting, job contracts, "
        f"labor regulations, minimum wage, working hours, "
        f"employee benefits, health insurance, sick leave, "
        f"parental leave, annual leave, vacation policies, "
        f"unemployment benefits, pensions, retirement, "
        f"remote work regulations, hybrid work, "
        f"employee stock options, profit sharing"
        
        # Intellectual property and data
        f"\n• Intellectual property, patents, trademarks, copyrights, "
        f"data protection, GDPR compliance, privacy regulations, "
        f"cybersecurity requirements, digital signatures"
        
        # Business compliance
        f"\n• Regulatory compliance, industry-specific regulations, "
        f"health and safety requirements, environmental regulations, "
        f"consumer protection laws, competition law, "
        f"anti-corruption laws, KYC requirements, AML regulations"
        
        # Business services and infrastructure
        f"\n• Business premises, commercial property, office space, "
        f"coworking spaces, business addresses, virtual offices, "
        f"utilities, telecommunications, internet services, "
        f"business insurance, liability insurance, property insurance, "
        f"recommendations and tips from local advisers, business acumen"
        
        # Business support and networking
        f"\n• Chambers of commerce, business associations, industry groups, "
        f"business networking, mentorship programs, business advisors, "
        f"business coaching, consultant services, business consultancy"
        
        # Business closure and restructuring
        f"\n• Business dissolution, bankruptcy process, insolvency, "
        f"business restructuring, mergers, acquisitions, "
        f"business succession planning, business valuation, "
        f"market information, internationalization of business, "
        f"investment opportunities, how to invest"
    )

    business_relevance_prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are evaluating whether questions are broadly related to business, entrepreneurship, or economic activities.

        Your task is to determine if a question has ANY connection to business or entrepreneurship topics, even if indirect or implied.

        Use an INCLUSIVE approach:
        - Say 'yes' if the question relates to ANY aspect of starting, running, managing, or closing businesses or startups
        - Say 'yes' if the question COULD be asked by someone interested in entrepreneurship
        - Say 'yes' if the question relates to topics that entrepreneurs or business people commonly need to know
        - Say 'yes' if the question involves economic activities, work, income, or finances
        - Say 'yes' if the question is about living, working, or operating in a country from a practical standpoint
        - Say 'yes' even if the business connection is implicit rather than explicit
        - Say 'yes' if you're uncertain but the question could reasonably have a business angle

        Only say 'no' if the question is CLEARLY unrelated to business, entrepreneurship, economics, work, or practical aspects of living in a country.

        Here are example topics that should be considered business-related:
        {business_topics}

        Answer ONLY with 'yes' or 'no'.
        """),
            ("human", "Question: {question}")
        ])
    
    country_relevance_prompt = ChatPromptTemplate.from_messages([
        ("system", f"Determine if the question is explicitly about a country OTHER THAN {country} or a city that is NOT in {country}. If no country is explicitly mentioned, the question is about {country}. Answer only 'yes' or 'no'."),
        ("human", "Question: {question}")
    ])
    
    # Function to check business topic relevance
    def is_business_related(q):
        try:
            result = (business_relevance_prompt | st.session_state.router_llm | StrOutputParser()).invoke({"question": q})
            return "yes" in result.lower()
        except Exception as e:
            print(f"Error in business relevance check: {e}")
            # Default to True in case of error
            return True
    
    # Function to check if question is about a different country/city
    def is_wrong_country(q):
        try:
            result = (country_relevance_prompt | st.session_state.router_llm | StrOutputParser()).invoke({"question": q})
            return "yes" in result.lower()
        except Exception as e:
            print(f"Error in country relevance check: {e}")
            # Default to False in case of error
            return False
    
    # Check both conditions separately
    business_related = is_business_related(question)
    different_country = is_wrong_country(question)
    
    # If question is about a different country or not business-related, mark as unrelated
    if different_country or not business_related:
        print(f"Question is {'about a different country' if different_country else 'not related to business topics'}, marking as unrelated")
        return "unrelated"
    
    # Now we know the question is business-related and not about a different country
    # For Estonia, always use web search for relevant questions
    if st.session_state.selected_country == "Estonia":
        return "websearch"

    # For Finland with hybrid or internet search enabled
    if st.session_state.selected_country == "Finland":
        if hybrid_search_enabled:
            return "hybrid_search"
        elif internet_search_enabled:
            return "websearch"
        else:
            return "retrieve"  # Default to retrieve for Finland without special search options
    
    # Default fallback
    return "retrieve"


workflow = StateGraph(GraphState)
# # Add nodes
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("route_after_grading", route_after_grading)
workflow.add_node("websearch", web_search)
workflow.add_node("generate", generate)
workflow.add_node("hybrid_search", hybrid_search)
workflow.add_node("unrelated", handle_unrelated)

# Set conditional entry points
workflow.set_conditional_entry_point(
    route_question,
    {
        "retrieve": "retrieve",
        "websearch": "websearch",
        "hybrid_search": "hybrid_search",
        "unrelated": "unrelated"
    },
)

# Add edges
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    route_after_grading,
    {"websearch": "websearch", "generate": "generate"},
)
workflow.add_edge("websearch", "generate")
workflow.add_edge("hybrid_search", "generate")
workflow.add_edge("unrelated", "generate")


# Compile app
app = workflow.compile()
