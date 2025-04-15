# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import io
import re
import sys
import time

import streamlit as st
import torch
import tornado
from langchain_openai import ChatOpenAI

from agentic_rag import initialize_app
from st_callback import get_streamlit_cb

# This code line below Fixes console "RuntimeError: Tried to instantiate class '__path__._path', but it does not exist!"
# reference: https://github.com/VikParuchuri/marker/issues/442#issuecomment-2636393925
torch.classes.__path__ = []

# -------------------- Initialization --------------------
st.set_option("client.showErrorDetails", False)  # Hide error detail

# Add country selection at startup
if "selected_country" not in st.session_state:
    st.session_state.selected_country = None

# Early session state initialization
if "messages" not in st.session_state:
    st.session_state.messages = []
if "followup_key" not in st.session_state:
    st.session_state.followup_key = 0
if "pending_followup" not in st.session_state:
    st.session_state.pending_followup = None
if "last_assistant" not in st.session_state:
    st.session_state.last_assistant = None
if "followup_questions" not in st.session_state:
    st.session_state.followup_questions = []
if "show_guidelines" not in st.session_state:
    st.session_state.show_guidelines = False

# -------------------- Helper Functions --------------------
def get_followup_questions(last_user, last_assistant):
    """
    Generate three concise follow-up questions dynamically based on the latest conversation.
    """
    prompt = f"""Based on the conversation below:
User: {last_user}
Assistant: {last_assistant}
Generate three concise follow-up questions that a user might ask next.
Each question should be on a separate line. The generated questions should be independent and can be answered without knowing the last question. Focus on brevity.
Follow-up Questions:"""
    try:
        # Use ChatOpenAI as a fallback if the selected models because otherwise it will fail. e.g Gemma might not support invoking method.
        if any(model_type in st.session_state.selected_model.lower()
               for model_type in ["gemma2", "deepseek", "mixtral"]):
            fallback_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
            response = fallback_llm.invoke(prompt)
        else:
            response = st.session_state.llm.invoke(prompt)

        text = response.content if hasattr(response, "content") else str(response)
        questions = [q.strip() for q in text.split('\n') if q.strip()]
        return questions[:3]
    except Exception as e:
        print(f"Failed to generate follow-up questions: {e}")
        return []


def process_question(question, answer_style):
    """
    Process a question (typed or follow-up):
      1. Append as a user message.
      2. Run the RAG workflow (via app.stream) and stream the assistant's response.
         If streaming produces no content (or errors occur), a fallback non-streaming
         approach is attempted.
    """
    # 1) Add user question to the chat
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(f"**You:** {question}")

    # Redirect stdout for debugging
    output_buffer = io.StringIO()
    sys.stdout = output_buffer
    assistant_response = ""

    # 2) Initialize empty assistant message for streaming the response
    st.session_state.messages.append({"role": "assistant", "content": ""})
    assistant_index = len(st.session_state.messages) - 1

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        debug_placeholder = st.empty()
        # CallBack handler get_streamlit_cb
        st_callback = get_streamlit_cb(st.empty())

        start_time = time.time()

        with st.spinner("Thinking..."):
            inputs = {
                "question": question,
                "hybrid_search": st.session_state.hybrid_search,
                "internet_search": st.session_state.internet_search,
                "answer_style": answer_style
            }
            try:
                # Attempt to stream response
                for idx, chunk in enumerate(app.stream(inputs, config={"callbacks": [st_callback]})):
                    debug_logs = output_buffer.getvalue()
                    debug_placeholder.text_area(
                        "Debug Logs", debug_logs, height=100, key=f"debug_logs_{idx}"
                    )
                    if "generate" in chunk and "generation" in chunk["generate"]:
                        assistant_response += chunk["generate"]["generation"]
                        styled_response = re.sub(
                            r'\[(.*?)\]',
                            r'<span class="reference">[\1]</span>',
                            assistant_response
                        )
                        response_placeholder.markdown(
                            f"**Assistant:** {styled_response}",
                            unsafe_allow_html=True
                        )
            except (tornado.websocket.WebSocketClosedError, tornado.iostream.StreamClosedError) as ws_error:
                # Log and silently handle known WebSocket errors without showing a modal.
                print(f"WebSocket connection closed: {ws_error}")
            except Exception as e:
                error_str = str(e)
                # Filter out non-critical errors (like "Bad message format") from showing in the UI.
                if "Bad message format" in error_str:
                    print(f"Non-critical error: {error_str}")
                else:
                    error_msg = f"Error generating response: {error_str}"
                    response_placeholder.error(error_msg)
                    st_callback.text = error_msg

            # If no response was produced by streaming, attempt fallback using invoke
            if not assistant_response.strip():
                try:
                    result = app.invoke(inputs)
                    if "generate" in result and "generation" in result["generate"]:
                        assistant_response = result["generate"]["generation"]
                        styled_response = re.sub(
                            r'\[(.*?)\]',
                            r'<span class="reference">[\1]</span>',
                            assistant_response
                        )
                        response_placeholder.markdown(
                            f"**Assistant:** {styled_response}",
                            unsafe_allow_html=True
                        )
                    else:
                        raise ValueError("No generation found in result")
                except Exception as fallback_error:
                    fallback_str = str(fallback_error)
                    if "Bad message format" in fallback_str:
                        print(f"Non-critical fallback error: {fallback_str}")
                    else:
                        print(f"Fallback also failed: {fallback_str}")
                        if not assistant_response.strip():
                            error_msg = ("Sorry, I encountered an error while generating a response. "
                                         "Please try again or select a different model.")
                            response_placeholder.error(error_msg)
                            assistant_response = error_msg

        # End timer and calculate generation time
        end_time = time.time()
        generation_time = end_time - start_time
        st.session_state["last_generation_time"] = generation_time

        # Optionally display the generation time if the timer is toggled on
        if st.session_state.get("show_timer", True):
            response_placeholder.markdown(
                f"*Generation time: {generation_time:.2f} seconds*")

        # Restore original stdout
        sys.stdout = sys.__stdout__

    # 3) Update the assistant message with the final response
    st.session_state.messages[assistant_index]["content"] = assistant_response
    st.session_state.followup_key += 1

# -------------------- Country Selection Screen --------------------
# Show country selection if not already selected
if st.session_state.selected_country is None:
    st.set_page_config(page_title="Smart Guide", layout="centered", page_icon="🌍")
    
    # Add CSS for styling
    st.markdown("""
    <style>
    .flag-image {
        border: 2px solid #333;
        border-radius: 4px;
    }
    .country-code {
        font-size: 28px;
        font-weight: bold;
        text-align: center;
        margin-top: 15px;
        margin-bottom: 5px;
    }
    .country-name {
        font-size: 22px;
        text-align: center;
        margin-bottom: 10px;
    }
    .country-desc {
        text-align: center;
        margin-bottom: 15px;
    }
    .footer-text {
        text-align: left;
    }
    .check-item {
        display: flex;
        align-items: flex-start;
        margin-bottom: 8px;
        text-align: left;
    }
    .check-icon {
        color: #00c851;
        margin-right: 10px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Title and introduction
    st.markdown("<h1 style='text-align: center;'>🌍 Smart Guide</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Welcome! Choose a country to access tailored business information</p>", unsafe_allow_html=True)
    
    # Country selection cards
    col1, col2 = st.columns(2)
    
    with col1:
        # Finnish flag with border
        st.markdown('<div class="flag-image">', unsafe_allow_html=True)
        st.image("images/finland.png")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Country information
        st.markdown('<div class="country-code">FI</div>', unsafe_allow_html=True)
        st.markdown('<div class="country-name">Finland</div>', unsafe_allow_html=True)
        st.markdown('<div class="country-desc">Access comprehensive business guides and web resources</div>', unsafe_allow_html=True)
        
        # Button
        if st.button("Select Finland", key="finland_btn", use_container_width=True):
            st.session_state.selected_country = "Finland"
            # Reset search settings for Finland (set defaults)
            st.session_state.hybrid_search = True
            st.session_state.internet_search = False
            st.rerun()
    
    with col2:
        # Estonian flag with border
        st.markdown('<div class="flag-image">', unsafe_allow_html=True)
        st.image("images/estonia.jpg")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Country information
        st.markdown('<div class="country-code">EE</div>', unsafe_allow_html=True)
        st.markdown('<div class="country-name">Estonia</div>', unsafe_allow_html=True)
        st.markdown('<div class="country-desc">Get the latest information from trusted Estonian sources</div>', unsafe_allow_html=True)
        
        # Button
        if st.button("Select Estonia", key="estonia_btn", use_container_width=True):
            st.session_state.selected_country = "Estonia"
            # For Estonia, force internet search only
            st.session_state.hybrid_search = False
            st.session_state.internet_search = True
            st.rerun()
    
    # Footer with left-aligned text
    st.markdown("---")
    st.markdown("<h3 class='footer-text'>Why choose our Smart Guide?</h3>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="check-item">
        <span class="check-icon">✅</span>
        <span>AI-powered answers to your business questions</span>
    </div>
    <div class="check-item">
        <span class="check-icon">✅</span>
        <span>Country-specific information from reliable sources</span>
    </div>
    <div class="check-item">
        <span class="check-icon">✅</span>
        <span>Up-to-date guidance on entrepreneurship and regulations</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Exit early - don't load anything else until country is selected
    st.stop()

# -------------------- Page Layout & Configuration --------------------
st.set_page_config(
    page_title=f"Smart Guide - {st.session_state.selected_country}",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🧠"
)

st.markdown("""
    <style>
    .reference {
        color: blue;
        font-weight: bold;
    }
    .country-header {
        display: flex;
        align-items: center;
        margin-bottom: 20px;
    }
    .country-flag {
        font-size: 40px;
        margin-right: 15px;
    }
    .feature-card {
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 10px;
        background-color: #f0f2f6;
        border-left: 4px solid #4169e1;
    }
    .sidebar-header {
        display: flex;
        align-items: center;
        margin-bottom: 20px;
    }
    .sidebar-emoji {
        font-size: 28px;
        margin-right: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------- Sidebar --------------------
with st.sidebar:
    try:
        st.image("images/LOGO_UPBEAT.jpg", width=150, use_container_width=True)
    except Exception as e:
        # If logo isn't available, show a styled header instead
        st.markdown(
            f"""<div class='sidebar-header'>
                <div class='sidebar-emoji'>{"🇫🇮" if st.session_state.selected_country == "Finland" else "🇪🇪"}</div>
                <div>Smart Guide</div>
            </div>""", 
            unsafe_allow_html=True
        )

    st.title(f"🗣️ Smart Guide for {st.session_state.selected_country}")

    
    st.markdown("**▶️ Actions:**")

    # Set default model selections if not present.
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = "gpt-4o"
    if "selected_routing_model" not in st.session_state:
        st.session_state.selected_routing_model = "gpt-4o"
    if "selected_grading_model" not in st.session_state:
        st.session_state.selected_grading_model = "gpt-4o"
    if "selected_embedding_model" not in st.session_state:
        st.session_state.selected_embedding_model = "text-embedding-3-large"

    answer_style = st.select_slider(
            "💬 Answer Style",
            options=["Concise", "Moderate", "Explanatory"],
            value="Explanatory",
            key="answer_style_slider"
        )
    st.session_state.answer_style = answer_style

    # Country-specific search options
    if st.session_state.selected_country == "Estonia":
        # For Estonia, only show internet search option with styled info box
        st.session_state.hybrid_search = False
        st.session_state.internet_search = True
        
        st.markdown("""
        <div style="background-color: #e6f3ff; padding: 10px; border-radius: 5px; border-left: 4px solid #0072CE;">
            <h4 style="margin-top: 0;">🇪🇪 Estonia Mode</h4>
            <p>We're using real-time web search to provide you with the latest Estonian business information.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # For Finland, show all search options
        search_option = st.radio(
            "Search options",
            ["Reliable documents", "Reliable web sources", "Reliable docs & web sources"],
            index=2
        )
        st.session_state.hybrid_search = (search_option == "Reliable docs & web sources")
        st.session_state.internet_search = (search_option == "Reliable web sources")


    col1, col2 = st.columns(2)
    
    # Add custom CSS for consistent button styling
    st.markdown("""
    <style>
        .sidebar-button {
            width: 100%;
            height: 46px;
            border-radius: 4px;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            text-decoration: none;
            font-weight: 500;
            color: #ffffff;
            background-color: #262730;
            border: 1px solid #4B5563;
            cursor: pointer;
            padding: 0.5rem 1rem;
            transition: all 0.2s ease;
        }
        .sidebar-button:hover {
            background-color: #36373F;
            border-color: #6B7280;
        }
        .sidebar-button-icon {
            margin-right: 8px;
        }
    </style>
    """, unsafe_allow_html=True)

    # Action buttons with consistent styling
    if st.button("🔄 Reset Chat", key="reset_button", use_container_width=True, 
                help="Clear the current conversation and start fresh"):
        st.session_state.messages = []

    if st.button("🌍 Change Country", key="change_country", use_container_width=True,
                help="Switch between Finland and Estonia"):
        # Reset ALL session state variables
        for key in list(st.session_state.keys()):
            # Keep only minimal state needed to show the country selection screen
            if key not in ["_is_running", "_script_run_ctx"]:
                del st.session_state[key]
        
        # Set selected_country to None to show country selection screen
        st.session_state.selected_country = None
        
        # Force complete reinitialization by rerunning
        st.rerun()

    # Add a separator for the manual download button
    st.markdown("---")

    # Add manual download button header
    st.markdown("**📚 Documentation:**")

    # Create download button for the manual PDF
    try:
        with open("manual.pdf", "rb") as pdf_file:
            pdf_bytes = pdf_file.read()
            
        st.download_button(
            label="📄 Download Manual",
            data=pdf_bytes,
            file_name="Smart_Guide_System_Manual.pdf",
            mime="application/pdf",
            help="Download the complete user manual as PDF",
            key="manual_download",
            use_container_width=True
        )
    except FileNotFoundError:
        st.error("Manual PDF not found. Please contact support.")
    except Exception as e:
        st.error(f"Error loading manual: {str(e)}")
    # Toggle for displaying generation time
    st.checkbox("Show generation time", value=True, key="show_timer")
    
    # RAG workflow initilizate.
    try:
        app = initialize_app(
            st.session_state.selected_model,
            st.session_state.selected_embedding_model,
            st.session_state.selected_routing_model,
            st.session_state.selected_grading_model,
            st.session_state.hybrid_search,
            st.session_state.internet_search,
            st.session_state.answer_style
        )
    except Exception as e:
        st.error("Error initializing model, continuing with previous model: " + str(e))
        # Initialize a fallback LLM for follow-up questions
        st.session_state.llm = ChatOpenAI(model="gpt-4o", temperature=0.5)

# -------------------- Main Title & Introduction --------------------
# flag_emoji = "🇫🇮" if st.session_state.selected_country == "Finland" else "🇪🇪"

# # Title with flag emoji
# st.title(f"{flag_emoji} Smart Guide for Entrepreneurship and Business Planning in {st.session_state.selected_country}")

# # Display guidelines if toggle is on (for either country)
# if st.session_state.show_guidelines:
#     # Show guidelines in an expander
#     with st.expander(f"📖 Smart Guide Help & Guidelines for {st.session_state.selected_country}", expanded=True):
#         if st.session_state.selected_country == "Finland":
#             show_finland_guidelines()
#         else:  # Estonia
#             show_estonia_guidelines()
            
#         if st.button("Close", key="close_guidelines"):
#             st.session_state.show_guidelines = False
#             st.rerun()


###############################
# In the main section after the title
flag_emoji = "🇫🇮" if st.session_state.selected_country == "Finland" else "🇪🇪"

# Title with flag emoji
st.title(f"{flag_emoji} Smart Guide for Entrepreneurship and Business Planning in {st.session_state.selected_country}")

# Display guidelines as expandable sections
st.subheader("📖 Smart Guide Help & Resources")

# Section 1: How it works
with st.expander("🔍 How It Works", expanded=False):
    st.write(f"Smart Guide combines AI with trusted {st.session_state.selected_country} business information to answer your entrepreneurship questions accurately and efficiently.")
    
    with st.container():
        st.markdown("### 🌟 Key Features")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"- **{st.session_state.selected_country}-Specific Answers:** Information tailored to {st.session_state.selected_country}'s business environment")
            if st.session_state.selected_country == "Finland":
                st.markdown("- **Multiple Search Options:** Choose between reliable documents, web sources, or both")
            else:
                st.markdown("- **Web Search:** Real-time information from official Estonian websites")
        with col2:
            st.markdown("- **Adjustable Detail Level:** Select Concise, Moderate, or Explanatory responses")
            st.markdown("- **Follow-up Questions:** Automatically generated based on your conversation")

# Section 2: Information Sources
with st.expander("🧠 Information Sources", expanded=False):
    if st.session_state.selected_country == "Finland":
        tab1, tab2 = st.tabs(["📚 Document Sources", "🌐 Web Sources"])
        
        with tab1:
            st.write("Comprehensive business guides about starting and running companies in Finland, including taxation, permits, and legal requirements.")
            
            option1, option2, option3 = st.columns(3)
                
            with option1:
                st.markdown("### 📄 Reliable Documents")
                st.markdown("Curated business guides with comprehensive information about entrepreneurship in Finland.")
                st.markdown("**Best for:** Foundational knowledge and established procedures.")
            
            with option2:
                st.markdown("### 🌐 Reliable Web Sources")
                st.markdown("Trusted Finnish government and business websites to find the most current and reliable information.")
                st.markdown("**Best for:** Recent updates, changes to regulations, or specific programs.")
            
            with option3:
                st.markdown("### 🔄 Reliable Docs & Web Sources")
                st.markdown("Combines both reliable document and web searches for the most comprehensive results, clearly indicating which information comes from each source.")
                st.markdown("**Best for:** Most questions, giving you both established knowledge and recent updates.")
        
        with tab2:
            st.write("Real-time information from trusted Finnish official websites, including:")
            
            # Create a grid of domains using columns
            domains = [
                "migri.fi", "enterfinland.fi", "businessfinland.fi", "kela.fi",
                "vero.fi", "suomi.fi", "valvira.fi", "finlex.fi",
                "hus.fi", "lvm.fi", "thefinlandbusinesspress.fi", "infofinland.fi",
                "ely-keskus.fi", "yritystulkki.fi", "tem.fi", "prh.fi",
                "avi.fi", "ruokavirasto.fi", "traficom.fi", "trade.gov",
                "finlex.fi", "te-palvelut.fi", "tilastokeskus.fi", "veronmaksajat.fi",
                "hel.fi", "ukko.fi", "yrityssalo.fi", "stm.fi",
                "eurofound.europa.eu", "oph.fi", "oikeusrekisterikeskus.fi"
            ]
            
            # Display domains in a multi-column layout
            cols = st.columns(4)
            for i, domain in enumerate(domains):
                cols[i % 4].markdown(f"- `{domain}`")
            
            # Add the reliable web sources explanation here
            st.markdown("### 🌐 Reliable Web Sources")
            st.markdown("Performs internet searches on trusted Finnish government and business websites to find the most current information.")
            st.markdown("**Best for:** Recent updates, changes to regulations, or specific programs.")
    else:  # Estonia
        st.write("Real-time information from trusted Estonian official websites, including:")
        
        # Create a grid of domains using columns
        domains = [
            "eesti.ee", "e-resident.gov.ee", "investinestonia.com", "mkm.ee",
            "tallinn.ee", "ebs.ee", "emta.ee", "learn.e-resident.gov.ee",
            "fi.ee", "riigiteataja.ee", "ttja.ee", "stat.ee",
            "ariregister.rik.ee", "tradewithestonia.com", "kul.ee", "pta.agri.ee",
            "terviseamet.ee"
        ]
        
        # Display domains in a multi-column layout
        cols = st.columns(4)
        for i, domain in enumerate(domains):
            cols[i % 4].markdown(f"- `{domain}`")
        
        # Add the reliable web sources explanation for Estonia
        st.markdown("### 🌐 Reliable Web Sources")
        st.markdown("Performs internet searches on trusted Estonian government and business websites to find the most current information.")
        st.markdown("**Best for:** Recent regulatory updates, changes to programs, and current information.")

# Section 3: Search Options (different per country)
with st.expander("🔎 Search Options Explained", expanded=False):
    if st.session_state.selected_country == "Finland":
        # Create tabs for different types of guidelines
        search_tab1, search_tab2 = st.tabs(["Search Mode Guidelines", "Answer Style Guidelines"])
        
        with search_tab1:
            st.markdown("### How to Choose the Right Search Mode")
            
            st.markdown("""
            • **Reliable Documents:** Best for foundational knowledge, established procedures, and core business regulations that don't change frequently
            
            • **Reliable Web Sources:** Ideal for recent updates, current tax rates, licensing fees, and time-sensitive information
            
            • **Reliable Docs & Web Sources:** Recommended for comprehensive research, comparing information from multiple sources, or when unsure which mode is best
            """)
            
            st.info("You can select your preferred search option in the sidebar under 'Search options'")
        
        with search_tab2:
            st.markdown("### How to Choose the Right Answer Style")
            
            st.markdown("""
            • **Concise:** Short, direct answers with essential information only – perfect when you need quick facts or have limited time
            
            • **Moderate:** Balanced explanations with key details and context – good for most questions and everyday use
            
            • **Explanatory:** In-depth responses with comprehensive information, examples, and relevant details – ideal for complex topics or deep understanding
            """)
            
            st.info("You can adjust the answer style in the sidebar using the slider")
            
    else:  # Estonia
        # Create tabs for Estonia
        search_tab1, search_tab2 = st.tabs(["Search Mode Guidelines", "Answer Style Guidelines"])
        
        with search_tab1:
            st.markdown("### Estonia Web Search")
            
            st.markdown("""
            • **Real-time Web Search:** The Estonia Smart Guide uses real-time web search from trusted Estonian government and business websites
            
            • **Reliable Sources:** Information comes from official Estonian government platforms, business resources, and e-Residency websites
            
            • **Up-to-date Information:** Get the latest regulations, tax rates, and business procedures for the Estonian market
            """)
            
            # Estonia-specific information
            st.subheader("🇪🇪 About Estonia's Business Environment")
            
            col1, col2 = st.columns(2)
            with col1:
                st.info("**Digital Nation:** Estonia is known for its advanced digital ecosystem, allowing most business operations to be conducted online.")
            with col2:
                st.info("**E-Residency:** Estonia offers a digital identity that allows entrepreneurs worldwide to establish and manage an Estonian-based business online.")
        
        with search_tab2:
            st.markdown("### How to Choose the Right Answer Style")
            
            st.markdown("""
            • **Concise:** Short, direct answers with essential information only – perfect when you need quick facts or have limited time
            
            • **Moderate:** Balanced explanations with key details and context – good for most questions and everyday use
            
            • **Explanatory:** In-depth responses with comprehensive information, examples, and relevant details – ideal for complex topics or deep understanding
            """)
            
            st.info("You can adjust the answer style in the sidebar using the slider")

# Section 4: Tips for Best Results
with st.expander("💡 Tips for Best Results", expanded=False):
    tip1, tip2, tip3 = st.columns(3)
    
    with tip1:
        st.info("**Be specific** with your questions for more precise answers")
    
    with tip2:
        st.info("**Adjust the answer style** in the sidebar to control detail level")
    
    with tip3:
        if st.session_state.selected_country == "Finland":
            st.info("**Try rephrasing your question** if you don't get the information you need or want a more accurate answer.")
        else:  # Estonia
            st.info("**Ask about e-Residency** if you're interested in starting a business remotely")
#########################

# Welcome message
st.header(f"Welcome to your {st.session_state.selected_country} Smart Guide!")

# What I can help with section
st.markdown("📝 **What I can help you with:**")

# Use Streamlit's native container without any custom HTML/CSS
with st.container():
    if st.session_state.selected_country == "Finland":
        st.markdown("""
        <ul style="list-style-position: inside; text-align: left; display: inline-block;">
            <li> 📊 Finding answers from business and entrepreneurship guides in Finland </li>
            <li> 🔍 Providing up-to-date information via AI-based internet search </li>
            <li> 💼 Tax-related information, permits, registrations, and more </li>
            <li> 🚀 Business setup processes specific to Finland </li>
        </ul>
        """, unsafe_allow_html=True)
    else:  # Estonia
        st.markdown("""
            <ul style="list-style-position: inside; text-align: left; display: inline-block;">
                <li> 🔍 Providing up-to-date information via AI-based internet search </li>
                <li> 💼 Tax-related information, permits, registrations, and more </li>
                <li> 🚀 Business setup processes specific to Estonia </li>
            </ul>
        """, unsafe_allow_html=True)

# Pro tip
st.markdown("💡 **Pro tip:** Ask specific questions for the most accurate answers!")
st.markdown("**Start by typing your question in the chat below!**")

# Sample question suggestions
st.subheader("Try asking:")

if st.session_state.selected_country == "Finland":
    sample_questions = [
        "How do I register a company in Finland?",
        "What taxes do entrepreneurs pay in Finland?",
        "What are the requirements for a foreigner to start a business in Finland?"
    ]
else:  # Estonia
    sample_questions = [
        "How do I register a company in Estonia?",
        "What is e-Residency in Estonia?",
        "What taxes do entrepreneurs pay in Estonia?"
    ]

# Create clickable sample questions
cols = st.columns(3)
for i, question in enumerate(sample_questions):
    if cols[i].button(f"💬 {question}", key=f"sample_q_{i}", use_container_width=True):
        st.session_state.pending_followup = question
        st.rerun()

# -------------------- Display Conversation History --------------------
for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.markdown(f"**You:** {message['content']}")
    elif message["role"] == "assistant":
        with st.chat_message("assistant"):
            # Process the response to add styling to references
            import re
            styled_response = re.sub(
                r'\[(.*?)\]',
                r'<span class="reference">[\1]</span>',
                message['content']
            )
            st.markdown(
                f"**Assistant:** {styled_response}",
                unsafe_allow_html=True
            )

# Display the last generation time outside the chat messages if enabled.
if st.session_state.get("show_timer", True) and "last_generation_time" in st.session_state:
    st.markdown(
        f"<small>Last Generation Time: {st.session_state.last_generation_time:.2f} seconds</small>", unsafe_allow_html=True)

# -------------------- Process a Pending Follow-Up (if any) --------------------
if st.session_state.pending_followup is not None:
    question = st.session_state.pending_followup
    st.session_state.pending_followup = None
    process_question(question, st.session_state.answer_style)

# -------------------- Process New User Input --------------------
user_input = st.chat_input("Type your question (Max. 200 char):")
if user_input:
    if len(user_input) > 200:
        st.error(
            "Your question exceeds 200 characters. Please shorten it and try again.")
    else:
        process_question(user_input, st.session_state.answer_style)

# -------------------- Helper function for Follow-Up --------------------
def handle_followup(question: str):
    st.session_state.pending_followup = question

# -------------------- Generate and Display Follow-Up Questions --------------------
if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
    try:
        last_assistant_message = st.session_state.messages[-1]["content"]

        # Don't generate followup questions if response is empty or contains error messages
        if not last_assistant_message.strip() or "Sorry, I encountered an error" in last_assistant_message:
            st.session_state.followup_questions = []
        # Don't generate followup questions for unrelated responses
        elif "I apologize, but I'm designed to answer questions" in last_assistant_message:
            st.session_state.followup_questions = []
        else:
            # Get the last user message
            last_user_message = next(
                (msg["content"] for msg in reversed(st.session_state.messages)
                 if msg["role"] == "user"),
                ""
            )

            # Generate new questions only if the last assistant message has changed
            if st.session_state.last_assistant != last_assistant_message:
                print("Generating new followup questions")
                st.session_state.last_assistant = last_assistant_message
                try:
                    st.session_state.followup_questions = get_followup_questions(
                        last_user_message,
                        last_assistant_message
                    )
                except Exception as e:
                    print(f"Failed to generate followup questions: {e}")
                    st.session_state.followup_questions = []

        # Display follow-up questions only if we have valid ones
        if st.session_state.followup_questions and len(st.session_state.followup_questions) > 0:
            st.markdown("#### Related Questions:")
            cols = st.columns(len(st.session_state.followup_questions))
            
            for i, question in enumerate(st.session_state.followup_questions):
                # Remove numbering e.g "1. ", "2. ", etc.
                clean_question = re.sub(r'^\d+\.\s*', '', question)
                with cols[i]:
                    if st.button(
                        f"💬 {clean_question}",
                        key=f"followup_{i}_{st.session_state.followup_key}",
                        use_container_width=True
                    ):
                        handle_followup(clean_question)
                        st.rerun()
    except Exception as e:
        print(f"Error in followup section: {e}")
        st.session_state.followup_questions = []

# Footer with attribution
st.markdown("---")
st.markdown(
    f"""
    <div style="text-align: center; color: gray; font-size: 12px;">
        <p>Smart Guide for {st.session_state.selected_country} | Powered by AI | &copy; 2025</p>
    </div>
    """,
    unsafe_allow_html=True
)






















# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# import io
# import re
# import sys
# import time

# import streamlit as st
# import torch
# import tornado
# from langchain_openai import ChatOpenAI

# from agentic_rag import initialize_app
# from st_callback import get_streamlit_cb

# # This code line below Fixes console "RuntimeError: Tried to instantiate class '__path__._path', but it does not exist!"
# # reference: https://github.com/VikParuchuri/marker/issues/442#issuecomment-2636393925
# torch.classes.__path__ = []

# # -------------------- Initialization --------------------
# st.set_option("client.showErrorDetails", False)  # Hide error detail

# # Add country selection at startup
# if "selected_country" not in st.session_state:
#     st.session_state.selected_country = None

# # Early session state initialization
# if "messages" not in st.session_state:
#     st.session_state.messages = []
# if "followup_key" not in st.session_state:
#     st.session_state.followup_key = 0
# if "pending_followup" not in st.session_state:
#     st.session_state.pending_followup = None
# if "last_assistant" not in st.session_state:
#     st.session_state.last_assistant = None
# if "followup_questions" not in st.session_state:
#     st.session_state.followup_questions = []

# # -------------------- Country Selection Screen --------------------
# # Show country selection if not already selected
# if st.session_state.selected_country is None:
#     st.set_page_config(page_title="Smart Guide", layout="centered", page_icon="🌍")
    
#     # Add CSS for styling
#     st.markdown("""
#     <style>
#     .flag-image {
#         border: 2px solid #333;
#         border-radius: 4px;
#     }
#     .country-code {
#         font-size: 28px;
#         font-weight: bold;
#         text-align: center;
#         margin-top: 15px;
#         margin-bottom: 5px;
#     }
#     .country-name {
#         font-size: 22px;
#         text-align: center;
#         margin-bottom: 10px;
#     }
#     .country-desc {
#         text-align: center;
#         margin-bottom: 15px;
#     }
#     .footer-text {
#         text-align: left;
#     }
#     .check-item {
#         display: flex;
#         align-items: flex-start;
#         margin-bottom: 8px;
#         text-align: left;
#     }
#     .check-icon {
#         color: #00c851;
#         margin-right: 10px;
#     }
#     </style>
#     """, unsafe_allow_html=True)
    
#     # Title and introduction
#     st.markdown("<h1 style='text-align: center;'>🌍 Smart Guide</h1>", unsafe_allow_html=True)
#     st.markdown("<p style='text-align: center;'>Welcome! Choose a country to access tailored business information</p>", unsafe_allow_html=True)
    
#     # Country selection cards
#     col1, col2 = st.columns(2)
    
#     with col1:
#         # Finnish flag with border
#         st.markdown('<div class="flag-image">', unsafe_allow_html=True)
#         st.image("images/finland.png")
#         st.markdown('</div>', unsafe_allow_html=True)
        
#         # Country information
#         st.markdown('<div class="country-code">FI</div>', unsafe_allow_html=True)
#         st.markdown('<div class="country-name">Finland</div>', unsafe_allow_html=True)
#         st.markdown('<div class="country-desc">Access comprehensive business guides and web resources</div>', unsafe_allow_html=True)
        
#         # Button
#         if st.button("Select Finland", key="finland_btn", use_container_width=True):
#             st.session_state.selected_country = "Finland"
#             # Reset search settings for Finland (set defaults)
#             st.session_state.hybrid_search = True
#             st.session_state.internet_search = False
#             st.rerun()
    
#     with col2:
#         # Estonian flag with border
#         st.markdown('<div class="flag-image">', unsafe_allow_html=True)
#         st.image("images/estonia.jpg")
#         st.markdown('</div>', unsafe_allow_html=True)
        
#         # Country information
#         st.markdown('<div class="country-code">EE</div>', unsafe_allow_html=True)
#         st.markdown('<div class="country-name">Estonia</div>', unsafe_allow_html=True)
#         st.markdown('<div class="country-desc">Get the latest information from trusted Estonian sources</div>', unsafe_allow_html=True)
        
#         # Button
#         if st.button("Select Estonia", key="estonia_btn", use_container_width=True):
#             st.session_state.selected_country = "Estonia"
#             # For Estonia, force internet search only
#             st.session_state.hybrid_search = False
#             st.session_state.internet_search = True
#             st.rerun()
    
#     # Footer with left-aligned text
#     st.markdown("---")
#     st.markdown("<h3 class='footer-text'>Why choose our Smart Guide?</h3>", unsafe_allow_html=True)
    
#     st.markdown("""
#     <div class="check-item">
#         <span class="check-icon">✅</span>
#         <span>AI-powered answers to your business questions</span>
#     </div>
#     <div class="check-item">
#         <span class="check-icon">✅</span>
#         <span>Country-specific information from reliable sources</span>
#     </div>
#     <div class="check-item">
#         <span class="check-icon">✅</span>
#         <span>Up-to-date guidance on entrepreneurship and regulations</span>
#     </div>
#     """, unsafe_allow_html=True)
    
#     # Exit early - don't load anything else until country is selected
#     st.stop()

# # -------------------- Helper Functions --------------------
# def get_followup_questions(last_user, last_assistant):
#     """
#     Generate three concise follow-up questions dynamically based on the latest conversation.
#     """
#     prompt = f"""Based on the conversation below:
# User: {last_user}
# Assistant: {last_assistant}
# Generate three concise follow-up questions that a user might ask next.
# Each question should be on a separate line. The generated questions should be independent and can be answered without knowing the last question. Focus on brevity.
# Follow-up Questions:"""
#     try:
#         # Use ChatOpenAI as a fallback if the selected models because otherwise it will fail. e.g Gemma might not support invoking method.
#         if any(model_type in st.session_state.selected_model.lower()
#                for model_type in ["gemma2", "deepseek", "mixtral"]):
#             fallback_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
#             response = fallback_llm.invoke(prompt)
#         else:
#             response = st.session_state.llm.invoke(prompt)

#         text = response.content if hasattr(response, "content") else str(response)
#         questions = [q.strip() for q in text.split('\n') if q.strip()]
#         return questions[:3]
#     except Exception as e:
#         print(f"Failed to generate follow-up questions: {e}")
#         return []


# def process_question(question, answer_style):
#     """
#     Process a question (typed or follow-up):
#       1. Append as a user message.
#       2. Run the RAG workflow (via app.stream) and stream the assistant's response.
#          If streaming produces no content (or errors occur), a fallback non-streaming
#          approach is attempted.
#     """
#     # 1) Add user question to the chat
#     st.session_state.messages.append({"role": "user", "content": question})
#     with st.chat_message("user"):
#         st.markdown(f"**You:** {question}")

#     # Redirect stdout for debugging
#     output_buffer = io.StringIO()
#     sys.stdout = output_buffer
#     assistant_response = ""

#     # 2) Initialize empty assistant message for streaming the response
#     st.session_state.messages.append({"role": "assistant", "content": ""})
#     assistant_index = len(st.session_state.messages) - 1

#     with st.chat_message("assistant"):
#         response_placeholder = st.empty()
#         debug_placeholder = st.empty()
#         # CallBack handler get_streamlit_cb
#         st_callback = get_streamlit_cb(st.empty())

#         start_time = time.time()

#         with st.spinner("Thinking..."):
#             inputs = {
#                 "question": question,
#                 "hybrid_search": st.session_state.hybrid_search,
#                 "internet_search": st.session_state.internet_search,
#                 "answer_style": answer_style
#             }
#             try:
#                 # Attempt to stream response
#                 for idx, chunk in enumerate(app.stream(inputs, config={"callbacks": [st_callback]})):
#                     debug_logs = output_buffer.getvalue()
#                     debug_placeholder.text_area(
#                         "Debug Logs", debug_logs, height=100, key=f"debug_logs_{idx}"
#                     )
#                     if "generate" in chunk and "generation" in chunk["generate"]:
#                         assistant_response += chunk["generate"]["generation"]
#                         styled_response = re.sub(
#                             r'\[(.*?)\]',
#                             r'<span class="reference">[\1]</span>',
#                             assistant_response
#                         )
#                         response_placeholder.markdown(
#                             f"**Assistant:** {styled_response}",
#                             unsafe_allow_html=True
#                         )
#             except (tornado.websocket.WebSocketClosedError, tornado.iostream.StreamClosedError) as ws_error:
#                 # Log and silently handle known WebSocket errors without showing a modal.
#                 print(f"WebSocket connection closed: {ws_error}")
#             except Exception as e:
#                 error_str = str(e)
#                 # Filter out non-critical errors (like "Bad message format") from showing in the UI.
#                 if "Bad message format" in error_str:
#                     print(f"Non-critical error: {error_str}")
#                 else:
#                     error_msg = f"Error generating response: {error_str}"
#                     response_placeholder.error(error_msg)
#                     st_callback.text = error_msg

#             # If no response was produced by streaming, attempt fallback using invoke
#             if not assistant_response.strip():
#                 try:
#                     result = app.invoke(inputs)
#                     if "generate" in result and "generation" in result["generate"]:
#                         assistant_response = result["generate"]["generation"]
#                         styled_response = re.sub(
#                             r'\[(.*?)\]',
#                             r'<span class="reference">[\1]</span>',
#                             assistant_response
#                         )
#                         response_placeholder.markdown(
#                             f"**Assistant:** {styled_response}",
#                             unsafe_allow_html=True
#                         )
#                     else:
#                         raise ValueError("No generation found in result")
#                 except Exception as fallback_error:
#                     fallback_str = str(fallback_error)
#                     if "Bad message format" in fallback_str:
#                         print(f"Non-critical fallback error: {fallback_str}")
#                     else:
#                         print(f"Fallback also failed: {fallback_str}")
#                         if not assistant_response.strip():
#                             error_msg = ("Sorry, I encountered an error while generating a response. "
#                                          "Please try again or select a different model.")
#                             response_placeholder.error(error_msg)
#                             assistant_response = error_msg

#         # End timer and calculate generation time
#         end_time = time.time()
#         generation_time = end_time - start_time
#         st.session_state["last_generation_time"] = generation_time

#         # Optionally display the generation time if the timer is toggled on
#         if st.session_state.get("show_timer", True):
#             response_placeholder.markdown(
#                 f"*Generation time: {generation_time:.2f} seconds*")

#         # Restore original stdout
#         sys.stdout = sys.__stdout__

#     # 3) Update the assistant message with the final response
#     st.session_state.messages[assistant_index]["content"] = assistant_response
#     st.session_state.followup_key += 1


# # -------------------- Page Layout & Configuration --------------------
# st.set_page_config(
#     page_title=f"Smart Guide - {st.session_state.selected_country}",
#     layout="wide",
#     initial_sidebar_state="expanded",
#     page_icon="🧠"
# )

# st.markdown("""
#     <style>
#     .reference {
#         color: blue;
#         font-weight: bold;
#     }
#     .country-header {
#         display: flex;
#         align-items: center;
#         margin-bottom: 20px;
#     }
#     .country-flag {
#         font-size: 40px;
#         margin-right: 15px;
#     }
#     .feature-card {
#         border-radius: 5px;
#         padding: 10px;
#         margin-bottom: 10px;
#         background-color: #f0f2f6;
#         border-left: 4px solid #4169e1;
#     }
#     .sidebar-header {
#         display: flex;
#         align-items: center;
#         margin-bottom: 20px;
#     }
#     .sidebar-emoji {
#         font-size: 28px;
#         margin-right: 10px;
#     }
#     </style>
# """, unsafe_allow_html=True)

# # -------------------- Sidebar --------------------
# with st.sidebar:
#     try:
#         st.image("images/LOGO_UPBEAT.jpg", width=150, use_container_width=True)
#     except Exception as e:
#         # If logo isn't available, show a styled header instead
#         st.markdown(
#             f"""<div class='sidebar-header'>
#                 <div class='sidebar-emoji'>{"🇫🇮" if st.session_state.selected_country == "Finland" else "🇪🇪"}</div>
#                 <div>Smart Guide</div>
#             </div>""", 
#             unsafe_allow_html=True
#         )

#     st.title(f"🗣️ Smart Guide for {st.session_state.selected_country}")
#     st.markdown("**▶️ Actions:**")

#     # Set default model selections if not present.
#     if "selected_model" not in st.session_state:
#         st.session_state.selected_model = "gpt-4o"
#     if "selected_routing_model" not in st.session_state:
#         st.session_state.selected_routing_model = "gpt-4o"
#     if "selected_grading_model" not in st.session_state:
#         st.session_state.selected_grading_model = "gpt-4o"
#     if "selected_embedding_model" not in st.session_state:
#         st.session_state.selected_embedding_model = "text-embedding-3-large"

#     model_list = [
#         "llama-3.1-8b-instant",
#         "llama-3.3-70b-versatile",
#         "llama3-70b-8192",
#         "llama3-8b-8192",
#         "mixtral-8x7b-32768",
#         "gemma2-9b-it",
#         "gpt-4o-mini",
#         "gpt-4o",
#         "deepseek-r1-distill-llama-70b"
#     ]
#     embed_list = [
#         "text-embedding-3-large",
#         "sentence-transformers/all-MiniLM-L6-v2"
#     ]

#     answer_style = st.select_slider(
#             "💬 Answer Style",
#             options=["Concise", "Moderate", "Explanatory"],
#             value="Explanatory",
#             key="answer_style_slider"
#         )
#     st.session_state.answer_style = answer_style

#     # Country-specific search options
#     if st.session_state.selected_country == "Estonia":
#         # For Estonia, only show internet search option with styled info box
#         st.session_state.hybrid_search = False
#         st.session_state.internet_search = True
        
#         st.markdown("""
#         <div style="background-color: #e6f3ff; padding: 10px; border-radius: 5px; border-left: 4px solid #0072CE;">
#             <h4 style="margin-top: 0;">🇪🇪 Estonia Mode</h4>
#             <p>We're using real-time web search to provide you with the latest Estonian business information.</p>
#         </div>
#         """, unsafe_allow_html=True)
#     else:
#         # For Finland, show all search options
#         search_option = st.radio(
#             "Search options",
#             ["Reliable documents", "Reliable web sources", "Reliable docs & web sources"],
#             index=2
#         )
#         st.session_state.hybrid_search = (search_option == "Reliable docs & web sources")
#         st.session_state.internet_search = (search_option == "Reliable web sources")

#     ############################# hard-coded selected model without the option to select from the dropdown menu.
#     if 'selected_model' not in st.session_state:
#         st.session_state.selected_model = "gpt-4o"  
    
#     if 'selected_routing_model' not in st.session_state:
#         st.session_state.selected_routing_model = "gpt-4o"  
        
#     if 'selected_grading_model' not in st.session_state:
#         st.session_state.selected_grading_model = "gpt-4o"  
        
#     if 'selected_embedding_model' not in st.session_state:
#         st.session_state.selected_embedding_model = "text-embedding-3-large"  
#     ############################

#     col1, col2 = st.columns(2)
    
#     with col1:
#         if st.button("🔄 Reset Chat", key="reset_button", use_container_width=True):
#             st.session_state.messages = []
    
#     # In the "Change Country" button section
#     with col2:
#         if st.button("🌍 Change Country", key="change_country", use_container_width=True):
#             # Reset ALL session state variables
#             for key in list(st.session_state.keys()):
#                 # Keep only minimal state needed to show the country selection screen
#                 if key not in ["_is_running", "_script_run_ctx"]:
#                     del st.session_state[key]
            
#             # Set selected_country to None to show country selection screen
#             st.session_state.selected_country = None
            
#             # Force complete reinitialization by rerunning
#             st.rerun()

#     # Toggle for displaying generation time
#     st.checkbox("Show generation time", value=True, key="show_timer")
    
#     # RAG workflow initilizate.
#     try:
#         app = initialize_app(
#             st.session_state.selected_model,
#             st.session_state.selected_embedding_model,
#             st.session_state.selected_routing_model,
#             st.session_state.selected_grading_model,
#             st.session_state.hybrid_search,
#             st.session_state.internet_search,
#             st.session_state.answer_style
#         )
#     except Exception as e:
#         st.error("Error initializing model, continuing with previous model: " + str(e))
#         # (Optional) Initialize your primary LLM if needed.
#         # st.session_state.llm = ChatOpenAI(model="gpt-4o", temperature=0.5)

# # -------------------- Main Title & Introduction --------------------
# flag_emoji = "🇫🇮" if st.session_state.selected_country == "Finland" else "🇪🇪"

# # Title with flag emoji
# st.title(f"{flag_emoji} Smart Guide for Entrepreneurship and Business Planning in {st.session_state.selected_country}")

# # Welcome message
# st.header(f"Welcome to your {st.session_state.selected_country} Smart Guide!")

# # What I can help with section
# st.markdown("📝 **What I can help you with:**")

# # Use Streamlit's native container without any custom HTML/CSS
# with st.container():
#     if st.session_state.selected_country == "Finland":
#         st.markdown("""
#         <ul style="list-style-position: inside; text-align: left; display: inline-block;">
#             <li> 📊 Finding answers from business and entrepreneurship guides in Finland </li>
#             <li> 🔍 Providing up-to-date information via AI-based internet search </li>
#             <li> 💼 Tax-related information, permits, registrations, and more </li>
#             <li> 🚀 Business setup processes specific to Finland </li>
#         </ul>
#         """, unsafe_allow_html=True)
#     else:  # Estonia
#         st.markdown("""
#             <ul style="list-style-position: inside; text-align: left; display: inline-block;">
#                 <li> 🔍 Providing up-to-date information via AI-based internet search </li>
#                 <li> 💼 Tax-related information, permits, registrations, and more </li>
#                 <li> 🚀 Business setup processes specific to Estonia </li>
#             </ul>
#         """, unsafe_allow_html=True)

# # Pro tip
# st.markdown("💡 **Pro tip:** Ask specific questions for the most accurate answers!")
# st.markdown("**Start by typing your question in the chat below!**")

# # Sample question suggestions
# st.subheader("Try asking:")

# if st.session_state.selected_country == "Finland":
#     sample_questions = [
#         "How do I register a company in Finland?",
#         "What taxes do entrepreneurs pay in Finland?",
#         "What are the requirements for a foreigner to start a business in Finland?"
#     ]
# else:  # Estonia
#     sample_questions = [
#         "How do I register a company in Estonia?",
#         "What is e-Residency in Estonia?",
#         "What taxes do entrepreneurs pay in Estonia?"
#     ]

# # Create clickable sample questions
# cols = st.columns(3)
# for i, question in enumerate(sample_questions):
#     if cols[i].button(f"💬 {question}", key=f"sample_q_{i}", use_container_width=True):
#         st.session_state.pending_followup = question
#         st.rerun()

# # -------------------- Display Conversation History --------------------
# for message in st.session_state.messages:
#     if message["role"] == "user":
#         with st.chat_message("user"):
#             st.markdown(f"**You:** {message['content']}")
#     elif message["role"] == "assistant":
#         with st.chat_message("assistant"):
#             # Process the response to add styling to references
#             import re
#             styled_response = re.sub(
#                 r'\[(.*?)\]',
#                 r'<span class="reference">[\1]</span>',
#                 message['content']
#             )
#             st.markdown(
#                 f"**Assistant:** {styled_response}",
#                 unsafe_allow_html=True
#             )

# # Display the last generation time outside the chat messages if enabled.
# if st.session_state.get("show_timer", True) and "last_generation_time" in st.session_state:
#     st.markdown(
#         f"<small>Last Generation Time: {st.session_state.last_generation_time:.2f} seconds</small>", unsafe_allow_html=True)

# # -------------------- Process a Pending Follow-Up (if any) --------------------
# if st.session_state.pending_followup is not None:
#     question = st.session_state.pending_followup
#     st.session_state.pending_followup = None
#     process_question(question, st.session_state.answer_style)

# # -------------------- Process New User Input --------------------
# user_input = st.chat_input("Type your question (Max. 200 char):")
# if user_input:
#     if len(user_input) > 200:
#         st.error(
#             "Your question exceeds 200 characters. Please shorten it and try again.")
#     else:
#         process_question(user_input, st.session_state.answer_style)

# # -------------------- Helper function for Follow-Up --------------------
# def handle_followup(question: str):
#     st.session_state.pending_followup = question

# # -------------------- Generate and Display Follow-Up Questions --------------------
# if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
#     try:
#         last_assistant_message = st.session_state.messages[-1]["content"]

#         # Don't generate followup questions if response is empty or contains error messages
#         if not last_assistant_message.strip() or "Sorry, I encountered an error" in last_assistant_message:
#             st.session_state.followup_questions = []
#         # Don't generate followup questions for unrelated responses
#         elif "I apologize, but I'm designed to answer questions" in last_assistant_message:
#             st.session_state.followup_questions = []
#         else:
#             # Get the last user message
#             last_user_message = next(
#                 (msg["content"] for msg in reversed(st.session_state.messages)
#                  if msg["role"] == "user"),
#                 ""
#             )

#             # Generate new questions only if the last assistant message has changed
#             if st.session_state.last_assistant != last_assistant_message:
#                 print("Generating new followup questions")
#                 st.session_state.last_assistant = last_assistant_message
#                 try:
#                     st.session_state.followup_questions = get_followup_questions(
#                         last_user_message,
#                         last_assistant_message
#                     )
#                 except Exception as e:
#                     print(f"Failed to generate followup questions: {e}")
#                     st.session_state.followup_questions = []

#         # Display follow-up questions only if we have valid ones
#         if st.session_state.followup_questions and len(st.session_state.followup_questions) > 0:
#             st.markdown("#### Related Questions:")
#             cols = st.columns(len(st.session_state.followup_questions))
            
#             for i, question in enumerate(st.session_state.followup_questions):
#                 # Remove numbering e.g "1. ", "2. ", etc.
#                 clean_question = re.sub(r'^\d+\.\s*', '', question)
#                 with cols[i]:
#                     if st.button(
#                         f"💬 {clean_question}",
#                         key=f"followup_{i}_{st.session_state.followup_key}",
#                         use_container_width=True
#                     ):
#                         handle_followup(clean_question)
#                         st.rerun()
#     except Exception as e:
#         print(f"Error in followup section: {e}")
#         st.session_state.followup_questions = []

# # Footer with attribution
# st.markdown("---")
# st.markdown(
#     f"""
#     <div style="text-align: center; color: gray; font-size: 12px;">
#         <p>Smart Guide for {st.session_state.selected_country} | Powered by AI | &copy; 2025</p>
#     </div>
#     """,
#     unsafe_allow_html=True
# )