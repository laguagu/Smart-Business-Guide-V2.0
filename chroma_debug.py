"""
Chroma DB Debug Utility

This script helps verify and debug Chroma DB. Run it directly to check 
if your Chroma DB is properly set up and working.
"""

import json
import os
import sys
from pathlib import Path

# Try to import required packages
try:
    import chromadb
    import streamlit as st
    from langchain_chroma import Chroma
    from langchain_openai import OpenAIEmbeddings
except ImportError as e:
    print(f"Required package not found: {e}")
    print("Please install required packages with:")
    print("pip install chromadb langchain-chroma langchain-openai streamlit")
    sys.exit(1)

# Load environment variables from .streamlit/secrets.toml if available
try:
    if not os.environ.get("OPENAI_API_KEY") and os.path.exists('.streamlit/secrets.toml'):
        import toml
        secrets = toml.load('.streamlit/secrets.toml')
        os.environ["OPENAI_API_KEY"] = secrets.get("OPENAI_API_KEY", "")
except Exception as e:
    print(f"Error loading secrets: {e}")

# Default paths to check
PATHS_TO_CHECK = [
    'data/chroma_db_llamaparse-openai',
    'data/chroma_db_llamaparse-huggincface',
]

def verify_chroma_db(persist_directory, collection_name="rag"):
    """
    Verify that Chroma DB is correctly set up and working.
    Returns a dict with verification results and diagnostics.
    """
    results = {
        "exists": False,
        "loadable": False,
        "collection_info": None,
        "document_count": 0,
        "file_structure": {},
        "error": None
    }
    
    try:
        # Check if directory exists
        if not os.path.exists(persist_directory):
            results["error"] = f"Directory {persist_directory} does not exist"
            return results
        
        results["exists"] = True
        
        # Analyze file structure
        for root, dirs, files in os.walk(persist_directory):
            rel_path = os.path.relpath(root, persist_directory)
            if rel_path == '.':
                results["file_structure"]["root_files"] = files
                results["file_structure"]["root_dirs"] = dirs
            elif "chroma.sqlite3" in files:
                results["file_structure"]["sqlite_path"] = os.path.join(rel_path, "chroma.sqlite3")
                results["file_structure"]["sqlite_size"] = os.path.getsize(
                    os.path.join(persist_directory, rel_path, "chroma.sqlite3")
                )
        
        # Try to load the vector store
        try:
            # Use a temporary embedding model for verification
            temp_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
            
            # First try using ChromaDB directly
            try:
                client = chromadb.PersistentClient(path=persist_directory)
                all_collections = client.list_collections()
                results["chromadb_collections"] = [c.name for c in all_collections]
                
                # Check if our collection exists
                collection_exists = any(c.name == collection_name for c in all_collections)
                results["collection_exists_in_chromadb"] = collection_exists
                
                if collection_exists:
                    collection = client.get_collection(collection_name)
                    count = collection.count()
                    results["chromadb_direct_count"] = count
            except Exception as e:
                results["chromadb_direct_error"] = str(e)
            
            # Then try using langchain_chroma
            vectorstore = Chroma(
                persist_directory=persist_directory,
                embedding_function=temp_embeddings,
                collection_name=collection_name
            )
            
            results["loadable"] = True
            # Get collection info
            results["collection_info"] = {
                "collection_name": vectorstore._collection.name,
            }
            
            # Count documents
            results["document_count"] = vectorstore._collection.count()
            
            # Optional: Get a sample document to verify content
            if results["document_count"] > 0:
                sample = vectorstore._collection.peek(1)
                results["sample_document"] = {
                    "ids": sample["ids"],
                    "metadata_sample": sample["metadatas"][0] if sample["metadatas"] else None,
                    "document_preview": sample["documents"][0][:200] + "..." if sample["documents"] else None
                }
                
        except Exception as e:
            results["error"] = f"Failed to load Chroma DB: {str(e)}"
            results["loadable"] = False
    
    except Exception as e:
        results["error"] = f"Verification failed: {str(e)}"
    
    return results

def main():
    print("\n=== Chroma DB Debug Utility ===\n")
    
    # Check if OPENAI_API_KEY is set
    if not os.environ.get("OPENAI_API_KEY"):
        print("WARNING: OPENAI_API_KEY environment variable is not set.")
        print("Some tests may fail without a valid API key.\n")
    
    # Check each path
    for path in PATHS_TO_CHECK:
        print(f"\nChecking Chroma DB at: {path}")
        
        # Make path absolute if relative
        abs_path = Path(path)
        if not abs_path.is_absolute():
            abs_path = Path(os.getcwd()) / path
        
        # Verify the DB
        results = verify_chroma_db(str(abs_path))
        
        # Print results
        if not results["exists"]:
            print(f"❌ Directory does not exist: {abs_path}")
            continue
        
        print(f"✅ Directory exists")
        
        if "sqlite_path" in results.get("file_structure", {}):
            sqlite_path = results["file_structure"]["sqlite_path"]
            sqlite_size = results["file_structure"]["sqlite_size"]
            print(f"✅ Found SQLite database: {sqlite_path} ({sqlite_size/1024/1024:.2f} MB)")
        else:
            print("❌ No SQLite database found in directory structure")
        
        if results["loadable"]:
            print(f"✅ Successfully loaded Chroma DB")
            print(f"✅ Collection: {results['collection_info']['collection_name']}")
            print(f"✅ Document count: {results['document_count']}")
            
            if results["document_count"] > 0 and "sample_document" in results:
                print("\nSample document preview:")
                print(f"ID: {results['sample_document']['ids'][0]}")
                if results['sample_document']['metadata_sample']:
                    print(f"Metadata: {json.dumps(results['sample_document']['metadata_sample'], indent=2)}")
                print(f"Content: {results['sample_document']['document_preview']}")
        else:
            print(f"❌ Failed to load Chroma DB: {results['error']}")
        
        # Print ChromaDB direct access results if available
        if "chromadb_collections" in results:
            print(f"\nCollections found via ChromaDB direct access: {results['chromadb_collections']}")
            if results.get("collection_exists_in_chromadb"):
                print(f"✅ Collection 'rag' exists in ChromaDB")
                print(f"✅ Document count (via ChromaDB): {results.get('chromadb_direct_count', 'N/A')}")
            else:
                print(f"❌ Collection 'rag' does not exist in ChromaDB")
        
        print("\n" + "-"*50)
    
    print("\nDebug complete. If you're experiencing issues, please check:")
    print("1. Ensure the directories exist and have proper permissions")
    print("2. Verify the SQLite database file is not corrupted")
    print("3. Check your embedding model and API keys are correctly configured")
    print("4. Make sure the collection name matches in your code and the database")

if __name__ == "__main__":
    main()
