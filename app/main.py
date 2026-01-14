"""
main.py - Streamlit Application Entry Point

This is the main entry point for the RAG chatbot Streamlit app.
Run with: streamlit run app/main.py

Features:
- Document ingestion & chunking
- Vector search with FAISS
- Hybrid search (Vector + BM25)
- Cross-encoder reranking
- Guardrails (prompt injection detection)

Giga Academy Cohort IV - RAG Chatbot Project
"""

import streamlit as st
import os
import sys
import time

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import app configuration and components
from app.config import (
    PAGE_TITLE,
    PAGE_ICON,
    LAYOUT,
    APP_TITLE,
    WELCOME_MESSAGE,
    NO_DOCUMENTS_MESSAGE,
    NO_VECTORSTORE_MESSAGE,
    PROCESSING_MESSAGE,
    SUCCESS_MESSAGE,
    ERROR_API_KEY,
    VECTORSTORE_DIR,
)
from app.components.sidebar import render_sidebar
from app.components.chat import (
    initialize_chat_history,
    render_chat_interface,
    add_message,
    display_user_message,
    display_assistant_response,
    display_welcome_message,
    display_error_message,
    display_success_message,
    display_warning_message,
    display_thinking_indicator,
    clear_chat_history,
)

# Import RAG modules
from rag.ingestion import load_uploaded_files
from rag.chunking import chunk_documents, get_chunking_stats
from rag.embeddings import get_embeddings_model, validate_api_key
from rag.vector_store import (
    create_vector_store,
    save_vector_store,
    load_vector_store,
    vector_store_exists,
    delete_vector_store,
)
from rag.retriever import create_retriever, retrieve_documents
from rag.qa_chain import create_qa_chain, ask_question, get_source_details
from rag.hybrid_search import create_hybrid_retriever
from rag.reranker import CrossEncoderReranker
from rag.guardrails import validate_input, wrap_context_safely, detect_prompt_injection


# =============================================================================
# Custom Styling
# =============================================================================
def apply_custom_styling():
    """Apply custom CSS for monochromatic (black, white, gray) theme."""
    custom_css = """
    <style>
    /* Main background and text colors */
    .stApp {
        background-color: #212121;
        color: #ffffff;
    }
    
    /* Sidebar styling (stable selectors) */
    section[data-testid="stSidebar"] {
        background-color: #181818 !important;
        border-right: 1px solid #2a2a2a !important;
    }
    section[data-testid="stSidebar"] * {
        color: #f0f0f0 !important;
    }
    section[data-testid="stSidebar"] .stButton button {
        background-color: #222222 !important;
        border: 1px solid #444444 !important;
        color: #f0f0f0 !important;
    }
    section[data-testid="stSidebar"] .stButton button:hover {
        background-color: #2a2a2a !important;
        border-color: #666666 !important;
    }
    section[data-testid="stSidebar"] .stSelectbox, 
    section[data-testid="stSidebar"] .stNumberInput, 
    section[data-testid="stSidebar"] .stTextInput {
        color: #f0f0f0 !important;
    }
    section[data-testid="stSidebar"] input, 
    section[data-testid="stSidebar"] select {
        background-color: #1f1f1f !important;
        border: 1px solid #444444 !important;
        color: #f0f0f0 !important;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        font-weight: 600;
    }

    /* App Header - transparent (no background or shadow) */
    header[data-testid="stHeader"],
    div[data-testid="stHeader"],
    .stAppHeader {
        background: transparent !important;
        background-color: transparent !important;
        box-shadow: none !important;
    }
    /* Target any nested emotion cache class in header */
    .stAppHeader .st-emotion-cache-40nadn {
        background: transparent !important;
        background-color: transparent !important;
    }

    /* Bottom container/footer - transparent */
    div[data-testid="stBottom"],
    footer,
    .stBottom,
    .st-emotion-cache-hzygls {
        background: #212121 !important;
        background-color: #212121 !important;
        box-shadow: none !important;
    }
    .st-emotion-cache-1353z0o{
        background: #rgb(48,48,48) !important;
        background-color: rgb(48,48,48) !important;
    
    /* Buttons - monochromatic */
    .stButton button {
        background-color: #ffffff !important;
        border: 1px solid #cccccc !important;
        color: #000000 !important;
    }
    .stButton button:hover {
        background-color: #f8f9fa !important;
        border-color: #999999 !important;
    }
    
    /* Input fields */
    .stTextInput input, .stNumberInput input, .stSelectbox select {
        background-color: #ffffff !important;
        border: 1px solid #cccccc !important;
        color: #000000 !important;
    }
    
    /* Checkboxes */
    .stCheckbox label {
        color: #000000 !important;
    }
    
    /* Sliders */
    .stSlider .st-bx {
        background-color: #cccccc !important;
    }
    
    /* Success/Warning/Error messages */
    .stSuccess, .stWarning, .stError {
        background-color: #f8f9fa !important;
        border-left: 4px solid #cccccc !important;
        color: #000000 !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #181818 !important;
        color: #f0f0f0 !important;
        border: 1px solid #2a2a2a !important;
    }
    
    /* Chat messages */
    .stChatMessage {
        background-color: #212121 !important;
        border: 1px solid #212121 !important;
        border-radius: 8px !important;
    }
    
    /* Chat input */
    .stChatInput input {
        background-color: #303030 !important;
        border: 1px solid #444444 !important;
        color: #f0f0f0 !important;
    }
    .stChatInput input::placeholder {
        color: #bbbbbb !important;
    }
    
    /* Input focus states - white background */
    .st-emotion-cache-19cfm8f.focused{
        border-color: #ffffff !important;
    }
    
    /* Streamlit element container */
    .st-ek {
        background-color: rgb(48,48,48) !important;
        color: #f0f0f0 !important;
    }
    
    
    /* Remove any colored accents */
    .stProgress .st-bo {
        background-color: #cccccc !important;
    }
    
    /* Links */
    a {
        color: #333333 !important;
    }
    a:hover {
        color: #000000 !important;
    }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)


# =============================================================================
# Page Configuration (must be first Streamlit command)
# =============================================================================
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout=LAYOUT,
)


# =============================================================================
# Session State Initialization
# =============================================================================
def initialize_session_state() -> None:
    """Initialize all session state variables."""
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    
    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None
    
    if "documents_processed" not in st.session_state:
        st.session_state.documents_processed = False
    
    if "doc_count" not in st.session_state:
        st.session_state.doc_count = 0
    
    if "chunks" not in st.session_state:
        st.session_state.chunks = []
    
    if "available_documents" not in st.session_state:
        st.session_state.available_documents = []
    
    # Reranker (lazy loaded)
    if "reranker" not in st.session_state:
        st.session_state.reranker = None
    
    # Initialize chat history
    initialize_chat_history()


# =============================================================================
# Vector Store Operations
# =============================================================================
def build_vector_store(
    file_paths: list,
    chunk_size: int,
    chunk_overlap: int
) -> bool:
    """
    Build the vector store from uploaded documents.
    """
    try:
        # Check API key first
        if not validate_api_key():
            display_error_message(ERROR_API_KEY)
            return False
        
        with st.spinner(PROCESSING_MESSAGE):
            # Step 1: Load documents
            status = st.empty()
            status.info("Loading documents...")
            
            documents = load_uploaded_files(file_paths)
            
            if not documents:
                display_error_message("No valid documents found. Please check your files.")
                return False
            
            status.info(f"Loaded {len(documents)} document(s)")
            
            # Step 2: Chunk documents
            status.info("Chunking documents...")
            
            chunks = chunk_documents(
                documents,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            
            if not chunks:
                display_error_message("No chunks created. Documents may be empty.")
                return False
            
            # Store chunks for hybrid search
            st.session_state.chunks = chunks
            
            # Extract unique document names for filters
            unique_docs = set(chunk.metadata.get("filename", "Unknown") for chunk in chunks)
            st.session_state.available_documents = sorted(list(unique_docs))
            
            # Get chunking stats
            stats = get_chunking_stats(chunks)
            status.info(
                f"Created {stats['total_chunks']} chunks "
                f"(avg {stats['avg_tokens']:.0f} tokens)"
            )
            
            # Step 3: Create embeddings model
            status.info("Creating embeddings...")
            embeddings = get_embeddings_model()
            
            # Step 4: Create vector store
            status.info("Building vector store...")
            vectorstore = create_vector_store(chunks, embeddings)
            
            # Step 5: Save to disk
            status.info("Saving vector store...")
            save_vector_store(vectorstore, VECTORSTORE_DIR)
            
            # Update session state
            st.session_state.vectorstore = vectorstore
            st.session_state.documents_processed = True
            st.session_state.doc_count = stats['total_chunks']
            st.session_state.qa_chain = None  # Reset chain
            
            # Clear chat history and memory when rebuilding
            clear_chat_history()
            
            status.empty()
            display_success_message(
                f"{SUCCESS_MESSAGE}\n\n"
                f"**Stats:** {stats['total_chunks']} chunks from "
                f"{stats['unique_sources']} document(s)"
            )
            
            return True
    
    except Exception as e:
        display_error_message(f"Error building vector store: {str(e)}")
        return False


def load_existing_vector_store() -> bool:
    """Load vector store from disk if it exists."""
    if not vector_store_exists(VECTORSTORE_DIR):
        return False
    
    try:
        if not validate_api_key():
            return False
        
        embeddings = get_embeddings_model()
        vectorstore = load_vector_store(embeddings, VECTORSTORE_DIR)
        
        st.session_state.vectorstore = vectorstore
        st.session_state.documents_processed = True
        
        try:
            st.session_state.doc_count = vectorstore.index.ntotal
        except:
            st.session_state.doc_count = 0
        
        return True
    
    except Exception as e:
        st.warning(f"Could not load existing vector store: {str(e)}")
        return False


def handle_clear_vector_store() -> None:
    """Clear the vector store and reset session state."""
    try:
        delete_vector_store(VECTORSTORE_DIR)
        st.session_state.vectorstore = None
        st.session_state.qa_chain = None
        st.session_state.documents_processed = False
        st.session_state.doc_count = 0
        st.session_state.chunks = []
        st.session_state.available_documents = []
        clear_chat_history()
        display_success_message("✅ Vector store cleared successfully!")
    except Exception as e:
        display_error_message(f"Error clearing vector store: {str(e)}")


# =============================================================================
# Question Answering with Advanced Features
# =============================================================================
def process_question(question: str, sidebar_state: dict) -> None:
    """
    Process a user question with all advanced features.
    """
    start_time = time.time()
    guardrail_triggered = False
    
    # Get settings from sidebar
    top_k = sidebar_state["top_k"]
    use_hybrid = sidebar_state["use_hybrid"]
    use_reranking = sidebar_state["use_reranking"]
    use_memory = sidebar_state["use_memory"]
    enable_guardrails = sidebar_state["enable_guardrails"]
    metadata_filter = sidebar_state.get("metadata_filter")
    
    # Step 1: Guardrails check
    if enable_guardrails:
        is_valid, message, sanitized = validate_input(question)
        if not is_valid:
            display_user_message(question)
            add_message("user", question)
            display_warning_message(message)
            add_message("assistant", message)
            guardrail_triggered = True
            return
        
        # Check for medium risk
        result = detect_prompt_injection(question)
        if result.risk_level in ["medium", "high"]:
            display_user_message(question)
            add_message("user", question)
            display_warning_message(
                "I detected potentially unsafe content in your question. "
                "Please rephrase your question."
            )
            add_message("assistant", "I detected potentially unsafe content in your question. Please rephrase your question.")
            guardrail_triggered = True
            return
    
    # Display user message
    display_user_message(question)
    add_message("user", question)
    
    # Step 2: Enhance query with memory context
    enhanced_query = question
    if use_memory and st.session_state.conversation_memory.turns:
        context_prefix = st.session_state.conversation_memory.get_context_for_query(question)
        if context_prefix:
            enhanced_query = context_prefix + question
    
    # Generate answer
    with display_thinking_indicator():
        try:
            # Step 3: Retrieve documents
            retrieval_method = "vector"
            
            if use_hybrid and st.session_state.chunks:
                # Use hybrid retrieval
                retrieval_method = "hybrid"
                hybrid_retriever = create_hybrid_retriever(
                    st.session_state.vectorstore,
                    st.session_state.chunks,
                )
                retrieved_docs = hybrid_retriever.retrieve(enhanced_query, k=top_k * 2 if use_reranking else top_k)
            else:
                # Use standard vector retrieval
                retrieved_docs = retrieve_documents(
                    st.session_state.vectorstore,
                    enhanced_query,
                    k=top_k * 2 if use_reranking else top_k,
                    filter_dict=metadata_filter,
                )
            
            # Step 4: Reranking (optional)
            if use_reranking and retrieved_docs:
                retrieval_method = "hybrid+rerank" if use_hybrid else "vector+rerank"
                
                # Lazy load reranker
                if st.session_state.reranker is None:
                    try:
                        st.session_state.reranker = CrossEncoderReranker()
                    except ImportError:
                        display_warning_message("Reranking unavailable. Install sentence-transformers.")
                        st.session_state.reranker = False
                
                if st.session_state.reranker:
                    retrieved_docs = st.session_state.reranker.rerank_documents(
                        question, retrieved_docs, top_k=top_k
                    )
                else:
                    retrieved_docs = retrieved_docs[:top_k]
            
            # Ensure we have top_k docs
            retrieved_docs = retrieved_docs[:top_k]
            
            # Step 5: Create retriever and QA chain
            retriever = create_retriever(
                st.session_state.vectorstore,
                k=top_k,
                filter_dict=metadata_filter,
            )
            
            qa_chain = create_qa_chain(retriever)
            
            # Step 6: Ask question
            response = ask_question(qa_chain, question)
            
            # Extract answer and sources
            answer = response.get("result", "I couldn't generate an answer.")
            sources = get_source_details(response)
            
            # Don't show sources for "I don't know" responses
            if "cannot find" in answer.lower() and "provided documents" in answer.lower():
                sources = []
            
            # Step 7: Update conversation memory
            if use_memory:
                st.session_state.conversation_memory.add_turn(
                    question=question,
                    answer=answer,
                    sources=sources,
                )
            
            # Calculate response time
            response_time_ms = (time.time() - start_time) * 1000
            
            # Display response with citations
            display_assistant_response(answer, sources)
            
            # Add to chat history
            add_message("assistant", answer, sources)
        
        except Exception as e:
            error_msg = f"Error generating answer: {str(e)}"
            display_error_message(error_msg)
            add_message("assistant", error_msg)


# =============================================================================
# Main Application
# =============================================================================
def main() -> None:
    """Main application entry point."""
    # Apply custom monochromatic styling
    apply_custom_styling()
    
    # Initialize session state
    initialize_session_state()
    
    # Display title and description
    st.title(APP_TITLE)
    
    # Render sidebar and get settings
    sidebar_state = render_sidebar()
    
    # Handle build button click
    if sidebar_state["build_button"]:
        if not sidebar_state["document_paths"]:
            display_error_message(NO_DOCUMENTS_MESSAGE)
        else:
            build_vector_store(
                sidebar_state["document_paths"],
                sidebar_state["chunk_size"],
                sidebar_state["chunk_overlap"],
            )
    
    # Handle clear button click
    if sidebar_state["clear_button"]:
        handle_clear_vector_store()
    
    # Try to load existing vector store on first run
    if not st.session_state.documents_processed:
        load_existing_vector_store()
    
    st.divider()
    
    # Show welcome message if no vector store
    if not st.session_state.documents_processed:
        display_welcome_message(WELCOME_MESSAGE)
    
    # Render chat interface
    user_question = render_chat_interface()
    
    # Process question if submitted
    if user_question:
        if not st.session_state.documents_processed:
            display_error_message(NO_VECTORSTORE_MESSAGE)
        else:
            process_question(user_question, sidebar_state)


# =============================================================================
# Entry Point
# =============================================================================
if __name__ == "__main__":
    main()
