"""
sidebar.py - Sidebar Component

This module handles the sidebar UI for settings and controls.
Documents are automatically loaded from the data/ folder.
Includes advanced features: hybrid search, reranking, guardrails.
"""

import streamlit as st
import os
from typing import Dict, Any, List

from app.config import (
    DEFAULT_TOP_K,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    ALLOWED_EXTENSIONS,
    DATA_DIR,
)


def get_documents_from_data_folder() -> List[str]:
    """
    Get all document files from the data/ folder.
    
    Returns:
        List of file paths
    """
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)
        return []
    
    file_paths = []
    for filename in os.listdir(DATA_DIR):
        ext = filename.split('.')[-1].lower()
        if ext in ALLOWED_EXTENSIONS:
            file_paths.append(os.path.join(DATA_DIR, filename))
    
    return file_paths


def get_available_documents() -> List[str]:
    """Get list of available document filenames from session state."""
    if "available_documents" in st.session_state:
        return st.session_state.available_documents
    return []


def render_sidebar() -> Dict[str, Any]:
    """
    Render the sidebar with settings and controls.
    
    Returns:
        Dictionary with sidebar state and user selections
    """
    # Disabled features (set to defaults)
    use_memory = False
    metadata_filter = None
    
    with st.sidebar:
        st.title("Settings")
        
        # =================================================================
        # Knowledge Base Section
        # =================================================================
        st.header("Knowledge Base")
        
        # Get documents from data/ folder
        document_paths = get_documents_from_data_folder()
        
        if document_paths:
            st.markdown(f"**{len(document_paths)} document(s) found**")
            
            # Show document names
            with st.expander("View documents", expanded=False):
                for path in document_paths:
                    filename = os.path.basename(path)
                    file_size = os.path.getsize(path) / 1024  # KB
                    st.text(f"• {filename} ({file_size:.1f} KB)")
        else:
            st.markdown("**No documents in data/ folder**")
            st.caption("Add PDF or Markdown files to the `data/` folder")
        
        st.divider()
        
        # =================================================================
        # Retrieval Settings
        # =================================================================
        st.header("Retrieval Settings")
        
        top_k = st.slider(
            "Top-K (chunks to retrieve)",
            min_value=1,
            max_value=10,
            value=DEFAULT_TOP_K,
            help="Number of most relevant document chunks to retrieve for each question."
        )
        
        # Retrieval method selection
        retrieval_method = st.radio(
            "Retrieval Method",
            options=["Vector Search", "Hybrid Search (Vector + BM25)"],
            index=0,
            help="Hybrid search combines semantic and keyword matching for better results."
        )
        use_hybrid = retrieval_method == "Hybrid Search (Vector + BM25)"
        
        # Reranking option
        use_reranking = st.checkbox(
            "Enable Reranking",
            value=False,
            help="Use cross-encoder to rerank retrieved documents for better relevance."
        )
        
        st.divider()
        
        # =================================================================
        # Memory & Guardrails
        # =================================================================
        with st.expander("Security", expanded=False):
            # Guardrails
            enable_guardrails = st.checkbox(
                "Enable Guardrails",
                value=True,
                help="Detect and prevent prompt injection attacks."
            )
        
        st.divider()
        
        # =================================================================
        # Advanced Chunking Settings
        # =================================================================
        with st.expander("Advanced Settings", expanded=False):
            st.caption("Chunking parameters (for document processing)")
            
            chunk_size = st.number_input(
                "Chunk Size (tokens)",
                min_value=200,
                max_value=800,
                value=DEFAULT_CHUNK_SIZE,
                step=50,
                help="Target size for each document chunk. Range: 400-500 recommended."
            )
            
            chunk_overlap = st.number_input(
                "Chunk Overlap (tokens)",
                min_value=0,
                max_value=200,
                value=DEFAULT_CHUNK_OVERLAP,
                step=25,
                help="Overlap between consecutive chunks. Range: 50-100 recommended."
            )
        
        st.divider()
        
        # =================================================================
        # Action Buttons
        # =================================================================
        st.header("Actions")
        
        build_button = st.button(
            "Build Knowledge Base",
            use_container_width=True,
            help="Process documents from data/ folder and create searchable index.",
            disabled=len(document_paths) == 0,
        )
        
        clear_button = st.button(
            "Clear Knowledge Base",
            use_container_width=True,
            help="Delete the current index and rebuild."
        )
        
        st.divider()
        
        # =================================================================
        # Status Information
        # =================================================================
        st.header("Status")
        
        # Show vector store status
        if "documents_processed" in st.session_state and st.session_state.documents_processed:
            st.markdown("**Vector store ready**")
            if "doc_count" in st.session_state:
                st.caption(f"{st.session_state.doc_count} chunks indexed")
        else:
            st.markdown("**No vector store loaded**")
        
        # Show active features
        active_features = []
        if use_hybrid:
            active_features.append("Hybrid")
        if use_reranking:
            active_features.append("Rerank")
        if enable_guardrails:
            active_features.append("Guardrails")
        
        if active_features:
            st.caption("Active: " + " | ".join(active_features))
        
        st.divider()
        
        # =================================================================
        # About Section
        # =================================================================
        with st.expander("About", expanded=False):
            st.markdown("""
            **RAG Chatbot**
            
            *Giga Academy Cohort IV Project*
            
            A Retrieval-Augmented Generation chatbot with advanced features.
            
            **Tech Stack:**
            - 🦜 LangChain
            - 🔍 FAISS Vector Store
            - 🧠 OpenAI Embeddings
            - 🎨 Streamlit
            
            **Features:**
            - ✅ Document ingestion (PDF, MD)
            - ✅ Token-based chunking
            - ✅ Vector search
            - ✅ Hybrid search (BM25 + Vector)
            - ✅ Cross-encoder reranking
            - ✅ Guardrails (security)
            - ✅ Source citations
            """)
        
        # Return all sidebar state
        return {
            "document_paths": document_paths,
            "top_k": top_k,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "build_button": build_button,
            "clear_button": clear_button,
            # Advanced features
            "use_hybrid": use_hybrid,
            "use_reranking": use_reranking,
            "use_memory": False,  # Disabled
            "memory_turns": 5,  # Default
            "enable_guardrails": enable_guardrails,
            "metadata_filter": None,  # Disabled
        }
