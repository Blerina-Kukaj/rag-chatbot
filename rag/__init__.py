"""
RAG (Retrieval-Augmented Generation) Package

This package provides all core functionality for the RAG chatbot:
- Document ingestion (PDF, Markdown)
- Text cleaning and chunking (token-based)
- Embedding generation (OpenAI-compatible)
- FAISS vector store management
- Top-K retrieval (vector, hybrid, reranked)
- QA chain with grounded responses and citations
- Guardrails for security

Project: Giga Academy Cohort IV - RAG Chatbot
"""

__version__ = "1.0.0"
__author__ = "Giga Academy Cohort IV"

from rag.ingestion import (
    load_document,
    load_documents_from_directory,
    load_uploaded_files,
    is_supported_file,
    SUPPORTED_EXTENSIONS,
)
from rag.chunking import chunk_documents
from rag.embeddings import get_embeddings_model
from rag.vector_store import (
    create_vector_store,
    save_vector_store,
    load_vector_store,
    vector_store_exists,
)
from rag.retriever import create_retriever
from rag.prompts import get_qa_prompt
from rag.qa_chain import create_qa_chain, ask_question, ConversationMemory
from rag.hybrid_search import HybridRetriever, create_hybrid_retriever
from rag.reranker import CrossEncoderReranker, rerank_documents
from rag.guardrails import (
    detect_prompt_injection,
    validate_input,
)

__all__ = [
    # Ingestion
    "load_document",
    "load_documents_from_directory",
    "load_uploaded_files",
    "is_supported_file",
    "SUPPORTED_EXTENSIONS",
    # Chunking
    "chunk_documents",
    # Embeddings
    "get_embeddings_model",
    # Vector Store
    "create_vector_store",
    "save_vector_store",
    "load_vector_store",
    "vector_store_exists",
    # Retriever
    "create_retriever",
    # Hybrid Search
    "HybridRetriever",
    "create_hybrid_retriever",
    # Reranking
    "CrossEncoderReranker",
    "rerank_documents",
    # Prompts
    "get_qa_prompt",
    # QA Chain
    "create_qa_chain",
    "ask_question",
    "ConversationMemory",
    # Guardrails
    "detect_prompt_injection",
    "validate_input",
]
