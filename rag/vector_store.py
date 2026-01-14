"""
vector_store.py - FAISS Vector Store Management

This module handles creation, persistence, and loading of FAISS vector stores.
The vector store enables efficient semantic search over embedded document chunks.

FAISS indexes are persisted to disk for reuse across sessions.
"""

import os
import shutil
from typing import List

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings


# Default path for persisting FAISS index
DEFAULT_VECTORSTORE_PATH = "vectorstore"


def create_vector_store(
    documents: List[Document],
    embeddings: OpenAIEmbeddings,
) -> FAISS:
    """
    Create a FAISS vector store from a list of documents.
    
    Each document is embedded using the provided embeddings model,
    and the resulting vectors are indexed in FAISS for efficient retrieval.
    
    Args:
        documents: List of Document objects to embed and index
        embeddings: Embeddings model for vectorization
        
    Returns:
        FAISS vector store instance
        
    Raises:
        ValueError: If documents list is empty
    """
    if not documents:
        raise ValueError("Cannot create vector store from empty document list")
    
    # Create FAISS index from documents
    # This performs embedding for all documents and builds the search index
    vectorstore = FAISS.from_documents(
        documents=documents,
        embedding=embeddings,
    )
    
    return vectorstore


def save_vector_store(
    vectorstore: FAISS,
    path: str = DEFAULT_VECTORSTORE_PATH,
) -> None:
    """
    Save a FAISS vector store to disk for persistence.
    
    The vector store is saved in a format that can be loaded later,
    avoiding the need to re-embed documents on every session.
    
    Args:
        vectorstore: FAISS vector store to save
        path: Directory path where the vector store will be saved
    """
    # Create directory if it doesn't exist
    os.makedirs(path, exist_ok=True)
    
    # Save the FAISS index and associated data
    vectorstore.save_local(path)


def load_vector_store(
    embeddings: OpenAIEmbeddings,
    path: str = DEFAULT_VECTORSTORE_PATH,
    allow_dangerous_deserialization: bool = True,
) -> FAISS:
    """
    Load a FAISS vector store from disk.
    
    Args:
        embeddings: Embeddings model (must match the one used to create the store)
        path: Directory path where the vector store is saved
        allow_dangerous_deserialization: Allow loading pickled data (required for FAISS)
        
    Returns:
        Loaded FAISS vector store instance
        
    Raises:
        FileNotFoundError: If the vector store doesn't exist at the specified path
        ValueError: If the vector store files are corrupted
    """
    if not vector_store_exists(path):
        raise FileNotFoundError(
            f"Vector store not found at {path}. "
            "Please create and save a vector store first."
        )
    
    # Load the FAISS index
    # Note: allow_dangerous_deserialization=True is required because FAISS uses pickle
    # Only load vector stores from trusted sources
    vectorstore = FAISS.load_local(
        path,
        embeddings,
        allow_dangerous_deserialization=allow_dangerous_deserialization,
    )
    
    return vectorstore


def vector_store_exists(path: str = DEFAULT_VECTORSTORE_PATH) -> bool:
    """
    Check if a FAISS vector store exists at the specified path.
    
    Args:
        path: Directory path to check
        
    Returns:
        True if vector store files exist, False otherwise
    """
    # FAISS saves two files: index.faiss and index.pkl
    index_file = os.path.join(path, "index.faiss")
    pkl_file = os.path.join(path, "index.pkl")
    
    return os.path.exists(index_file) and os.path.exists(pkl_file)


def delete_vector_store(path: str = DEFAULT_VECTORSTORE_PATH) -> None:
    """
    Delete a persisted FAISS vector store.
    
    This is useful when you want to rebuild the index from scratch.
    
    Args:
        path: Directory path of the vector store to delete
    """
    if os.path.exists(path):
        shutil.rmtree(path)


def add_documents_to_store(
    vectorstore: FAISS,
    documents: List[Document],
) -> FAISS:
    """
    Add new documents to an existing FAISS vector store.
    
    This allows incremental updates without rebuilding the entire index.
    
    Args:
        vectorstore: Existing FAISS vector store
        documents: New documents to add
        
    Returns:
        Updated FAISS vector store
    """
    if not documents:
        return vectorstore
    
    # Add documents to the existing index
    vectorstore.add_documents(documents)
    
    return vectorstore


def get_store_stats(vectorstore: FAISS) -> dict:
    """
    Get statistics about the vector store.
    
    Args:
        vectorstore: FAISS vector store
        
    Returns:
        Dictionary with store statistics (document count, etc.)
    """
    # Get the number of documents in the store
    try:
        doc_count = vectorstore.index.ntotal
    except AttributeError:
        doc_count = 0
    
    return {
        "document_count": doc_count,
        "index_type": type(vectorstore.index).__name__ if hasattr(vectorstore, 'index') else "Unknown",
    }
