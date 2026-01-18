"""
retriever.py - Document Retrieval Module

This module handles Top-K retrieval from the FAISS vector store.
Returns the most semantically similar document chunks for a given query.
Supports metadata filtering for targeted retrieval.
"""

from typing import List, Tuple, Optional, Dict, Any

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_core.retrievers import BaseRetriever


# Default number of documents to retrieve
DEFAULT_TOP_K = 3


def create_retriever(
    vectorstore: FAISS,
    k: int = DEFAULT_TOP_K,
    search_type: str = "similarity",
    filter_dict: Optional[Dict[str, Any]] = None,
) -> BaseRetriever:
    """
    Create a retriever from a FAISS vector store.
    
    The retriever performs semantic search to find the most relevant
    document chunks for a given query.
    
    Args:
        vectorstore: FAISS vector store to search
        k: Number of top documents to retrieve (Top-K)
        search_type: Type of search ("similarity" or "mmr")
                    - "similarity": Pure semantic similarity
                    - "mmr": Maximal Marginal Relevance (diversity-aware)
        filter_dict: Optional metadata filter (e.g., {"filename": "doc.pdf"})
        
    Returns:
        Configured retriever instance
    """
    search_kwargs = {"k": k}
    
    # Add filter if provided
    if filter_dict:
        search_kwargs["filter"] = filter_dict
    
    # Create retriever with specified parameters
    retriever = vectorstore.as_retriever(
        search_type=search_type,
        search_kwargs=search_kwargs,
    )
    
    return retriever


def retrieve_documents(
    vectorstore: FAISS,
    query: str,
    k: int = DEFAULT_TOP_K,
    filter_dict: Optional[Dict[str, Any]] = None,
) -> List[Document]:
    """
    Retrieve the top-K most relevant documents for a query.
    
    Args:
        vectorstore: FAISS vector store to search
        query: User's question or search query
        k: Number of documents to retrieve
        filter_dict: Optional metadata filter
        
    Returns:
        List of the most relevant Document objects
    """
    if not query or not query.strip():
        return []
    
    # Perform similarity search with optional filter
    if filter_dict:
        documents = vectorstore.similarity_search(query, k=k, filter=filter_dict)
    else:
        documents = vectorstore.similarity_search(query, k=k)
    
    return documents
