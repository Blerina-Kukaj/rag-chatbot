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


def retrieve_documents_with_scores(
    vectorstore: FAISS,
    query: str,
    k: int = DEFAULT_TOP_K,
    filter_dict: Optional[Dict[str, Any]] = None,
) -> List[Tuple[Document, float]]:
    """
    Retrieve documents with their similarity scores.
    
    This is useful for understanding how confident the retrieval is
    and for filtering low-quality matches.
    
    Args:
        vectorstore: FAISS vector store to search
        query: User's question or search query
        k: Number of documents to retrieve
        filter_dict: Optional metadata filter
        
    Returns:
        List of tuples (Document, similarity_score)
        Scores are distances (lower is better for FAISS L2 distance)
    """
    if not query or not query.strip():
        return []
    
    # Perform similarity search with scores
    # Note: FAISS doesn't support filtering with scores directly,
    # so we fetch more and filter manually
    fetch_k = k * 3 if filter_dict else k
    documents_with_scores = vectorstore.similarity_search_with_score(query, k=fetch_k)
    
    # Apply manual filter if needed
    if filter_dict:
        filtered = []
        for doc, score in documents_with_scores:
            matches = all(
                doc.metadata.get(key) == value 
                for key, value in filter_dict.items()
            )
            if matches:
                filtered.append((doc, score))
        documents_with_scores = filtered[:k]
    
    return documents_with_scores


def filter_documents_by_metadata(
    documents: List[Document],
    filter_dict: Dict[str, Any],
) -> List[Document]:
    """
    Filter documents by metadata criteria.
    
    Args:
        documents: List of documents to filter
        filter_dict: Metadata key-value pairs to match
        
    Returns:
        Filtered list of documents
    """
    if not filter_dict:
        return documents
    
    filtered = []
    for doc in documents:
        matches = all(
            doc.metadata.get(key) == value
            for key, value in filter_dict.items()
        )
        if matches:
            filtered.append(doc)
    
    return filtered


def get_available_filters(documents: List[Document]) -> Dict[str, List[Any]]:
    """
    Get available filter options from documents.
    
    Useful for building filter UI components.
    
    Args:
        documents: List of documents to analyze
        
    Returns:
        Dictionary of metadata keys to unique values
    """
    filters = {
        "filename": set(),
        "page_display": set(),
    }
    
    for doc in documents:
        if "filename" in doc.metadata:
            filters["filename"].add(doc.metadata["filename"])
        if "page_display" in doc.metadata and doc.metadata["page_display"] is not None:
            filters["page_display"].add(doc.metadata["page_display"])
    
    return {
        key: sorted(list(values)) 
        for key, values in filters.items() 
        if values
    }


def format_retrieved_context(documents: List[Document]) -> str:
    """
    Format retrieved documents into a context string for the LLM.
    
    Each document is formatted with its metadata for easy reference
    and citation in the generated answer.
    
    Args:
        documents: List of retrieved Document objects
        
    Returns:
        Formatted context string
    """
    if not documents:
        return "No relevant documents found."
    
    context_parts = []
    
    for i, doc in enumerate(documents, 1):
        # Extract metadata
        filename = doc.metadata.get("filename", "Unknown")
        page = doc.metadata.get("page_display")
        chunk_id = doc.metadata.get("chunk_id", 0)
        
        # Build citation reference
        citation = f"[{filename}"
        if page is not None:
            citation += f", Page {page}"
        citation += f", Chunk {chunk_id}]"
        
        # Format document with citation
        context_parts.append(
            f"Document {i} {citation}:\n{doc.page_content}\n"
        )
    
    return "\n".join(context_parts)


def get_unique_sources(documents: List[Document]) -> List[dict]:
    """
    Extract unique source information from retrieved documents.
    
    This is useful for displaying citations in the UI.
    
    Args:
        documents: List of retrieved Document objects
        
    Returns:
        List of dictionaries with source information
    """
    sources = []
    seen = set()
    
    for doc in documents:
        filename = doc.metadata.get("filename", "Unknown")
        page = doc.metadata.get("page_display")
        chunk_id = doc.metadata.get("chunk_id", 0)
        
        # Create unique identifier
        source_id = f"{filename}_{page}_{chunk_id}"
        
        if source_id not in seen:
            seen.add(source_id)
            sources.append({
                "filename": filename,
                "page": page,
                "chunk_id": chunk_id,
                "content_preview": (
                    doc.page_content[:200] + "..." 
                    if len(doc.page_content) > 200 
                    else doc.page_content
                ),
            })
    
    return sources
