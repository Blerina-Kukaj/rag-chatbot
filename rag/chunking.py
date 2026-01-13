"""
chunking.py - Text Cleaning and Chunking Module

This module handles text preprocessing and intelligent chunking of documents.
Uses token-based splitting to ensure chunks stay within 400-500 tokens
with 50-100 token overlap as per project requirements.

Token-based chunking ensures compatibility with LLM context windows and embedding models.
"""

import re
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tiktoken


# Token limits for chunking (per project requirements: 400-500 tokens, 50-100 overlap)
CHUNK_SIZE = 450  # Target: 400-500 tokens (middle of range)
CHUNK_OVERLAP = 75  # Target: 50-100 tokens (middle of range)

# Tiktoken encoding for OpenAI models (text-embedding-3-small uses cl100k_base)
ENCODING_NAME = "cl100k_base"


def clean_text(text: str) -> str:
    """
    Clean and normalize text content.
    
    Removes excessive whitespace, normalizes line breaks, and handles
    common formatting issues from PDF/Markdown extraction.
    
    Args:
        text: Raw text to clean
        
    Returns:
        Cleaned text string
    """
    if not text:
        return ""
    
    # Replace multiple spaces with single space
    text = re.sub(r' +', ' ', text)
    
    # Replace multiple newlines with double newline (paragraph break)
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    
    # Remove spaces at the start/end of lines
    text = re.sub(r'(?m)^ +| +$', '', text)
    
    # Remove common PDF artifacts (form feeds, etc.)
    text = text.replace('\f', '\n')
    text = text.replace('\r', '\n')
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def count_tokens(text: str, encoding_name: str = ENCODING_NAME) -> int:
    """
    Count the number of tokens in a text string using tiktoken.
    
    Args:
        text: Text to count tokens for
        encoding_name: Name of the tiktoken encoding to use
        
    Returns:
        Number of tokens in the text
    """
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(text))


def chunk_documents(
    documents: List[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Document]:
    """
    Split documents into smaller chunks based on token count.
    
    Uses RecursiveCharacterTextSplitter with token-based length function
    to ensure chunks are appropriately sized for embedding and retrieval.
    
    Each chunk preserves the original document's metadata and adds:
    - chunk_id: Sequential identifier for the chunk within the source document
    
    Args:
        documents: List of Document objects to chunk
        chunk_size: Target size in tokens (default: 450, range: 400-500)
        chunk_overlap: Overlap size in tokens (default: 75, range: 50-100)
        
    Returns:
        List of chunked Document objects
    """
    if not documents:
        return []
    
    # Clean all document texts before chunking
    for doc in documents:
        doc.page_content = clean_text(doc.page_content)
    
    # Filter out empty documents
    documents = [doc for doc in documents if doc.page_content.strip()]
    
    if not documents:
        return []
    
    # Create text splitter with token-based length function
    # RecursiveCharacterTextSplitter tries to split on natural boundaries
    # (paragraphs, sentences, etc.) while respecting token limits
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=count_tokens,
        separators=[
            "\n\n",  # Paragraph breaks (highest priority)
            "\n",    # Line breaks
            ". ",    # Sentences
            "! ",    # Exclamations
            "? ",    # Questions
            "; ",    # Semi-colons
            ", ",    # Clauses
            " ",     # Words
            "",      # Characters (fallback)
        ],
    )
    
    # Split documents and preserve metadata
    chunked_docs = text_splitter.split_documents(documents)
    
    # Add chunk_id to metadata for citation purposes
    # Group chunks by source document and page
    current_source = None
    chunk_counter = 0
    
    for doc in chunked_docs:
        # Create unique identifier for source document + page combination
        source_identifier = (
            doc.metadata.get("source", "") + 
            str(doc.metadata.get("page", ""))
        )
        
        # Reset counter for new source document/page
        if source_identifier != current_source:
            current_source = source_identifier
            chunk_counter = 0
        
        chunk_counter += 1
        doc.metadata["chunk_id"] = chunk_counter
        
        # Add token count to metadata for transparency
        doc.metadata["token_count"] = count_tokens(doc.page_content)
    
    return chunked_docs


def get_chunk_info(chunk: Document) -> dict:
    """
    Extract structured information about a chunk for citation purposes.
    
    Args:
        chunk: Document chunk
        
    Returns:
        Dictionary with citation info (filename, page, chunk_id, token_count)
    """
    content = chunk.page_content
    return {
        "filename": chunk.metadata.get("filename", "Unknown"),
        "page": chunk.metadata.get("page_display"),  # Can be None for Markdown
        "chunk_id": chunk.metadata.get("chunk_id", 0),
        "token_count": chunk.metadata.get("token_count", count_tokens(content)),
        "content_preview": content[:100] + "..." if len(content) > 100 else content,
    }


def get_chunking_stats(chunks: List[Document]) -> dict:
    """
    Get statistics about chunked documents.
    
    Useful for verifying chunking is working correctly.
    
    Args:
        chunks: List of chunked Document objects
        
    Returns:
        Dictionary with chunking statistics
    """
    if not chunks:
        return {
            "total_chunks": 0,
            "avg_tokens": 0,
            "min_tokens": 0,
            "max_tokens": 0,
            "unique_sources": 0,
        }
    
    token_counts = [
        chunk.metadata.get("token_count", count_tokens(chunk.page_content))
        for chunk in chunks
    ]
    
    unique_sources = set(
        chunk.metadata.get("filename", "Unknown") for chunk in chunks
    )
    
    return {
        "total_chunks": len(chunks),
        "avg_tokens": sum(token_counts) / len(token_counts),
        "min_tokens": min(token_counts),
        "max_tokens": max(token_counts),
        "unique_sources": len(unique_sources),
    }
