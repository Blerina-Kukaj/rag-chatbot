"""
embeddings.py - Embedding Model Module

This module provides a wrapper for OpenAI-compatible embedding models.
Uses text-embedding-3-small by default for efficient, high-quality embeddings.

API key is loaded from environment variables.
"""

import os
from typing import Optional

from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()

# Default embedding model (OpenAI text-embedding-3-small)
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


def get_embeddings_model(
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
) -> OpenAIEmbeddings:
    """
    Create and configure an OpenAI embeddings model.
    
    The model is used to convert text chunks into vector representations
    for semantic search in the FAISS vector store.
    
    Args:
        model_name: Name of the embedding model (default: text-embedding-3-small)
        api_key: OpenAI API key (if None, reads from OPENAI_API_KEY env var)
        api_base: Custom API base URL for OpenAI-compatible APIs
                  (if None, reads from OPENAI_API_BASE env var)
    
    Returns:
        Configured OpenAIEmbeddings instance
        
    Raises:
        ValueError: If API key is not provided and not found in environment
    """
    # Get API key from parameter or environment
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        raise ValueError(
            "OpenAI API key not found. Please set OPENAI_API_KEY environment variable "
            "or pass it as a parameter."
        )
    
    # Get custom API base if provided (for OpenAI-compatible APIs)
    if api_base is None:
        api_base = os.getenv("OPENAI_API_BASE")
    
    # Configure embeddings model
    embeddings_config = {
        "model": model_name,
        "openai_api_key": api_key,
    }
    
    # Add custom base URL if provided
    if api_base:
        embeddings_config["openai_api_base"] = api_base
    
    embeddings = OpenAIEmbeddings(**embeddings_config)
    
    return embeddings


def validate_api_key(api_key: Optional[str] = None) -> bool:
    """
    Validate that an API key is available.
    
    Args:
        api_key: API key to validate (if None, checks environment)
        
    Returns:
        True if API key is available, False otherwise
    """
    if api_key:
        return True
    
    return bool(os.getenv("OPENAI_API_KEY"))


def get_embedding_dimension(model_name: str = DEFAULT_EMBEDDING_MODEL) -> int:
    """
    Get the dimension of embeddings for a given model.
    
    This is useful for initializing vector stores with the correct dimensions.
    
    Args:
        model_name: Name of the embedding model
        
    Returns:
        Embedding dimension (number of vector components)
    """
    # Known dimensions for OpenAI embedding models
    model_dimensions = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }
    
    return model_dimensions.get(model_name, 1536)  # Default to 1536 if unknown
