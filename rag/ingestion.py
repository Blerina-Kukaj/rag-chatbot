"""
ingestion.py - Document Loading Module

This module handles loading documents from various file formats (PDF, Markdown).
Uses LangChain's document loaders to extract text and metadata from files.

Supported formats:
- PDF (.pdf) - Uses PyPDFLoader for page-by-page extraction
- Markdown (.md) - Uses UnstructuredMarkdownLoader for text extraction
"""

import os
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, UnstructuredMarkdownLoader


# Supported file extensions
SUPPORTED_EXTENSIONS = {".pdf", ".md"}


def get_file_extension(file_path: str) -> str:
    """
    Extract the file extension from a file path.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Lowercase file extension (e.g., '.pdf')
    """
    return Path(file_path).suffix.lower()


def is_supported_file(file_path: str) -> bool:
    """
    Check if a file has a supported extension.
    
    Args:
        file_path: Path to the file
        
    Returns:
        True if the file type is supported, False otherwise
    """
    return get_file_extension(file_path) in SUPPORTED_EXTENSIONS


def load_pdf(file_path: str) -> List[Document]:
    """
    Load a PDF file and extract text page by page.
    
    Each page becomes a separate Document with metadata containing:
    - source: Original file path
    - page: Page number (0-indexed)
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        List of Document objects, one per page
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file is not a PDF
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found: {file_path}")
    
    if get_file_extension(file_path) != ".pdf":
        raise ValueError(f"Expected PDF file, got: {file_path}")
    
    # PyPDFLoader extracts text from each page separately
    # This preserves page numbers for citation purposes
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    
    # Enrich metadata with the filename (not full path) for cleaner citations
    filename = Path(file_path).name
    for doc in documents:
        doc.metadata["filename"] = filename
        # Convert 0-indexed page to 1-indexed for human-readable citations
        doc.metadata["page_display"] = doc.metadata.get("page", 0) + 1
    
    return documents


def load_markdown(file_path: str) -> List[Document]:
    """
    Load a Markdown file and extract text content.
    
    The entire file becomes a single Document with metadata containing:
    - source: Original file path
    - filename: Just the filename for citations
    
    Args:
        file_path: Path to the Markdown file
        
    Returns:
        List containing a single Document object
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file is not a Markdown file
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Markdown file not found: {file_path}")
    
    if get_file_extension(file_path) != ".md":
        raise ValueError(f"Expected Markdown file, got: {file_path}")
    
    # UnstructuredMarkdownLoader handles Markdown parsing
    loader = UnstructuredMarkdownLoader(file_path)
    documents = loader.load()
    
    # Enrich metadata with the filename for citations
    filename = Path(file_path).name
    for doc in documents:
        doc.metadata["filename"] = filename
        # Markdown files don't have pages, so we mark this for citation handling
        doc.metadata["page_display"] = None
    
    return documents


def load_document(file_path: str) -> List[Document]:
    """
    Load a single document (PDF or Markdown) based on its extension.
    
    This is the main entry point for loading individual files.
    It automatically detects the file type and uses the appropriate loader.
    
    Args:
        file_path: Path to the document file
        
    Returns:
        List of Document objects extracted from the file
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file type is not supported
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    extension = get_file_extension(file_path)
    
    if extension == ".pdf":
        return load_pdf(file_path)
    elif extension == ".md":
        return load_markdown(file_path)
    else:
        raise ValueError(
            f"Unsupported file type: {extension}. "
            f"Supported types: {', '.join(SUPPORTED_EXTENSIONS)}"
        )


def load_documents_from_directory(
    directory_path: str,
    recursive: bool = False
) -> List[Document]:
    """
    Load all supported documents from a directory.
    
    Args:
        directory_path: Path to the directory containing documents
        recursive: If True, also search subdirectories
        
    Returns:
        List of all Document objects from all supported files
        
    Raises:
        FileNotFoundError: If the directory doesn't exist
    """
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"Directory not found: {directory_path}")
    
    if not os.path.isdir(directory_path):
        raise ValueError(f"Path is not a directory: {directory_path}")
    
    all_documents: List[Document] = []
    
    # Choose between recursive and non-recursive file discovery
    if recursive:
        # Walk through all subdirectories
        for root, _, files in os.walk(directory_path):
            for filename in files:
                file_path = os.path.join(root, filename)
                if is_supported_file(file_path):
                    try:
                        docs = load_document(file_path)
                        all_documents.extend(docs)
                    except Exception as e:
                        # Log warning but continue with other files
                        print(f"Warning: Failed to load {file_path}: {e}")
    else:
        # Only look at files in the immediate directory
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            if os.path.isfile(file_path) and is_supported_file(file_path):
                try:
                    docs = load_document(file_path)
                    all_documents.extend(docs)
                except Exception as e:
                    # Log warning but continue with other files
                    print(f"Warning: Failed to load {file_path}: {e}")
    
    return all_documents


def load_uploaded_files(file_paths: List[str]) -> List[Document]:
    """
    Load multiple documents from a list of file paths.
    
    This function is designed for use with Streamlit's file uploader,
    where users may select multiple files at once.
    
    Args:
        file_paths: List of paths to document files
        
    Returns:
        List of all Document objects from all successfully loaded files
    """
    all_documents: List[Document] = []
    
    for file_path in file_paths:
        if is_supported_file(file_path):
            try:
                docs = load_document(file_path)
                all_documents.extend(docs)
            except Exception as e:
                # Log warning but continue with other files
                print(f"Warning: Failed to load {file_path}: {e}")
        else:
            print(f"Warning: Skipping unsupported file: {file_path}")
    
    return all_documents
