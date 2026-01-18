"""
reranker.py - Document Reranking Module

Uses cross-encoder models to rerank retrieved documents for improved relevance.
Reranking is applied after initial retrieval to refine the top results.
"""

from typing import List, Tuple, Optional
import os

from langchain_core.documents import Document


class CrossEncoderReranker:
    """
    Reranker using cross-encoder models for semantic similarity scoring.
    
    Cross-encoders jointly encode query and document for more accurate
    relevance scoring compared to bi-encoders (embeddings).
    """
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize the cross-encoder reranker.
        
        Args:
            model_name: HuggingFace model name for cross-encoder
        """
        self.model_name = model_name
        self._model = None
    
    def _load_model(self):
        """Lazy load the cross-encoder model."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name)
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for reranking. "
                    "Install with: pip install sentence-transformers"
                )
        return self._model
    
    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: Optional[int] = None,
    ) -> List[Tuple[Document, float]]:
        """
        Rerank documents based on relevance to the query.
        
        Args:
            query: Search query
            documents: List of documents to rerank
            top_k: Number of top documents to return (None = all)
            
        Returns:
            List of (document, score) tuples sorted by relevance
        """
        if not documents:
            return []
        
        if not query or not query.strip():
            return [(doc, 0.0) for doc in documents]
        
        model = self._load_model()
        
        # Prepare pairs for cross-encoder
        pairs = [(query, doc.page_content) for doc in documents]
        
        # Get relevance scores
        scores = model.predict(pairs)
        
        # Combine documents with scores
        doc_scores = list(zip(documents, scores))
        
        # Sort by score (descending)
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top_k if specified
        if top_k is not None:
            doc_scores = doc_scores[:top_k]
        
        return doc_scores
    
    def rerank_documents(
        self,
        query: str,
        documents: List[Document],
        top_k: Optional[int] = None,
    ) -> List[Document]:
        """
        Rerank and return only the documents (without scores).
        
        Args:
            query: Search query
            documents: List of documents to rerank
            top_k: Number of top documents to return
            
        Returns:
            List of reranked documents
        """
        reranked = self.rerank(query, documents, top_k)
        return [doc for doc, score in reranked]


def get_reranker() -> CrossEncoderReranker:
    """
    Get a reranker instance.
    
    Returns:
        CrossEncoderReranker instance
    """
    return CrossEncoderReranker()


def rerank_documents(
    query: str,
    documents: List[Document],
    top_k: Optional[int] = None,
) -> List[Document]:
    """
    Convenience function to rerank documents.
    
    Args:
        query: Search query
        documents: Documents to rerank
        top_k: Number of top documents to return
        
    Returns:
        Reranked documents
    """
    reranker = CrossEncoderReranker()
    return reranker.rerank_documents(query, documents, top_k)
