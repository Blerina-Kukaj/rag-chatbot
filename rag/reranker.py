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


# Simple LLM-based reranker (fallback if sentence-transformers not available)
class LLMReranker:
    """
    Simple reranker using OpenAI to score relevance.
    
    Fallback option when cross-encoder models aren't available.
    """
    
    def __init__(self):
        """Initialize LLM reranker."""
        from langchain_openai import ChatOpenAI
        from dotenv import load_dotenv
        
        load_dotenv()
        
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )
    
    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: Optional[int] = None,
    ) -> List[Tuple[Document, float]]:
        """
        Rerank documents using LLM scoring.
        
        Note: This is more expensive than cross-encoder reranking.
        """
        if not documents:
            return []
        
        scored_docs = []
        
        for doc in documents:
            prompt = f"""Rate the relevance of this document to the query on a scale of 0-10.
            
Query: {query}

Document: {doc.page_content[:500]}

Respond with ONLY a number from 0-10."""
            
            try:
                response = self.llm.invoke(prompt)
                score = float(response.content.strip())
            except:
                score = 5.0  # Default score on error
            
            scored_docs.append((doc, score))
        
        # Sort by score
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        if top_k:
            scored_docs = scored_docs[:top_k]
        
        return scored_docs


def get_reranker(use_cross_encoder: bool = True) -> CrossEncoderReranker:
    """
    Get a reranker instance.
    
    Args:
        use_cross_encoder: Whether to use cross-encoder (recommended)
        
    Returns:
        Reranker instance
    """
    if use_cross_encoder:
        return CrossEncoderReranker()
    else:
        return LLMReranker()


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
