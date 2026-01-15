"""
hybrid_search.py - Hybrid Search Module

Combines BM25 (keyword/lexical) search with vector (semantic) search
for improved retrieval quality. Uses Reciprocal Rank Fusion (RRF)
to merge results from both methods.
"""

from typing import List, Tuple, Optional
from collections import defaultdict

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever


# Default weights for hybrid search
DEFAULT_VECTOR_WEIGHT = 0.5
DEFAULT_BM25_WEIGHT = 0.5


class HybridRetriever:
    """
    Hybrid retriever combining BM25 and vector search.
    
    Uses Reciprocal Rank Fusion (RRF) to combine rankings from
    both retrieval methods for better results.
    """
    
    def __init__(
        self,
        vectorstore: FAISS,
        documents: List[Document],
        vector_weight: float = DEFAULT_VECTOR_WEIGHT,
        bm25_weight: float = DEFAULT_BM25_WEIGHT,
    ):
        """
        Initialize hybrid retriever.
        
        Args:
            vectorstore: FAISS vector store for semantic search
            documents: Original documents for BM25 index
            vector_weight: Weight for vector search results (0-1)
            bm25_weight: Weight for BM25 results (0-1)
        """
        self.vectorstore = vectorstore
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        
        # Create BM25 retriever from documents
        self.bm25_retriever = BM25Retriever.from_documents(documents)
    
    def _reciprocal_rank_fusion(
        self,
        rankings: List[List[Document]],
        weights: List[float],
        k: int = 60,  # RRF constant
    ) -> List[Tuple[Document, float]]:
        """
        Combine multiple rankings using Reciprocal Rank Fusion.
        
        RRF Score = sum(weight_i / (k + rank_i)) for each ranking
        
        Args:
            rankings: List of document rankings from different methods
            weights: Weights for each ranking
            k: RRF constant (higher = more equal weighting)
            
        Returns:
            List of (document, score) tuples sorted by combined score
        """
        # Track scores by document content (as unique identifier)
        doc_scores = defaultdict(float)
        doc_map = {}  # content -> document
        
        for ranking, weight in zip(rankings, weights):
            for rank, doc in enumerate(ranking, 1):
                # Use content hash as key
                key = hash(doc.page_content)
                doc_scores[key] += weight / (k + rank)
                doc_map[key] = doc
        
        # Sort by combined score
        sorted_docs = sorted(
            doc_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [(doc_map[key], score) for key, score in sorted_docs]
    
    def retrieve(
        self,
        query: str,
        k: int = 3,
    ) -> List[Document]:
        """
        Retrieve documents using hybrid search.
        
        Args:
            query: Search query
            k: Number of documents to return
            
        Returns:
            List of top-k documents
        """
        if not query or not query.strip():
            return []
        
        # Get results from both methods (fetch more for better fusion)
        fetch_k = k * 3
        
        # Vector search
        vector_results = self.vectorstore.similarity_search(query, k=fetch_k)
        
        # BM25 search
        self.bm25_retriever.k = fetch_k
        bm25_results = self.bm25_retriever.invoke(query)
        
        # Combine using RRF
        combined = self._reciprocal_rank_fusion(
            rankings=[vector_results, bm25_results],
            weights=[self.vector_weight, self.bm25_weight],
        )
        
        # Return top-k
        return [doc for doc, score in combined[:k]]
    
    def retrieve_with_scores(
        self,
        query: str,
        k: int = 3,
    ) -> List[Tuple[Document, float]]:
        """
        Retrieve documents with their hybrid scores.
        
        Args:
            query: Search query
            k: Number of documents to return
            
        Returns:
            List of (document, score) tuples
        """
        if not query or not query.strip():
            return []
        
        fetch_k = k * 3
        
        vector_results = self.vectorstore.similarity_search(query, k=fetch_k)
        self.bm25_retriever.k = fetch_k
        bm25_results = self.bm25_retriever.invoke(query)
        
        combined = self._reciprocal_rank_fusion(
            rankings=[vector_results, bm25_results],
            weights=[self.vector_weight, self.bm25_weight],
        )
        
        return combined[:k]


def create_hybrid_retriever(
    vectorstore: FAISS,
    documents: List[Document],
    vector_weight: float = DEFAULT_VECTOR_WEIGHT,
    bm25_weight: float = DEFAULT_BM25_WEIGHT,
) -> HybridRetriever:
    """
    Create a hybrid retriever combining vector and BM25 search.
    
    Args:
        vectorstore: FAISS vector store
        documents: Original chunked documents
        vector_weight: Weight for vector search (0-1)
        bm25_weight: Weight for BM25 search (0-1)
        
    Returns:
        Configured HybridRetriever instance
    """
    return HybridRetriever(
        vectorstore=vectorstore,
        documents=documents,
        vector_weight=vector_weight,
        bm25_weight=bm25_weight,
    )
