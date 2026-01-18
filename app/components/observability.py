"""
observability.py - Observability Dashboard Component

This module provides metrics tracking and visualization for the RAG system.
Displays query history, retrieval statistics, and document coverage analytics.
"""

import streamlit as st
from typing import Dict, Any, List
from collections import Counter
import time


def initialize_metrics() -> None:
    """Initialize metrics storage in session state."""
    if "metrics" not in st.session_state:
        st.session_state.metrics = {
            "queries": [],  # List of query records
            "total_queries": 0,
            "total_retrieval_time": 0.0,
            "total_generation_time": 0.0,
            "documents_retrieved": Counter(),  # filename -> count
            "retrieval_methods": Counter(),  # method -> count
            "guardrail_blocks": 0,
        }


def log_query(
    question: str,
    answer: str,
    sources: List[Dict[str, Any]],
    retrieval_method: str,
    retrieval_time: float,
    generation_time: float,
    guardrail_triggered: bool = False,
) -> None:
    """
    Log a query with its metadata for observability.
    
    Args:
        question: User's question
        answer: Generated answer
        sources: Retrieved source documents
        retrieval_method: Method used (vector, hybrid, hybrid+rerank)
        retrieval_time: Time taken for retrieval (seconds)
        generation_time: Time taken for answer generation (seconds)
        guardrail_triggered: Whether guardrails blocked the query
    """
    initialize_metrics()
    
    metrics = st.session_state.metrics
    
    # Record query
    query_record = {
        "timestamp": time.time(),
        "question": question,
        "answer": answer[:200] + "..." if len(answer) > 200 else answer,
        "num_sources": len(sources),
        "retrieval_method": retrieval_method,
        "retrieval_time": retrieval_time,
        "generation_time": generation_time,
        "total_time": retrieval_time + generation_time,
        "guardrail_triggered": guardrail_triggered,
        "sources": [s.get("filename", "Unknown") for s in sources],
    }
    
    metrics["queries"].append(query_record)
    metrics["total_queries"] += 1
    metrics["total_retrieval_time"] += retrieval_time
    metrics["total_generation_time"] += generation_time
    
    # Track document usage
    for source in sources:
        filename = source.get("filename", "Unknown")
        metrics["documents_retrieved"][filename] += 1
    
    # Track retrieval methods
    metrics["retrieval_methods"][retrieval_method] += 1
    
    # Track guardrail blocks
    if guardrail_triggered:
        metrics["guardrail_blocks"] += 1
    
    # Keep only last 100 queries to avoid memory issues
    if len(metrics["queries"]) > 100:
        metrics["queries"] = metrics["queries"][-100:]


def render_observability_dashboard() -> None:
    """Render the observability dashboard with metrics and charts."""
    initialize_metrics()
    
    st.header("Observability Dashboard")
    st.caption("Real-time metrics and analytics for the RAG system")
    
    metrics = st.session_state.metrics
    
    # =================================================================
    # Summary Metrics
    # =================================================================
    st.subheader("Summary Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Queries", metrics["total_queries"])
    
    with col2:
        avg_retrieval = (
            metrics["total_retrieval_time"] / metrics["total_queries"]
            if metrics["total_queries"] > 0
            else 0
        )
        st.metric("Avg Retrieval Time", f"{avg_retrieval:.2f}s")
    
    with col3:
        avg_generation = (
            metrics["total_generation_time"] / metrics["total_queries"]
            if metrics["total_queries"] > 0
            else 0
        )
        st.metric("Avg Generation Time", f"{avg_generation:.2f}s")
    
    with col4:
        st.metric("Guardrail Blocks", metrics["guardrail_blocks"])
    
    st.divider()
    
    # =================================================================
    # Query History
    # =================================================================
    st.subheader("Recent Query History")
    
    if metrics["queries"]:
        # Show last 10 queries
        recent_queries = metrics["queries"][-10:][::-1]  # Reverse for newest first
        
        for i, query in enumerate(recent_queries, 1):
            with st.expander(
                f"Query {len(metrics['queries']) - i + 1}: {query['question'][:60]}...",
                expanded=False
            ):
                st.markdown(f"**Question:** {query['question']}")
                st.markdown(f"**Answer:** {query['answer']}")
                st.markdown(f"**Retrieval Method:** {query['retrieval_method']}")
                st.markdown(f"**Sources Retrieved:** {query['num_sources']}")
                st.markdown(f"**Retrieval Time:** {query['retrieval_time']:.3f}s")
                st.markdown(f"**Generation Time:** {query['generation_time']:.3f}s")
                st.markdown(f"**Total Time:** {query['total_time']:.3f}s")
                
                if query['sources']:
                    st.markdown("**Documents Used:**")
                    for doc in set(query['sources']):
                        st.text(f"  • {doc}")
    else:
        st.info("No queries yet. Start asking questions to see metrics!")
    
    st.divider()
    
    # =================================================================
    # Document Coverage
    # =================================================================
    st.subheader("Document Coverage")
    
    if metrics["documents_retrieved"]:
        st.caption("How often each document is retrieved in answers")
        
        # Create table data
        doc_data = []
        for doc, count in metrics["documents_retrieved"].most_common():
            usage_pct = (count / metrics["total_queries"]) * 100 if metrics["total_queries"] > 0 else 0
            doc_data.append({
                "Document": doc,
                "Times Retrieved": count,
                "Usage %": f"{usage_pct:.1f}%"
            })
        
        # Display as dataframe
        st.dataframe(doc_data, use_container_width=True, hide_index=True)
        
        # Simple bar chart
        if len(doc_data) > 0:
            st.bar_chart(
                data={row["Document"]: row["Times Retrieved"] for row in doc_data},
                height=300
            )
    else:
        st.info("No document usage data yet.")
    
    st.divider()
    
    # =================================================================
    # Retrieval Method Distribution
    # =================================================================
    st.subheader("Retrieval Method Distribution")
    
    if metrics["retrieval_methods"]:
        st.caption("Distribution of retrieval methods used")
        
        method_data = []
        for method, count in metrics["retrieval_methods"].most_common():
            pct = (count / metrics["total_queries"]) * 100 if metrics["total_queries"] > 0 else 0
            method_data.append({
                "Method": method,
                "Count": count,
                "Percentage": f"{pct:.1f}%"
            })
        
        st.dataframe(method_data, use_container_width=True, hide_index=True)
        
        # Pie chart representation
        if len(method_data) > 0:
            method_chart_data = {row["Method"]: row["Count"] for row in method_data}
            st.bar_chart(method_chart_data, height=250, use_container_width=True)
    else:
        st.info("No retrieval method data yet.")
    
    st.divider()
    
    # =================================================================
    # Performance Analytics
    # =================================================================
    st.subheader("Performance Analytics")
    
    if len(metrics["queries"]) > 1:
        st.caption("Response time trends over recent queries")
        
        # Get last 20 queries for trend analysis
        recent = metrics["queries"][-20:]
        
        # Create time series data
        time_data = {
            "Query #": list(range(len(recent))),
            "Retrieval Time (s)": [q["retrieval_time"] for q in recent],
            "Generation Time (s)": [q["generation_time"] for q in recent],
            "Total Time (s)": [q["total_time"] for q in recent],
        }
        
        st.line_chart(
            data={
                "Retrieval": time_data["Retrieval Time (s)"],
                "Generation": time_data["Generation Time (s)"],
            },
            height=300
        )
        
        # Performance statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            min_time = min(q["total_time"] for q in recent)
            st.metric("Fastest Query", f"{min_time:.3f}s")
        
        with col2:
            max_time = max(q["total_time"] for q in recent)
            st.metric("Slowest Query", f"{max_time:.3f}s")
        
        with col3:
            avg_time = sum(q["total_time"] for q in recent) / len(recent)
            st.metric("Average Time", f"{avg_time:.3f}s")
    else:
        st.info("Need more queries for performance analytics.")


def get_metrics_summary() -> Dict[str, Any]:
    """
    Get a summary of metrics for display in other components.
    
    Returns:
        Dictionary with key metrics
    """
    initialize_metrics()
    metrics = st.session_state.metrics
    
    return {
        "total_queries": metrics["total_queries"],
        "avg_retrieval_time": (
            metrics["total_retrieval_time"] / metrics["total_queries"]
            if metrics["total_queries"] > 0
            else 0
        ),
        "avg_generation_time": (
            metrics["total_generation_time"] / metrics["total_queries"]
            if metrics["total_queries"] > 0
            else 0
        ),
        "guardrail_blocks": metrics["guardrail_blocks"],
        "most_used_document": (
            metrics["documents_retrieved"].most_common(1)[0][0]
            if metrics["documents_retrieved"]
            else None
        ),
        "primary_retrieval_method": (
            metrics["retrieval_methods"].most_common(1)[0][0]
            if metrics["retrieval_methods"]
            else None
        ),
    }
