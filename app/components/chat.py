"""
chat.py - Chat Interface Component

This module handles the chat interface with message display and citation rendering.
Provides a conversational UI with source attribution for each answer.
"""

import streamlit as st
from typing import List, Dict, Any, Optional


def initialize_chat_history() -> None:
    """Initialize chat history in session state if not present."""
    if "messages" not in st.session_state:
        st.session_state.messages = []


def add_message(
    role: str,
    content: str,
    sources: Optional[List[Dict[str, Any]]] = None
) -> None:
    """
    Add a message to chat history.
    
    Args:
        role: "user" or "assistant"
        content: Message content
        sources: List of source documents (for assistant messages)
    """
    message = {
        "role": role,
        "content": content,
    }
    
    if sources:
        message["sources"] = sources
    
    st.session_state.messages.append(message)


def display_citations(sources: List[Dict[str, Any]]) -> None:
    """
    Display citations in a formatted expandable section.
    
    Args:
        sources: List of source document dictionaries with keys:
                 - filename, page, chunk_id, content_preview
    """
    if not sources:
        return
    
    st.markdown("---")
    st.markdown("**Sources:**")
    
    for i, source in enumerate(sources, 1):
        filename = source.get("filename", "Unknown")
        page = source.get("page")
        chunk_id = source.get("chunk_id", 0)
        content_preview = source.get("content_preview", "")
        
        # Build citation label
        if page is not None:
            citation_label = f"{filename} | Page {page} | Chunk {chunk_id}"
        else:
            citation_label = f"{filename} | Chunk {chunk_id}"
        
        # Display as expander with content preview
        with st.expander(citation_label, expanded=False):
            st.text(content_preview)


def display_message(message: Dict[str, Any]) -> None:
    """
    Display a single message with optional citations.
    
    Args:
        message: Dictionary with role, content, and optional sources
    """
    role = message.get("role", "user")
    content = message.get("content", "")
    sources = message.get("sources", [])
    
    # Custom avatars for unique branding
    if role == "user":
        avatar = "👤"  # Person icon for user
    elif role == "assistant":
        avatar = "🤖"  # Robot icon for assistant
    else:
        avatar = "❓"  # Question mark for unknown
    
    with st.chat_message(role, avatar=avatar):
        st.markdown(content)
        
        # Display citations for assistant messages
        # Only show sources if the answer doesn't indicate "I don't know"
        if role == "assistant" and sources:
            # Check if response indicates no answer was found
            no_answer_phrases = [
                "I cannot find this information",
                "I cannot find this information in the provided",
                "I don't know",
                "I do not know",
                "not found in the provided",
                "cannot answer this question",
                "no information available",
                "unable to find"
            ]
            
            has_answer = not any(phrase.lower() in content.lower() for phrase in no_answer_phrases)
            
            if has_answer:
                display_citations(sources)


def display_chat_history() -> None:
    """Display all messages from chat history."""
    for message in st.session_state.messages:
        display_message(message)


def clear_chat_history() -> None:
    """Clear all messages from chat history."""
    st.session_state.messages = []


def render_chat_input() -> Optional[str]:
    """
    Render the chat input field.
    
    Returns:
        User's input question if submitted, otherwise None
    """
    return st.chat_input("Ask a question about your documents...")


def display_user_message(question: str) -> None:
    """
    Display a user message immediately (before adding to history).
    
    Args:
        question: User's question
    """
    with st.chat_message("user", avatar="👤"):
        st.markdown(question)


def display_assistant_response(
    answer: str,
    sources: List[Dict[str, Any]]
) -> None:
    """
    Display an assistant response with citations.
    
    Args:
        answer: The generated answer
        sources: List of source documents
    """
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(answer)
        
        if sources:
            display_citations(sources)


def display_thinking_indicator() -> Any:
    """
    Display a thinking/processing indicator.
    
    Returns:
        Spinner context manager
    """
    return st.spinner("Searching documents and generating answer...")


def display_welcome_message(message: str) -> None:
    """
    Display a welcome/info message.
    
    Args:
        message: Welcome message to display
    """
    st.markdown(f"**{message}**")


def display_error_message(message: str) -> None:
    """
    Display an error message.
    
    Args:
        message: Error message to display
    """
    st.markdown(f"**Error:** {message}")


def display_success_message(message: str) -> None:
    """
    Display a success message.
    
    Args:
        message: Success message to display
    """
    st.markdown(f"**{message}**")


def display_warning_message(message: str) -> None:
    """
    Display a warning message.
    
    Args:
        message: Warning message to display
    """
    st.markdown(f"**Warning:** {message}")


def render_chat_input() -> Optional[str]:
    """
    Render the chat input field.
    
    Returns:
        User's input question if submitted, otherwise None
    """
    # Add some spacing
    st.markdown("")
    
    # Create the chat input
    user_input = st.chat_input(
        placeholder="Ask a question about the documents...",
        key="chat_input"
    )
    
    return user_input if user_input and user_input.strip() else None


def render_chat_interface() -> Optional[str]:
    """
    Render the complete chat interface.
    
    This includes:
    - Chat history display
    - Chat input field
    
    Returns:
        User's input question if submitted, otherwise None
    """
    # Initialize chat history
    initialize_chat_history()
    
    # Display chat history
    display_chat_history()
    
    # Render chat input and return user's question
    return render_chat_input()
