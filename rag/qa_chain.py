"""
qa_chain.py - QA Chain Orchestration Module

This module combines the retriever, LLM, and prompt template into a complete
Question-Answering chain using LangChain's RetrievalQA.

The chain ensures grounded responses with proper citations.
"""

import os
from typing import Optional, Dict, Any, List

from langchain_classic.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_core.retrievers import BaseRetriever
from dotenv import load_dotenv

from rag.prompts import get_qa_prompt


# Load environment variables
load_dotenv()

# Default LLM settings
DEFAULT_MODEL = "gpt-3.5-turbo"
DEFAULT_TEMPERATURE = 0.0  # Low temperature for factual, grounded responses


class ConversationMemory:
    """
    Simple conversation memory for maintaining context across turns.
    Stores question-answer pairs and formats them for the prompt.
    """
    
    def __init__(self, max_turns: int = 3):
        self.turns = []
        self.max_turns = max_turns
    
    def add_turn(self, question: str, answer: str, sources: list = None):
        """Add a Q&A turn to memory."""
        self.turns.append({
            "question": question,
            "answer": answer,
            "sources": sources or []
        })
        
        # Keep only recent turns
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns:]
    
    def get_formatted_history(self) -> str:
        """Get formatted chat history for the prompt."""
        if not self.turns:
            return "No previous conversation."
        
        formatted = []
        for turn in self.turns:
            formatted.append(f"User: {turn['question']}")
            formatted.append(f"Assistant: {turn['answer']}")
        
        return "\n".join(formatted)
    
    def clear(self):
        """Clear all conversation history."""
        self.turns = []


def create_llm(
    model_name: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
) -> ChatOpenAI:
    """
    Create and configure an OpenAI-compatible LLM.
    
    Args:
        model_name: Name of the model (default: gpt-3.5-turbo)
        temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative)
        api_key: OpenAI API key (if None, reads from OPENAI_API_KEY env var)
        api_base: Custom API base URL for OpenAI-compatible APIs
        
    Returns:
        Configured ChatOpenAI instance
        
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
    
    # Get custom API base if provided
    if api_base is None:
        api_base = os.getenv("OPENAI_API_BASE")
    
    # Configure LLM
    llm_config = {
        "model": model_name,
        "temperature": temperature,
        "openai_api_key": api_key,
    }
    
    # Add custom base URL if provided
    if api_base:
        llm_config["openai_api_base"] = api_base
    
    llm = ChatOpenAI(**llm_config)
    
    return llm


def create_qa_chain(
    retriever: BaseRetriever,
    llm: Optional[ChatOpenAI] = None,
    return_source_documents: bool = True,
) -> RetrievalQA:
    """
    Create a RetrievalQA chain for question answering.
    
    The chain combines:
    1. Retriever: Finds relevant document chunks
    2. Prompt: Instructs LLM to answer from context only
    3. LLM: Generates grounded answers with citations
    
    Args:
        retriever: Document retriever (from create_retriever())
        llm: Language model (if None, creates default ChatOpenAI)
        return_source_documents: Whether to return source documents with answers
        
    Returns:
        Configured RetrievalQA chain
    """
    # Create default LLM if not provided
    if llm is None:
        llm = create_llm()
    
    # Get the QA prompt template
    prompt = get_qa_prompt()
    
    # Create RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # "stuff" = insert all retrieved docs into prompt
        retriever=retriever,
        return_source_documents=return_source_documents,
        chain_type_kwargs={"prompt": prompt},
    )
    
    return qa_chain


def ask_question(
    qa_chain: RetrievalQA,
    question: str,
    chat_history: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Ask a question using the QA chain.
    
    Args:
        qa_chain: Configured RetrievalQA chain
        question: User's question
        chat_history: Formatted chat history to provide context
        
    Returns:
        Dictionary with:
        - 'query': The original question
        - 'result': The LLM's answer
        - 'source_documents': List of retrieved documents (if enabled)
    """
    if not question or not question.strip():
        return {
            "query": question,
            "result": "Please provide a valid question.",
            "source_documents": [],
        }
    
    # If chat history is provided, prepend it to the question for context
    enhanced_question = question
    if chat_history and chat_history != "No previous conversation.":
        enhanced_question = f"""Previous conversation:
{chat_history}

Current question: {question}

Note: Use the previous conversation to understand context, but answer only from the provided documents."""
    
    # Invoke the chain
    response = qa_chain.invoke({"query": enhanced_question})
    
    # Store the original question in the response
    response["original_query"] = question
    
    return response


def get_source_details(response: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract detailed source information from QA chain response.
    
    This is useful for displaying citations in the UI with content previews.
    
    Args:
        response: Response from ask_question()
        
    Returns:
        List of dictionaries with source details
    """
    source_docs = response.get("source_documents", [])
    
    sources = []
    for i, doc in enumerate(source_docs, 1):
        content = doc.page_content
        
        # Clean content preview by removing Wikipedia attribution
        content_lines = content.split('\n')
        # Filter out lines that contain Wikipedia attribution
        filtered_lines = [line for line in content_lines if not any(phrase in line.lower() for phrase in [
            'source:', 'wikipedia', 'public domain', 'this article is written like'
        ])]
        clean_content = '\n'.join(filtered_lines).strip()
        
        sources.append({
            "index": i,
            "filename": doc.metadata.get("filename", "Unknown"),
            "page": doc.metadata.get("page_display"),
            "chunk_id": doc.metadata.get("chunk_id", 0),
            "content": content,
            "content_preview": clean_content[:300] + "..." if len(clean_content) > 300 else clean_content,
        })
    
    return sources
