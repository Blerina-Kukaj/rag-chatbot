"""
prompts.py - Prompt Templates Module

This module defines prompt templates for the QA chain.
The prompts are designed to enforce grounded responses (no hallucinations)
and ensure proper citation of sources.

Key requirements:
- Answer ONLY from provided context
- Say "I don't know" if answer not in documents
- Always cite sources
"""

from langchain_core.prompts import PromptTemplate, ChatPromptTemplate


# System instruction for grounded QA (prevents hallucinations)
SYSTEM_MESSAGE = """You are a helpful AI assistant that answers questions based ONLY on the provided context documents.

CRITICAL RULES YOU MUST FOLLOW:
1. Answer ONLY using information explicitly stated in the provided context
2. If the answer is NOT in the context, you MUST say: "I cannot find this information in the provided documents."
3. NEVER make up information or use knowledge outside the provided context
4. Always cite your sources using the document references provided (e.g., [filename, Page X, Chunk Y])
5. Be concise and direct in your answers
6. If the context is ambiguous or incomplete, acknowledge this limitation

Remember: It is better to say "I don't know" than to provide incorrect information."""


# Main QA prompt template
QA_PROMPT_TEMPLATE = """Use the following context to answer the question. You must ONLY use information from the context provided below.

CONTEXT:
{context}

QUESTION: {question}

INSTRUCTIONS:
- Answer based ONLY on the context above
- If the answer is not in the context, say "I cannot find this information in the provided documents."
- Include citations in your answer using the format: [Document Name, Page X, Chunk Y]
- Be concise and accurate

ANSWER:"""


def get_qa_prompt() -> PromptTemplate:
    """
    Get the prompt template for the QA chain.
    
    This template instructs the LLM to:
    - Answer only from the provided context
    - Cite sources properly
    - Admit when information is not available (no hallucinations)
    
    Returns:
        PromptTemplate configured for grounded QA
    """
    prompt = PromptTemplate(
        template=QA_PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )
    
    return prompt


def get_chat_qa_prompt() -> ChatPromptTemplate:
    """
    Get a chat-style prompt template for chat models.
    
    This uses separate system and human messages for better
    instruction following with chat-based models like GPT-3.5/4.
    
    Returns:
        ChatPromptTemplate configured for grounded QA
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_MESSAGE),
        ("human", """Context:
{context}

Question: {question}

Please answer the question using ONLY the context provided above. Include citations."""),
    ])
    
    return prompt


# Conversational QA prompt (with chat history support)
CONVERSATIONAL_QA_TEMPLATE = """Use the following context to answer the question. You must ONLY use information from the context provided.

PREVIOUS CONVERSATION:
{chat_history}

CONTEXT:
{context}

CURRENT QUESTION: {question}

INSTRUCTIONS:
- Answer based ONLY on the context above
- You may reference the previous conversation for context about what the user is asking
- If the answer is not in the context, say "I cannot find this information in the provided documents."
- Include citations using the format: [Document Name, Page X, Chunk Y]

ANSWER:"""


def get_conversational_qa_prompt() -> PromptTemplate:
    """
    Get the prompt template for conversational QA with chat history.
    
    This template includes chat history for multi-turn conversations
    while maintaining grounding in the retrieved documents.
    
    Returns:
        PromptTemplate configured for conversational QA
    """
    prompt = PromptTemplate(
        template=CONVERSATIONAL_QA_TEMPLATE,
        input_variables=["chat_history", "context", "question"],
    )
    
    return prompt


def format_chat_history(history: list) -> str:
    """
    Format chat history for inclusion in the prompt.
    
    This enables conversational QA where the model can reference
    previous questions and answers.
    
    Args:
        history: List of (question, answer) tuples or dicts with 'question'/'answer' keys
        
    Returns:
        Formatted chat history string
    """
    if not history:
        return "No previous conversation."
    
    formatted = []
    
    # Take last 3 exchanges to avoid context overflow
    recent_history = history[-3:] if len(history) > 3 else history
    
    for i, exchange in enumerate(recent_history, 1):
        if isinstance(exchange, tuple):
            question, answer = exchange
        elif isinstance(exchange, dict):
            question = exchange.get("question", "")
            answer = exchange.get("answer", "")
        else:
            continue
            
        formatted.append(f"User: {question}")
        formatted.append(f"Assistant: {answer}")
    
    return "\n".join(formatted)


def get_system_message() -> str:
    """
    Get the system message for chat models.
    
    Returns:
        System message string with grounding instructions
    """
    return SYSTEM_MESSAGE
