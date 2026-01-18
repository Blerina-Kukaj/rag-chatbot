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

from langchain_core.prompts import PromptTemplate


# System instruction for grounded QA (prevents hallucinations)
SYSTEM_MESSAGE = """You are a helpful AI assistant that answers questions based ONLY on the provided scientific research context.

CRITICAL RULES YOU MUST FOLLOW:
1. Answer ONLY using information explicitly stated in the provided research papers
2. If the answer is NOT in the context, you MUST say: "I cannot find this information in the provided research documents."
3. NEVER make up information or use knowledge outside the provided scientific context
4. Focus on evidence-based findings from the research papers
5. When discussing AI applications, reference specific methodologies or results when available
6. Be precise about study findings, limitations, and conclusions
7. NEVER include any citation formatting in your answers - citations are added automatically

Remember: Base answers on actual research findings, not general knowledge."""


# Main QA prompt template
QA_PROMPT_TEMPLATE = """Use the following scientific research context to answer the question. You must ONLY use information from the research papers provided.

RESEARCH CONTEXT:
{context}

QUESTION: {question}

INSTRUCTIONS:
- Answer based ONLY on the research context above
- Reference specific findings, methodologies, or results when relevant
- If the answer is not in the research context, say "I cannot find this information in the provided research documents."
- Be evidence-based and cite study implications when appropriate
- IMPORTANT: Do NOT include ANY citation formatting in your answer
- Just provide the research-based answer

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
