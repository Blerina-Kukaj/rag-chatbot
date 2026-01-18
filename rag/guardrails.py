"""
guardrails.py - Security and Safety Module

Implements prompt injection detection and defense mechanisms to prevent
malicious manipulation of the RAG system.

Features:
- Prompt injection detection in user queries
- Defense against "ignore previous instructions" attacks
- Document instruction filtering
"""

import re
from typing import Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class GuardrailResult:
    """Result of guardrail check."""
    is_safe: bool
    risk_level: str  # "low", "medium", "high"
    detected_issues: List[str]
    sanitized_input: Optional[str] = None


# Patterns that indicate potential prompt injection
INJECTION_PATTERNS = [
    # Ignore instructions patterns
    r"ignore\s+(all\s+)?(previous|above|prior|the\s+)?\s*(instructions?|prompts?|rules?)",
    r"disregard\s+(all\s+)?(previous|above|prior|the\s+)?\s*(instructions?|prompts?|rules?)",
    r"forget\s+(all\s+)?(previous|above|prior|the\s+)?\s*(instructions?|prompts?|rules?)",
    r"override\s+(all\s+)?(previous|above|prior|the\s+)?\s*(instructions?|prompts?|rules?)",
    
    # Role manipulation patterns
    r"you\s+are\s+(now|no longer)\s+a",
    r"act\s+as\s+(if\s+you\s+are|a)",
    r"pretend\s+(to\s+be|you\s+are)",
    r"roleplay\s+as",
    r"switch\s+to\s+.+\s+mode",
    
    # System prompt extraction
    r"(show|reveal|display|print|output)\s+(me\s+)?(your|the)\s+(system\s+)?(prompt|instructions?)",
    r"what\s+(are|is)\s+your\s+(system\s+)?(prompt|instructions?)",
    
    # Jailbreak attempts
    r"DAN\s+mode",
    r"developer\s+mode",
    r"bypass\s+(safety|content|filter)",
    r"override\s+(safety|restrictions?|limitations?)",
    
    # Code execution attempts
    r"```\s*(python|bash|shell|exec)",
    r"(run|execute)\s+(this\s+)?(code|command|script)",
    
    # Output manipulation
    r"(start|begin)\s+your\s+(response|answer)\s+with",
    r"respond\s+only\s+with",
    r"your\s+(first|next)\s+word\s+(must|should)\s+be",
    
    # Medical/scientific injection attempts
    r"cure\s+(all\s+)?(diseases?|cancers?|conditions?)",
    r"cure\s+my\s+(cancer|disease|condition|illness)",
    r"how\s+(do\s+I|can\s+I|should\s+I)\s+(cure|treat|heal)\s+(my\s+)?(cancer|disease|condition)",
    r"can\s+(you|AI)\s+cure\s+(my\s+)?(cancer|disease|condition)",
    r"diagnose\s+(my|me|this)\s+(condition|disease|problem|symptoms?|illness|pain)",
    r"prescribe\s+(medication|drugs?|treatment)",
    r"medical\s+advice",
    r"health\s+(recommendation|prescription)",
]

# Patterns in documents that try to manipulate the AI
DOCUMENT_INJECTION_PATTERNS = [
    r"\[SYSTEM\]",
    r"\[INSTRUCTION\]",
    r"\[IGNORE\s+PREVIOUS\]",
    r"AI:\s*ignore",
    r"ASSISTANT:\s*ignore",
    r"<\s*system\s*>",
    r"<\s*instruction\s*>",
]


def detect_prompt_injection(text: str) -> GuardrailResult:
    """
    Detect potential prompt injection in user input.
    
    Args:
        text: User input text to check
        
    Returns:
        GuardrailResult with safety assessment
    """
    if not text:
        return GuardrailResult(
            is_safe=True,
            risk_level="low",
            detected_issues=[],
            sanitized_input=text,
        )
    
    text_lower = text.lower()
    detected_issues = []
    
    # Check against injection patterns
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            detected_issues.append(f"Matched pattern: {pattern[:50]}...")
    
    # Determine risk level
    if len(detected_issues) == 0:
        risk_level = "low"
        is_safe = True
    elif len(detected_issues) <= 2:
        risk_level = "medium"
        is_safe = True  # Allow but flag
    else:
        risk_level = "high"
        is_safe = False
    
    return GuardrailResult(
        is_safe=is_safe,
        risk_level=risk_level,
        detected_issues=detected_issues,
        sanitized_input=text if is_safe else None,
    )


def sanitize_document_content(content: str) -> str:
    """
    Sanitize document content to prevent injection via documents.
    
    Removes or neutralizes potential instruction injection attempts
    embedded in documents.
    
    Args:
        content: Document content to sanitize
        
    Returns:
        Sanitized content
    """
    if not content:
        return content
    
    sanitized = content
    
    # Remove potential injection markers
    for pattern in DOCUMENT_INJECTION_PATTERNS:
        sanitized = re.sub(pattern, "[REMOVED]", sanitized, flags=re.IGNORECASE)
    
    return sanitized


def validate_input(user_input: str) -> Tuple[bool, str, Optional[str]]:
    """
    Validate and sanitize user input.
    
    Args:
        user_input: Raw user input
        
    Returns:
        Tuple of (is_valid, message, sanitized_input)
    """
    # Check for empty input
    if not user_input or not user_input.strip():
        return False, "Please provide a valid question.", None
    
    # Check for prompt injection
    result = detect_prompt_injection(user_input)
    
    if not result.is_safe or result.risk_level in ["medium", "high"]:
        return False, (
            "I detected potentially unsafe content in your question. "
            "Please rephrase your question."
        ), None
    
    if result.risk_level == "medium":
        # Log warning but allow (could add logging here)
        pass
    
    return True, "Input validated", user_input.strip()
