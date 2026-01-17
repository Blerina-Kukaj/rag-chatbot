"""
config.py - Application Configuration

This module contains configuration settings for the Streamlit app.
Centralizes all app-level constants and default values.
"""

# =============================================================================
# Page Configuration
# =============================================================================
PAGE_TITLE = "RAG Chatbot - Giga Academy"
PAGE_ICON = "⬛"
LAYOUT = "wide"

# =============================================================================
# App Display Settings
# =============================================================================
APP_TITLE = "RAG Chatbot"

# =============================================================================
# Default Parameters
# =============================================================================
DEFAULT_TOP_K = 5  # Increased for scientific content complexity
DEFAULT_CHUNK_SIZE = 550  # Tokens per chunk (range: 400-500)
DEFAULT_CHUNK_OVERLAP = 100  # Overlap tokens (range: 50-100)

# =============================================================================
# Model Settings
# =============================================================================
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_LLM_MODEL = "gpt-3.5-turbo"
DEFAULT_TEMPERATURE = 0.0  # Low for factual responses

# =============================================================================
# File Upload Settings
# =============================================================================
MAX_FILE_SIZE_MB = 200  # Maximum file size in MB
ALLOWED_EXTENSIONS = ["pdf", "md"]

# =============================================================================
# Directory Paths
# =============================================================================
DATA_DIR = "data"  # Where uploaded documents are saved
VECTORSTORE_DIR = "vectorstore"  # Where FAISS index is persisted

# =============================================================================
# UI Messages
# =============================================================================
WELCOME_MESSAGE = """
 **Welcome to the RAG Chatbot!**

**How to use:**
1.  Click **"Build Knowledge Base"** in the sidebar (first time only)
2.  Ask questions in the chat below
3.  Get grounded answers with citations

*The chatbot answers only from the pre-loaded knowledge base.*
"""

NO_DOCUMENTS_MESSAGE = "⚠️ No documents found in data/ folder. Please add PDF or Markdown files."
NO_VECTORSTORE_MESSAGE = "⚠️ Knowledge base not ready. Please click 'Build Knowledge Base' in the sidebar."
PROCESSING_MESSAGE = "⏳ Processing documents... This may take a moment."
SUCCESS_MESSAGE = "✅ Vector store created successfully! You can now ask questions."
ERROR_API_KEY = "❌ OpenAI API key not found. Please set OPENAI_API_KEY in your .env file."

# =============================================================================
# Citation Display
# =============================================================================
CITATION_HEADER = "**Sources:**"
NO_SOURCES_MESSAGE = "No sources available for this response."
