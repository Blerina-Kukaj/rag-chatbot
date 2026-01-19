# RAG Chatbot for Scientific Research

A sophisticated Retrieval-Augmented Generation (RAG) chatbot optimized for scientific content, specifically designed for **Giga Academy Cohort IV - Project #4**. This system provides grounded, evidence-based answers from a curated dataset of AI healthcare research papers with proper citations and guardrails.

 **Live Demo**: [Deployed on Streamlit Cloud]

## Project Overview

This RAG chatbot answers questions using a collection of 37 PMC AI healthcare research papers, providing:
- **Grounded answers** with citations from scientific literature
- **"I don't know"** responses when information isn't available
- **Hybrid search** combining keyword and semantic matching
- **Cross-encoder reranking** for improved relevance
- **Guardrails** protecting against prompt injection and medical advice

### Must-Have Requirements (Completed)
- Document ingestion pipeline from PDFs
- Vector search with FAISS
- Grounded responses with citations
- "I don't know" handling for unavailable information
- Simple, clean web UI

### Nice-to-Have Features (5 Implemented!)

**Core Features (3 required):**
- **Hybrid Search**: BM25 + semantic search with Reciprocal Rank Fusion
- **Reranking**: Cross-encoder model for improved document ranking  
- **Guardrails**: Advanced prompt injection detection with medical patterns

**Additional Features :**
- **Conversation Memory**: Short-term context (3 turns) while maintaining retrieval grounding
- **Observability Dashboard**: Real-time metrics, query history, and performance analytics

## Dataset

**37 PMC AI Healthcare Research Papers** (1,218+ chunks)
- Scientific content optimization (550-token chunks, 100-token overlap)
- Topics: AI in medical diagnosis, drug discovery, patient monitoring, radiology, personalized medicine
- Source: PubMed Central (PMC) - peer-reviewed research

## Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Framework** | LangChain 0.2+ | RAG orchestration |
| **Vector Store** | FAISS | Efficient similarity search |
| **Embeddings** | OpenAI text-embedding-3-small | Semantic encoding |
| **LLM** | OpenAI gpt-3.5-turbo | Answer generation |
| **Hybrid Search** | rank-bm25 | Keyword-based retrieval |
| **Reranking** | sentence-transformers | Cross-encoder reranking |
| **UI** | Streamlit 1.32+ | Web interface |
| **Document Processing** | PyPDF, LangChain | PDF ingestion |
| **Language** | Python 3.10+ | Core implementation |

## Quick Start

### Option 1: Use Deployed Version
Visit the [live demo] - the vector store is pre-built and loads instantly!

### Option 2: Run Locally

#### Prerequisites
- Python 3.11+
- OpenAI API key

#### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Blerina-Kukaj/rag-chatbot.git
cd rag-chatbot

# 2. Create virtual environment
python -m venv venv

# 3. Activate environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Configure environment
cp .env.example .env
# Edit .env and add your OpenAI API key:
# OPENAI_API_KEY=your_openai_api_key_here
```

### Running the Application

```bash
# Start the Streamlit app
streamlit run app/main.py

# The app will open in your browser at http://localhost:8501
```

### First-Time Setup (Local)
1. **Pre-built Vector Store**: The vector store is included in the repository (1,218 chunks from 37 PDFs)
2. **Instant Loading**: No waiting for document processing - the index loads immediately!
3. **Ready to Query**: Start asking questions about AI in healthcare right away

*To rebuild the vector store from scratch (optional):* Use the "Clear Vector Store" button in the sidebar, then click "Build Knowledge Base".

## Usage Examples

### Sample Questions
```
"What is AI in healthcare?"
"How does AI help with medical imaging?"
"What role does AI play in drug discovery?"
"How is AI used in personalized medicine?"
```

### Advanced Features
- **Hybrid Search**: Toggle in sidebar for keyword + semantic retrieval
- **Reranking**: Enable for improved answer relevance
- **Conversation Memory**: Maintains context across 3 turns
- **Guardrails**: Automatically blocks unsafe prompts
- **Observability Dashboard**: Real-time metrics, query history, performance charts

## Testing

Comprehensive test scenarios are documented in [TEST_SCENARIOS.md](TEST_SCENARIOS.md), covering:
- Basic functionality (grounding, citations, "I don't know" responses)
- Hybrid search and reranking
- Guardrails (prompt injection, medical advice)
- Conversation memory (context maintenance without hallucination)
- Observability dashboard


## Architecture

```
📁 rag-chatbot/
├── 📁 data/                 # PMC AI healthcare PDFs (37 files)
├── 📁 vectorstore/          # FAISS index & metadata
├── 📁 rag/                  # Core RAG components
│   ├── ingestion.py         # Document loading
│   ├── chunking.py          # Scientific text chunking
│   ├── embeddings.py        # OpenAI embeddings
│   ├── vector_store.py      # FAISS management
│   ├── retriever.py         # Document retrieval
│   ├── hybrid_search.py     # BM25 + vector search
│   ├── reranker.py          # Cross-encoder reranking
│   ├── qa_chain.py          # LangChain QA pipeline
│   ├── prompts.py           # Scientific prompts
│   └── guardrails.py        # Safety & injection protection
├── 📁 app/                  # Streamlit application
│   ├── main.py             # Main app entry point
│   ├── components/         # UI components
│   │   ├── chat.py         # Chat interface
│   │   ├── sidebar.py      # Settings panel
│   │   └── observability.py # Dashboard & metrics
│   └── config.py           # App configuration
└── requirements.txt        # Python dependencies
```

### Data Flow
1. **Ingestion**: Load 37 PMC PDFs
2. **Chunking**: Split into 550-token chunks (100 overlap)
3. **Embedding**: Convert to vectors using OpenAI
4. **Indexing**: Store in FAISS vector database
5. **Retrieval**: Hybrid search (BM25 + semantic)
6. **Reranking**: Cross-encoder for relevance
7. **Generation**: GPT-3.5-turbo with scientific prompts
8. **Guardrails**: Safety filtering
9. **UI**: Streamlit with citations

## UI Features

### Chat Interface
- **Clean Design**: Monochromatic theme optimized for readability
- **Citation Display**: Expandable sources with content previews
- **Message History**: Persistent conversation tracking
- **Error Handling**: Clear feedback for API issues

### Sidebar Controls
- **Search Settings**: Toggle hybrid search and reranking
- **Vector Store Management**: Rebuild or clear index
- **Document Info**: View dataset statistics
- **Settings**: API key configuration

## Security & Guardrails

- **Prompt Injection Detection**: Blocks malicious inputs
- **Medical Advice Filtering**: Prevents healthcare recommendations
- **API Key Protection**: Secure environment variable handling (Streamlit Secrets for deployment)
- **Input Validation**: Sanitizes user queries

## Deployment

This project is deployed on **Streamlit Cloud** with:
- Pre-built FAISS vector store for instant loading
- Secure API key management via Streamlit Secrets
- Python 3.11 runtime
- All 37 PDFs and processed chunks included

## Acknowledgments

- **Giga Academy Cohort IV** for the project requirements
- **PubMed Central (PMC)** for the AI healthcare research papers
- **OpenAI** for embeddings and LLM access
- **LangChain** for the RAG framework
- **FAISS** for efficient vector search

