# RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot project for **Giga Academy Cohort IV - Project #4**.

## Project Goal

Build a chatbot that answers questions using a curated set of documents with:
- Retrieval of most relevant passages
- Grounded answers with citations
- "I don't know" when information isn't available

## Tech Stack

| Component | Technology |
|-----------|------------|
| Framework | LangChain |
| Vector Store | FAISS |
| Embeddings | OpenAI text-embedding-3-small |
| LLM | OpenAI gpt-3.5-turbo |
| Language | Python 3.10+ |

## Setup

```bash
# 1. Clone the repo and enter folder
git clone https://github.com/Blerina-Kukaj/rag-chatbot.git
cd rag-chatbot

# 2. Create virtual environment
python -m venv venv
# macOS/Linux
source venv/bin/activate
# Windows
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
# Edit .env and add your OpenAI API key:
# OPENAI_API_KEY=YOUR_KEY
