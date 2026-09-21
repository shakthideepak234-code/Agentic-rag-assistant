# Agentic RAG Assistant ("Apple")

An intelligent, multi-source Agentic Retrieval-Augmented Generation (RAG) assistant built with Python, Google Gemini 3.5 Flash Lite, ChromaDB, LangChain, FastAPI, Vercel, and Streamlit.

The assistant dynamically evaluates user queries to route them between a private vector knowledge base (ChromaDB) and live web search (DuckDuckGo), maintaining conversational context across turns.

---

## Key Features

- Retrieval-Augmented Generation (RAG): Document chunking and embedding ingestion into ChromaDB using LangChain text splitters.
- Live Web Search Integration: Fallback to real-time DuckDuckGo web search for current events, news, or out-of-domain queries.
- Agentic Routing & Decision Loop: Automatically decides whether to pull from internal documents or live web results based on query relevance.
- Session Memory: Remembers multi-turn conversation context across interactions.
- Multiple Deployments & Interfaces:
  - Vercel Web App: Serverless FastAPI backend with static HTML/JS frontend (`api/index.py`, `public/index.html`).
  - Streamlit Dashboard: Interactive web application (`streamlit_app.py`).
  - CLI Terminal: Fast command-line interface (`agent.py`).

---

## System Architecture

```text
               User Question (Vercel Web UI / Streamlit / Terminal CLI)
                                  │
                                  ▼
                     Agent Decision & Routing Engine
                                  │
        ┌────────────────────────┴────────────────────────┐
        ▼                                                 ▼
  Knowledge Base Retrieval                         Live Web Search
  (ChromaDB Vector Store)                        (DuckDuckGo Search)
        │                                                 │
        └────────────────────────┬────────────────────────┘
                                  │
                                  ▼
                      Gemini 3.5 Flash Lite LLM
                         + Memory Context
                                  │
                                  ▼
                         Source-Attributed Answer
```

---

## Getting Started

### 1. Prerequisites
- Python 3.10+
- Google Gemini API Key ([Get a key from Google AI Studio](https://aistudio.google.com/apikey))

### 2. Installation & Setup

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/shakthideepak234-code/Agentic-rag-assistant.git
   cd Agentic-rag-assistant
   ```

2. **Create & Activate Virtual Environment:**
   ```bash
   python -m venv .venv
   # Windows (PowerShell)
   .\.venv\Scripts\Activate.ps1
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables:**
   Create a `.env` file in the root directory:
   ```env
   GOOGLE_API_KEY=your_gemini_api_key_here
   ```

---

## Usage & Running Locally

### 1. Ingest Knowledge Base Documents
Populate ChromaDB with documents from `documents/ai_notes.txt`:
```bash
python rag.py
```

### 2. Run the Vercel API Locally (FastAPI)
```bash
uvicorn api.index:app --reload
```

### 3. Run the Streamlit App
Launch the Streamlit interface:
```bash
streamlit run streamlit_app.py
```

### 4. Run the Terminal Agent (CLI)
Chat directly in your terminal:
```bash
python agent.py
```

---

## Deployment Guide

### Deploying to Vercel
1. Import `shakthideepak234-code/Agentic-rag-assistant` into Vercel.
2. Under **Environment Variables**, add `GOOGLE_API_KEY`.
3. Click **Deploy**. Vercel will automatically build the FastAPI backend (`api/index.py`) and serve the frontend (`public/index.html`).

### Deploying to Streamlit Community Cloud
1. Connect your repository to [Streamlit Community Cloud](https://share.streamlit.io/).
2. Set `streamlit_app.py` as the main file.
3. Add `GOOGLE_API_KEY` under Secrets.

---

## Project Structure

```text
Agentic-rag-assistant/
├── api/
│   └── index.py          # FastAPI Serverless API for Vercel
├── public/
│   └── index.html        # Static HTML/JS frontend for Vercel
├── documents/
│   └── ai_notes.txt      # Knowledge base source document
├── agent.py              # CLI Agentic RAG engine with memory & web search
├── app.py                # Basic Gemini chatbot script
├── rag.py                # Document chunking & ChromaDB ingestion
├── streamlit_app.py      # Streamlit Web Application interface
├── chroma_db/            # Persistent ChromaDB vector database directory
├── .env                  # Environment variables (Git ignored)
├── .gitignore            # Git exclusion rules
├── .vercelignore         # Vercel build exclusion rules
├── vercel.json           # Vercel serverless routing configuration
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

---

## Resume Description

> **Agentic RAG Assistant** | *Python, Google Gemini API, ChromaDB, LangChain, FastAPI, Streamlit, Vercel*
> - Engineered an Agentic Retrieval-Augmented Generation (RAG) assistant that dynamically routes queries between a ChromaDB vector store and live web search APIs.
> - Implemented document chunking using LangChain `RecursiveCharacterTextSplitter` and persistent vector indexing for low-latency retrieval.
> - Built multi-turn session memory and source-attributed responses with Gemini 3.5 Flash Lite.
> - Deployed dual production web applications: a serverless FastAPI app on Vercel and an interactive dashboard on Streamlit Cloud.
