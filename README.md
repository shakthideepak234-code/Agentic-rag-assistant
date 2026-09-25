---
title: Apple Agentic RAG Assistant
emoji: 🍎
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.42.0
app_file: app.py
pinned: false
license: mit
---

# Agentic RAG Assistant ("Apple")

An intelligent, multi-source Agentic Retrieval-Augmented Generation (RAG) assistant built with Python, Google Gemini, ChromaDB, LangChain, FastAPI, Vercel, and Streamlit.

The assistant dynamically evaluates user queries to route them between a private vector knowledge base (ChromaDB) and live web search (DuckDuckGo), maintaining conversational context across turns.

---

## Key Features

- **Retrieval-Augmented Generation (RAG)**: Document chunking and embedding ingestion into ChromaDB using LangChain text splitters.
- **Live Web Search Integration**: Fallback to real-time DuckDuckGo web search for current events, news, or out-of-domain queries.
- **Agentic Routing & Decision Loop**: Automatically decides whether to pull from internal documents or live web results based on query relevance.
- **Multimodal Voice Input & TTS Output**: Record voice questions directly and hear spoken responses.
- **Session Memory**: Remembers multi-turn conversation context across interactions.
- **Multiple Deployments & Interfaces**:
  - **Hugging Face Spaces**: Instant one-click interactive Streamlit deployment (`app.py` / `streamlit_app.py`).
  - **Streamlit Community Cloud**: Interactive dashboard (`streamlit_app.py`).
  - **Vercel Web App**: Serverless FastAPI backend with static HTML/JS frontend (`api/index.py`, `public/index.html`).
  - **CLI Terminal**: Fast command-line interface (`agent.py`).

---

## System Architecture

```text
               User Question (Web UI / Streamlit / Voice / Terminal CLI)
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
                         Gemini LLM Engine
                         + Memory Context
                                  │
                                  ▼
                         Source-Attributed Answer
```

---

## Getting Started

### 1. Prerequisites
- Python 3.10+
- Google Gemini API Key ([Get a free key from Google AI Studio](https://aistudio.google.com/apikey))

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
   # Linux/macOS
   source .venv/bin/activate
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

### 1. Run the Streamlit Web App
Launch the interactive Streamlit interface:
```bash
streamlit run app.py
```
*(or `streamlit run streamlit_app.py`)*

### 2. Ingest Knowledge Base Documents
Populate ChromaDB with documents from `documents/ai_notes.txt`:
```bash
python rag.py
```

### 3. Run the Terminal Agent (CLI)
Chat directly in your terminal:
```bash
python agent.py
```

### 4. Run the Vercel API Locally (FastAPI)
```bash
uvicorn api.index:app --reload
```

---

## Deployment Guide

### Deploying to Hugging Face Spaces (Streamlit)

1. Go to [Hugging Face Spaces](https://huggingface.co/new-space).
2. Enter a Space Name (e.g. `agentic-rag-assistant`).
3. Select **Streamlit** as the Space SDK.
4. Choose **Public** or **Private** and click **Create Space**.
5. Push this repository to your Hugging Face Space repository:
   ```bash
   git remote add space https://huggingface.co/spaces/<YOUR_USERNAME>/<YOUR_SPACE_NAME>
   git push space main
   ```
6. In your Hugging Face Space, navigate to **Settings** -> **Variables and secrets**.
7. Under **Secrets**, add:
   - **Key:** `GOOGLE_API_KEY`
   - **Value:** *Your Google Gemini API Key*
8. The Space will automatically build and launch! (If no secret is set, you can also enter the API key directly in the web app sidebar).

---

### Deploying to Streamlit Community Cloud
1. Connect your repository to [Streamlit Community Cloud](https://share.streamlit.io/).
2. Set `app.py` or `streamlit_app.py` as the main file path.
3. Under **Advanced settings -> Secrets**, add:
   ```toml
   GOOGLE_API_KEY = "your_gemini_api_key_here"
   ```

---

### Deploying to Vercel
1. Import `shakthideepak234-code/Agentic-rag-assistant` into Vercel.
2. Under **Environment Variables**, add `GOOGLE_API_KEY`.
3. Click **Deploy**.

---

## Project Structure

```text
Agentic-rag-assistant/
├── app.py                # Hugging Face & Streamlit entrypoint
├── streamlit_app.py      # Streamlit Web Application interface (UI, Voice, RAG)
├── api/
│   └── index.py          # FastAPI Serverless API for Vercel
├── public/
│   └── index.html        # Static HTML/JS frontend for Vercel
├── documents/
│   └── ai_notes.txt      # Knowledge base source document
├── agent.py              # CLI Agentic RAG engine with memory & web search
├── rag.py                # Document chunking & ChromaDB ingestion
├── chroma_db/            # Persistent ChromaDB vector database directory
├── .env                  # Environment variables (Git ignored)
├── .gitignore            # Git exclusion rules
├── .vercelignore         # Vercel build exclusion rules
├── vercel.json           # Vercel serverless routing configuration
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation & HF Space config
```
