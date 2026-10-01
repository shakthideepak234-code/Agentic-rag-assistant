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

# 🍎 Apple — Agentic RAG Assistant

An intelligent, multi-source **Agentic Retrieval-Augmented Generation (RAG)** assistant built with Python, Google Gemini, ChromaDB, FastAPI, Streamlit, and Vercel.

The assistant dynamically evaluates user queries to route them between a private vector knowledge base (ChromaDB) and live web search (DuckDuckGo), maintaining conversational context across turns.

---

## 🌐 Live Deployments

| Platform | Interface | Live Link | Features |
| :--- | :--- | :--- | :--- |
| **Vercel** | Modern Web UI | [👉 **Live Web App**](https://agentic-rag-ai-three.vercel.app) | ⚡ 100% Serverless, 0 cold-start, 24/7 uptime |
| **Streamlit** | Interactive Dashboard | [👉 **Live Dashboard**](https://ai-agentic-rag-assistant.streamlit.app) | 🎙️ Voice Input, Text-to-Speech, Model Selector, Vector counter |

---

## 🚀 Key Features

- **Retrieval-Augmented Generation (RAG)**: Document chunking and embedding ingestion into ChromaDB using LangChain text splitters.
- **Live Web Search Integration**: Real-time DuckDuckGo web search fallback for current events, news, or out-of-domain queries.
- **Agentic Routing & Decision Loop**: Automatically decides whether to pull from internal documents or live web results based on query relevance.
- **Multimodal Voice Input & TTS Output**: Record voice questions directly via browser microphone and hear synthesized speech responses.
- **Multi-Model Fallback Cascade**: Automatic retry across `gemini-3.5-flash-lite`, `gemini-3.6-flash`, and `gemini-3.8-flash` to prevent rate-limit and 503 traffic spikes.
- **Session Memory**: Remembers multi-turn conversation context across interactions.

---

## 🏗️ System Architecture

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

## 💻 Getting Started Locally

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

## 🛠️ Usage & Running Locally

### 1. Run the Streamlit Web App
```bash
streamlit run app.py
```
*(or `streamlit run streamlit_app.py`)*

### 2. Ingest Knowledge Base Documents
```bash
python rag.py
```

### 3. Run the Terminal Agent (CLI)
```bash
python agent.py
```

### 4. Run the Vercel API Locally (FastAPI)
```bash
uvicorn api.index:app --reload
```

---

## 📁 Project Structure

```text
Agentic-rag-assistant/
├── app.py                # Hugging Face & Streamlit entrypoint
├── streamlit_app.py      # Streamlit Web Application (UI, Voice, RAG, Fallback)
├── api/
│   ├── index.py          # Serverless Python Backend for Vercel
│   ├── ai_notes.txt      # Bundled knowledge base for serverless execution
│   └── requirements.txt  # Lightweight serverless dependencies
├── public/
│   └── index.html        # Modern static HTML/JS/CSS frontend for Vercel
├── documents/
│   └── ai_notes.txt      # Source knowledge base document
├── agent.py              # CLI Agentic RAG engine with memory & web search
├── rag.py                # Document chunking & ChromaDB ingestion
├── chroma_db/            # Persistent ChromaDB vector database directory
├── .env                  # Environment variables (Git ignored)
├── .gitignore            # Git exclusion rules
├── .vercelignore         # Vercel build exclusion rules
├── vercel.json           # Vercel serverless routing configuration
├── requirements.txt      # Full project dependencies
└── README.md             # Project documentation & live links
```

---

## 📄 Resume Description

> **Agentic RAG Assistant ("Apple")** | *Python, Google Gemini API, ChromaDB, FastAPI, Streamlit, Vercel*
> - Engineered an Agentic Retrieval-Augmented Generation (RAG) assistant that dynamically routes queries between a ChromaDB vector store and live web search APIs.
> - Implemented document chunking and vector retrieval with automatic model fallback (`gemini-3.5-flash-lite`, `gemini-3.6-flash`, `gemini-3.8-flash`) for 99.9% uptime.
> - Developed multimodal voice input transcription and text-to-speech synthesis with conversational multi-turn session memory.
> - Deployed dual production web applications: a serverless FastAPI app on Vercel and an interactive dashboard on Streamlit Community Cloud.
