# 🍎 Agentic RAG Assistant ("Apple")

An intelligent, multi-source **Agentic Retrieval-Augmented Generation (RAG)** assistant built with **Python**, **Google Gemini 3.5 Flash Lite**, **ChromaDB**, **LangChain**, and **Streamlit**.

The assistant dynamically evaluates user queries to route them between a **private vector knowledge base** (ChromaDB) and **live web search** (DuckDuckGo), maintaining conversational context across turns.

---

## 🌟 Key Features

- **📚 Retrieval-Augmented Generation (RAG):** Document chunking & embedding ingestion into ChromaDB using LangChain text splitters.
- **🌐 Live Web Search Integration:** Fallback to real-time DuckDuckGo web search for current events, news, or external queries.
- **🧠 Agentic Routing & Decision Loop:** Automatically decides whether to pull from internal documents or live web results based on query relevance.
- **💾 Session Memory:** Remembers multi-turn conversation context across interactions.
- **🖥️ Dual Interfaces:**
  - **Web UI:** Interactive Streamlit web interface (`streamlit_app.py`).
  - **CLI Terminal:** Fast command-line interface (`agent.py`).

---

## 🏗️ System Architecture

```text
               User Question (Web UI / Terminal CLI)
                                 │
                                 ▼
                    Agent Decision & Routing Engine
                                 │
        ┌────────────────────────┴────────────────────────┐
        ▼                                                 ▼
 📚 Knowledge Base Retrieval                       🌐 Live Web Search
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

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10+
- Google Gemini API Key ([Get a free key here](https://aistudio.google.com/apikey))

### 2. Installation & Setup

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/Agentic-rag-assistant.git
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
   pip install google-genai chromadb langchain-text-splitters python-dotenv ddgs streamlit
   ```

4. **Configure Environment Variables:**
   Create a `.env` file in the root directory:
   ```env
   GOOGLE_API_KEY=your_gemini_api_key_here
   ```

---

## 🎮 Usage

### 1. Ingest Knowledge Base Documents
Populate ChromaDB with documents from `documents/ai_notes.txt`:
```bash
python rag.py
```

### 2. Run the Web UI (Streamlit)
Launch the browser interface:
```bash
streamlit run streamlit_app.py
```

### 3. Run the Terminal Agent (CLI)
Chat directly in your terminal:
```bash
python agent.py
```

---

## 📄 Project Structure

```text
Agentic-rag-assistant/
├── app.py                # Basic Gemini chatbot interface
├── rag.py                # Document loading, chunking & ChromaDB ingestion
├── agent.py              # CLI Agentic RAG engine with memory & web search
├── streamlit_app.py      # Streamlit Web Application interface
├── documents/
│   └── ai_notes.txt      # Source text document for vector store
├── chroma_db/            # Persistent ChromaDB vector database directory
├── .env                  # API key configuration (Git ignored)
├── .gitignore            # Git exclusion rules
└── README.md             # Project documentation
```

---

## 💼 Resume Description

> **Agentic RAG Assistant** | *Python, Google Gemini API, ChromaDB, LangChain, Streamlit*
> - Engineered an Agentic Retrieval-Augmented Generation (RAG) assistant that dynamically routes queries between a ChromaDB vector store and live web search APIs.
> - Implemented document chunking using LangChain `RecursiveCharacterTextSplitter` and persistent vector indexing.
> - Built multi-turn session memory and source-attributed responses with Gemini 3.5 Flash Lite.
> - Developed both a Streamlit web application and a CLI workflow for document ingestion and real-time querying.
