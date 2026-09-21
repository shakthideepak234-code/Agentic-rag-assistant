import os
from pathlib import Path
from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(title="Agentic RAG Assistant API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Gemini Client
api_key = os.getenv("GOOGLE_API_KEY")

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[Dict[str, str]]] = []

# Paths setup
CHROMA_PATH = Path("./chroma_db")
DOCS_PATH = Path("documents/ai_notes.txt")

# Safe optional import of chromadb
try:
    import chromadb
except Exception as e:
    chromadb = None

def get_knowledge_context(question: str) -> str:
    """Retrieve relevant context from ChromaDB or fallback to direct document matching."""
    if chromadb is not None:
        try:
            if CHROMA_PATH.exists() and any(CHROMA_PATH.iterdir()):
                chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))
                collection = chroma_client.get_or_create_collection(name="ai_knowledge")
                results = collection.query(query_texts=[question], n_results=2)
                if results and results.get("documents") and results["documents"][0]:
                    return "\n\n".join(results["documents"][0])
        except Exception as e:
            print(f"ChromaDB lookup exception: {e}")

    # Fallback to direct document text splitting if ChromaDB is missing or unavailable
    if DOCS_PATH.exists():
        try:
            text = DOCS_PATH.read_text(encoding="utf-8")
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
            chunks = splitter.split_text(text)
            keywords = [w.lower() for w in question.split() if len(w) > 3]
            matching_chunks = [c for c in chunks if any(kw in c.lower() for kw in keywords)]
            if matching_chunks:
                return "\n\n".join(matching_chunks[:2])
            return "\n\n".join(chunks[:2])
        except Exception as e:
            print(f"Document fallback error: {e}")

    return "No knowledge base documents found."

def search_web(query: str) -> str:
    """Search live web using DuckDuckGo."""
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
        if not results:
            return "No web results found."
        return "\n\n".join([f"{r['title']}: {r['body']}" for r in results])
    except Exception as e:
        return f"Web search failed: {e}"

@app.get("/")
def read_root():
    return {"status": "ok", "service": "Agentic RAG Assistant API"}

@app.get("/api/health")
def health_check():
    return {"status": "ok", "service": "Agentic RAG Assistant API"}

@app.post("/api/chat")
def chat_endpoint(request: ChatRequest):
    if not api_key:
        raise HTTPException(status_code=500, detail="GOOGLE_API_KEY environment variable is not configured.")

    from google import genai
    client = genai.Client(api_key=api_key)
    question = request.message

    # Memory representation
    memory_lines = []
    if request.history:
        for msg in request.history[-6:]:
            role = msg.get("role", "user").capitalize()
            content = msg.get("content", "")
            memory_lines.append(f"{role}: {content}")
    memory = "\n".join(memory_lines)

    kb_context = get_knowledge_context(question)
    web_context = search_web(question)

    prompt = f"""You are Apple, an intelligent AI assistant with access to a knowledge base and live web search.

Conversation memory:
{memory}

--- Knowledge Base ---
{kb_context}

--- Live Web Search ---
{web_context}

Instructions:
- Use the Knowledge Base if it contains a relevant answer.
- Use the Live Web Search for current events, news, or topics not in the knowledge base.
- If you use web results, start your reply with: (from web)
- If you use knowledge base, start your reply with: (from knowledge base)
- Do not use any emojis in your response text.

Answer the user's question clearly and helpfully.
User: {question}
Apple:"""

    try:
        chat = client.chats.create(model="gemini-3.5-flash-lite")
        response = chat.send_message(prompt)
        reply_text = response.text if response else "No response generated."

        source = "knowledge_base"
        if "(from web)" in reply_text:
            source = "web"
        elif "(from knowledge base)" in reply_text:
            source = "knowledge_base"

        return {
            "reply": reply_text,
            "source": source,
            "kb_context": kb_context,
            "web_context": web_context
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")
