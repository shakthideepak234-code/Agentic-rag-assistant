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

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[Dict[str, str]]] = []

# Paths setup with robust absolute resolution for Vercel Serverless
BASE_DIR = Path(__file__).resolve().parent.parent
DOCS_PATH = BASE_DIR / "documents" / "ai_notes.txt"
if not DOCS_PATH.exists():
    DOCS_PATH = Path("documents/ai_notes.txt")

def get_knowledge_context(question: str) -> str:
    """Retrieve relevant context from documents."""
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
            print(f"Document fallback notice: {e}")

    return "No knowledge base documents found."

def search_web(query: str) -> str:
    """Search live web using DuckDuckGo."""
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
        if not results:
            return "No web search results found."
        return "\n\n".join([f"{r.get('title', '')}: {r.get('body', '')}" for r in results])
    except Exception as e:
        return f"Web search notice: {e}"

@app.get("/")
def read_root():
    return {"status": "ok", "service": "Agentic RAG Assistant API"}

@app.get("/api/health")
def health_check():
    return {"status": "ok", "service": "Agentic RAG Assistant API"}

@app.post("/api/chat")
def chat_endpoint(request: ChatRequest):
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGEL_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="GOOGLE_API_KEY environment variable is not configured in Vercel project settings."
        )

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
1. If the user shares or asks about personal details from conversation memory, answer directly from memory.
2. If using knowledge base, answer accurately based on the documents.
3. If using web results, answer based on current information.
4. Do not use any emojis in your response text.

Answer clearly, friendly, and concisely.
User: {question}
Apple:"""

    candidate_models = ["gemini-3.5-flash-lite", "gemini-3.6-flash", "gemini-3.8-flash", "gemini-3.1-flash-lite"]
    reply_text = None
    last_err = None
    
    for model in candidate_models:
        try:
            chat = client.chats.create(model=model)
            response = chat.send_message(prompt)
            if response and response.text:
                reply_text = response.text
                break
        except Exception as e:
            last_err = e
            continue

    if not reply_text:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(last_err)}")

    source = "knowledge_base"
    if "(from web)" in reply_text.lower():
        source = "web"
    elif "(from knowledge base)" in reply_text.lower():
        source = "knowledge_base"
    elif "memory" in reply_text.lower():
        source = "memory"

    return {
        "reply": reply_text,
        "source": source,
        "kb_context": kb_context,
        "web_context": web_context
    }
