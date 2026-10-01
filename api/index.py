import os
import sys
from pathlib import Path
from typing import List, Dict, Optional, Any
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(title="Agentic RAG Assistant API")

# Enable CORS for all origins
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

# Embedded default knowledge context as a foolproof fallback
DEFAULT_DOC_TEXT = """Artificial Intelligence (AI) is a field of computer science focused on creating systems that can perform tasks that normally require human intelligence.

Retrieval-Augmented Generation (RAG) combines information retrieval with a large language model. Instead of relying only on the model's existing knowledge, RAG retrieves relevant information from an external knowledge base and provides it to the model as context.

Agentic RAG adds an AI agent that can decide which tools or information sources to use. The agent may choose between documents, memory, APIs, or other tools before generating an answer.

Large Language Models (LLMs) are AI models trained on large amounts of text to understand and generate natural language.

AI is expected to have a profound impact on healthcare, with applications ranging from disease diagnosis to drug development and patient monitoring.

Improved Diagnostics: AI can enhance the accuracy of disease detection, such as lung cancer detection, with sensitivity rates between 81% and 99%. It can also reduce false negatives, improving the chances of early detection and treatment.

Accelerated Drug Development: AI is revolutionizing the drug discovery process, allowing for the identification of new drug targets and advancing candidates into preclinical trials in a fraction of the time it typically takes."""

def get_knowledge_context(question: str) -> str:
    """Retrieve relevant context from documents with absolute and embedded fallback."""
    text = DEFAULT_DOC_TEXT
    
    # Try finding document from files
    candidate_paths = [
        Path(__file__).resolve().parent / "ai_notes.txt",
        Path(__file__).resolve().parent.parent / "documents" / "ai_notes.txt",
        Path("api/ai_notes.txt"),
        Path("documents/ai_notes.txt"),
    ]
    for p in candidate_paths:
        try:
            if p.exists():
                text = p.read_text(encoding="utf-8")
                break
        except Exception:
            continue

    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
        chunks = splitter.split_text(text)
    except Exception:
        chunks = [c.strip() for c in text.split("\n\n") if c.strip()]

    keywords = [w.lower() for w in question.split() if len(w) > 3]
    matching = [c for c in chunks if any(kw in c.lower() for kw in keywords)]
    if matching:
        return "\n\n".join(matching[:2])
    return "\n\n".join(chunks[:2]) if chunks else "No knowledge base documents found."

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

def generate_ai_reply(api_key: str, prompt: str) -> str:
    """Generate response using Google GenAI SDK with multi-model fallback."""
    from google import genai
    client = genai.Client(api_key=api_key)
    
    models = ["gemini-3.5-flash-lite", "gemini-3.6-flash", "gemini-3.8-flash", "gemini-3.1-flash-lite"]
    last_err = None
    
    for m in models:
        try:
            chat = client.chats.create(model=m)
            res = chat.send_message(prompt)
            if res and res.text:
                return res.text
        except Exception as e:
            last_err = e
            continue
            
    return f"Unable to generate AI response: {last_err}"

@app.get("/")
@app.get("/api")
@app.get("/api/health")
def health_check():
    has_key = bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGEL_API_KEY"))
    return {"status": "ok", "service": "Apple Agentic RAG API", "api_key_configured": has_key}

@app.api_route("/api/chat", methods=["GET", "POST"])
@app.api_route("/chat", methods=["GET", "POST"])
@app.post("/")
async def handle_chat(request: Request):
    # Handle GET request with status info
    if request.method == "GET":
        return {"status": "ok", "message": "Send a POST request with {'message': 'your question'}"}

    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            status_code=400,
            content={"detail": "Invalid JSON body. Expected {'message': 'your question'}."}
        )

    question = body.get("message", "").strip()
    if not question:
        return JSONResponse(
            status_code=400,
            content={"detail": "Field 'message' cannot be empty."}
        )

    history = body.get("history", [])

    # Check API key from env or header
    api_key = (
        os.getenv("GOOGLE_API_KEY") 
        or os.getenv("GOOGEL_API_KEY")
        or request.headers.get("x-api-key")
    )
    
    if not api_key:
        return {
            "reply": "⚠️ **GOOGLE_API_KEY** is not configured in your Vercel Project Settings.\n\nPlease go to **Vercel Dashboard -> Settings -> Environment Variables**, add `GOOGLE_API_KEY`, and click **Redeploy**.",
            "source": "knowledge_base",
            "kb_context": "",
            "web_context": ""
        }

    # Format memory lines
    memory_lines = []
    if history and isinstance(history, list):
        for msg in history[-6:]:
            if isinstance(msg, dict):
                role = msg.get("role", "user").capitalize()
                content = msg.get("content", "")
                memory_lines.append(f"{role}: {content}")
    memory_text = "\n".join(memory_lines)

    kb_context = get_knowledge_context(question)
    web_context = search_web(question)

    prompt = f"""You are Apple, an intelligent AI assistant with access to a knowledge base and live web search.

Conversation memory:
{memory_text}

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

    try:
        reply_text = generate_ai_reply(api_key, prompt)
    except Exception as e:
        reply_text = f"Error processing request: {e}"

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
