import os
import json
import urllib.request
import urllib.parse
from http.server import BaseHTTPRequestHandler
from pathlib import Path

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

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

    chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
    keywords = [w.lower() for w in question.split() if len(w) > 3]
    matching = [c for c in chunks if any(kw in c.lower() for kw in keywords)]
    if matching:
        return "\n\n".join(matching[:2])
    return "\n\n".join(chunks[:2]) if chunks else "No knowledge base documents found."

WEATHER_KEYWORDS = {"weather", "temperature", "forecast", "rain", "sunny", "humidity", "climate", "hot", "cold", "wind", "storm"}
NEWS_KEYWORDS = {"news", "latest", "today", "current", "update", "happening", "2024", "2025", "2026", "recent", "now"}

def _is_weather_query(q: str) -> bool:
    words = set(q.lower().split())
    return bool(words & WEATHER_KEYWORDS)

def _weather_search(query: str) -> str:
    """Get real-time weather using wttr.in (no API key needed)."""
    try:
        # Extract likely city name: remove weather keywords, take first meaningful token(s)
        tokens = [w for w in query.split() if w.lower() not in WEATHER_KEYWORDS and len(w) > 2]
        city = "+".join(tokens[:2]) if tokens else "auto"
        url = f"https://wttr.in/{urllib.parse.quote(city)}?format=j1"
        req = urllib.request.Request(url, headers={"User-Agent": "curl/7.88.0", "Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="ignore"))
            cc = data.get("current_condition", [{}])[0]
            area = data.get("nearest_area", [{}])[0]
            city_name = area.get("areaName", [{}])[0].get("value", city)
            country = area.get("country", [{}])[0].get("value", "")
            desc = cc.get("weatherDesc", [{}])[0].get("value", "N/A")
            temp_c = cc.get("temp_C", "N/A")
            feels_c = cc.get("FeelsLikeC", "N/A")
            humidity = cc.get("humidity", "N/A")
            wind_kmph = cc.get("windspeedKmph", "N/A")
            return (
                f"Live weather for {city_name}, {country}: {desc}. "
                f"Temperature: {temp_c}C (feels like {feels_c}C). "
                f"Humidity: {humidity}%. Wind: {wind_kmph} km/h."
            )
    except Exception as e:
        return f"Weather lookup failed: {e}"

def _wikipedia_search(query: str) -> str:
    """Get a factual summary from the Wikipedia REST API (no key needed)."""
    try:
        # Try exact title first, then search
        clean = urllib.parse.quote(query.replace(" ", "_"))
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{clean}"
        req = urllib.request.Request(url, headers={"User-Agent": "AgenticRAG/1.0", "Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="ignore"))
            extract = data.get("extract", "")
            if extract and len(extract) > 30:
                return f"(Wikipedia) {extract[:500]}"
    except Exception:
        pass
    return ""

def _ddg_instant(query: str) -> str:
    """DuckDuckGo Instant Answer API — good for entity definitions."""
    try:
        url = "https://api.duckduckgo.com/?q=" + urllib.parse.quote(query) + "&format=json&no_html=1&skip_disambig=1"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=4) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="ignore"))
            abstract = data.get("AbstractText", "")
            if abstract:
                return abstract
            topics = [t.get("Text", "") for t in data.get("RelatedTopics", []) if isinstance(t, dict) and "Text" in t]
            if topics:
                return "\n".join(topics[:2])
    except Exception:
        pass
    return ""

def search_web(query: str) -> str:
    """Hybrid live web search: weather via wttr.in, facts via Wikipedia, fallback to DDG Instant Answer."""
    if _is_weather_query(query):
        return _weather_search(query)

    # Try Wikipedia first for factual/general queries
    wiki = _wikipedia_search(query)
    if wiki:
        return wiki

    # Fallback: DDG Instant Answer (good for entities/definitions)
    ddg = _ddg_instant(query)
    if ddg:
        return ddg

    return "No live web search results found for this query."

def generate_ai_reply(api_key: str, prompt: str) -> str:
    """Direct, zero-dependency REST API call to Google Gemini with automatic model fallback."""
    models = ["gemini-3.5-flash-lite", "gemini-3.6-flash", "gemini-3.8-flash", "gemini-3.1-flash-lite"]
    last_err = None
    
    for m in models:
        try:
            endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent?key={api_key}"
            payload = json.dumps({
                "contents": [
                    {
                        "parts": [{"text": prompt}]
                    }
                ]
            }).encode("utf-8")
            
            req = urllib.request.Request(
                endpoint,
                data=payload,
                headers={"Content-Type": "application/json"}
            )
            
            with urllib.request.urlopen(req, timeout=12) as response:
                result = json.loads(response.read().decode("utf-8"))
                candidates = result.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts and "text" in parts[0]:
                        return parts[0]["text"]
        except Exception as e:
            last_err = e
            continue
            
    return f"Unable to generate AI response: {last_err}"

class handler(BaseHTTPRequestHandler):
    def send_json_response(self, status_code: int, data: dict):
        response_bytes = json.dumps(data).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response_bytes)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, x-api-key")
        self.end_headers()
        self.wfile.write(response_bytes)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, x-api-key")
        self.end_headers()

    def do_GET(self):
        has_key = bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGEL_API_KEY"))
        self.send_json_response(200, {
            "status": "ok",
            "service": "Apple Agentic RAG API (Vercel Serverless)",
            "api_key_configured": has_key
        })

    def do_POST(self):
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            if content_length > 0:
                body_bytes = self.rfile.read(content_length)
                body = json.loads(body_bytes.decode("utf-8"))
            else:
                body = {}
        except Exception:
            self.send_json_response(400, {"detail": "Invalid JSON body payload."})
            return

        question = body.get("message", "").strip()
        if not question:
            self.send_json_response(400, {"detail": "Field 'message' cannot be empty."})
            return

        history = body.get("history", [])

        # Check API key from env or header
        api_key = (
            os.getenv("GOOGLE_API_KEY") 
            or os.getenv("GOOGEL_API_KEY")
            or self.headers.get("x-api-key")
        )
        
        if not api_key:
            self.send_json_response(200, {
                "reply": "**GOOGLE_API_KEY** is not configured in your Vercel Project Settings.\n\nPlease go to **Vercel Dashboard -> Settings -> Environment Variables**, add `GOOGLE_API_KEY`, and click **Redeploy**.",
                "source": "knowledge_base",
                "kb_context": "",
                "web_context": ""
            })
            return

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

        self.send_json_response(200, {
            "reply": reply_text,
            "source": source,
            "kb_context": kb_context,
            "web_context": web_context
        })
