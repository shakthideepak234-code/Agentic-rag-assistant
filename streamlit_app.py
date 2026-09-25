import sys
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import os
import io
import re
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types
from gtts import gTTS

# 1. Page Config & Custom Modern CSS
st.set_page_config(
    page_title="Apple AI — Agentic RAG Platform",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Modern Clean UI
st.markdown("""
<style>
    /* Global Styles */
    .main {
        background-color: #0E1117;
    }
    
    /* Header Container */
    .header-container {
        padding: 1.2rem 0rem 0.8rem 0rem;
        border-bottom: 1px solid #1E2638;
        margin-bottom: 1.5rem;
    }
    .header-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #FFFFFF;
        margin-bottom: 0.2rem;
        letter-spacing: -0.5px;
    }
    .header-subtitle {
        font-size: 0.95rem;
        color: #8B949E;
    }

    /* Badges */
    .badge-kb {
        background-color: rgba(56, 139, 253, 0.15);
        color: #58A6FF;
        border: 1px solid rgba(56, 139, 253, 0.4);
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.82rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 8px;
    }
    .badge-web {
        background-color: rgba(46, 160, 67, 0.15);
        color: #3FB950;
        border: 1px solid rgba(46, 160, 67, 0.4);
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.82rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 8px;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #161B22;
        border-right: 1px solid #1E2638;
    }

    /* Metric Cards */
    div[data-testid="stMetricValue"] {
        font-size: 1.6rem !important;
        font-weight: 700 !important;
        color: #58A6FF !important;
    }
    
    /* Card Container */
    .css-card {
        background-color: #161B22;
        border: 1px solid #30363D;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# 2. Load API key & setup Gemini client
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Check Streamlit secrets if running on Streamlit Cloud / Hugging Face Spaces
if not api_key:
    try:
        api_key = st.secrets.get("GOOGLE_API_KEY")
    except Exception:
        pass

# If still not found, check session state or ask in sidebar
with st.sidebar:
    st.title("Control Panel")
    st.caption("Agentic RAG System Overview")
    st.markdown("---")

    if not api_key:
        st.warning("⚠️ Google API Key not found in environment secrets.")
        user_key_input = st.text_input(
            "Enter Gemini API Key",
            type="password",
            help="Get your free API key at https://aistudio.google.com/apikey"
        )
        if user_key_input:
            api_key = user_key_input
            st.session_state["user_api_key"] = user_key_input
        elif "user_api_key" in st.session_state and st.session_state["user_api_key"]:
            api_key = st.session_state["user_api_key"]

if not api_key:
    st.info("👋 Welcome to **Apple Agentic RAG Assistant**!\n\nPlease enter your **Google Gemini API Key** in the sidebar (or configure the `GOOGLE_API_KEY` secret in your deployment settings) to start.")
    st.stop()

@st.cache_resource
def get_genai_client(key: str):
    return genai.Client(api_key=key)

@st.cache_resource
def get_chroma_collection():
    collection = None
    try:
        import chromadb
        chroma_client = chromadb.PersistentClient(path="./chroma_db")
        collection = chroma_client.get_or_create_collection(name="ai_knowledge")
        
        # Auto-populate if empty (e.g. fresh cloud deployment)
        if collection.count() == 0:
            doc_path = Path("documents/ai_notes.txt")
            if doc_path.exists():
                text = doc_path.read_text(encoding="utf-8")
                from langchain_text_splitters import RecursiveCharacterTextSplitter
                splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
                chunks = splitter.split_text(text)
                for i, chunk in enumerate(chunks):
                    collection.upsert(ids=[f"chunk_{i}"], documents=[chunk])
    except Exception as e:
        print(f"ChromaDB initialization notice: {e}")
    return collection

client = get_genai_client(api_key)
collection = get_chroma_collection()

# 3. Resilient AI Generation with Auto-Fallback
def generate_response_with_fallback(prompt: str, preferred_model: str) -> str:
    """Generate LLM response with automatic fallback to healthy models on 503/429 spikes."""
    candidate_models = [preferred_model, "gemini-3.5-flash-lite", "gemini-3.6-flash", "gemini-3.1-flash-lite", "gemini-3.8-flash"]
    seen = set()
    models_to_try = [m for m in candidate_models if not (m in seen or seen.add(m))]
    
    last_error = None
    for model in models_to_try:
        try:
            chat = client.chats.create(model=model)
            response = chat.send_message(prompt)
            if response and response.text:
                return response.text
        except Exception as e:
            last_error = e
            continue
            
    return f"Unable to generate response right now ({last_error}). Please retry in a moment."

def transcribe_speech_with_fallback(audio_bytes: bytes, preferred_model: str) -> str:
    """Transcribe voice audio with automatic model fallback."""
    candidate_models = [preferred_model, "gemini-3.5-flash-lite", "gemini-3.6-flash", "gemini-3.1-flash-lite", "gemini-3.8-flash"]
    seen = set()
    models_to_try = [m for m in candidate_models if not (m in seen or seen.add(m))]
    
    for model in models_to_try:
        try:
            response = client.models.generate_content(
                model=model,
                contents=[
                    types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav"),
                    "Transcribe this audio recording into exact plain text. Output ONLY the transcribed text and nothing else."
                ]
            )
            if response and response.text:
                return response.text.strip()
        except Exception:
            continue
    raise RuntimeError("Voice transcription unavailable right now. Please try typing your question.")

def search_knowledge(question: str) -> str:
    """Search ChromaDB knowledge base with fallback to document parsing."""
    if collection is not None:
        try:
            results = collection.query(query_texts=[question], n_results=2)
            if results and results.get("documents") and results["documents"][0]:
                return "\n\n".join(results["documents"][0])
        except Exception:
            pass

    # Fallback to direct document matching
    doc_path = Path("documents/ai_notes.txt")
    if doc_path.exists():
        try:
            text = doc_path.read_text(encoding="utf-8")
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
            chunks = splitter.split_text(text)
            keywords = [w.lower() for w in question.split() if len(w) > 3]
            matching = [c for c in chunks if any(kw in c.lower() for kw in keywords)]
            if matching:
                return "\n\n".join(matching[:2])
            return "\n\n".join(chunks[:2])
        except Exception:
            pass

    return "No knowledge base documents found."

def search_web(query: str) -> str:
    """Live web search using DuckDuckGo."""
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

def text_to_speech(text: str):
    """Convert text into speech audio bytes."""
    try:
        clean_text = re.sub(r'<[^>]+>', '', text)
        clean_text = clean_text.replace("**", "")
        clean_text = re.sub(r'SOURCE:\s*(KNOWLEDGE BASE|LIVE WEB|MEMORY)', '', clean_text, flags=re.IGNORECASE)
        clean_text = clean_text.strip()
        
        if not clean_text:
            return None

        tts = gTTS(text=clean_text, lang='en')
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        return fp.read()
    except Exception:
        return None

# 4. Sidebar Controls & Metrics
with st.sidebar:
    col1, col2 = st.columns(2)
    with col1:
        st.metric("API Status", "Active 🟢")
    with col2:
        try:
            doc_count = collection.count() if collection else 0
        except Exception:
            doc_count = 0
        st.metric("Vector Chunks", doc_count)

    st.markdown("---")
    
    st.subheader("Model & Voice")
    model_choice = st.selectbox(
        "Gemini Model",
        options=["gemini-3.5-flash-lite", "gemini-3.6-flash", "gemini-3.8-flash", "gemini-3.1-flash-lite"],
        index=0,
        help="gemini-3.5-flash-lite is recommended for ultra-fast, high availability responses."
    )
    enable_tts = st.toggle("Voice Output (Text-to-Speech)", value=True)
    auto_play = st.toggle("Auto-play Audio Response", value=False)

    st.markdown("---")
    st.subheader("Sample Queries")
    st.markdown("""
    **Knowledge Base (RAG):**
    - *What is RAG?*
    - *What is an LLM?*
    
    **Live Search (Agentic):**
    - *What is the latest tech news today?*
    - *Weather in Tokyo right now*
    """)

    st.markdown("---")
    if st.button("Clear Chat History", use_container_width=True, type="secondary"):
        st.session_state.messages = []
        st.rerun()

# 5. Header Area
st.markdown("""
<div class="header-container">
    <div class="header-title">🍎 Apple — Agentic RAG Platform</div>
    <div class="header-subtitle">Intelligent hybrid retrieval engine combining ChromaDB Vector Search & Real-Time Web Intelligence</div>
</div>
""", unsafe_allow_html=True)

# 6. Initialize Session History
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am Apple, your Agentic RAG assistant. Ask me questions about your knowledge base or live web data."}
    ]

# Display Chat Messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)
        if "audio" in message and message["audio"]:
            st.audio(message["audio"], format="audio/mp3")

# 7. Voice & Text Input Section
st.markdown("### 🎙️ Voice Input")
audio_input = st.audio_input("Record Voice Question")

# Always render text input at the bottom
text_input = st.chat_input("Ask a question or enter a command...")

user_input = None

# If user typed in chat input, use text
if text_input:
    user_input = text_input

# If user recorded audio, transcribe voice
elif audio_input is not None:
    audio_bytes = audio_input.getvalue()
    audio_hash = hash(audio_bytes)
    if st.session_state.get("last_audio_hash") != audio_hash:
        st.session_state["last_audio_hash"] = audio_hash
        with st.spinner("Transcribing speech with Gemini Multimodal..."):
            try:
                user_input = transcribe_speech_with_fallback(audio_bytes, model_choice)
                st.toast(f"Transcribed Voice: '{user_input}'", icon="🎙️")
            except Exception as e:
                st.error(f"Speech transcription failed: {e}")

# 8. Core Agent Processing Loop
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing query, searching knowledge base & web..."):
            memory_text = "\n".join([
                f"{m['role'].capitalize()}: {m['content']}"
                for m in st.session_state.messages[-6:]
            ])

            kb_context = search_knowledge(user_input)
            web_context = search_web(user_input)

            prompt = f"""You are Apple, an intelligent AI assistant with conversation memory, a knowledge base, and live web search.

Conversation Memory (Highest priority for personal info & follow-ups):
{memory_text}

--- Knowledge Base ---
{kb_context}

--- Live Web Search ---
{web_context}

Instructions:
1. FIRST check Conversation Memory. If the user shares or asks about personal details (like their name, preferences, or past messages in memory), answer directly from Conversation Memory and start your reply with: <span class="badge-kb">SOURCE: MEMORY</span>
2. If the question is about documents or technical topics in the Knowledge Base, start your reply with: <span class="badge-kb">SOURCE: KNOWLEDGE BASE</span>
3. If the question is about current events, weather, or real-time news, start your reply with: <span class="badge-web">SOURCE: LIVE WEB</span>
4. Do NOT include emojis in your text.

Answer clearly, friendly, and concisely.
User: {user_input}
Apple:"""

            reply_text = generate_response_with_fallback(prompt, model_choice)

            st.markdown(reply_text, unsafe_allow_html=True)

            audio_data = None
            if enable_tts:
                with st.spinner("Synthesizing audio..."):
                    audio_data = text_to_speech(reply_text)
                    if audio_data:
                        st.audio(audio_data, format="audio/mp3", autoplay=auto_play)

    st.session_state.messages.append({
        "role": "assistant",
        "content": reply_text,
        "audio": audio_data if enable_tts else None
    })
