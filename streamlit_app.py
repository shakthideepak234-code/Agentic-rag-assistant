import os
import io
import streamlit as st
import chromadb
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
        padding: 1.5rem 0rem 1rem 0rem;
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

if not api_key:
    st.error("GOOGLE_API_KEY not found in .env file! Please configure environment variables.")
    st.stop()

@st.cache_resource
def get_genai_client():
    return genai.Client(api_key=api_key)

@st.cache_resource
def get_chroma_collection():
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_or_create_collection(name="ai_knowledge")
    
    # Auto-populate if empty (e.g. fresh cloud deployment)
    if collection.count() == 0:
        try:
            from pathlib import Path
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            doc_path = Path("documents/ai_notes.txt")
            if doc_path.exists():
                text = doc_path.read_text(encoding="utf-8")
                splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
                chunks = splitter.split_text(text)
                for i, chunk in enumerate(chunks):
                    collection.upsert(ids=[f"chunk_{i}"], documents=[chunk])
        except Exception:
            pass
            
    return collection

client = get_genai_client()
collection = get_chroma_collection()

# 3. Helper Functions
def search_knowledge(question):
    try:
        results = collection.query(query_texts=[question], n_results=2)
        if results and results.get("documents") and results["documents"][0]:
            return "\n\n".join(results["documents"][0])
    except Exception as e:
        return f"Error reading knowledge base: {e}"
    return "No knowledge base documents found."

def search_web(query):
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
        if not results:
            return "No results found."
        return "\n\n".join([f"{r.get('title', '')}: {r.get('body', '')}" for r in results])
    except Exception as e:
        return f"Web search failed: {e}"

def text_to_speech(text):
    try:
        import re
        # Remove HTML tags (like <span class="badge-kb">...</span>)
        clean_text = re.sub(r'<[^>]+>', '', text)
        # Remove SOURCE headers and markdown stars
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

# 4. Sidebar Dashboard
with st.sidebar:
    st.title("Control Panel")
    st.caption("Agentic RAG System Overview")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("API Status", "Active")
    with col2:
        doc_count = collection.count()
        st.metric("Vector Chunks", doc_count)

    st.markdown("---")
    
    st.subheader("Voice Configuration")
    enable_tts = st.toggle("Voice Output (Text-to-Speech)", value=True)
    auto_play = st.toggle("Auto-play Audio Response", value=True)

    st.markdown("---")
    st.subheader("Sample Queries")
    st.markdown("""
    **Knowledge Base (RAG):**
    - *What is RAG?*
    - *What is an LLM?*
    
    **Live Search (Agentic):**
    - *Weather in Chennai today*
    - *Latest tech news*
    """)

    st.markdown("---")
    if st.button("Clear Chat History", use_container_width=True, type="secondary"):
        st.session_state.messages = []
        st.rerun()

# 5. Header Area
st.markdown("""
<div class="header-container">
    <div class="header-title">Apple — Agentic RAG Platform</div>
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
elif audio_input is not None and "last_audio" not in st.session_state:
    with st.spinner("Transcribing speech with Gemini Multimodal..."):
        try:
            audio_bytes = audio_input.read()
            response = client.models.generate_content(
                model="gemini-3.5-flash-lite",
                contents=[
                    types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav"),
                    "Transcribe this audio recording into exact plain text. Output ONLY the transcribed text and nothing else."
                ]
            )
            user_input = response.text.strip()
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

            try:
                chat = client.chats.create(model="gemini-3.5-flash-lite")
                response = chat.send_message(prompt)
                reply_text = response.text
            except Exception as e:
                reply_text = f"Error generating response: {e}"

            st.markdown(reply_text, unsafe_allow_html=True)

            audio_bytes = None
            if enable_tts:
                with st.spinner("Synthesizing audio..."):
                    audio_bytes = text_to_speech(reply_text)
                    if audio_bytes:
                        st.audio(audio_bytes, format="audio/mp3", autoplay=auto_play)

    st.session_state.messages.append({
        "role": "assistant",
        "content": reply_text,
        "audio": audio_bytes if enable_tts else None
    })
