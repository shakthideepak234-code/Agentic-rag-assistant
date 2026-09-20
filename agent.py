import os
import chromadb
from dotenv import load_dotenv
from google import genai

# Load API key
load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# Connect to ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="ai_knowledge")

# Conversation memory
conversation_history = []


# Tool: Search the knowledge base
def search_knowledge(question):
    results = collection.query(query_texts=[question], n_results=2)
    return "\n\n".join(results["documents"][0])


# Tool: Live web search using DuckDuckGo (no API key needed)
def search_web(query):
    try:
        from ddgs import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
        if not results:
            return "No results found."
        return "\n\n".join([f"{r['title']}: {r['body']}" for r in results])
    except Exception as e:
        return f"Web search failed: {e}"


# Agent: single API call — decides and answers together
def agent(question):
    # Add user question to memory
    conversation_history.append({"role": "user", "content": question})

    # Build memory context (last 6 messages)
    memory = "\n".join([
        f"{m['role'].capitalize()}: {m['content']}"
        for m in conversation_history[-6:]
    ])

    # Search knowledge base
    kb_context = search_knowledge(question)

    # Also get web results as backup
    web_context = search_web(question)

    # Single prompt: decide + answer in one call
    prompt = f"""You are Apple, an intelligent AI assistant with two sources of information.

Conversation so far:
{memory}

--- Knowledge Base ---
{kb_context}

--- Live Web Search ---
{web_context}

Instructions:
- Use the Knowledge Base if it has a relevant answer.
- Use the Live Web Search for current events, news, or things not in the knowledge base.
- If you use web results, start your reply with: 🌐 (from web)
- If you use knowledge base, start your reply with: 📚 (from knowledge base)

Answer the user's question clearly and helpfully.
User: {question}
Apple:"""

    chat = client.chats.create(model="gemini-3.5-flash-lite")
    response = chat.send_message(prompt)
    reply = response.text

    # Save reply to memory
    conversation_history.append({"role": "assistant", "content": reply})

    return reply


# Chat loop
print("Apple Agent is ready! Type 'exit' to stop.\n")

while True:
    question = input("You: ")

    if question.lower() == "exit":
        print("Apple: Goodbye!")
        break

    answer = agent(question)
    print("\nApple:", answer)
    print()