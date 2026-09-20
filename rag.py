import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from google import genai

# Load API key
load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# 1. Load document
document_path = Path("documents/ai_notes.txt")
text = document_path.read_text(encoding="utf-8")
print(f"Loaded document: {document_path.name}")

# 2. Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)
chunks = splitter.split_text(text)
print(f"Split into {len(chunks)} chunks")

# 3. Connect to ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="ai_knowledge")

# 4. Store chunks in ChromaDB
for i, chunk in enumerate(chunks):
    collection.upsert(ids=[f"chunk_{i}"], documents=[chunk])
print(f"Stored {len(chunks)} chunks in ChromaDB.\n")

# 5. Ask a test question
question = input("Test your knowledge base — Ask me: ")

# 6. Search ChromaDB
results = collection.query(query_texts=[question], n_results=2)
context = "\n\n".join(results["documents"][0])

print("\n--- Retrieved Context ---")
print(context)
print("-------------------------\n")

# 7. Send to Gemini using chat
prompt = f"""You are Apple, an AI assistant using a knowledge base.

Use the following retrieved information to answer the user's question.

Knowledge:
{context}

User question:
{question}

Answer clearly and simply.
"""

chat = client.chats.create(model="gemini-3.5-flash-lite")
response = chat.send_message(prompt)

# 8. Print final answer
print("Apple:", response.text)