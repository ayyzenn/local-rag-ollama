# rag_local_ollama.py

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import subprocess
from preprocess import preprocess_documents

# -----------------------------
# Step 1: Prepare Sample Documents
# -----------------------------

documents = preprocess_documents("docs")

# -----------------------------
# Step 2: Initialize ChromaDB
# -----------------------------
# Disable telemetry
client = chromadb.Client(Settings(anonymized_telemetry=False))
collection = client.get_or_create_collection(name="my_knowledge")

# -----------------------------
# Step 3: Create Embeddings & Store in Vector DB
# -----------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

for i, doc in enumerate(documents):
    embedding = model.encode(doc).tolist()
    collection.add(documents=[doc], ids=[str(i)], embeddings=[embedding])

# -----------------------------
# Step 4: Accept Query
# -----------------------------
query = input("Ask your question: ")
query_embed = model.encode(query).tolist()

results = collection.query(query_embeddings=[query_embed], n_results=2)
contexts = results['documents'][0]

# Combine top documents into one context string
context = "\n".join(contexts)

# -----------------------------
# Step 5: Build Prompt for Ollama
# -----------------------------
prompt = f"""Answer the following question using only the information in the context. Be concise and factual.

Context:
{context}

Question:
{query}

Answer:"""

# -----------------------------
# Step 6: Run Ollama CLI for Generation
# -----------------------------
result = subprocess.run(
    ["ollama", "run", "llama3.1"],
    input=prompt,
    capture_output=True,
    text=True
)

print("\n🧠 LLM Response:")
print(result.stdout.strip())

