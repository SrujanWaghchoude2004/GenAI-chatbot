# chat.py
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

EMBED_MODEL = "all-MiniLM-L6-v2"
INDEX_FILE = "faiss_index.idx"
META_FILE = "meta.json"

# Load embedder
embedder = SentenceTransformer(EMBED_MODEL)

# Load FAISS index
index = faiss.read_index(INDEX_FILE)
meta = json.load(open(META_FILE, "r", encoding="utf-8"))

# Load GPT-2 model
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

def retrieve(query, top_k=3):
    q_emb = embedder.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, top_k)
    return [meta[i] for i in I[0]]

def generate_answer(query, context):
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=120)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

print("RAG GPT-2 Chatbot Ready!")
print("Type 'exit' to quit.\n")

while True:
    q = input("You: ")
    if q.lower() in ["exit", "quit"]:
        break

    docs = retrieve(q)
    context = "\n".join([f"{d['source']}: {d['text']}" for d in docs])

    ans = generate_answer(q, context)
    print("\nBot:", ans)
    print("\n" + "-"*40 + "\n")
