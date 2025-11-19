# ingest.py
import os
import json
import faiss
import numpy as np
from tqdm import tqdm
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer

DATA_DIR = "data"
INDEX_FILE = "faiss_index.idx"
META_FILE = "meta.json"

EMBED_MODEL = "all-MiniLM-L6-v2"
EMBED_DIM = 384

def load_pdf(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        t = page.extract_text()
        if t:
            text += " " + t
    return text

def chunk(text, size=300):
    words = text.split()
    return [" ".join(words[i:i+size]) for i in range(0, len(words), size)]

def main():
    model = SentenceTransformer(EMBED_MODEL)
    all_chunks = []
    meta = []

    print("Loading files...")
    for file in os.listdir(DATA_DIR):
        path = os.path.join(DATA_DIR, file)
        if file.endswith(".pdf"):
            text = load_pdf(path)
        else:
            text = open(path, "r", encoding="utf-8").read()

        chunks = chunk(text)
        for i, ch in enumerate(chunks):
            all_chunks.append(ch)
            meta.append({"text": ch, "source": f"{file} (chunk {i})"})

    print("Embedding...")
    embeddings = model.encode(all_chunks, convert_to_numpy=True)
    embeddings = embeddings.astype("float32")
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(embeddings)

    faiss.write_index(index, INDEX_FILE)
    json.dump(meta, open(META_FILE, "w", encoding="utf-8"))

    print("Index created successfully.")

if __name__ == "__main__":
    main()
