import os
import time
import pickle
import numpy as np
import faiss
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

DOCS_FOLDER = "documents"
INDEX_FILE = "faiss_index.bin"
META_FILE = "faiss_meta.pkl"
CHECKPOINT_FILE = "checkpoint.pkl"

import google.genai
from typing import List

class GeminiEmbeddings:
    def __init__(self, api_key: str, model: str = "models/gemini-embedding-001"):
        self.client = google.genai.Client(api_key=api_key)
        self.model = model
    
    def embed_query(self, text: str) -> List[float]:
        response = self.client.models.embed_content(
            model=self.model,
            contents=text
        )
        return response.embeddings[0].values

def load_documents():
    all_docs = []
    for filename in os.listdir(DOCS_FOLDER):
        path = os.path.join(DOCS_FOLDER, filename)

        if filename.endswith(".pdf"):
            loader = PyPDFLoader(path)
            docs = loader.load()
            all_docs.extend(docs)
            print(f"Loaded PDF: {filename}")

        elif filename.endswith(".md") or filename.endswith(".txt"):
            loader = TextLoader(path, encoding="utf-8")
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = f"{filename}"
            all_docs.extend(docs)
            print(f"Loaded Text File: {filename}")

    return all_docs

def main():
    print("Loading documents...")
    documents = load_documents()
    print(f"Total pages/documents: {len(documents)}")

    print("Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(documents)
    total = len(chunks)
    print(f"Total chunks: {total}")

    # 🔥 FINAL FIX: Using correct model name from the list
    embeddings = GeminiEmbeddings(
        api_key=os.getenv("GOOGLE_API_KEY"),
        model="models/gemini-embedding-001"
    )

    # Load checkpoint if exists
    if os.path.exists(CHECKPOINT_FILE):
        print("Checkpoint found — resuming...")
        with open(CHECKPOINT_FILE, "rb") as f:
            checkpoint = pickle.load(f)
        texts = checkpoint["texts"]
        metadatas = checkpoint["metadatas"]
        vectors = checkpoint["vectors"]
        start_idx = checkpoint["next_idx"]
        print(f"Resuming from chunk {start_idx + 1}/{total}")
    else:
        texts, metadatas, vectors = [], [], []
        start_idx = 0

    print(f"Embedding chunks {start_idx + 1} to {total}...")
    for idx in range(start_idx, total):
        chunk = chunks[idx]
        success = False

        while not success:
            try:
                emb = embeddings.embed_query(chunk.page_content)
                texts.append(chunk.page_content)
                
                metadatas.append({
                    "source": str(chunk.metadata.get("source", "unknown")),
                    "page": chunk.metadata.get("page", -1),
                    "chunk_id": idx
                })
                vectors.append(emb)
                print(f"[{idx + 1}/{total}] Embedded.", flush=True)
                success = True

                if (idx + 1) % 10 == 0:
                    with open(CHECKPOINT_FILE, "wb") as f:
                        pickle.dump({
                            "texts": texts,
                            "metadatas": metadatas,
                            "vectors": vectors,
                            "next_idx": idx + 1
                        }, f)

            except Exception as e:
                error_str = str(e)
                if "RESOURCE_EXHAUSTED" in error_str or "429" in error_str:
                    print(f"[{idx + 1}/{total}] Rate limit — waiting 30s...", flush=True)
                    time.sleep(30)
                elif "503" in error_str or "UNAVAILABLE" in error_str:
                    print(f"[{idx + 1}/{total}] Server unavailable — waiting 15s...", flush=True)
                    time.sleep(15)
                else:
                    print(f"[{idx + 1}/{total}] Unexpected error: {e}", flush=True)
                    time.sleep(10)

        time.sleep(0.5)

    print("Building FAISS index...", flush=True)
    vectors_np = np.array(vectors, dtype="float32")
    norms = np.linalg.norm(vectors_np, axis=1, keepdims=True)
    vectors_np = vectors_np / norms

    dim = vectors_np.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors_np)

    print("Saving index + metadata...", flush=True)
    faiss.write_index(index, INDEX_FILE)
    with open(META_FILE, "wb") as f:
        pickle.dump({"texts": texts, "metadatas": metadatas}, f)

    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)

    print(f"Done! Indexed {index.ntotal} vectors.")

if __name__ == "__main__":
    main()