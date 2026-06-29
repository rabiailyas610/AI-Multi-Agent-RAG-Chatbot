import os
import time
import pickle
import numpy as np
import faiss
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from google.genai.errors import ClientError

load_dotenv()

DOCS_FOLDER = "documents"
INDEX_FILE = "faiss_index.bin"
META_FILE = "faiss_meta.pkl"


def load_documents():
    all_docs = []
    for filename in os.listdir(DOCS_FOLDER):
        if filename.endswith(".pdf"):
            path = os.path.join(DOCS_FOLDER, filename)
            loader = PyPDFLoader(path)
            docs = loader.load()
            all_docs.extend(docs)
            print(f"Loaded: {filename} ({len(docs)} pages)")
    return all_docs


def main():
    print("Loading PDFs...")
    documents = load_documents()
    print(f"Total pages: {len(documents)}")

    print("Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(documents)
    total = len(chunks)
    print(f"Total chunks: {total}")

    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-2-preview",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        request_options={"timeout": 60}
    )

    texts = []
    metadatas = []
    vectors = []

    print(f"Embedding {total} chunks one-by-one...", flush=True)
    for idx, chunk in enumerate(chunks, start=1):
        success = False
        while not success:
            try:
                emb = embeddings.embed_query(chunk.page_content)
                texts.append(chunk.page_content)
                metadatas.append({
                    "source": str(chunk.metadata.get("source", "unknown")),
                    "page": chunk.metadata.get("page", -1)
                })
                vectors.append(emb)
                print(f"[{idx}/{total}] Embedded.", flush=True)
                success = True
            except ClientError as e:
                if "RESOURCE_EXHAUSTED" in str(e):
                    print(f"[{idx}/{total}] Rate limit hit, waiting 30s...", flush=True)
                    time.sleep(30)
                else:
                    print(f"[{idx}/{total}] Error: {e}", flush=True)
                    raise
        time.sleep(0.5)

    print("Building FAISS index...", flush=True)
    vectors_np = np.array(vectors, dtype="float32")
    norms = np.linalg.norm(vectors_np, axis=1, keepdims=True)
    vectors_np = vectors_np / norms

    dim = vectors_np.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors_np)

    print("Saving index + metadata to disk...", flush=True)
    faiss.write_index(index, INDEX_FILE)
    with open(META_FILE, "wb") as f:
        pickle.dump({"texts": texts, "metadatas": metadatas}, f)

    print("Done! FAISS index built and saved.")
    print(f"Total vectors indexed: {index.ntotal}")


if __name__ == "__main__":
    main()