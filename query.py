import os
import pickle
import time
import numpy as np
import faiss
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

load_dotenv()

INDEX_FILE = "faiss_index.bin"
META_FILE = "faiss_meta.pkl"
TOP_K = 3


def main():
    print("Loading FAISS index...")
    index = faiss.read_index(INDEX_FILE)
    with open(META_FILE, "rb") as f:
        meta = pickle.load(f)
    texts = meta["texts"]
    metadatas = meta["metadatas"]
    print(f"Loaded {index.ntotal} vectors.")

    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-2-preview",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        request_timeout=60
    )
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        timeout=60
    )

    def search(query, k=TOP_K):
        q_emb = np.array(embeddings.embed_query(query), dtype="float32")
        q_emb = q_emb / np.linalg.norm(q_emb)
        q_emb = q_emb.reshape(1, -1)
        scores, indices = index.search(q_emb, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append({"text": texts[idx], "metadata": metadatas[idx], "score": float(score)})
        return results

    def ask_llm_with_retry(prompt, max_retries=3):
        for attempt in range(1, max_retries + 1):
            try:
                return llm.invoke(prompt)
            except Exception as e:
                print(f"(Connection hiccup, retry {attempt}/{max_retries}: {e})")
                time.sleep(3)
        raise RuntimeError("Failed after multiple retries — check your internet connection.")

    print("Ready. Ask a question (type 'exit' to quit).")
    while True:
        question = input("\nYou: ")
        if question.lower() == "exit":
            break

        results = search(question)
        context = "\n\n---\n\n".join(
            f"Source: {r['metadata'].get('source','unknown')}\nText: {r['text']}" for r in results
        )

        prompt = f"""You are a helpful assistant with access to a set of documents.

Use the following context if it's relevant to answer the question.
If the context contains the answer, base your response on it and end your answer with:
[Source: Document]

If the context does NOT contain relevant information, answer the question using your own general knowledge instead, and end your answer with:
[Source: General knowledge — not from the uploaded documents]

Context:
{context}

Question: {question}

Answer:"""

        try:
            response = ask_llm_with_retry(prompt)
            answer = response.content
            print(f"\nAI: {answer}")

            if "[Source: Document]" in answer:
                print("Sources:")
                for r in results:
                    print(f"  - {r['metadata'].get('source')} (score: {r['score']:.2f})")
        except Exception as e:
            print(f"\n⚠️ Could not get a response: {e}")
            print("This is likely a temporary network issue — try asking again.")


if __name__ == "__main__":
    main()