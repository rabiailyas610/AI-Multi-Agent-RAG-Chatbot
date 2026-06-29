import os
import pickle
import time
import numpy as np
import faiss
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

load_dotenv()

INDEX_FILE = "faiss_index.bin"
META_FILE = "faiss_meta.pkl"
TOP_K = 3

st.set_page_config(page_title="RAG Support Agent", page_icon="🤖")
st.title("🤖 AI Support Agent")
st.caption("Ask questions about your uploaded documents — answers grounded in your PDFs, with fallback to general knowledge.")


@st.cache_resource
def load_resources():
    index = faiss.read_index(INDEX_FILE)
    with open(META_FILE, "rb") as f:
        meta = pickle.load(f)
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
    return index, meta, embeddings, llm


index, meta, embeddings, llm = load_resources()
texts = meta["texts"]
metadatas = meta["metadatas"]


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
        except Exception:
            time.sleep(3)
    raise RuntimeError("Failed after multiple retries — check your internet connection.")


if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ask a question about your documents...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
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
            except Exception as e:
                answer = f"⚠️ Could not get a response: {e}"

            st.markdown(answer)

            if "[Source: Document]" in answer:
                with st.expander("📚 Sources used"):
                    for r in results:
                        st.write(f"- {r['metadata'].get('source')} (score: {r['score']:.2f})")

    st.session_state.messages.append({"role": "assistant", "content": answer})