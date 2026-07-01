# utils.py
import os
import pickle
import json
import datetime
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import faiss
from config import INDEX_FILE, META_FILE, CHATS_FILE, ORDERS_FILE

load_dotenv()

# --- CHAT HISTORY HELPERS ---
def load_all_chats():
    if os.path.exists(CHATS_FILE):
        try:
            with open(CHATS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_all_chats(chats):
    with open(CHATS_FILE, "w", encoding="utf-8") as f:
        json.dump(chats, f, indent=2, ensure_ascii=False)

def save_current_chat(st_session_state):
    if st_session_state.current_chat_id and st_session_state.messages:
        chats = load_all_chats()
        chat_id = st_session_state.current_chat_id
        chats[chat_id] = {
            "messages": st_session_state.messages,
            "title": st_session_state.chat_titles.get(chat_id, "New Chat"),
            "timestamp": datetime.datetime.now().isoformat()
        }
        save_all_chats(chats)

# --- ORDERS HELPERS ---
def load_orders():
    if os.path.exists(ORDERS_FILE):
        try:
            with open(ORDERS_FILE, "r") as f:
                return json.load(f)
        except:
            return []
    return []

def save_orders(orders):
    with open(ORDERS_FILE, "w") as f:
        json.dump(orders, f, indent=2)

def generate_order_id():
    import random
    return "ORD-" + str(random.randint(1000, 9999))

# --- FAISS + GEMINI LOADER ---
@st.cache_resource
def load_resources():
    index = faiss.read_index(INDEX_FILE)
    with open(META_FILE, "rb") as f:
        meta = pickle.load(f)
    
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-2-preview",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        request_options={"timeout": 25}
    )
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        request_options={"timeout": 25}
    )
    return index, meta, embeddings, llm