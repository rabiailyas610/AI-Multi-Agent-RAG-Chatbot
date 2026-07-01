import os
import pickle
import time
import random
import datetime
import json
import numpy as np
import faiss
import streamlit as st
import plotly.express as px
import pandas as pd
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from collections import Counter

load_dotenv()

INDEX_FILE = "faiss_index.bin"
META_FILE = "faiss_meta.pkl"
CHATS_FILE = "chats.json"
ORDERS_FILE = "orders.json"
TOP_K = 3

st.set_page_config(page_title="AI Support Agent", page_icon="🤖", layout="wide")

# ─── SESSION STATE INIT ──────────────────────────────────────────────────────
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_titles" not in st.session_state:
    st.session_state.chat_titles = {}
if "route_history" not in st.session_state:
    st.session_state.route_history = []
if "source_usage" not in st.session_state:
    st.session_state.source_usage = []
if "query_times" not in st.session_state:
    st.session_state.query_times = []
if "activity_log" not in st.session_state:
    st.session_state.activity_log = []
if "processing" not in st.session_state:
    st.session_state.processing = False
if "show_source_list" not in st.session_state:
    st.session_state.show_source_list = False

# ─── CUSTOM CSS (Strong Sticky Tabs) ──────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #f8fafc; }
    section[data-testid="stSidebar"] {
        position: sticky !important;
        top: 0 !important;
        height: 100vh !important;
        overflow-y: auto !important;
    }
    .stChatMessage {
        border-radius: 12px !important;
        padding: 14px 18px !important;
        margin-bottom: 12px !important;
        background-color: white !important;
        border: 1px solid #e2e8f0 !important;
    }
    .stChatMessage[data-testid="user"] { background-color: #ffffff !important; }
    .stChatMessage[data-testid="assistant"] { border-left: 4px solid #3b82f6 !important; }
    .stChatInput {
        position: sticky !important;
        bottom: 0 !important;
        background-color: #f8fafc !important;
        padding: 16px 0 8px 0 !important;
        z-index: 99 !important;
        border-top: 1px solid #e2e8f0 !important;
    }
    
    /* 🔥 FIX: STICKY TABS */
    .stTabs [data-baseweb="tab-list"] {
        position: sticky !important;
        top: 0px !important;
        background-color: #f8fafc !important;
        z-index: 1000 !important;
        padding: 12px 0 !important;
        margin-bottom: 16px !important;
        border-bottom: 2px solid #e2e8f0 !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04) !important;
        backdrop-filter: blur(8px) !important;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 24px !important;
        border-radius: 8px 8px 0 0 !important;
        font-weight: 500 !important;
    }
    .stTabs [aria-selected="true"] { background-color: #3b82f6 !important; color: white !important; }
    .stTabs [aria-selected="false"] { color: #64748b !important; }
    
    .metric-card { background: white; padding: 18px 16px; border-radius: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.04); border: 1px solid #e2e8f0; margin-bottom: 8px; }
    .metric-value { font-size: 28px; font-weight: 700; color: #0f172a; }
    .metric-label { font-size: 14px; color: #64748b; }
    .config-card { background: white; padding: 14px 16px; border-radius: 10px; border: 1px solid #e2e8f0; margin-bottom: 10px; }
    .config-label { font-size: 11px; color: #94a3b8; text-transform: uppercase; font-weight: 600; }
    .config-value { font-size: 15px; font-weight: 500; color: #0f172a; margin-top: 2px; }
    .activity-item { background: white; padding: 10px 14px; border-radius: 8px; border-left: 3px solid #3b82f6; margin-bottom: 8px; font-size: 14px; }
    .activity-time { font-size: 12px; color: #94a3b8; margin-top: 4px; }
    .source-badge-pdf { background-color: #eff6ff; color: #2563eb; padding: 2px 10px; border-radius: 12px; font-size: 11px; font-weight: 600; display: inline-block; }
    .source-badge-web { background-color: #f0fdf4; color: #16a34a; padding: 2px 10px; border-radius: 12px; font-size: 11px; font-weight: 600; display: inline-block; }
    .confidence-high { color: #22c55e; font-weight: 600; }
    .confidence-medium { color: #eab308; font-weight: 600; }
    .confidence-low { color: #ef4444; font-weight: 600; }
    
    .chat-list-item { padding: 8px 12px; margin: 4px 0; border-radius: 6px; cursor: pointer; background: transparent; border: none; text-align: left; width: 100%; font-size: 14px; color: #1e293b; transition: background 0.1s; display: flex; justify-content: space-between; align-items: center; }
    .chat-list-item:hover { background-color: #f1f5f9; }
    .chat-list-item.active { background-color: #e2e8f0; font-weight: 600; }
    .chat-delete-btn { background: none; border: none; color: #94a3b8; cursor: pointer; font-size: 14px; padding: 0 4px; }
    .chat-delete-btn:hover { color: #ef4444; }
</style>
""", unsafe_allow_html=True)

# ─── LOAD RESOURCES ──────────────────────────────────────────────────────────
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

try:
    index, meta, embeddings, llm = load_resources()
    texts = meta["texts"]
    metadatas = meta["metadatas"]
    SYSTEM_READY = True
except:
    SYSTEM_READY = False
    st.warning("⚠️ Index not found. Run `python ingest.py` first.")

# ─── CHAT STORAGE ────────────────────────────────────────────────────────────
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

def save_current_chat():
    if st.session_state.current_chat_id and st.session_state.messages:
        chats = load_all_chats()
        chat_id = st.session_state.current_chat_id
        chats[chat_id] = {
            "messages": st.session_state.messages,
            "title": st.session_state.chat_titles.get(chat_id, "New Chat"),
            "timestamp": datetime.datetime.now().isoformat()
        }
        save_all_chats(chats)

def load_chat(chat_id):
    chats = load_all_chats()
    if chat_id in chats:
        st.session_state.messages = chats[chat_id]["messages"]
        st.session_state.current_chat_id = chat_id
        if "title" in chats[chat_id]:
            st.session_state.chat_titles[chat_id] = chats[chat_id]["title"]
        return True
    return False

def delete_chat(chat_id):
    chats = load_all_chats()
    if chat_id in chats:
        del chats[chat_id]
        save_all_chats(chats)
        if st.session_state.current_chat_id == chat_id:
            st.session_state.messages = []
            st.session_state.current_chat_id = None
        return True
    return False

def new_chat():
    if st.session_state.messages:
        save_current_chat()
    chat_id = str(int(time.time() * 1000))
    st.session_state.current_chat_id = chat_id
    st.session_state.messages = []
    st.session_state.chat_titles[chat_id] = "New Chat"
    chats = load_all_chats()
    chats[chat_id] = {"messages": [], "title": "New Chat", "timestamp": datetime.datetime.now().isoformat()}
    save_all_chats(chats)
    st.rerun()

def ensure_chat_id():
    if st.session_state.current_chat_id is None:
        chat_id = str(int(time.time() * 1000))
        st.session_state.current_chat_id = chat_id
        st.session_state.chat_titles[chat_id] = "New Chat"
        chats = load_all_chats()
        chats[chat_id] = {"messages": [], "title": "New Chat", "timestamp": datetime.datetime.now().isoformat()}
        save_all_chats(chats)
        return True
    return False

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 💬 Chats")
    st.markdown("---")
    if st.button("➕ New Chat", use_container_width=True):
        new_chat()
    st.markdown("---")
    if SYSTEM_READY:
        st.caption(f"🟢 Online · 📚 {len(texts)} chunks")
    else:
        st.caption("🔴 Offline")
    st.markdown("---")
    chats = load_all_chats()
    if chats:
        sorted_chats = sorted(chats.items(), key=lambda x: x[1].get("timestamp", ""), reverse=True)
        for chat_id, chat_data in sorted_chats:
            title = chat_data.get("title", "New Chat")
            display_title = title[:35] + "..." if len(title) > 35 else title
            is_active = chat_id == st.session_state.current_chat_id
            col1, col2 = st.columns([0.85, 0.15])
            with col1:
                if st.button(display_title, key=f"chat_{chat_id}", use_container_width=True,
                             type="secondary" if not is_active else "primary"):
                    load_chat(chat_id)
                    st.rerun()
            with col2:
                if st.button("🗑️", key=f"del_{chat_id}"):
                    delete_chat(chat_id)
                    st.rerun()
    else:
        st.caption("No chats yet. Start a new conversation!")

# ─── AGENTS ──────────────────────────────────────────────────────────────────
def get_recent_history(max_turns=3):
    history = st.session_state.messages[-(max_turns * 2):]
    lines = []
    for msg in history:
        role = "Customer" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)

def search(query, k=TOP_K):
    if not SYSTEM_READY:
        return []
    q_emb = np.array(embeddings.embed_query(query), dtype="float32")
    q_emb = q_emb / np.linalg.norm(q_emb)
    q_emb = q_emb.reshape(1, -1)
    scores, indices = index.search(q_emb, k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        src = metadatas[idx].get('source', 'Unknown')
        if "scraped" in src or ".txt" in src:
            display_name = "🌐 " + src.replace("scraped_", "").replace(".md", "").replace(".txt", "")
            source_type = "web"
        else:
            display_name = "📄 " + src
            source_type = "pdf"
        results.append({
            "text": texts[idx],
            "metadata": metadatas[idx],
            "display_name": display_name,
            "source_type": source_type,
            "score": float(score)
        })
    return results

def call_llm_with_retry(prompt, max_retries=2):
    for attempt in range(1, max_retries + 1):
        try:
            return llm.invoke(prompt).content
        except Exception as e:
            if "timeout" in str(e).lower() or "429" in str(e) or "resource_exhausted" in str(e).lower():
                raise RuntimeError("⚠️ API slow/quota khatam. 5 minute baad try karein.")
            else:
                if attempt == max_retries:
                    raise RuntimeError(f"⚠️ Error: {e}")
                time.sleep(3)

def classifier_agent(question):
    q_lower = question.lower()
    order_tracking = ["track", "tracking", "status", "delivery", "shipped", "parcel", "where is", "not delivered", "received", "shipment", "courier", "dispatch", "arrive", "late", "delay"]
    if any(kw in q_lower for kw in order_tracking):
        return "order"
    doc_keywords = ["shipping", "return", "refund", "privacy", "policy", "terms", "apple", "amazon", "daraz", "document", "pdf", "how to order", "place an order", "order process", "step", "guide", "procedure"]
    if any(kw in q_lower for kw in doc_keywords):
        return "document"
    return "general"

def document_agent(question):
    results = search(question)
    if not results:
        return "I don't have that information in my documents. Is there something else I can help you with?", []
    for r in results:
        st.session_state.source_usage.append({
            "source_type": r["source_type"],
            "display_name": r["display_name"],
            "score": r["score"]
        })
    context = "\n\n---\n\n".join([f"Text: {r['text']}" for r in results])
    history = get_recent_history()
    prompt = f"""You are a customer support assistant. Your answer MUST be based ONLY on the context provided below.

Recent conversation (for context – use this to understand follow-up questions):
{history}

STRICT RULES:
1. ONLY use the text from the context below for factual answers. Do NOT use your general knowledge.
2. Use the recent conversation above to understand what the customer is referring to (e.g. "it", "that order", "when will it arrive").
3. If the context contains the answer, extract and summarize it in simple, natural English. Be specific – provide details, not just a link or a vague statement.
4. If the context does NOT contain the answer, say EXACTLY: "I don't have that information in my documents. Is there something else I can help you with?"

Context:
{context}

Customer Question: {question}

Your Reply:"""
    answer = call_llm_with_retry(prompt)
    return answer, results

# ─── ORDER AGENT (Updates orders.json) ─────────────────────────────────────
class OrderAgent:
    def __init__(self):
        self.orders_file = ORDERS_FILE
        self.load_orders()
    
    def load_orders(self):
        if os.path.exists(self.orders_file):
            try:
                with open(self.orders_file, "r") as f:
                    self.orders = json.load(f)
                return
            except:
                pass
        self.orders = []
    
    def save_orders(self):
        with open(self.orders_file, "w") as f:
            json.dump(self.orders, f, indent=2)
    
    def answer(self, query):
        # Product list
        products = [
            "iPhone 15 Pro", "Samsung Galaxy S24", "Sony WH-1000XM5",
            "MacBook Air M3", "Nike Air Max", "Adidas Ultraboost",
            "Dell XPS 16", "iPad Pro", "Apple Watch Series 9",
            "PlayStation 5", "Wireless Mouse", "Mechanical Keyboard",
            "4K Monitor", "Bluetooth Speaker", "Smart TV 55 inch"
        ]
        product = random.choice(products)
        statuses = ["Processing", "Shipped", "In Transit", "Out for Delivery", "Delivered"]
        status = random.choice(statuses)
        date = (datetime.datetime.now() - datetime.timedelta(days=random.randint(0, 15))).strftime("%Y-%m-%d")
        total = round(random.uniform(29.99, 1599.99), 2)
        
        order_id = "ORD-" + str(random.randint(1000, 9999))
        
        new_order = {
            "id": order_id,
            "product": product,
            "status": status,
            "date": date,
            "total": total
        }
        self.orders.append(new_order)
        self.save_orders()
        
        return f"📦 **Order {order_id}** ({product}) is **{status}**. Placed on {date}. Total: ${total:.2f}", []

def general_agent(question):
    history = get_recent_history()
    prompt = f"""You are a professional and warm customer support assistant.

Recent conversation:
{history}

Answer the user's question in a friendly but professional tone.
Use the recent conversation to understand follow-up questions (e.g. "when", "it", "that").
Keep it simple, clear, and natural. One direct answer, no brackets/options.

Question: {question}

Your Reply:"""
    return call_llm_with_retry(prompt), []

def run_agents(question):
    route = classifier_agent(question)
    if route == "order":
        return "order", *OrderAgent().answer(question)
    elif route == "document":
        return "document", *document_agent(question)
    else:
        return "general", *general_agent(question)

# ─── CHAT RENDER ──────────────────────────────────────────────────────────────
def render_chat():
    st.title("💬 Chat")
    st.caption("Ask about documents, orders, or general knowledge. Powered by FAISS + Gemini.")
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            with st.chat_message("user", avatar="👤"):
                st.markdown(msg["content"])
        else:
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(msg["content"])
                if "sources" in msg and msg["sources"]:
                    with st.container():
                        st.caption("📚 Sources:")
                        unique_sources = {}
                        for s in msg["sources"]:
                            key = s['display_name']
                            if key not in unique_sources or s['score'] > unique_sources[key]['score']:
                                unique_sources[key] = s
                        for s in unique_sources.values():
                            badge = "📄 PDF" if s["source_type"] == "pdf" else "🌐 WEB"
                            st.caption(f"• {badge} {s['display_name']} (score: {s['score']:.2f})")
    
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user" and not st.session_state.get("processing", False):
        st.session_state.processing = True
        question = st.session_state.messages[-1]["content"]
        if len(st.session_state.messages) == 1 and st.session_state.current_chat_id:
            chats = load_all_chats()
            if st.session_state.current_chat_id in chats:
                new_title = question[:40] + "..." if len(question) > 40 else question
                chats[st.session_state.current_chat_id]["title"] = new_title
                st.session_state.chat_titles[st.session_state.current_chat_id] = new_title
                save_all_chats(chats)
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("🧠 Agents thinking..."):
                start = time.time()
                try:
                    route, answer, results = run_agents(question)
                    st.session_state.route_history.append(route)
                except Exception as e:
                    route = "general"
                    answer = f"⚠️ {str(e)}"
                    results = []
                elapsed = time.time() - start
                st.session_state.query_times.append({"timestamp": datetime.datetime.now().strftime("%H:%M"), "latency": elapsed})
                if route == "document":
                    st.session_state.activity_log.append({
                        "message": f"📚 Document search: {len(results)} sources in {elapsed:.1f}s",
                        "time": datetime.datetime.now().strftime("%I:%M %p")
                    })
                st.markdown(answer)
                if route == "document" and results:
                    avg_score = sum(r["score"] for r in results) / len(results)
                    if avg_score > 0.8: color = "confidence-high"; label = "High"
                    elif avg_score > 0.5: color = "confidence-medium"; label = "Medium"
                    else: color = "confidence-low"; label = "Low"
                    st.markdown(f"**Confidence:** <span class='{color}'>{label} ({avg_score:.2f})</span>", unsafe_allow_html=True)
                    st.progress(min(avg_score, 1.0))
                    st.markdown("---")
                    st.caption("📚 **Sources Used:**")
                    unique_results = {}
                    for r in results:
                        key = r['display_name']
                        if key not in unique_results or r['score'] > unique_results[key]['score']:
                            unique_results[key] = r
                    cols = st.columns(len(unique_results))
                    for i, (key, r) in enumerate(unique_results.items()):
                        with cols[i]:
                            badge = "📄 PDF" if r["source_type"] == "pdf" else "🌐 WEB"
                            st.markdown(f"<span class='source-badge-pdf'>{badge}</span>", unsafe_allow_html=True)
                            st.caption(r["display_name"][:25])
                            st.caption(f"Score: {r['score']:.2f}")
            assistant_msg = {"role": "assistant", "content": answer, "sources": results if route == "document" else []}
            st.session_state.messages.append(assistant_msg)
            save_current_chat()
            st.session_state.processing = False
            st.rerun()
    
    # ─── CHAT INPUT (PROFESSIONAL - No Disable) ─────────────────────────────
    question = st.chat_input("How can I help you today?", key="global_chat_input")
    if question:
        if st.session_state.get("processing", False):
            st.toast("⏳ I'm still thinking about your last question! Please wait a moment.")
        else:
            ensure_chat_id()
            st.session_state.activity_log.append({
                "message": f"Query: \"{question[:50]}...\"",
                "time": datetime.datetime.now().strftime("%I:%M %p")
            })
            st.session_state.messages.append({"role": "user", "content": question})
            save_current_chat()
            st.rerun()

# ─── DASHBOARD (Customer View) ──────────────────────────────────────────────
def render_dashboard():
    st.title("📊 Customer Dashboard")
    st.caption("Your orders, cart, and purchase history.")
    st.markdown("---")
    
    # Load orders
    orders = []
    if os.path.exists(ORDERS_FILE):
        try:
            with open(ORDERS_FILE, "r") as f:
                orders = json.load(f)
        except:
            pass
    
    # --- Order Summary Cards ---
    total_orders = len(orders)
    delivered = sum(1 for o in orders if o["status"] == "Delivered")
    shipped = sum(1 for o in orders if o["status"] in ["Shipped", "Out for Delivery", "In Transit"])
    pending = sum(1 for o in orders if o["status"] == "Processing")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class="metric-card"><div class="metric-value">{total_orders}</div><div class="metric-label">📦 Total Orders</div></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card"><div class="metric-value">{pending}</div><div class="metric-label">⏳ Pending</div></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="metric-card"><div class="metric-value">{shipped}</div><div class="metric-label">🚚 Shipped</div></div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class="metric-card"><div class="metric-value">{delivered}</div><div class="metric-label">✅ Delivered</div></div>""", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # --- All Purchased Items Summary ---
    if orders:
        st.subheader("🛍️ All Purchased Items")
        product_counts = {}
        for o in orders:
            p = o["product"]
            product_counts[p] = product_counts.get(p, 0) + 1
        
        # Show as a nice list
        cols = st.columns(3)
        idx = 0
        for product, count in product_counts.items():
            with cols[idx % 3]:
                st.write(f"• **{product}** (x{count})")
            idx += 1
        
        if not product_counts:
            st.info("No purchases yet. Ask me to 'track my order' to generate a demo order!")
        
        st.markdown("---")
    else:
        st.info("No purchases yet. Ask me to 'track my order' to generate a demo order!")
        st.markdown("---")
    
    # --- Cart Section (Mock) ---
    st.subheader("🛒 Your Cart")
    cart_items = [
        {"product": "Wireless Mouse", "qty": 1, "price": 25.99},
        {"product": "USB-C Cable", "qty": 2, "price": 12.50}
    ]
    if cart_items:
        for item in cart_items:
            st.write(f"• **{item['product']}** x {item['qty']} — ${item['price']:.2f} each")
        total_cart = sum(item["qty"] * item["price"] for item in cart_items)
        st.info(f"🛒 Cart Total: ${total_cart:.2f}")
    else:
        st.info("Your cart is empty.")
    
    st.markdown("---")
    
    # --- Recent Purchase History ---
    st.subheader("📜 Order History")
    if orders:
        # Show latest orders (newest first)
        recent = sorted(orders, key=lambda x: x["date"], reverse=True)
        
        # Create a table using columns
        cols = st.columns([2, 3, 2, 2, 2])
        with cols[0]: st.write("**Order ID**")
        with cols[1]: st.write("**Product**")
        with cols[2]: st.write("**Date**")
        with cols[3]: st.write("**Status**")
        with cols[4]: st.write("**Total**")
        st.markdown("---")
        
        for o in recent:
            c1, c2, c3, c4, c5 = st.columns([2, 3, 2, 2, 2])
            with c1: st.write(o["id"])
            with c2: st.write(o["product"])
            with c3: st.write(o["date"])
            with c4: st.write(o["status"])
            with c5: st.write(f"${o['total']:.2f}")
    else:
        st.info("No orders yet. Ask me to 'track my order' to generate a demo order!")

# ─── MAIN TABS ──────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["💬 Chat", "📊 Dashboard"])
with tab1:
    render_chat()
with tab2:
    render_dashboard()