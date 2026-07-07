# agents.py
import random
import datetime
import numpy as np
import streamlit as st
import re
from utils import load_orders, save_orders, generate_order_id
from config import TOP_K

# --- HELPER: Recent History ---
def get_recent_history(st_session_state, max_turns=3):
    history = st_session_state.messages[-(max_turns * 2):]
    lines = []
    for msg in history:
        role = "Customer" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)

# --- HELPER: FAISS Search ---
def search(query, embeddings, texts, metadatas, index, k=TOP_K):
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

# --- HELPER: LLM Call with Retry ---
def call_llm_with_retry(llm, prompt, max_retries=2):
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

# ============================================================
# 🔥 ORDER MANAGER (Full CRUD)
# ============================================================
class OrderManager:
    def __init__(self):
        self.orders = load_orders()
        self.auto_update_status()
    
    def save(self):
        save_orders(self.orders)
    
    def auto_update_status(self):
        now = datetime.datetime.now()
        status_flow = ["Processing", "Shipped", "In Transit", "Delivered"]
        for order in self.orders:
            if order["status"] == "Delivered" or order["status"] == "Cancelled":
                continue
            if "created_at" not in order:
                if "date" in order:
                    try:
                        created = datetime.datetime.strptime(order["date"], "%Y-%m-%d %H:%M")
                    except:
                        created = now - datetime.timedelta(seconds=random.randint(0, 60))
                else:
                    created = now - datetime.timedelta(seconds=random.randint(0, 60))
                order["created_at"] = created.isoformat()
                self.save()
            created = datetime.datetime.fromisoformat(order["created_at"])
            elapsed = (now - created).total_seconds()
            step = int(elapsed // 60)
            if step >= len(status_flow):
                order["status"] = "Delivered"
            else:
                order["status"] = status_flow[step]
        self.save()
    
    def add_order(self, product, status, total, created_at=None):
        if created_at is None:
            created_at = datetime.datetime.now().isoformat()
        order = {
            "id": generate_order_id(),
            "product": product,
            "status": status,
            "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "total": round(total, 2),
            "created_at": created_at
        }
        self.orders.append(order)
        self.save()
        return order
    
    def cancel_order(self, order_id):
        for o in self.orders:
            if o["id"] == order_id and o["status"] not in ["Delivered", "Cancelled"]:
                o["status"] = "Cancelled"
                self.save()
                return True
        return False
    
    def get_active_orders(self):
        return [o for o in self.orders if o["status"] not in ["Delivered", "Cancelled"]]
    
    def get_completed_orders(self):
        return [o for o in self.orders if o["status"] == "Delivered"]
    
    def get_cancelled_orders(self):
        return [o for o in self.orders if o["status"] == "Cancelled"]
    
    def get_stats(self):
        total = len(self.orders)
        active = len(self.get_active_orders())
        completed = len(self.get_completed_orders())
        cancelled = len(self.get_cancelled_orders())
        total_spent = sum(o["total"] for o in self.orders if o["status"] == "Delivered")
        return {
            "total": total,
            "active": active,
            "completed": completed,
            "cancelled": cancelled,
            "total_spent": round(total_spent, 2)
        }

# ============================================================
# AGENT 1: CLASSIFIER
# ============================================================
def classifier_agent(question):
    q_lower = question.lower()
    doc_keywords = [
        "policy", "policies", "terms", "conditions", "privacy", 
        "return", "refund", "exchange", "cancellation", 
        "carriage", "liability", "insurance", "customs",
        "charges", "fee", "fees", "cost", "pricing", "price", 
        "delivery charges", "shipping charges", "shipping fee",
        "payment", "billing", "tax", "vat", "duties",
        "document", "pdf", "section", "clause", "paragraph",
        "how to order", "place an order", "order process", "step", "guide", "procedure"
    ]
    if any(kw in q_lower for kw in doc_keywords):
        return "document"
    
    order_keywords = [
        "track", "tracking", "status", "delivery", "shipped", "parcel", 
        "where is", "not delivered", "received", "shipment", "courier", 
        "dispatch", "arrive", "late", "delay", "orders", "order history",
        "how many", "count", "list", "history", "cancel", "total"
    ]
    if any(kw in q_lower for kw in order_keywords):
        return "order"
    
    return "general"

# --- AGENT 2: Document ---
def document_agent(question, embeddings, texts, metadatas, index, llm, st_session_state):
    results = search(question, embeddings, texts, metadatas, index)
    if not results:
        return "I don't have that information in my documents. Is there something else I can help you with?", []
    
    for r in results:
        st_session_state.source_usage.append({
            "source_type": r["source_type"],
            "display_name": r["display_name"],
            "score": r["score"]
        })
    
    context = "\n\n---\n\n".join([f"Text: {r['text']}" for r in results])
    history = get_recent_history(st_session_state)
    
    prompt = f"""You are a customer support assistant. Answer based ONLY on the context.

Recent conversation:
{history}

STRICT RULES:
1. ONLY use the context.
2. If context has answer, summarize clearly.
3. If not, say: "I don't have that information in my documents."

Context:
{context}

Question: {question}

Answer:"""
    
    answer = call_llm_with_retry(llm, prompt)
    return answer, results

# ============================================================
# 🔥 AGENT 3: Order (FIXED — Full breakdown)
# ============================================================
class OrderAgent:
    def __init__(self):
        self.manager = OrderManager()
    
    def answer(self, query):
        q_lower = query.lower()
        
        # 🔥 FULL BREAKDOWN: Total, Active, Delivered, Cancelled
        if "how many" in q_lower or "count" in q_lower:
            stats = self.manager.get_stats()
            total_orders = stats['active'] + stats['completed']
            if total_orders == 0 and stats['cancelled'] == 0:
                return "You haven't placed any orders yet.", []
            return (
                f"You have placed **{total_orders}** orders in total.\n"
                f"🔄 **Active:** {stats['active']}\n"
                f"✅ **Delivered:** {stats['completed']}\n"
                f"❌ **Cancelled:** {stats['cancelled']}",
                []
            )
        
        # Total spent
        if q_lower == "total" or "total spent" in q_lower:
            stats = self.manager.get_stats()
            return f"💰 Total spent: ${stats['total_spent']:.2f}", []
        
        # Cancel - NOT ALLOWED
        if "cancel" in q_lower and "ord-" in q_lower:
            return "I don't have permission to cancel orders. Please use the website.", []
        
        # Place/Add - NOT ALLOWED
        if "place" in q_lower or "add" in q_lower:
            return "I'm just an assistant and don't have permission to place orders.", []
        
        # List orders
        if "list" in q_lower or "all" in q_lower or "history" in q_lower:
            if not self.manager.orders:
                return "No orders found.", []
            lines = ["📜 **Your Order History:**"]
            for o in self.manager.orders[-5:]:
                lines.append(f"• `{o['id']}` | {o['product']} | {o['status']} | {o['date']} | ${o['total']:.2f}")
            return "\n".join(lines), []
        
        # Track/Status - NOT ALLOWED
        if any(kw in q_lower for kw in ["track", "tracking", "status", "where is", "delivery", "shipped"]):
            return "I can't track specific orders. Please use the website.", []
        
        # Default
        return "Ask 'how many orders' or 'list my orders'.", []

# --- AGENT 4: General ---
def general_agent(question, llm, st_session_state):
    history = get_recent_history(st_session_state)
    prompt = f"""You are a professional customer support assistant.

Recent conversation:
{history}

Answer warmly and professionally. One direct answer.

Question: {question}

Answer:"""
    return call_llm_with_retry(llm, prompt), []

# --- MAIN ROUTER ---
def run_agents(question, embeddings, texts, metadatas, index, llm, st_session_state):
    route = classifier_agent(question)
    if route == "order":
        agent = OrderAgent()
        return "order", *agent.answer(question)
    elif route == "document":
        return "document", *document_agent(question, embeddings, texts, metadatas, index, llm, st_session_state)
    else:
        return "general", *general_agent(question, llm, st_session_state)