import random
import datetime
import numpy as np
import streamlit as st
import time
import re
import os
from dotenv import load_dotenv

load_dotenv()

from utils import load_orders, save_orders, generate_order_id
from config import TOP_K

# ============================================================
# HELPER: Recent History
# ============================================================
def get_recent_history(st_session_state, max_turns=3):
    history = st_session_state.messages[-(max_turns * 2):]
    lines = []
    for msg in history:
        role = "Customer" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)

# ============================================================
# HELPER: FAISS Search
# ============================================================
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
        if ".pdf" in src:
            display_name = "📄 " + src
            source_type = "pdf"
        else:
            display_name = "🌐 " + src.replace("scraped_", "").replace(".md", "").replace(".txt", "")
            source_type = "web"
        results.append({
            "text": texts[idx],
            "metadata": metadatas[idx],
            "display_name": display_name,
            "source_type": source_type,
            "score": float(score)
        })
    return results

# ============================================================
# HELPER: LLM Call with Retry
# ============================================================
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
# 🔥 ORDER MANAGER
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

# ============================================================
# AGENT 2: DOCUMENT (SMART PROMPT)
# ============================================================
def document_agent(question, embeddings, texts, metadatas, index, llm, st_session_state):
    results = search(question, embeddings, texts, metadatas, index)
    
    if not results:
        return "I don't have that information in my documents. Please contact our support team.", []
    
    for r in results:
        st_session_state.source_usage.append({
            "source_type": r["source_type"],
            "display_name": r["display_name"],
            "score": r["score"]
        })
    
    context = "\n\n---\n\n".join([f"Text: {r['text']}" for r in results])
    history = get_recent_history(st_session_state)
    
    # 🔥 SMART PROMPT - Connects related information
    prompt = f"""You are a customer support assistant. Answer the user's question based **ONLY** on the provided context below.

Recent conversation:
{history}

INSTRUCTIONS:
1. Read the context carefully.
2. If the context contains **any information related to the user's question** (for example, if the user asks about "privacy" and the context talks about "data collection", "cookies", "ads", or "usage of information"), use that information to form a clear, concise, and helpful answer.
3. **DO NOT** just look for the exact wording "Privacy Policy". Connect the dots and summarize what the context says about the user's topic.
4. If the context is **completely unrelated** to the question, only then say: "I don't have that information in my documents. Please contact our support team."

Context:
{context}

Question: {question}

Answer:"""
    
    answer = call_llm_with_retry(llm, prompt)
    return answer, results

# ============================================================
# AGENT 3: ORDER (SMART TRACKING - ACTUAL STATUS)
# ============================================================
class OrderAgent:
    def __init__(self):
        self.manager = OrderManager()
    
    def answer(self, query):
        q_lower = query.lower()
        
        # 1. Total spent
        if q_lower == "total" or "total spent" in q_lower:
            stats = self.manager.get_stats()
            return f"💰 Your total spent across all delivered orders is **${stats['total_spent']:.2f}**.", []
        
        # 2. Cancel order
        if "cancel" in q_lower and "ord-" in q_lower:
            return (
                "🛑 I'm sorry, but I don't have permission to cancel orders directly. "
                "You can cancel your order by visiting your account on our website.",
                []
            )
        
        # 3. Place order
        if "place" in q_lower or "add" in q_lower:
            return (
                "🛑 I'm just an assistant and don't have permission to place orders. "
                "You can place an order directly on our website or mobile app.",
                []
            )
        
        # 4. How many orders
        if "how many" in q_lower or "count" in q_lower:
            stats = self.manager.get_stats()
            total_orders = stats['active'] + stats['completed']
            return f"You have placed **{total_orders}** orders. Active: {stats['active']} | Delivered: {stats['completed']} | Cancelled: {stats['cancelled']}", []
        
        # 5. List all orders
        if "list" in q_lower or "all" in q_lower or "history" in q_lower:
            if not self.manager.orders:
                return "No orders found.", []
            lines = ["📜 **Your Order History:**"]
            for o in self.manager.orders[-5:]:
                lines.append(f"• `{o['id']}` | {o['product']} | {o['status']} | {o['date']} | ${o['total']:.2f}")
            return "\n".join(lines), []
        
        # 🔥 6. TRACK ORDER (NEW SMART LOGIC - REAL STATUS)
        if any(kw in q_lower for kw in ["track", "tracking", "status", "where is", "delivery", "shipped", "received"]):
            
            # Agar user ne specific order ID di hai (jaise "track ORD-123")
            match = re.search(r'(ORD-\d+)', query)
            if match:
                order_id = match.group(1)
                for o in self.manager.orders:
                    if o['id'] == order_id:
                        status = o['status']
                        product = o['product']
                        if status == "Processing":
                            return f"📦 Order {order_id} for '{product}' is currently **Processing**. It will be shipped soon.", []
                        elif status == "Shipped":
                            return f"🚚 Order {order_id} for '{product}' has been **Shipped**! It's on its way.", []
                        elif status == "In Transit":
                            return f"🚛 Order {order_id} for '{product}' is **In Transit**. Almost there!", []
                        elif status == "Delivered":
                            return f"✅ Order {order_id} for '{product}' has been **Delivered**! Thank you for shopping with us.", []
                        else:
                            return f"Order {order_id} is currently {status}.", []
                return f"Order {order_id} not found. Please check the ID.", []
            
            # Agar user ne koi specific ID nahi di, toh active orders dikhao
            active_orders = self.manager.get_active_orders()
            if not active_orders:
                # Agar active orders nahi hain, toh check karo ke koi delivered order hai ya nahi
                delivered = self.manager.get_completed_orders()
                if delivered:
                    latest = delivered[-1]
                    return f"✅ You don't have any active orders. Your last order '{latest['product']}' was delivered on {latest['date']}. Thank you!", []
                else:
                    return "📭 You don't have any orders yet. Visit the Shop to place your first order!", []
            
            # Active orders ka summary
            lines = ["📦 **Your Active Orders:**"]
            for o in active_orders:
                status_emoji = "⏳" if o['status'] == "Processing" else "🚚" if o['status'] == "Shipped" else "🚛"
                lines.append(f"• {status_emoji} `{o['id']}` | {o['product']} | **{o['status']}**")
            lines.append("\n💡 To track a specific order, say 'track ORD-XXX'")
            return "\n".join(lines), []
        
        # 7. Default
        return (
            "I can help with order info! Ask about:\n"
            "- 'How many orders do I have?'\n"
            "- 'List my orders'\n"
            "- 'Total spent'\n"
            "- 'Track my order'\n"
            "- 'Track ORD-123'",
            []
        )

# ============================================================
# AGENT 4: GENERAL
# ============================================================
def general_agent(question, llm, st_session_state):
    history = get_recent_history(st_session_state)
    prompt = f"""You are a professional customer support assistant.

Recent conversation:
{history}

Answer warmly and professionally.

Question: {question}

Answer:"""
    return call_llm_with_retry(llm, prompt), []

# ============================================================
# 🔥 MAIN ROUTER
# ============================================================
def run_agents(question, embeddings, texts, metadatas, index, llm, st_session_state):
    route = classifier_agent(question)
    
    if route == "order":
        agent = OrderAgent()
        return "order", *agent.answer(question)
    
    elif route == "document":
        return "document", *document_agent(question, embeddings, texts, metadatas, index, llm, st_session_state)
    
    else:
        return "general", *general_agent(question, llm, st_session_state)