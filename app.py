import os
import time
import datetime
import streamlit as st

from config import TOP_K
from utils import load_resources, load_all_chats, save_all_chats, save_current_chat
from agents import run_agents, OrderManager

st.set_page_config(page_title="AI Support Agent", page_icon="🤖", layout="wide")

# ============================================================
# SESSION STATE INIT
# ============================================================
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
if "selected_order" not in st.session_state:
    st.session_state.selected_order = None
if "cart" not in st.session_state:
    st.session_state.cart = []
if "active_tab" not in st.session_state:
    st.session_state.active_tab = 0

# ============================================================
# CSS
# ============================================================
st.markdown("""
<style>
    .stApp { background-color: #f8fafc; }
    section[data-testid="stSidebar"] { position: sticky !important; top: 0 !important; height: 100vh !important; overflow-y: auto !important; }
    .stChatMessage { border-radius: 12px !important; padding: 14px 18px !important; margin-bottom: 12px !important; background-color: white !important; border: 1px solid #e2e8f0 !important; }
    .stChatMessage[data-testid="user"] { background-color: #ffffff !important; }
    .stChatMessage[data-testid="assistant"] { border-left: 4px solid #3b82f6 !important; }
    .stChatInput { position: sticky !important; bottom: 0 !important; background-color: #f8fafc !important; padding: 16px 0 8px 0 !important; z-index: 99 !important; border-top: 1px solid #e2e8f0 !important; }
    
    .stTabs {
        position: sticky !important;
        top: 0px !important;
        z-index: 9999 !important;
        background-color: #f8fafc !important;
        padding: 0 !important;
        margin: 0 !important;
        display: block !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        position: sticky !important;
        top: 0px !important;
        background-color: #f8fafc !important;
        z-index: 10000 !important;
        padding: 6px 0 !important;
        margin: 0 !important;
        border-bottom: 2px solid #e2e8f0 !important;
        gap: 2px !important;
        flex-wrap: nowrap !important;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 6px 16px !important;
        border-radius: 8px 8px 0 0 !important;
        font-weight: 500 !important;
        font-size: 14px !important;
        white-space: nowrap !important;
    }
    .stTabs [aria-selected="true"] { background-color: #3b82f6 !important; color: white !important; }
    .stTabs [aria-selected="false"] { color: #64748b !important; }
    
    .order-card {
        background: white;
        padding: 2px 6px !important;
        border-radius: 4px;
        border: 1px solid #e2e8f0;
        margin: 0 !important;
        display: flex;
        justify-content: space-between;
        align-items: center;
        min-height: 24px !important;
    }
    .order-card:hover { background-color: #f8fafc; border-color: #3b82f6; }
    .order-id { font-weight: 600; color: #0f172a; font-size: 11px; }
    .order-status { font-size: 9px; padding: 0px 5px; border-radius: 4px; font-weight: 600; }
    .status-processing { background: #fef3c7; color: #d97706; }
    .status-shipped { background: #dbeafe; color: #2563eb; }
    .status-transit { background: #e0e7ff; color: #4f46e5; }
    .status-delivered { background: #d1fae5; color: #059669; }
    .status-cancelled { background: #fee2e2; color: #dc2626; }
    
    .stColumns { gap: 0px !important; }
    .stColumns > div { padding: 0 2px !important; }
    
    .metric-card { background: white; padding: 8px 12px; border-radius: 8px; border: 1px solid #e2e8f0; margin-bottom: 4px; }
    .metric-value { font-size: 20px; font-weight: 700; color: #0f172a; }
    .metric-label { font-size: 11px; color: #64748b; }
    
    .product-card {
        background: white;
        padding: 8px;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        text-align: center;
        margin-bottom: 6px;
    }
    .product-card:hover { border-color: #3b82f6; }
    .product-name { font-weight: 600; font-size: 12px; }
    .product-price { color: #3b82f6; font-size: 14px; font-weight: 700; }
    
    .cart-item {
        background: white;
        padding: 2px 8px;
        border-radius: 4px;
        border: 1px solid #e2e8f0;
        margin-bottom: 0px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        min-height: 28px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD RESOURCES
# ============================================================
try:
    index, meta, embeddings, llm = load_resources()
    texts = meta["texts"]
    metadatas = meta["metadatas"]
    SYSTEM_READY = True
except:
    SYSTEM_READY = False
    st.warning("⚠️ Index not found. Run `python ingest.py` first.")

# ============================================================
# CHAT HELPERS
# ============================================================
def load_all_chats_local():
    from utils import load_all_chats
    return load_all_chats()

def save_all_chats_local(chats):
    from utils import save_all_chats
    save_all_chats(chats)

def save_current_chat_local():
    from utils import save_current_chat
    save_current_chat(st.session_state)

def ensure_chat_id():
    if st.session_state.current_chat_id is None:
        chat_id = str(int(time.time() * 1000))
        st.session_state.current_chat_id = chat_id
        st.session_state.chat_titles[chat_id] = "New Chat"
        chats = load_all_chats_local()
        chats[chat_id] = {"messages": [], "title": "New Chat", "timestamp": datetime.datetime.now().isoformat()}
        save_all_chats_local(chats)
        return True
    return False

def new_chat():
    if st.session_state.messages:
        save_current_chat_local()
    chat_id = str(int(time.time() * 1000))
    st.session_state.current_chat_id = chat_id
    st.session_state.messages = []
    st.session_state.chat_titles[chat_id] = "New Chat"
    st.session_state.route_history = []
    st.session_state.source_usage = []
    st.session_state.query_times = []
    st.session_state.activity_log = []
    st.session_state.processing = False
    chats = load_all_chats_local()
    chats[chat_id] = {"messages": [], "title": "New Chat", "timestamp": datetime.datetime.now().isoformat()}
    save_all_chats_local(chats)
    st.rerun()

def delete_chat(chat_id):
    chats = load_all_chats_local()
    if chat_id in chats:
        del chats[chat_id]
        save_all_chats_local(chats)
        if st.session_state.current_chat_id == chat_id:
            st.session_state.messages = []
            st.session_state.current_chat_id = None
        return True
    return False

def load_chat(chat_id):
    chats = load_all_chats_local()
    if chat_id in chats:
        st.session_state.messages = chats[chat_id]["messages"]
        st.session_state.current_chat_id = chat_id
        if "title" in chats[chat_id]:
            st.session_state.chat_titles[chat_id] = chats[chat_id]["title"]
        return True
    return False

# ============================================================
# ORDER & CART HELPERS
# ============================================================
def render_order_detail(order):
    with st.container():
        st.markdown(f"""
        <div style="background:white; padding:12px; border-radius:8px; border:1px solid #e2e8f0; margin:6px 0;">
            <h4>📦 {order['id']}</h4>
            <p><strong>Product:</strong> {order['product']}</p>
            <p><strong>Status:</strong> <span class="order-status status-{order['status'].lower().replace(' ', '-')}">{order['status']}</span></p>
            <p><strong>Date:</strong> {order['date']}</p>
            <p><strong>Total:</strong> ${order['total']:.2f}</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🔙 Back to Orders"):
            st.session_state.selected_order = None
            st.rerun()

def render_order_cards(orders, show_cancel=False):
    if not orders:
        st.info("🚫 No orders found.")
        return
    for idx, order in enumerate(orders):
        col1, col2, col3, col4 = st.columns([2, 2.5, 1.5, 1], gap="small")
        with col1:
            st.markdown(f"<span class='order-id'>{order['id']}</span>", unsafe_allow_html=True)
        with col2:
            st.write(order['product'])
        with col3:
            status_class = f"status-{order['status'].lower().replace(' ', '-')}"
            st.markdown(f"<span class='order-status {status_class}'>{order['status']}</span>", unsafe_allow_html=True)
        with col4:
            if st.button("📋", key=f"view_{order['id']}_{idx}"):
                st.session_state.selected_order = order
                st.rerun()
            if show_cancel and order['status'] not in ["Delivered", "Cancelled"]:
                if st.button("❌", key=f"cancel_{order['id']}_{idx}"):
                    manager = OrderManager()
                    if manager.cancel_order(order['id']):
                        st.toast(f"✅ Order {order['id']} cancelled!")
                        st.rerun()
                    else:
                        st.toast("⚠️ Cannot cancel this order.")

def add_to_cart(product, price):
    for item in st.session_state.cart:
        if item["name"] == product:
            item["qty"] += 1
            return
    st.session_state.cart.append({"name": product, "price": price, "qty": 1})

def remove_from_cart(product):
    st.session_state.cart = [item for item in st.session_state.cart if item["name"] != product]

def update_cart_qty(product, qty):
    for item in st.session_state.cart:
        if item["name"] == product:
            if qty <= 0:
                remove_from_cart(product)
            else:
                item["qty"] = qty
            return

def get_cart_total():
    return sum(item["price"] * item["qty"] for item in st.session_state.cart)

def clear_cart():
    st.session_state.cart = []

# ============================================================
# SIDEBAR
# ============================================================
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
    chats = load_all_chats_local()
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

# ============================================================
# VIEW FUNCTIONS
# ============================================================
def render_chat():
    st.title("💬 Chat")
    st.caption("Ask about documents, orders, or general knowledge.")
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
            chats = load_all_chats_local()
            if st.session_state.current_chat_id in chats:
                new_title = question[:40] + "..." if len(question) > 40 else question
                chats[st.session_state.current_chat_id]["title"] = new_title
                st.session_state.chat_titles[st.session_state.current_chat_id] = new_title
                save_all_chats_local(chats)
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("🧠 Agents thinking..."):
                start = time.time()
                try:
                    route, answer, results = run_agents(question, embeddings, texts, metadatas, index, llm, st.session_state)
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
        save_current_chat_local()
        st.session_state.processing = False
        st.rerun()
    question = st.chat_input("How can I help you today?", key="global_chat_input")
    if question:
        if st.session_state.get("processing", False):
            st.toast("⏳ Still thinking! Please wait.")
        else:
            ensure_chat_id()
            st.session_state.activity_log.append({
                "message": f"Query: \"{question[:50]}...\"",
                "time": datetime.datetime.now().strftime("%I:%M %p")
            })
            st.session_state.messages.append({"role": "user", "content": question})
            save_current_chat_local()
            st.rerun()

def render_dashboard():
    st.title("📊 Customer Dashboard")
    st.caption("Manage your orders, view cart, and track purchases.")
    st.markdown("---")
    manager = OrderManager()
    stats = manager.get_stats()
    total_orders = stats['active'] + stats['completed']
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class="metric-card"><div class="metric-value">{total_orders}</div><div class="metric-label">📦 Total Orders</div></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card"><div class="metric-value">{stats['active']}</div><div class="metric-label">🔄 Active</div></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="metric-card"><div class="metric-value">{stats['completed']}</div><div class="metric-label">✅ Completed</div></div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class="metric-card"><div class="metric-value">${stats['total_spent']:.2f}</div><div class="metric-label">💰 Total Spent</div></div>""", unsafe_allow_html=True)
    st.markdown("---")
    st.subheader("🛒 Your Cart")
    if st.session_state.cart:
        for item in st.session_state.cart:
            col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
            with col1: st.write(f"**{item['name']}**")
            with col2: st.write(f"${item['price']:.2f}")
            with col3:
                qty = st.number_input("Qty", min_value=0, value=item['qty'], key=f"qty_{item['name']}", label_visibility="collapsed")
                if qty != item['qty']:
                    update_cart_qty(item['name'], qty)
                    st.rerun()
            with col4:
                if st.button("❌", key=f"remove_{item['name']}"):
                    remove_from_cart(item['name'])
                    st.rerun()
        total_cart = get_cart_total()
        st.info(f"🛒 **Cart Total: ${total_cart:.2f}**")
        col1, col2 = st.columns([3,1])
        with col1:
            if st.button("🛍️ Checkout All", use_container_width=True):
                manager = OrderManager()
                for item in st.session_state.cart:
                    manager.add_order(item['name'], "Processing", item['price'] * item['qty'])
                st.toast(f"✅ {len(st.session_state.cart)} items ordered!")
                clear_cart()
                st.rerun()
        with col2:
            if st.button("🗑️ Clear Cart", use_container_width=True):
                clear_cart()
                st.rerun()
    else:
        st.info("🛒 Cart is empty. Browse the Shop!")
    st.markdown("---")
    if st.session_state.selected_order:
        render_order_detail(st.session_state.selected_order)
        return
    tab1, tab2, tab3 = st.tabs(["🔄 Active", "✅ Completed", "❌ Cancelled"])
    with tab1:
        active = manager.get_active_orders()
        if active:
            st.caption(f"{len(active)} active orders")
            render_order_cards(active, show_cancel=True)
        else:
            st.info("No active orders.")
    with tab2:
        completed = manager.get_completed_orders()
        if completed:
            st.caption(f"{len(completed)} completed orders")
            render_order_cards(completed, show_cancel=False)
        else:
            st.info("No completed orders.")
    with tab3:
        cancelled = manager.get_cancelled_orders()
        if cancelled:
            st.caption(f"{len(cancelled)} cancelled orders")
            render_order_cards(cancelled, show_cancel=False)
        else:
            st.info("No cancelled orders.")

def render_shop():
    st.title("🛍️ Shop")
    st.caption("Browse products. Add to cart or buy instantly.")
    st.markdown("---")
    if st.session_state.cart:
        total_items = sum(item["qty"] for item in st.session_state.cart)
        st.info(f"🛒 {total_items} items in cart. Go to Dashboard to checkout!")
    products = [
        {"name": "iPhone 15 Pro", "price": 999.00, "image": "📱"},
        {"name": "Samsung Galaxy S24", "price": 899.00, "image": "📱"},
        {"name": "MacBook Air M3", "price": 1099.00, "image": "💻"},
        {"name": "Sony WH-1000XM5", "price": 399.00, "image": "🎧"},
        {"name": "Nike Air Max", "price": 150.00, "image": "👟"},
        {"name": "Smart TV 55 inch", "price": 699.00, "image": "📺"},
        {"name": "Wireless Earbuds", "price": 79.99, "image": "🎧"},
        {"name": "Gaming Chair", "price": 249.00, "image": "🪑"},
        {"name": "Coffee Maker", "price": 89.99, "image": "☕"},
        {"name": "Fitness Tracker", "price": 59.99, "image": "⌚"},
        {"name": "Mechanical Keyboard", "price": 129.99, "image": "⌨️"},
        {"name": "Wireless Mouse", "price": 49.99, "image": "🖱️"},
        {"name": "External SSD 1TB", "price": 159.99, "image": "💾"},
        {"name": "Smart Watch", "price": 299.99, "image": "⌚"},
        {"name": "Robot Vacuum", "price": 399.99, "image": "🤖"},
        {"name": "Bluetooth Speaker", "price": 89.99, "image": "🔊"}
    ]
    cols = st.columns(4)
    for i, product in enumerate(products):
        with cols[i % 4]:
            st.markdown(f"""
            <div class="product-card">
                <div style="font-size: 36px;">{product['image']}</div>
                <div class="product-name">{product['name']}</div>
                <div class="product-price">${product['price']:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                if st.button("🛒 Add", key=f"add_{i}"):
                    add_to_cart(product['name'], product['price'])
                    st.toast(f"✅ Added {product['name']}!")
                    st.rerun()
            with c2:
                if st.button("⚡ Buy", key=f"buy_{i}"):
                    manager = OrderManager()
                    manager.add_order(product['name'], "Processing", product['price'])
                    st.toast(f"✅ Order placed!")
                    st.rerun()
    st.markdown("---")
    st.info("💡 Orders: Processing → (1 min) → Shipped → (1 min) → Delivered.")

# ============================================================
# MAIN LAYOUT
# ============================================================
tab_param = st.query_params.get("tab", "Chat")
if tab_param == "Shop":
    active_index = 2
elif tab_param == "Dashboard":
    active_index = 1
else:
    active_index = 0

tabs = st.tabs(["💬 Chat", "📊 Dashboard", "🛍️ Shop"])

for i, tab in enumerate(tabs):
    with tab:
        if i == 0:
            render_chat()
        elif i == 1:
            render_dashboard()
        elif i == 2:
            render_shop()

if tab_param != "Chat":
    st.markdown(f"""
    <script>
        const tabs = window.parent.document.querySelectorAll('button[role="tab"]');
        if (tabs && tabs.length > {active_index}) {{
            tabs[{active_index}].click();
        }}
    </script>
    """, unsafe_allow_html=True)
    st.query_params.clear()