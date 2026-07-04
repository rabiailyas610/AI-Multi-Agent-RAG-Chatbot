<div align="center">
  <h1>🤖AI-Multi-Agent-RAG-Chatbot</h1>
  <p><strong>Multi-Agent RAG Chatbot with FAISS + Google Gemini</strong></p>
  <p>
    <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">
    <img src="https://img.shields.io/badge/Gemini-8E75B2?style=for-the-badge&logo=google&logoColor=white" alt="Gemini">
    <img src="https://img.shields.io/badge/FAISS-0769AD?style=for-the-badge&logo=meta&logoColor=white" alt="FAISS">
    <img src="https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white" alt="LangChain">
  </p>
  <p>
    <a href="https://ai-multi-agent-rag-chatbot-msjanvxvbtbduatqlrvzkv.streamlit.app/"><img src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg" alt="Live Demo"></a>
  </p>
</div>


## 📖 Overview

**AI Support Agent** is a production-ready **Retrieval-Augmented Generation (RAG)** chatbot that answers customer queries using uploaded documents (PDFs, scraped web pages). It features a **multi-agent architecture** with intelligent routing, FAISS-based vector search, and Gemini-powered embeddings + LLM.

**Live Demo:** [https://rag-support-agent-jrwz52uvxuljdrmxjknsvm.streamlit.app/](https://rag-support-agent-jrwz52uvxuljdrmxjknsvm.streamlit.app/)


## 🚀 Features

| Feature | Description |
| :--- | :--- |
| 💬 **Multi-Agent Chat** | Four specialized agents: Classifier, Document (RAG), Order (Mock), and General. |
| 📚 **Document RAG** | Upload PDFs or scrape web pages — system automatically indexes them for retrieval. |
| 🛍️ **Shop & Cart** | Browse products, add to cart, and checkout with simulated order processing. |
| 📊 **Customer Dashboard** | Track orders, view purchase history, cancel orders, and see total spending. |
| 🔍 **FAISS Vector Search** | Lightning-fast semantic search over 353+ document chunks using Facebook's FAISS. |
| 🧠 **Gemini Embeddings** | Google's `gemini-embedding-001` model powers high-quality text embeddings. |
| 📱 **Mobile Responsive** | Optimized for both desktop and mobile devices. |
| 🔐 **Secure API Handling** | Environment variables and Streamlit Secrets for safe credential management. |
| 🔄 **Checkpoint System** | Resume capability for interrupted indexing (rate limits, server errors). |


## 🧠 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Query                             │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│              Agent 1: Classifier (Router)                      │
│   Determines query type: Document, Order, or General          │
└─────────────────────────────────────────────────────────────────┘
          │                    │                    │
          ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│  📚 Document    │  │  📦 Order       │  │  💬 General     │
│  Agent (RAG)    │  │  Agent (Mock)   │  │  Agent (Chat)   │
│  FAISS + Gemini │  │  Order Management│  │  LLM Response   │
└─────────────────┘  └─────────────────┘  └─────────────────┘
          │                    │                    │
          └────────────────────┼────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Response to User                             │
└─────────────────────────────────────────────────────────────────┘
```


## 🛠️ Tech Stack

| Category | Technology |
| :--- | :--- |
| **Frontend** | Streamlit 1.58.0 |
| **Vector Database** | FAISS (Facebook AI Similarity Search) |
| **LLM & Embeddings** | Google Gemini (`gemini-2.5-flash` + `gemini-embedding-001`) |
| **Framework** | LangChain (Document loaders, text splitters) |
| **Scraping** | Firecrawl API (JavaScript-heavy websites) |
| **Data Storage** | JSON (Chat history, orders, metadata) |
| **Deployment** | Streamlit Community Cloud |


## 📦 Installation

### Prerequisites
- Python 3.10+
- Google Gemini API key ([Get it here](https://aistudio.google.com/))
- (Optional) Firecrawl API key for web scraping

### Steps

```bash
# Clone the repository
git clone https://github.com/rabiailyas610/rag-support-agent.git
cd rag-support-agent

# Create virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
echo "GOOGLE_API_KEY=your_api_key_here" > .env
echo "FIRECRAWL_API_KEY=your_firecrawl_key_here" >> .env  # Optional

# Build FAISS index from documents
python ingest.py

# Run the application
streamlit run app.py
```


## 📁 Project Structure

```
rag-support-agent/
├── app.py                 # Main Streamlit UI
├── agents.py              # Multi-agent logic (Classifier, Document, Order, General)
├── utils.py               # Helpers: JSON, FAISS, Gemini loader
├── config.py              # Constants (file paths, TOP_K)
├── ingest.py              # PDF/Text ingestion + FAISS index builder
├── scrape_web.py          # Web scraping with Firecrawl API
├── requirements.txt       # Python dependencies
├── documents/             # PDFs and scraped Markdown files
├── faiss_index.bin        # FAISS vector index
├── faiss_meta.pkl         # FAISS metadata (texts, sources)
├── chats.json             # Chat history (auto-generated)
├── orders.json            # Order history (auto-generated)
└── .env                   # Environment variables (API keys)
```


## 🧪 Example Queries

| Query | Expected Agent |
| :--- | :--- |
| `"What is Amazon's privacy policy?"` | Document Agent (RAG) |
| `"Track my order"` | Order Agent (generates mock order) |
| `"How many orders do I have?"` | Order Agent (counts orders) |
| `"Cancel order ORD-1234"` | Order Agent (marks as cancelled) |
| `"Hi, how are you?"` | General Agent (casual chat) |
| `"What is Apple's shipping policy?"` | Document Agent (RAG) |


## 🎯 Key Engineering Decisions

1. **FAISS over ChromaDB**: FAISS is lightweight, local, and faster for smaller datasets. It avoids the DLL locking issues seen with ChromaDB on Windows.

2. **Multi-Agent Architecture**: Separating concerns (Classifier, Document, Order, General) makes the system modular, testable, and extensible.

3. **Strict Grounding**: Document Agent uses a strict prompt to prevent hallucinations — LLM only answers from the provided context.

4. **Checkpoint System**: `ingest.py` saves progress every 10 chunks, allowing resume capability if the process is interrupted (rate limits, server errors).

5. **Firecrawl over BeautifulSoup**: Firecrawl renders JavaScript-heavy sites (Amazon, Apple), ensuring clean Markdown output for RAG.


## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.


## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Google Gemini for embeddings and LLM
- Meta (Facebook) for FAISS
- Streamlit for the amazing UI framework
- Firecrawl for dynamic web scraping


## 📬 Contact

**Rabia Ilyas**  
[GitHub](https://github.com/rabiailyas610) | [LinkedIn](https://www.linkedin.com/in/rabia-ilyas-37436131a/)


<div align="center">
  ⭐ If you found this project useful, please star the repository! ⭐
</div>
