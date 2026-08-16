# Local RAG Document Analyzer

Upload a PDF or TXT file and chat with it using a fully local AI — no internet or cloud required after setup.

## How it works

1. You upload a document → it gets chunked and embedded using `nomic-embed-text` (via Ollama)
2. You ask a question → the app finds the most relevant chunks and feeds them to your chosen LLM
3. The LLM answers based on the document content

Everything runs on your machine. No data leaves your computer.

---

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/download) installed and running

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/paul-andrei-jison/RAG_document_analyzer_project.git
cd RAG_document_analyzer_project
```

### 2. Pull the required Ollama models

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

Optionally pull other models shown in the dropdown:

```bash
ollama pull llama3.1
ollama pull mistral
ollama pull phi3
```

### 3. Create and activate a Python virtual environment

**Windows:**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Mac/Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running the app

### 1. Start Ollama (if not already running)

Ollama usually starts automatically. If not:

```bash
ollama serve
```

### 2. Start the backend server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

### 3. Open the app

Go to [http://localhost:8000](http://localhost:8000) in your browser.

> **No tunnel needed.** The frontend is served directly by FastAPI, so the browser and backend are on the same machine. A Cloudflare tunnel or ngrok is only needed if you want to expose the app to the internet.

**Accessing from another device on your local network (phone, tablet, etc.):**

Find your machine's local IP address:

```powershell
# Windows
ipconfig
```

Then open `http://<your-local-ip>:8000` on the other device (e.g. `http://192.168.1.5:8000`).

---

## Usage

1. Select a model from the dropdown (Llama 3.2 is a good default)
2. Upload a PDF or TXT file using the left sidebar
3. Wait for ingestion and summary to complete
4. Ask questions in the chat box
5. Use the **Doc Only** toggle to switch between:
   - **Doc Only** — answers strictly from your document
   - **Open** — the AI can also use its general knowledge

---

## Project structure

```
├── main.py          # FastAPI server and API endpoints
├── ai_provider.py   # Ollama provider (LLM + embeddings)
├── ingestion.py     # PDF/TXT parsing, chunking, embedding
├── rag_engine.py    # Summarization and RAG chat logic
├── frontend/
│   └── index.html   # Single-page UI
├── requirements.txt
└── local_vectordb/  # ChromaDB vector store (auto-created)
```
