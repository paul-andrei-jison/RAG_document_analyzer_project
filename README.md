# DocuMind

Upload PDFs and text files, then ask questions across all of them at once. Runs entirely on your machine — no cloud, no API keys.

## How it works

1. You upload a document → it gets chunked and embedded using `nomic-embed-text` (via Ollama)
2. You ask a question → the app finds the most relevant chunks across all documents simultaneously
3. Your chosen LLM reads those chunks and answers

Nothing leaves your computer.

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
ollama pull nomic-embed-text
ollama pull llama3.2
```

Other models available in the dropdown:

```bash
ollama pull llama3.1
ollama pull mistral
ollama pull phi3
```

### 3. Create a virtual environment

**Windows:**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Mac / Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Open [http://localhost:8000](http://localhost:8000) in your browser.

> The frontend is served directly by FastAPI using relative URLs, so no tunnel or extra config is needed for local use.

**Accessing from another device on your network (phone, tablet, another PC):**

Find your machine's local IP:

```powershell
# Windows
ipconfig
```

Then open `http://<your-local-ip>:8000` on the other device.

---

## Usage

1. Upload a PDF or TXT file using the **+** button in the sidebar
2. Wait for indexing to finish
3. Type a question in the chat box
4. Use the **Doc only / Open knowledge** toggle to control whether the AI answers strictly from your documents or can supplement with its own knowledge

---

## Project structure

```
├── main.py           # FastAPI server and API endpoints
├── ai_provider.py    # Ollama provider (LLM + embeddings)
├── ingestion.py      # PDF/TXT parsing, chunking, embedding
├── rag_engine.py     # RAG chat logic
├── frontend/
│   └── index.html    # Single-page UI
├── docs-site/
│   └── index.html    # Landing / how-to page (deployed on Amplify)
├── requirements.txt
└── local_vectordb/   # ChromaDB vector store (auto-created, gitignored)
```
