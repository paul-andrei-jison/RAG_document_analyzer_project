import os
import shutil
import ollama
import chromadb
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ai_provider import get_ai_provider
from ingestion import ingest_document
from rag_engine import summarize_document, chat_with_document

app = FastAPI(title="Local RAG Chat API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("frontend", exist_ok=True)
app.mount("/static", StaticFiles(directory="frontend", html=True), name="static")

@app.get("/")
async def serve_frontend():
    return FileResponse("frontend/index.html")

# Global Ollama provider used only for embedding during ingestion
ai = get_ai_provider("ollama")
db_client = chromadb.PersistentClient(path="./local_vectordb")
collection = db_client.get_or_create_collection(name="documents")

UPLOAD_DIR = "./temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


class SummarizeRequest(BaseModel):
    doc_id: str
    provider: str = "ollama"
    model_id: str = "llama3.2"


class ChatRequest(BaseModel):
    doc_id: str
    query: str
    provider: str = "ollama"
    model_id: str = "llama3.2"


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.endswith(('.pdf', '.txt')):
        raise HTTPException(status_code=400, detail="Only PDF and TXT files are supported.")

    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        stats = ingest_document(ai, collection, file_path)
        os.remove(file_path)
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/summarize")
async def summarize_endpoint(request: SummarizeRequest):
    try:
        request_ai = get_ai_provider(request.provider, request.model_id)
        summary = summarize_document(request_ai, collection, request.doc_id)
        return {"summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        request_ai = get_ai_provider(request.provider, request.model_id)
        answer = chat_with_document(request_ai, collection, request.doc_id, request.query)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status")
async def status_endpoint():
    try:
        ollama.Client().list()
        return {"status": "online"}
    except Exception:
        return {"status": "offline"}
