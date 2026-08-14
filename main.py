import os
import shutil
import chromadb
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import our custom modules from the previous chunks
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
    """Serves the main HTML interface."""
    return FileResponse("frontend/index.html")

# Initialize our AI Provider and Vector DB on startup
ai = get_ai_provider("ollama")
db_client = chromadb.PersistentClient(path="./local_vectordb")
collection = db_client.get_or_create_collection(name="documents")

# Temporary storage for uploaded files before ingestion
UPLOAD_DIR = "./temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# --- Pydantic Models for Request Bodies ---
class SummarizeRequest(BaseModel):
    doc_id: str

class ChatRequest(BaseModel):
    doc_id: str
    query: str

# --- API Endpoints ---

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Accepts a file upload (PDF/TXT), saves it temporarily, and ingests it into the vector database.
    """
    if not file.filename.endswith(('.pdf', '.txt')):
        raise HTTPException(status_code=400, detail="Only PDF and TXT files are supported.")
    
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    
    # Save the uploaded file to disk
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        # Run our ingestion pipeline
        stats = ingest_document(ai, collection, file_path)
        
        # Clean up the temporary file
        os.remove(file_path)
        
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/summarize")
async def summarize_endpoint(request: SummarizeRequest):
    """
    Runs the Map-Reduce summarization on a specific document.
    """
    try:
        summary = summarize_document(ai, collection, request.doc_id)
        return {"summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """
    Runs the RAG retrieval and chat logic for a specific document.
    """
    try:
        answer = chat_with_document(ai, collection, request.doc_id, request.query)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))