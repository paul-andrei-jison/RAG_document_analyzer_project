import os
import uuid
import pypdf
import chromadb
from typing import List, Dict, Any
from ai_provider import AIProvider

def extract_text(filepath: str) -> str:
    """Extracts raw text from a given PDF or TXT file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    ext = filepath.lower().split('.')[-1]
    
    if ext == 'txt':
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
            
    elif ext == 'pdf':
        reader = pypdf.PdfReader(filepath)
        text = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)
        return " ".join(text)
        
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    A simple sliding-window character chunker.
    Keeps chunks around `chunk_size` characters with `overlap` characters of overlap.
    """
    # Clean up whitespace and newlines a bit
    text = " ".join(text.split())
    
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + chunk_size
        chunks.append(text[start:end])
        # Move forward, but step back by the overlap amount
        start += chunk_size - overlap
        
    return chunks

def ingest_document(
    provider: AIProvider, 
    collection: chromadb.Collection, 
    filepath: str
) -> Dict[str, Any]:
    """
    Extracts, chunks, embeds, and stores a document in the ChromaDB collection.
    Returns metadata about the ingested document.
    """
    print(f"📄 Extracting text from {filepath}...")
    raw_text = extract_text(filepath)
    
    print("✂️ Chunking text...")
    chunks = chunk_text(raw_text, chunk_size=1000, overlap=200)
    
    print(f"🧠 Embedding and storing {len(chunks)} chunks (this might take a moment)...")
    
    doc_id = str(uuid.uuid4())
    filename = os.path.basename(filepath)
    
    # Process chunks one by one. 
    # Batching locally can overwhelm modest hardware RAM.
    for i, chunk in enumerate(chunks):
        # 1. Ask our AI Provider to embed the chunk
        embedding = provider.embed(chunk)
        
        # 2. Save it to ChromaDB
        collection.add(
            ids=[f"{doc_id}-chunk-{i}"],
            embeddings=[embedding],
            documents=[chunk],
            metadatas=[{"source": filename, "doc_id": doc_id, "chunk_index": i}]
        )
        
    print("✅ Ingestion complete.")
    
    return {
        "doc_id": doc_id,
        "filename": filename,
        "chunk_count": len(chunks)
    }