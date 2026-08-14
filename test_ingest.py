import os
import chromadb
from ai_provider import get_ai_provider
from ingestion import ingest_document

def test_ingestion():
    # 1. Create a dummy text file
    sample_text = (
        "Retrieval-Augmented Generation (RAG) is an AI framework. "
        "It retrieves facts from an external database and passes them to an LLM. "
        "This grounds the LLM's answers in truth and reduces hallucinations. " * 20
    )
    with open("sample_rag.txt", "w", encoding="utf-8") as f:
        f.write(sample_text)

    # 2. Setup Provider and DB
    ai = get_ai_provider("ollama")
    
    # We use PersistentClient so the vectors save to disk in the "./local_vectordb" folder
    db_client = chromadb.PersistentClient(path="./local_vectordb")
    
    # get_or_create prevents errors if you run this multiple times
    collection = db_client.get_or_create_collection(name="documents")

    # 3. Run the ingestion
    stats = ingest_document(ai, collection, "sample_rag.txt")
    
    print(f"\nResult: Document '{stats['filename']}' saved with ID: {stats['doc_id']}")
    print(f"Total chunks created: {stats['chunk_count']}")
    
    # 4. Verify the database has the data
    count = collection.count()
    print(f"Total vectors in the ChromaDB collection: {count}")

if __name__ == "__main__":
    test_ingestion()