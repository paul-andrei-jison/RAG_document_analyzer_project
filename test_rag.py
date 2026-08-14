import chromadb
from ai_provider import get_ai_provider
from rag_engine import summarize_document, chat_with_document

def test_pipeline():
    ai = get_ai_provider("ollama")
    db_client = chromadb.PersistentClient(path="./local_vectordb")
    collection = db_client.get_collection(name="documents")
    
    # Note: We need the doc_id from the previous ingestion step. 
    # Let's dynamically grab the first doc_id stored in the DB.
    all_metadata = collection.get()['metadatas']
    if not all_metadata:
        print("No documents found in DB. Run test_ingest.py first.")
        return
        
    doc_id = all_metadata[0]["doc_id"]
    print(f"Using Document ID: {doc_id}\n")
    
    print("--- 1. Testing Summarization ---")
    summary = summarize_document(ai, collection, doc_id)
    print("\n[DOCUMENT SUMMARY]")
    print(summary)
    
    print("\n--- 2. Testing Chat ---")
    question = "What does RAG stand for and what does it do?"
    answer = chat_with_document(ai, collection, doc_id, question)
    print(f"\nQuestion: {question}")
    print(f"Answer: {answer}")

if __name__ == "__main__":
    test_pipeline()