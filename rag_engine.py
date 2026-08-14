import chromadb
from typing import List
from ai_provider import AIProvider

def summarize_document(provider: AIProvider, collection: chromadb.Collection, doc_id: str) -> str:
    """
    Implements a Map-Reduce summarization strategy for local hardware.
    """
    # 1. Fetch all chunks belonging to this document from ChromaDB
    results = collection.get(
        where={"doc_id": doc_id}
    )
    
    chunks = results['documents']
    if not chunks:
        return "No chunks found for this document."

    print(f"🗺️ Map Phase: Summarizing {len(chunks)} chunks individually...")
    chunk_summaries = []
    
    for i, chunk in enumerate(chunks):
        # We keep the prompt simple to save local compute time
        map_prompt = f"Extract the most important facts from this text in 1-2 short sentences:\n\n{chunk}"
        summary = provider.generate(map_prompt)
        chunk_summaries.append(summary)
        print(f"  - Chunk {i+1}/{len(chunks)} summarized.")

    print("📉 Reduce Phase: Combining into final notes...")
    combined_summaries = "\n".join(chunk_summaries)
    
    reduce_prompt = f"""
You are an expert summarizer. Take the following raw notes and synthesize them 
into a clean, cohesive, bullet-point summary. Remove duplicates.

Raw Notes:
{combined_summaries}

Final Bulleted Summary:
"""
    final_summary = provider.generate(reduce_prompt)
    return final_summary


def chat_with_document(provider: AIProvider, collection: chromadb.Collection, doc_id: str, query: str) -> str:
    """
    Retrieval-Augmented Generation (RAG) chat logic.
    """
    print(f"🔍 Searching for answers to: '{query}'")
    
    # 1. Embed the user's question
    query_embedding = provider.embed(query)
    
    # 2. Retrieve the top 3 most relevant chunks specifically from this document
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
        where={"doc_id": doc_id} 
    )
    
    retrieved_chunks = results['documents'][0] if results['documents'] else []
    
    if not retrieved_chunks:
         return "I couldn't find any relevant information in the document to answer that."

    # 3. Format the context
    context = "\n\n---\n\n".join(retrieved_chunks)
    
    # 4. Prompt the LLM grounded in the context
    prompt = f"""
Answer the user's query based ONLY on the provided context below. 
If the answer is not in the context, say "I cannot answer this based on the document."
Do not use outside knowledge.

Context:
{context}

Query: {query}

Answer:
"""
    return provider.generate(prompt)