import chromadb
from ai_provider import AIProvider


def summarize_document(provider: AIProvider, collection: chromadb.Collection, doc_id: str) -> str:
    results = collection.get(where={"doc_id": doc_id})
    chunks = results['documents']
    if not chunks:
        return "No chunks found for this document."

    print(f"🗺️ Map Phase: Summarizing {len(chunks)} chunks...")
    chunk_summaries = []
    for i, chunk in enumerate(chunks):
        summary = provider.generate(f"Extract the most important facts from this text in 1-2 short sentences:\n\n{chunk}")
        chunk_summaries.append(summary)
        print(f"  - Chunk {i+1}/{len(chunks)} summarized.")

    print("📉 Reduce Phase: Combining into final summary...")
    reduce_prompt = f"""You are an expert summarizer. Synthesize the following notes into a clean, cohesive, bullet-point summary. Remove duplicates.

Raw Notes:
{chr(10).join(chunk_summaries)}

Final Bulleted Summary:"""
    return provider.generate(reduce_prompt)


def chat_with_documents(llm_provider: AIProvider, embed_provider: AIProvider, collection: chromadb.Collection, query: str, doc_only: bool = True) -> str:
    """
    Searches across ALL documents in the collection simultaneously.
    doc_only=True  → answer strictly from document content
    doc_only=False → AI may supplement with its own knowledge
    """
    print(f"🔍 Searching across all documents: '{query}'")

    total = collection.count()
    if total == 0:
        return "No documents have been uploaded yet."

    query_embedding = embed_provider.embed(query)
    n = min(5, total)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n
    )

    retrieved_chunks = results['documents'][0] if results['documents'] else []

    if not retrieved_chunks:
        if doc_only:
            return "I couldn't find any relevant information in your documents to answer that."
        else:
            return llm_provider.generate(f"Answer the following question using your own knowledge:\n\nQuestion: {query}\n\nAnswer:")

    # Include source filenames so the AI knows which doc each chunk came from
    metadatas = results['metadatas'][0] if results['metadatas'] else []
    context_parts = []
    for chunk, meta in zip(retrieved_chunks, metadatas):
        source = meta.get('source', 'Unknown') if meta else 'Unknown'
        context_parts.append(f"[{source}]\n{chunk}")
    context = "\n\n---\n\n".join(context_parts)

    if doc_only:
        # ── DOC ONLY PROMPT ──────────────────────────────────────────────────────
        # Restricts the AI to only use content found in the uploaded documents.
        # If the answer isn't there, it must say so rather than guessing.
        prompt = f"""Answer the user's query based ONLY on the provided document excerpts below.
If the answer is not present in the excerpts, say "I cannot find that in your documents."
Do not use outside knowledge.

Document Excerpts:
{context}

Query: {query}

Answer:"""
    else:
        # ── OPEN PROMPT ──────────────────────────────────────────────────────────
        # Lets the AI use the documents as primary context but also draw on its
        # own training knowledge to fill gaps or add clarification.
        prompt = f"""Answer the user's query using the provided document excerpts as your primary source.
You may also draw on your own knowledge to supplement or clarify, but prioritize what's in the documents.
If you use information beyond the documents, briefly note it.

Document Excerpts:
{context}

Query: {query}

Answer:"""

    return llm_provider.generate(prompt)
