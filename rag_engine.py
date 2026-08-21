import chromadb
from ai_provider import AIProvider


def chat_with_documents(llm_provider: AIProvider, embed_provider: AIProvider, collection: chromadb.Collection, query: str, doc_only: bool = True) -> dict:
    """
    Searches across ALL documents in the collection simultaneously.
    doc_only=True  → answer strictly from document content
    doc_only=False → AI may supplement with its own knowledge
    Returns dict: {answer, sources, chunks_retrieved}
    """
    print(f"🔍 Searching across all documents: '{query}'")

    total = collection.count()
    if total == 0:
        return {"answer": "No documents have been uploaded yet.", "sources": [], "chunks_retrieved": 0}

    query_embedding = embed_provider.embed(query)
    n = min(5, total)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n
    )

    retrieved_chunks = results['documents'][0] if results['documents'] else []

    if not retrieved_chunks:
        if doc_only:
            return {"answer": "I couldn't find any relevant information in your documents to answer that.", "sources": [], "chunks_retrieved": 0}
        else:
            answer = llm_provider.generate(f"Answer the following question using your own knowledge:\n\nQuestion: {query}\n\nAnswer:")
            return {"answer": answer, "sources": [], "chunks_retrieved": 0}

    metadatas = results['metadatas'][0] if results['metadatas'] else []
    context_parts = []
    unique_sources = []
    for chunk, meta in zip(retrieved_chunks, metadatas):
        source = meta.get('source', 'Unknown') if meta else 'Unknown'
        context_parts.append(f"[{source}]\n{chunk}")
        if source not in unique_sources:
            unique_sources.append(source)
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

    answer = llm_provider.generate(prompt)
    return {"answer": answer, "sources": unique_sources, "chunks_retrieved": len(retrieved_chunks)}
