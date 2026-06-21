"""
query.py
--------
Query-time pipeline: embeds the user's question, retrieves the closest
chunks from ChromaDB, filters out low-relevance noise, builds a strictly
grounded prompt, and calls Gemini to produce a cited answer.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import google.generativeai as genai

from src import config
from src.embeddings import embed_query

SYSTEM_PROMPT = (
    "You are a precise, professional document Q&A assistant. "
    "Use ONLY the provided context to answer the user's question. "
    "Cite the source inline next to every fact you state, in the format "
    "(filename, Page X). "
    "If the answer cannot be found in the context, respond with exactly: "
    "'I cannot find the answer in the provided documents.' "
    "Do not use your own outside knowledge, and do not make up facts, "
    "filenames, or page numbers."
)

_NOT_FOUND_MESSAGE = "I cannot find the answer in the provided documents."


def _get_collection():
    """Opens the existing persistent ChromaDB collection (does not create one)."""
    import chromadb

    client = chromadb.PersistentClient(path=str(config.DB_DIR))

    try:
        return client.get_collection(name=config.COLLECTION_NAME)
    except Exception as e:
        raise RuntimeError(
            "Vector database not found or empty. Run `python -m src.ingest` "
            "first to index your documents."
        ) from e


def _format_context(documents: list[str], metadatas: list[dict], distances: list[float]):
    """
    Builds the citation-labeled context block handed to the LLM, and a
    parallel list of human-readable citation strings. Chunks whose distance
    exceeds MAX_DISTANCE_THRESHOLD are dropped as irrelevant noise, even if
    they technically made it into the top-k results.
    """
    context_blocks = []
    citations = []

    for doc, meta, dist in zip(documents, metadatas, distances):
        if dist is not None and dist > config.MAX_DISTANCE_THRESHOLD:
            continue

        source = meta.get("source", "unknown")
        page = meta.get("page", "N/A")
        citation_str = f"{source}, Page {page}"

        context_blocks.append(f"[Source: {source}, Page: {page}]\n{doc}")
        citations.append(citation_str)

    return "\n\n---\n\n".join(context_blocks), citations


def query_rag_pipeline(user_query: str, k: int | None = None) -> dict:
    """
    Runs the full retrieve -> ground -> generate pipeline for a single
    question.

    Returns:
        {
            "answer": str,
            "citations": list[str],   # de-duplicated-but-ordered "(file, Page N)" strings
            "raw_context": list[str], # the raw chunk text actually used
        }
    """
    if not user_query or not user_query.strip():
        raise ValueError("user_query must be a non-empty string.")

    config.require_api_key()
    k = k or config.TOP_K

    collection = _get_collection()
    query_embedding = embed_query(user_query)
    results = collection.query(query_embeddings=[query_embedding], n_results=k)

    documents = results["documents"][0] if results.get("documents") else []
    metadatas = results["metadatas"][0] if results.get("metadatas") else []
    distances = (
        results["distances"][0]
        if results.get("distances")
        else [None] * len(documents)
    )

    if not documents:
        return {"answer": _NOT_FOUND_MESSAGE, "citations": [], "raw_context": []}

    context_payload, citations = _format_context(documents, metadatas, distances)

    if not context_payload:
        # Everything retrieved was below the relevance threshold.
        return {"answer": _NOT_FOUND_MESSAGE, "citations": [], "raw_context": documents}

    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"CONTEXT INFORMATION:\n{context_payload}\n\n"
        f"USER QUESTION: {user_query}\n\n"
        f"GROUNDED ANSWER:"
    )

    model = genai.GenerativeModel(config.GENERATION_MODEL)
    response = model.generate_content(prompt)

    return {
        "answer": response.text,
        "citations": citations,
        "raw_context": documents,
    }
