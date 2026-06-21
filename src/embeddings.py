"""
embeddings.py
-------------
Small wrapper around Gemini embeddings so ingestion and query-time retrieval
don't depend on ChromaDB's Google helper, which currently has a client
compatibility issue in this environment.
"""

from __future__ import annotations

import google.generativeai as genai

from src import config


def _configure_genai() -> None:
    """Configure the Gemini client from the project's .env-backed settings."""
    config.require_api_key()
    genai.configure(api_key=config.GEMINI_API_KEY)


def embed_texts(texts: list[str], task_type: str) -> list[list[float]]:
    """Embed a batch of texts for either document indexing or query retrieval."""
    _configure_genai()

    embeddings: list[list[float]] = []
    for text in texts:
        result = genai.embed_content(
            model=config.EMBEDDING_MODEL,
            content=text,
            task_type=task_type,
        )
        embeddings.append(result["embedding"])

    return embeddings


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embeddings tuned for stored document chunks."""
    return embed_texts(texts, task_type="RETRIEVAL_DOCUMENT")


def embed_query(text: str) -> list[float]:
    """Embedding tuned for a single user query."""
    return embed_texts([text], task_type="RETRIEVAL_QUERY")[0]
