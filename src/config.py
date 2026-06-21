"""
config.py
---------
Centralized configuration for the Document Q&A RAG Bot.

Everything that another developer might want to tune (chunk size, retrieval
depth, model names, paths) lives here, so the rest of the codebase never
hardcodes "magic" values.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load variables from a local .env file (see .env.example) into the process
# environment. This is a no-op if .env does not exist, which keeps things
# friendly for first-time setup.
load_dotenv()

# --- Paths -------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DB_DIR = BASE_DIR / "db"

# --- API -----------------------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- Models --------------------------------------------------------------
# Embedding model used for both indexing chunks and embedding user queries.
# These MUST match, otherwise similarity search is meaningless.
EMBEDDING_MODEL = "models/gemini-embedding-001"

# Generation model used to produce the final, grounded answer.
GENERATION_MODEL = "models/gemini-2.5-flash"

# --- Vector store ----------------------------------------------------------
COLLECTION_NAME = "document_knowledge_base"

# --- Chunking --------------------------------------------------------------
# Characters per chunk and overlap between consecutive chunks. See README.md
# ("Design Decisions") for the reasoning behind these defaults.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# --- Retrieval ---------------------------------------------------------
# Number of chunks to retrieve per query.
TOP_K = 4

# ChromaDB's default distance metric here is cosine distance (0 = identical,
# 2 = opposite). Chunks with a distance above this threshold are treated as
# noise and dropped before being shown to the LLM, even if they were in the
# top-k. Tune this if the bot is either too trigger-happy or too conservative
# about saying "I cannot find the answer."
MAX_DISTANCE_THRESHOLD = 1.0

# --- Ingestion -----------------------------------------------------------
# How many chunks to embed/upload to ChromaDB per API batch.
EMBED_BATCH_SIZE = 50

SUPPORTED_EXTENSIONS = {".pdf", ".docx"}


def require_api_key() -> None:
    """Raise a clear, actionable error if no Gemini API key is configured."""
    if not GEMINI_API_KEY:
        raise EnvironmentError(
            "GEMINI_API_KEY is not set. Copy .env.example to .env and add "
            "your Google Gemini API key, e.g.:\n\n"
            "    cp .env.example .env\n"
            "    # then edit .env and set GEMINI_API_KEY=your_key_here"
        )
