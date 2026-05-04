"""Central configuration for boepie.

Paths, embedding model settings, and ChromaDB collection config.
All values can be overridden via environment variables prefixed with
BOEPIE_ (e.g. BOEPIE_EMBEDDING_PROVIDER=ollama).
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Project root (two levels up from this file: src/boepie/config.py)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Where raw source documents live for RAG indexing.
SOURCES_DIR: Path = Path(
    os.environ.get("BOEPIE_SOURCES_DIR", str(_PROJECT_ROOT / "sources"))
)

# Where the ChromaDB persistent store is written.
CHROMA_DIR: Path = Path(
    os.environ.get("BOEPIE_CHROMA_DIR", str(_PROJECT_ROOT / "chroma_data"))
)

# ---------------------------------------------------------------------------
# Embedding model
# ---------------------------------------------------------------------------

# "openai" or "ollama"
EMBEDDING_PROVIDER: str = os.environ.get("BOEPIE_EMBEDDING_PROVIDER", "openai")

# OpenAI settings (default - easiest for collaborators)
OPENAI_EMBEDDING_MODEL: str = os.environ.get(
    "BOEPIE_OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
)
# API key read at runtime from standard OPENAI_API_KEY env var.

# Ollama settings (local/offline fallback)
OLLAMA_BASE_URL: str = os.environ.get("BOEPIE_OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_EMBEDDING_MODEL: str = os.environ.get(
    "BOEPIE_OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"
)

# ---------------------------------------------------------------------------
# ChromaDB
# ---------------------------------------------------------------------------

COLLECTION_NAME: str = "stimela_docs"

# ---------------------------------------------------------------------------
# Chunking defaults
# ---------------------------------------------------------------------------

PROSE_CHUNK_SIZE: int = 1500       # characters
PROSE_CHUNK_OVERLAP: int = 200     # characters

STRUCTURED_CHUNK_SIZE: int = 1000
STRUCTURED_CHUNK_OVERLAP: int = 100

# ---------------------------------------------------------------------------
# RAG query defaults
# ---------------------------------------------------------------------------

DEFAULT_TOP_K: int = 5
