"""Central configuration for boepie.

Paths, the embedding model binding, and retrieval defaults for the RAG engine.
All values can be overridden via environment variables prefixed with
BOEPIE_ (e.g. BOEPIE_EMBEDDING_MODEL=mxbai-embed-large).
"""

from __future__ import annotations

import os
from pathlib import Path

from platformdirs import user_cache_dir, user_data_dir

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Project root (two levels up from this file: src/boepie/config.py)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Where the literature corpus lives (one subdir per document, as produced by
# scripts/corpus_to_md.py: {citekey}/{citekey}.md + images/ + metadata.json).
# Dev/build-time only - never touched on the query-only end-user path.
LITERATURE_DIR: Path = Path(
    os.environ.get("BOEPIE_LITERATURE_DIR", str(_PROJECT_ROOT / "stimela-corpus"))
)

# Where the upstream docs corpus lives (one subdir per project, holding
# {project_name}/*.md pages plus {project_name}/metadata.json).
# Dev/build-time only - never touched on the query-only end-user path.
DOCS_DIR: Path = Path(
    os.environ.get("BOEPIE_DOCS_DIR", str(_PROJECT_ROOT / "docs-corpus"))
)

# Where built/fetched indices live, one subdirectory per named collection.
# Defaults to a platform-appropriate user data directory (not repo-relative)
# so an installed (not cloned) boepie has somewhere sensible to keep a
# fetched index across runs without redownloading it.
INDEX_DIR: Path = Path(
    os.environ.get("BOEPIE_INDEX_DIR", user_data_dir("boepie"))
)

# Explicit pointer at the `.boepie/` bundle directory to use, overriding the
# upward search from the cwd that `boepie.knowledge.find_bundle` otherwise
# does. The escape hatch for an MCP server whose client launched it from
# somewhere other than the project directory it is meant to serve.
#
# Unlike the paths above this is resolved per call, not at import: the server
# is long-lived and both the cwd and the bundle can change under it.
BUNDLE_DIR_ENV_VAR = "BOEPIE_BUNDLE_DIR"


def bundle_dir_override() -> Path | None:
    """The bundle directory `BOEPIE_BUNDLE_DIR` names, or None when unset."""
    raw_path = os.environ.get(BUNDLE_DIR_ENV_VAR)
    return Path(raw_path).expanduser() if raw_path else None


# Machine-global cache for curated knowledge-bundle content fetched via
# `boepie knowledge fetch` (the knowledge-content.tar.gz release asset
# extracts here). Distinct from INDEX_DIR: this holds source markdown that
# `knowledge init`/`apply` copy from, not a built search index.
CONTENT_DIR: Path = Path(
    os.environ.get("BOEPIE_CONTENT_DIR", str(Path(user_data_dir("boepie")) / "content"))
)

# Where the fastembed backend caches its downloaded ONNX model files.
# fastembed's own default is a tempdir (cleared on reboot on most systems,
# forcing a redownload); pointing it at a platform cache dir instead means
# the one-time download really only happens once.
EMBEDDING_CACHE_DIR: Path = Path(
    os.environ.get("BOEPIE_EMBEDDING_CACHE_DIR", str(Path(user_cache_dir("boepie")) / "embedding-models"))
)

# ---------------------------------------------------------------------------
# Embedding binding (build-time AND query-time)
# ---------------------------------------------------------------------------

# "fastembed", "ollama", or "openai". Default is fastembed: a small ONNX
# model run locally on CPU via the fastembed package, needing no running
# service, API key, or GPU - just a one-time model download cached locally.
# "ollama"/"openai" remain available for anyone already running local LLM
# infra or who wants a hosted API's quality.
EMBEDDING_BINDING: str = os.environ.get("BOEPIE_EMBEDDING_BINDING", "fastembed")
EMBEDDING_MODEL: str = os.environ.get("BOEPIE_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
EMBEDDING_HOST: str = os.environ.get("BOEPIE_EMBEDDING_HOST", "http://localhost:11434")

# Embedding dimension, recorded in the manifest and used to shape empty
# matrices. Defaults to BAAI/bge-small-en-v1.5's known dimension; override
# when overriding BOEPIE_EMBEDDING_MODEL to something else.
EMBEDDING_DIM: int = int(os.environ.get("BOEPIE_EMBEDDING_DIM", "384"))

# ---------------------------------------------------------------------------
# Chunking defaults (characters)
# ---------------------------------------------------------------------------

PROSE_CHUNK_SIZE: int = 1500
PROSE_CHUNK_OVERLAP: int = 200

# ---------------------------------------------------------------------------
# Retrieval defaults
# ---------------------------------------------------------------------------

DEFAULT_TOP_K: int = 5

# Number of candidates pulled from each retriever before fusion.
FUSION_CANDIDATES: int = 50

# Reciprocal Rank Fusion constant: score = sum 1 / (RRF_K + rank).
RRF_K: int = 60
