"""Embeddings-only hybrid (dense + BM25) retrieval engine.

Public API:

    from boepie.rag import build, search, LiteratureLoader, ModelBinding

    await build(LiteratureLoader(), embedding=...)      # dev-only, once
    results = await search("how does wsclean clean?")   # end-user path
"""

from __future__ import annotations

from boepie.rag.embedding import (
    ModelBinding,
    default_embedding_binding,
    embedding_options,
    resolve_host,
    validate_host,
)
from boepie.rag.engine import (
    BuildManifest,
    EmptyCollectionError,
    DocumentSpan,
    QueryHandle,
    build,
    clear_cache,
    get_or_load,
    index_id_for,
    load_for_query,
    query,
    read,
    read_span,
    search,
)
from boepie.rag.loaders import ContextLoader, DocsLoader, LiteratureLoader, Loader, NotesLoader
from boepie.rag.models import Chunk, Document, Filter, SearchResult

__all__ = [
    "build",
    "load_for_query",
    "get_or_load",
    "clear_cache",
    "query",
    "search",
    "read",
    "read_span",
    "ModelBinding",
    "BuildManifest",
    "EmptyCollectionError",
    "DocumentSpan",
    "QueryHandle",
    "default_embedding_binding",
    "embedding_options",
    "resolve_host",
    "validate_host",
    "index_id_for",
    "Loader",
    "LiteratureLoader",
    "DocsLoader",
    "ContextLoader",
    "NotesLoader",
    "Document",
    "Chunk",
    "SearchResult",
    "Filter",
]
