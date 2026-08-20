"""Shared on-disk layout and reconciliation for the machine-global corpus
collections (literature, docs, notes): directory-as-group, full-title
filenames, a surrogate `id` in frontmatter, and `managed_by: boepie | user`
provenance (Phase 1's field, extended here to corpus content). See
`boepie.corpus.layout` for the recursive group-walking rule,
`boepie.corpus.document` for reading/writing one leaf document, and
`boepie.corpus.reconcile` for the literature/docs manifest-diff sync.

The `.boepie/` context bundle is explicitly excluded from this scheme -
`boepie.context` stays path-addressed by design.
"""

from __future__ import annotations

from boepie.corpus.document import CorpusDocument, read_document, write_leaf_document
from boepie.corpus.ids import generate_id, unique_id
from boepie.corpus.layout import (
    WRAPPED_DOCUMENT_FILENAME,
    CorpusFrontmatter,
    DocumentLocation,
    IndexedDocument,
    classify_child,
    collection_index,
    full_title_filename,
    iter_documents,
    natural_key_of,
    title_needs_dot_stripped,
    unique_document_name,
)
from boepie.corpus.reconcile import (
    DocsPageSyncResult,
    DocsSyncResult,
    LiteratureSyncResult,
    normalize_force_path,
    resolve_force_paths,
    sync_docs,
    sync_docs_project,
    sync_literature,
)

__all__ = [
    "CorpusDocument",
    "CorpusFrontmatter",
    "DocsPageSyncResult",
    "DocsSyncResult",
    "DocumentLocation",
    "IndexedDocument",
    "LiteratureSyncResult",
    "WRAPPED_DOCUMENT_FILENAME",
    "classify_child",
    "collection_index",
    "full_title_filename",
    "generate_id",
    "iter_documents",
    "natural_key_of",
    "normalize_force_path",
    "read_document",
    "resolve_force_paths",
    "sync_docs",
    "sync_docs_project",
    "sync_literature",
    "title_needs_dot_stripped",
    "unique_document_name",
    "unique_id",
    "write_leaf_document",
]
