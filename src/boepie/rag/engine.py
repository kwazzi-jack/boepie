"""Hybrid (dense + BM25) retrieval engine with an optional embedding backend.

Our own chunker (``chunking.chunk_document``) produces the retrieval units.
The dense leg, when an embedding backend is configured, is a plain embedding
store: every chunk is embedded once at build time into an ``embeddings.npy``
matrix, and a query is answered by embedding the question and taking a
brute-force cosine over that matrix (sub-millisecond at this corpus scale, no
vector database needed). A BM25 sidecar built from the same chunk list
supplies the lexical leg, and the two are fused with reciprocal rank fusion
(``rag.search``).

A collection can also be built and queried with no embedding backend at all
(``embedding=None``): a lexical-only index, BM25 alone, no Ollama/OpenAI call
and no ``embeddings.npy`` on disk. This is what backs ``search_context`` over
the small curated bundle, where BM25 is sufficient and a live backend would
just be dead weight.

Two lifecycles:

* ``build()`` - dev-only. With an embedding binding it needs that backend
  reachable; with ``embedding=None`` it needs nothing but the corpus. Produces
  a self-contained ``index_root/<collection>/<index_id>/`` directory
  (``chunks.jsonl`` + ``bm25/`` + ``manifest.json``, plus ``embeddings.npy``
  when embedded) that can be tarred up and shipped as a release asset. No LLM,
  so it's cheap enough to run offline on a laptop.
* ``load_for_query()`` / ``query()`` (or the ``search()`` convenience wrapper) -
  the end-user path. For an embedded index, loads the matrix and embeds each
  question with the same backend the index was built with (enforced via the
  manifest). For a lexical-only index, no embedding binding is resolved or
  verified, and ``mode="hybrid"``/``"dense"`` are rejected in favour of
  ``mode="bm25"``.
"""

from __future__ import annotations

import asyncio
import json
import re
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np

from boepie.config import DEFAULT_TOP_K, FUSION_CANDIDATES, INDEX_DIR
from boepie.rag.bm25 import Bm25Index
from boepie.rag.chunking import chunk_document
from boepie.rag.embedding import ModelBinding, default_embedding_binding, embed_texts
from boepie.rag.loaders import (
    CorpusRevision,
    Loader,
    Revisioned,
    SourceDescribing,
    body_digest,
    loader_for,
)
from boepie.rag.models import Chunk, Filter, SearchResult, combine_filters
from boepie.rag.search import _rank_of, _reciprocal_rank_fusion

_MANIFEST_FILE = "manifest.json"
_CHUNKS_FILE = "chunks.jsonl"
_BM25_DIR = "bm25"
_EMBEDDINGS_FILE = "embeddings.npy"
_LATEST_FILE = "latest.json"

Mode = Literal["hybrid", "dense", "bm25"]


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def index_id_for(embedding: ModelBinding) -> str:
    """Derive a stable index directory name from the embedding binding.

    Only the embedding model affects what dense retrieval returns, so it alone
    determines the on-disk index id.
    """
    return f"{embedding.kind}-{_slugify(embedding.model)}"


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """L2-normalize each row so a dot product against a normalized query vector
    equals cosine similarity. Zero rows are left as-is (norm clamped to 1)."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


# ---------------------------------------------------------------------------
# Build (dev-only)
# ---------------------------------------------------------------------------


_LEXICAL_ONLY_INDEX_ID = "bm25"


@dataclass
class BuildManifest:
    name: str
    count: int
    index_id: str
    embedding_kind: str | None
    embedding_model: str | None
    embedding_dim: int | None
    lexical_only: bool
    built_at: str
    # Provenance for what went in: corpus-level context from the loader plus
    # the document ids actually indexed. Defaults to None so indices built
    # before provenance was recorded still load.
    sources: dict[str, Any] | None = None
    # Which corpus, in which state, this index was built over. The one field
    # here that prevents wrong answers rather than describing them - see
    # `_stale_reason`. None when the loader could not be fingerprinted, which
    # is read as "unverifiable", never as "fresh".
    built_from: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _collect_sources(loader: Loader) -> dict[str, Any]:
    """Corpus-level provenance to record in the manifest for this build.

    The surrounding context only - which upstream site and version a docs page
    came from, which bibliography a citekey came out of. *Which* documents
    were indexed belongs to `built_from`, which records them with the digests
    that make the record checkable; keeping a second copy of the id list here
    would be one more thing that can disagree with the index beside it.

    A loader that cannot describe itself - a test double, or a corpus with
    nothing external to point at - contributes nothing rather than failing.
    """
    return (
        dict(loader.describe_sources()) if isinstance(loader, SourceDescribing) else {}
    )


def _record_revision(
    loader: Loader, indexed: dict[str, str]
) -> dict[str, Any] | None:
    """The `built_from` block for this build, if the loader can supply one.

    `indexed` is what the build loop actually read, not a fresh walk: the
    record then describes the index sitting next to it rather than the corpus
    as it stood a moment later.
    """
    if not isinstance(loader, Revisioned):
        return None
    source_dir = loader.corpus_path()
    if not source_dir.is_dir():
        return None
    return asdict(CorpusRevision(path=str(source_dir.resolve()), documents=indexed))


class StaleIndexError(ValueError):
    """The corpus moved on after the index was built, so its hits are wrong.

    A ValueError subclass for the same reason `EmptyCollectionError` is: the
    CLI and the MCP read/search paths already turn a ValueError into a clean
    message rather than a traceback, and this one carries the command that
    fixes it.
    """


type IndexState = Literal["in step", "stale", "corpus absent", "unrecorded"]


@dataclass(frozen=True)
class IndexFreshness:
    """Whether an index still matches the corpus it was built over.

    Four states, and only one of them is a fault:

    ``in step``       every document the index holds is still on disk, unchanged.
    ``stale``         at least one is changed or gone, so hits are wrong.
    ``corpus absent`` no corpus here to compare against - the ordinary shape of
                      a machine that indexed a corpus it no longer keeps.
    ``unrecorded``    the index predates this check, or was built by a loader
                      that cannot be walked again. Unverifiable, never "fresh".

    Documents *added* since the build are counted separately and never make
    the state `stale`. They leave the index incomplete rather than wrong,
    which is the normal state between a `corpus add` and the `index build`
    that follows it - so serving decides on `stale` alone, while a caller
    asking "is this index complete" (`boepie setup`, `index status`) reads
    `added` as well.
    """

    state: IndexState
    corpus_path: str | None = None
    document_count: int = 0
    changed: int = 0
    gone: int = 0
    added: int = 0


def index_freshness(
    built_from: dict[str, Any] | None, collection: str
) -> IndexFreshness:
    """Compare an index's recorded corpus against what is on disk now.

    Takes the manifest's `built_from` block rather than the whole manifest:
    that is the only field this needs, and `index status` reads manifests
    straight off disk, where a truncated or foreign one must produce a status
    line rather than a crash.
    """
    if not built_from:
        return IndexFreshness(state="unrecorded")
    recorded = CorpusRevision(**built_from)
    loader = loader_for(collection, Path(recorded.path))
    if loader is None:
        # A collection boepie has no loader for cannot be verified, and
        # failing every such index would be worse than not checking it.
        return IndexFreshness(state="unrecorded", corpus_path=recorded.path)

    current = loader.corpus_revision() if isinstance(loader, Revisioned) else None
    if current is None:
        return IndexFreshness(
            state="corpus absent",
            corpus_path=recorded.path,
            document_count=len(recorded.documents),
        )

    gone = set(recorded.documents) - set(current.documents)
    changed = {
        document_id
        for document_id, digest in recorded.documents.items()
        if document_id in current.documents
        and current.documents[document_id] != digest
    }
    return IndexFreshness(
        state="stale" if (gone or changed) else "in step",
        corpus_path=recorded.path,
        document_count=len(recorded.documents),
        changed=len(changed),
        gone=len(gone),
        added=len(set(current.documents) - set(recorded.documents)),
    )


def _stale_reason(manifest: BuildManifest, collection: str) -> str | None:
    """Why `manifest`'s index must not be served, or None if it may be."""
    freshness = index_freshness(manifest.built_from, collection)
    if freshness.state != "stale":
        return None
    counts = [
        f"{freshness.changed} changed" if freshness.changed else "",
        f"{freshness.gone} gone" if freshness.gone else "",
    ]
    return (
        f"the '{collection}' index is stale: of the "
        f"{freshness.document_count} document(s) it was built over, "
        f"{' and '.join(part for part in counts if part)} on disk. "
        f"{_rebuild_hint(collection)}"
    )


class EmptyCollectionError(ValueError):
    """`build` was handed a loader that yielded no indexable chunks.

    A ValueError subclass so existing `except ValueError` handlers (the CLI's
    `_run`) keep reporting it as a clean message, while a caller building
    several collections at once can catch just this one and skip on.
    """

    def __init__(self, collection: str) -> None:
        self.collection = collection
        super().__init__(
            f"no indexable documents found for '{collection}' - the corpus "
            f"directory is empty, or its documents predate the current "
            f"frontmatter schema and carry no 'id'."
        )


async def build(
    loader: Loader,
    *,
    index_root: Path | str | None = None,
    embedding: ModelBinding | None = None,
    index_id: str | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> BuildManifest:
    """Chunk every document from ``loader`` and write a self-contained,
    shippable collection directory.

    With ``embedding`` given, chunks are also embedded into ``embeddings.npy``
    for the dense leg. With ``embedding=None`` the collection is BM25-only: no
    backend call, no embeddings file, and ``index_id`` defaults to "bm25"
    instead of one derived from a model binding.

    ``on_progress(done, total)``, if given, is called with ``(0, count)`` up
    front and after each embedding batch completes, so a CLI can render a bar.
    Unused when ``embedding is None``.
    """
    index_root = Path(index_root) if index_root is not None else INDEX_DIR
    if embedding is not None and embedding.dim is None:
        raise ValueError("embedding binding requires dim")

    chunks: list[Chunk] = []
    indexed: dict[str, str] = {}
    for document in loader.iter_documents():
        indexed[document.id] = body_digest(document.text)
        for chunk in chunk_document(document):
            chunk.collection = loader.name
            chunk.id = f"{document.id}::{len(chunks)}"
            chunk.chunk_index = len(chunks)
            chunks.append(chunk)

    # Guarded before anything is written or embedded: bm25s answers an empty
    # corpus with `max() iterable argument is empty` from deep inside its own
    # vocabulary build, which says nothing about the collection being empty.
    # An unreadable corpus reaches here the same way an absent one does - a
    # loader skips a document whose frontmatter it cannot parse - so the
    # message names the directory rather than asserting the files are missing.
    if not chunks:
        raise EmptyCollectionError(loader.name)

    resolved_id = index_id or (
        index_id_for(embedding) if embedding is not None else _LEXICAL_ONLY_INDEX_ID
    )
    collection_dir = Path(index_root) / loader.name / resolved_id
    if collection_dir.exists():
        shutil.rmtree(collection_dir)
    collection_dir.mkdir(parents=True)

    texts = [c.text for c in chunks]
    if embedding is not None:
        try:
            matrix = await embed_texts(embedding, texts, on_progress=on_progress)
        except Exception as error:  # surface a backend outage with useful context
            raise RuntimeError(
                f"Embedding backend failed ({embedding.kind}:{embedding.model} "
                f"@ {embedding.host}): {error}"
            ) from error
        np.save(collection_dir / _EMBEDDINGS_FILE, matrix)

    _write_chunks(collection_dir / _CHUNKS_FILE, chunks)
    bm25 = Bm25Index.build(texts)
    bm25.save(collection_dir / _BM25_DIR)

    manifest = BuildManifest(
        name=loader.name,
        count=len(chunks),
        index_id=resolved_id,
        embedding_kind=embedding.kind if embedding is not None else None,
        embedding_model=embedding.model if embedding is not None else None,
        embedding_dim=embedding.dim if embedding is not None else None,
        lexical_only=embedding is None,
        built_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        sources=_collect_sources(loader),
        built_from=_record_revision(loader, indexed),
    )
    (collection_dir / _MANIFEST_FILE).write_text(
        json.dumps(manifest.to_dict(), indent=2), encoding="utf-8"
    )

    latest_path = Path(index_root) / loader.name / _LATEST_FILE
    latest_path.write_text(json.dumps({"index_id": resolved_id}, indent=2), encoding="utf-8")

    # The handle cache holds chunks read at load time, so anything cached for
    # what we just overwrote is now stale. Evict both the explicitly-keyed
    # entry and the "latest" one (index_id omitted), which resolved here too.
    async with _CACHE_LOCK:
        _CACHE.pop((str(index_root), loader.name, resolved_id), None)
        _CACHE.pop((str(index_root), loader.name, ""), None)

    return manifest


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------


@dataclass
class QueryHandle:
    chunks: list[Chunk]
    matrix: np.ndarray | None  # (n, dim), rows L2-normalized; None for a lexical-only index
    bm25: Bm25Index
    manifest: BuildManifest
    embedding: ModelBinding | None  # None for a lexical-only index


def _build_hint(collection: str) -> str:
    """The command that actually produces `collection`'s index.

    Always a local build: boepie publishes no prebuilt index, so every index
    on a machine is built on it. The context collection is the odd one out
    only in which command does the building - its index is derived state
    inside a project's own `.boepie/` bundle, rebuilt by the bundle commands.
    """
    if collection == "context":
        return "Run `boepie context apply` (or `init` if there is no bundle yet)."
    return f"Run `boepie index build --collection {collection}` first."


def _rebuild_hint(collection: str) -> str:
    """The command that brings `collection`'s index back in step with its corpus.

    Separate from `_build_hint` because the two say different things for the
    context bundle, whose index is rebuilt by `context apply` rather than by
    `index build`, and because a rebuild is a different instruction from a
    first build even when the command happens to be the same.
    """
    if collection == "context":
        return "Run `boepie context apply` to rebuild it."
    return f"Run `boepie index build --collection {collection}` to rebuild it."


def _resolve_index_id(index_root: Path, collection: str, index_id: str | None) -> str:
    if index_id is not None:
        return index_id
    latest_path = index_root / collection / _LATEST_FILE
    if not latest_path.exists():
        raise FileNotFoundError(
            f"No index for collection '{collection}' at {index_root / collection}. "
            f"{_build_hint(collection)}"
        )
    return json.loads(latest_path.read_text(encoding="utf-8"))["index_id"]


async def load_for_query(
    index_root: Path | str | None = None,
    collection: str = "literature",
    *,
    embedding: ModelBinding | None = None,
    index_id: str | None = None,
) -> QueryHandle:
    """Load a built collection for querying.

    For a lexical-only index (``manifest.lexical_only``), ``embedding`` is
    ignored entirely: no default binding is resolved, nothing is verified
    against the manifest, and no matrix is loaded (``handle.matrix is None``).
    For an embedded index, ``embedding`` defaults to the active config and must
    match what the index was built with.
    """
    index_root = Path(index_root) if index_root is not None else INDEX_DIR
    resolved_id = _resolve_index_id(index_root, collection, index_id)
    collection_dir = index_root / collection / resolved_id
    manifest_path = collection_dir / _MANIFEST_FILE
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No index '{resolved_id}' for collection '{collection}' at "
            f"{collection_dir}. {_build_hint(collection)}"
        )
    manifest = BuildManifest(**json.loads(manifest_path.read_text(encoding="utf-8")))
    stale = _stale_reason(manifest, collection)
    if stale is not None:
        raise StaleIndexError(stale)
    chunks = _read_chunks(collection_dir / _CHUNKS_FILE)
    bm25 = Bm25Index.load(collection_dir / _BM25_DIR, len(chunks))

    if manifest.lexical_only:
        return QueryHandle(
            chunks=chunks, matrix=None, bm25=bm25, manifest=manifest, embedding=None
        )

    embedding = embedding or default_embedding_binding()
    if manifest.embedding_kind != embedding.kind or manifest.embedding_model != embedding.model:
        raise ValueError(
            f"Index '{resolved_id}' was built with embedding "
            f"'{manifest.embedding_kind}:{manifest.embedding_model}' but the "
            f"active config is '{embedding.kind}:{embedding.model}'. Rebuild "
            f"the index or set BOEPIE_EMBEDDING_BINDING/BOEPIE_EMBEDDING_MODEL "
            f"to match."
        )

    matrix = _normalize_rows(np.load(collection_dir / _EMBEDDINGS_FILE).astype(np.float32))

    return QueryHandle(
        chunks=chunks, matrix=matrix, bm25=bm25, manifest=manifest, embedding=embedding
    )


async def query(
    handle: QueryHandle,
    question: str,
    *,
    top_k: int = DEFAULT_TOP_K,
    filters: list[Filter] | None = None,
    mode: Mode = "hybrid",
) -> list[SearchResult]:
    """Run hybrid/dense/bm25 retrieval against an already-loaded handle.

    ``mode="hybrid"``/``"dense"`` against a lexical-only handle (``handle.matrix
    is None``) raises ``ValueError``: there is no dense leg to query, so use
    ``mode="bm25"`` instead.
    """
    if mode in ("hybrid", "dense") and handle.matrix is None:
        raise ValueError(
            f"Index '{handle.manifest.index_id}' is lexical-only (built with "
            f"embedding=None, no dense leg); mode='{mode}' needs an embedding "
            f"matrix. Use mode='bm25' instead."
        )

    predicate = combine_filters(filters)
    candidates = max(top_k, FUSION_CANDIDATES)

    dense_ranks: list[int] = []
    dense_scores: dict[int, float] = {}
    if mode in ("hybrid", "dense") and handle.matrix.shape[0]:
        query_vec = (await embed_texts(handle.embedding, [question]))[0]
        norm = np.linalg.norm(query_vec) or 1.0
        scores = handle.matrix @ (query_vec / norm)
        order = np.argsort(-scores)[:candidates]
        dense_ranks = [int(i) for i in order]
        if predicate is not None:
            dense_ranks = [idx for idx in dense_ranks if predicate(handle.chunks[idx])]
        dense_scores = {idx: float(scores[idx]) for idx in dense_ranks}

    bm25_ranks: list[int] = []
    bm25_scores: dict[int, float] = {}
    if mode in ("hybrid", "bm25"):
        bm25_hits = [
            (idx, raw_score)
            for idx, raw_score in handle.bm25.retrieve(question, candidates)
            if predicate is None or predicate(handle.chunks[idx])
        ]
        bm25_ranks = [idx for idx, _ in bm25_hits]
        bm25_scores = dict(bm25_hits)

    fused = _reciprocal_rank_fusion(dense_ranks, bm25_ranks)
    return [
        SearchResult(
            chunk=handle.chunks[idx],
            score=score,
            dense_rank=_rank_of(idx, dense_ranks),
            bm25_rank=_rank_of(idx, bm25_ranks),
            bm25_score=bm25_scores.get(idx),
            dense_score=dense_scores.get(idx),
        )
        for idx, score in fused[:top_k]
    ]


# ---------------------------------------------------------------------------
# Guided lookup: expand a search hit into wider context
# ---------------------------------------------------------------------------


@dataclass
class DocumentSpan:
    """A contiguous run of one document's chunks, stitched back into text.

    Produced by ``read_span`` to give an agent more context around a search
    hit than the single retrieved chunk. All coordinates are the same ones
    ``search`` surfaces, so a hit's ``document_id``/``chunk_index`` round-trips
    straight into a read request.
    """

    document_id: str
    source_path: str
    chunk_start: int  # chunk_index of the first chunk included
    chunk_end: int  # chunk_index of the last chunk included
    char_start: int
    char_end: int
    sections: list[str]
    text: str
    metadata: dict[str, Any]


def _stitch(chunks: list[Chunk]) -> str:
    """Concatenate document-ordered chunks into continuous text.

    Chunks overlap (sliding window) and their char offsets index the original
    document exactly, so overlaps are dropped by appending only the tail of
    each chunk past what's already covered. Whitespace-only regions the chunker
    skipped leave a small gap, bridged with a newline so paragraphs don't fuse.
    """
    parts: list[str] = []
    covered_end = -1
    for chunk in chunks:
        if chunk.char_start >= covered_end:
            if parts and chunk.char_start > covered_end:
                parts.append("\n")  # bridge a dropped whitespace gap
            parts.append(chunk.text)
        else:
            parts.append(chunk.text[covered_end - chunk.char_start :])
        covered_end = max(covered_end, chunk.char_end)
    return "".join(parts)


def build_alias_map(handle: QueryHandle) -> dict[str, str]:
    """Human-writable keys -> surrogate document id, for one loaded index.

    A corpus document is addressed by an opaque id, but the things people and
    curated notes actually write down are citekeys, arXiv ids, project/page
    pairs and titles - the context bundle's own paper stubs cite a citekey as
    the read handle. Resolving those keeps such references working without
    giving up the id's stability across renames.

    Built from chunk metadata already in memory, so it costs one pass and is
    only consulted after a literal id lookup has missed. Lowercased variants
    are included so casing need not be remembered exactly; a key that would
    map to two different documents is dropped rather than resolved
    arbitrarily.
    """
    candidates: dict[str, set[str]] = {}

    def record(key: object, document_id: str) -> None:
        if not isinstance(key, str) or not key:
            return
        for variant in (key, key.lower()):
            candidates.setdefault(variant, set()).add(document_id)

    for chunk in handle.chunks:
        metadata = chunk.metadata
        record(metadata.get("title"), chunk.document_id)
        bib = metadata.get("bib")
        if isinstance(bib, dict):
            record(bib.get("citekey"), chunk.document_id)
            record(bib.get("arxiv_id"), chunk.document_id)
            record(bib.get("doi"), chunk.document_id)
        docs = metadata.get("docs")
        if isinstance(docs, dict):
            project, page = docs.get("project"), docs.get("page")
            if project and page:
                record(f"{project}/{page}", chunk.document_id)

    return {key: next(iter(ids)) for key, ids in candidates.items() if len(ids) == 1}


def read_span(
    handle: QueryHandle,
    document_id: str,
    *,
    chunk_index: int | None = None,
    before: int = 1,
    after: int = 1,
) -> DocumentSpan:
    """Return a window of ``document_id``'s chunks, stitched into continuous text.

    ``chunk_index`` anchors the window (typically a chunk_index from a search
    hit); ``before``/``after`` extend it by that many neighbouring chunks within
    the same document. Pass ``chunk_index=None`` to read the whole document.

    ``document_id`` may also be a citekey, arXiv id, DOI, ``project/page``
    pair, or exact title; see ``build_alias_map``. Resolution happens only
    after a literal id lookup misses, so a real id is never shadowed.
    """
    doc_chunks = sorted(
        (c for c in handle.chunks if c.document_id == document_id),
        key=lambda c: (c.char_start, c.chunk_index),
    )
    if not doc_chunks:
        aliases = build_alias_map(handle)
        resolved = aliases.get(document_id) or aliases.get(document_id.lower())
        if resolved is not None:
            document_id = resolved
            doc_chunks = sorted(
                (c for c in handle.chunks if c.document_id == document_id),
                key=lambda c: (c.char_start, c.chunk_index),
            )
    if not doc_chunks:
        raise KeyError(f"No document '{document_id}' in this collection.")

    if chunk_index is None:
        window = doc_chunks
    else:
        positions = {c.chunk_index: i for i, c in enumerate(doc_chunks)}
        if chunk_index not in positions:
            raise KeyError(f"Chunk {chunk_index} is not part of document '{document_id}'.")
        anchor = positions[chunk_index]
        window = doc_chunks[max(0, anchor - before) : anchor + after + 1]

    return DocumentSpan(
        document_id=document_id,
        source_path=window[0].source_path,
        chunk_start=window[0].chunk_index,
        chunk_end=window[-1].chunk_index,
        char_start=window[0].char_start,
        char_end=window[-1].char_end,
        sections=list(dict.fromkeys(c.section for c in window if c.section)),
        text=_stitch(window),
        metadata=window[0].metadata,
    )


# ---------------------------------------------------------------------------
# Cached handles + the search() convenience wrapper
# ---------------------------------------------------------------------------

_CACHE: dict[tuple[str, str, str], QueryHandle] = {}
_CACHE_LOCK = asyncio.Lock()


async def get_or_load(
    index_root: Path | str | None = None,
    collection: str = "literature",
    *,
    embedding: ModelBinding | None = None,
    index_id: str | None = None,
) -> QueryHandle:
    index_root = Path(index_root) if index_root is not None else INDEX_DIR
    key = (str(index_root), collection, index_id or "")
    async with _CACHE_LOCK:
        if key not in _CACHE:
            _CACHE[key] = await load_for_query(
                index_root, collection, embedding=embedding, index_id=index_id
            )
        return _CACHE[key]


async def clear_cache() -> None:
    async with _CACHE_LOCK:
        _CACHE.clear()


async def search(
    question: str,
    collection: str = "literature",
    *,
    top_k: int = DEFAULT_TOP_K,
    filters: list[Filter] | None = None,
    mode: Mode = "hybrid",
    index_root: Path | str | None = None,
    embedding: ModelBinding | None = None,
    index_id: str | None = None,
) -> list[SearchResult]:
    """Load (or reuse a cached) collection and search it - the common entrypoint."""
    handle = await get_or_load(index_root, collection, embedding=embedding, index_id=index_id)
    return await query(handle, question, top_k=top_k, filters=filters, mode=mode)


async def read(
    document_id: str,
    *,
    chunk_index: int | None = None,
    before: int = 1,
    after: int = 1,
    collection: str = "literature",
    index_root: Path | str | None = None,
    embedding: ModelBinding | None = None,
    index_id: str | None = None,
) -> DocumentSpan:
    """Expand a single search hit into wider context (see ``read_span``).

    Reuses the same cached handle ``search`` loads, so a read after a search in
    the same session costs nothing extra. Batch by calling ``get_or_load`` once
    and looping ``read_span`` yourself.
    """
    handle = await get_or_load(index_root, collection, embedding=embedding, index_id=index_id)
    return read_span(handle, document_id, chunk_index=chunk_index, before=before, after=after)


def _write_chunks(path: Path, chunks: list[Chunk]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(chunk.model_dump_json() + "\n")


def _read_chunks(path: Path) -> list[Chunk]:
    with path.open(encoding="utf-8") as handle:
        return [Chunk.model_validate_json(line) for line in handle if line.strip()]
