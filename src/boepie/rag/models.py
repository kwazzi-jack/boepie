"""Data models for the generic retrieval engine.

These are deliberately source-agnostic: a ``Document``/``Chunk`` carries a
small set of typed, search-facing fields plus a free-form ``metadata`` dict
for whatever the source loader wants to attach (citekey, title, year, image
paths for literature; anything else for future loaders).
"""

from __future__ import annotations

from typing import Any, Callable, Literal

from pydantic import BaseModel, Field

# A predicate over a chunk used for metadata filtering.
ChunkPredicate = Callable[["Chunk"], bool]


class Document(BaseModel):
    """A single source document yielded by a loader, before chunking."""

    id: str
    """Stable identifier, unique within a collection (e.g. the citekey)."""

    text: str
    source_path: str
    """Path to the underlying file, surfaced to the user as 'where found'."""

    base_path: str | None = None
    """Directory for resolving relative assets (e.g. a doc's images/ dir)."""

    metadata: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    """A retrievable span of text with provenance back to its document."""

    id: str
    """``f"{document_id}::{chunk_index}"``."""

    collection: str
    document_id: str
    chunk_index: int
    text: str

    # Provenance - "where it was found".
    source_path: str
    char_start: int
    char_end: int
    section: str | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchResult(BaseModel):
    """A ranked chunk returned from a search, with debugging provenance.

    ``score`` is the Reciprocal Rank Fusion value used for ordering; it is
    only comparable across hits of the *same* query, never across
    collections or tools. ``bm25_score``/``dense_score`` are the raw
    per-leg values (BM25 term score, raw cosine similarity) the RRF was
    computed from, for maintainer-facing detail (CLI/``--json``); either is
    ``None`` when that leg did not run for this query (e.g. ``dense_score``
    is always ``None`` in ``mode='bm25'``).
    """

    chunk: Chunk
    score: float
    dense_rank: int | None = None
    bm25_rank: int | None = None
    bm25_score: float | None = None
    dense_score: float | None = None


FilterOp = Literal["eq", "in", "contains", "gte", "lte"]


class Filter(BaseModel):
    """A generic predicate over a chunk's ``metadata``.

    ``contains`` does a case-insensitive substring match; ``gte``/``lte``
    compare numerically when both sides parse as numbers, else lexically.
    """

    field: str
    op: FilterOp
    value: Any

    def predicate(self) -> ChunkPredicate:
        field, op, value = self.field, self.op, self.value

        def check(chunk: Chunk) -> bool:
            actual = chunk.metadata.get(field)
            if actual is None:
                return False
            if op == "eq":
                return actual == value
            if op == "in":
                return actual in value
            if op == "contains":
                return str(value).lower() in str(actual).lower()
            # gte / lte: try numeric, fall back to string comparison.
            left, right = _coerce_pair(actual, value)
            return left >= right if op == "gte" else left <= right

        return check


def combine_filters(filters: list[Filter] | None) -> ChunkPredicate | None:
    """AND a list of filters into a single predicate (None means no filter)."""
    if not filters:
        return None
    predicates = [f.predicate() for f in filters]
    return lambda chunk: all(p(chunk) for p in predicates)


def _coerce_pair(actual: Any, value: Any) -> tuple[Any, Any]:
    """Return (actual, value) as floats if both parse numerically, else strings."""
    try:
        return float(actual), float(value)
    except (TypeError, ValueError):
        return str(actual), str(value)
