"""Data models for the generic retrieval engine.

These are deliberately source-agnostic: a ``Document``/``Chunk`` carries a
small set of typed, search-facing fields plus a free-form ``metadata`` dict
for whatever the source loader wants to attach (citekey, title, year, image
paths for literature; anything else for future loaders).
"""

from __future__ import annotations

import functools
import re

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


FilterOp = Literal["eq", "in", "contains", "gte", "lte", "glob"]


def _lookup(metadata: dict[str, Any], dotted_field: str) -> Any:
    """Follow a dotted field name into nested metadata, e.g. "bib.year".

    Corpus frontmatter namespaces its collection-specific fields into blocks,
    so a filter's field name is a path rather than a single key. A plain name
    with no dots still resolves as a single top-level lookup.
    """
    current: Any = metadata
    for segment in dotted_field.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(segment)
        if current is None:
            return None
    return current


class Filter(BaseModel):
    """A generic predicate over a chunk's ``metadata``.

    ``field`` may be a dotted path into nested metadata (``bib.year``,
    ``docs.project``) as well as a plain top-level key.

    ``contains`` does a case-insensitive substring match; ``gte``/``lte``
    compare numerically when both sides parse as numbers, else lexically.
    ``glob`` matches a shell-style pattern against the whole value, with
    ``**`` spanning path separators - the form used for group paths.
    """

    field: str
    op: FilterOp
    value: Any

    def predicate(self) -> ChunkPredicate:
        field, op, value = self.field, self.op, self.value

        def check(chunk: Chunk) -> bool:
            actual = _lookup(chunk.metadata, field)
            if actual is None:
                return False
            if op == "eq":
                return actual == value
            if op == "in":
                return actual in value
            if op == "contains":
                return str(value).lower() in str(actual).lower()
            if op == "glob":
                return _glob_match(str(actual), str(value))
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


def _glob_match(actual: str, pattern: str) -> bool:
    """Shell-style match of a group path, selecting descendants too.

    Two rules, both chosen to match what the pattern looks like it should do:

    - `*` and `?` stop at a separator, `**` spans them, so `quartical/*` is
      one level and `**/gains` is any depth. Plain `fnmatch` cannot express
      this - its `*` crosses `/` - so patterns are translated to a regex.
    - A pattern that matches a group also selects everything filed under it:
      `--group calibration` reaches `calibration/gains/...`. Selecting a group
      and not its contents would be useless for filtering, so `*` and `**`
      end up equivalent here, and that is deliberate.
    """
    regex = _globstar_regex(pattern.rstrip("/"))
    if regex.fullmatch(actual):
        return True
    segments = actual.split("/")
    return any(
        regex.fullmatch("/".join(segments[:depth]))
        for depth in range(1, len(segments))
    )


@functools.lru_cache(maxsize=256)
def _globstar_regex(pattern: str) -> re.Pattern[str]:
    """`pattern` as a regex where `**` crosses separators and `*` does not.

    `**/` collapses to an optional run of leading segments, so `**/gains`
    matches a top-level `gains` as well as a nested one - the .gitignore
    reading, and the one anybody typing it expects.
    """
    parts: list[str] = []
    index = 0
    while index < len(pattern):
        if pattern.startswith("**/", index):
            parts.append("(?:.*/)?")
            index += 3
        elif pattern.startswith("**", index):
            parts.append(".*")
            index += 2
        elif pattern[index] == "*":
            parts.append("[^/]*")
            index += 1
        elif pattern[index] == "?":
            parts.append("[^/]")
            index += 1
        else:
            parts.append(re.escape(pattern[index]))
            index += 1
    return re.compile("".join(parts))


def _coerce_pair(actual: Any, value: Any) -> tuple[Any, Any]:
    """Return (actual, value) as floats if both parse numerically, else strings."""
    try:
        return float(actual), float(value)
    except (TypeError, ValueError):
        return str(actual), str(value)
