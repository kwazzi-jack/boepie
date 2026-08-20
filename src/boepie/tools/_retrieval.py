"""Shared rendering and error handling for the three retrieval tool pairs.

``search_literature`` / ``search_docs`` / ``search_context`` all emit output
family F2 (ranked hits) and ``read_literature`` / ``read_docs`` both emit F4
(a document span), per ``design/interface-spec.md``. Everything those tools have
in common lives here; what differs per collection (which metadata field
supplies the display title, which directory the source paths hang off, whether
there is a ``read_*`` tool to hand a handle to) is passed in.

Payload size is a correctness constraint here, not an aesthetic one: boepie is
evaluated against models with limited context windows, so every line these
functions emit is paid on every call of every session.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal, Sequence

from openai import APIConnectionError
from pydantic import BaseModel, Field

from boepie.config import DOCS_DIR, LITERATURE_DIR, NOTES_DIR
from boepie.rag import ModelBinding, get_or_load, read_span, search
from boepie.rag.engine import DocumentSpan, Mode
from boepie.rag.models import Chunk, Filter, SearchResult

# ---------------------------------------------------------------------------
# Shared parameter vocabulary (spec section 2)
# ---------------------------------------------------------------------------

Snippet = Literal["none", "short", "full"]

TOP_K_DESCRIPTION = "How many hits to return."
MODE_DESCRIPTION = (
    "'hybrid' (dense + lexical, default), 'dense' (embeddings only), "
    "or 'bm25' (lexical only)."
)
SNIPPET_DESCRIPTION = (
    "How much of each hit's text to include: 'none' (handles only), "
    "'short' (~200 chars, default), 'full' (the whole chunk)."
)

# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

SHORT_SNIPPET_CHARS = 200

# One blank line between spans in a batched read. Cheaper than a rule, and
# unambiguous because every span block opens with a "document_id=" line.
SPAN_SEPARATOR = "\n\n"

_INDENT = "    "
_WHITESPACE = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# Per-collection view: the one place that knows how each collection renders
# ---------------------------------------------------------------------------
#
# Both the MCP tools and the CLI's `search`/`read` commands render hits and
# spans through the same `format_hits`/`format_span`, so the per-collection
# knobs those take (which metadata field titles a hit, which directory the
# source paths hang off, whether there is a `read_*` tool, the command that
# rebuilds a missing index) live here as a single source of truth. Duplicating
# them per caller is exactly how the CLI and server drift apart.


def _title_or_id(chunk: Chunk) -> str:
    """Frontmatter/corpus title when the source recorded one, else the id."""
    return chunk.metadata.get("title") or chunk.document_id


def _docs_title(chunk: Chunk) -> str:
    """A docs hit's title, qualified by project.

    Page titles repeat heavily across projects ("Installation", "Overview"),
    so the project is what makes a hit identifiable at a glance - it is the
    part `search_docs`'s own `project` filter is phrased in.
    """
    docs_block = chunk.metadata.get("docs")
    project = docs_block.get("project") if isinstance(docs_block, dict) else None
    title = _title_or_id(chunk)
    return f"{project}: {title}" if project else title


@dataclass(frozen=True)
class CollectionView:
    """How one collection presents its hits and spans."""

    collection: str
    source_root: str
    keep_source_root: bool
    read_handles: bool
    missing_index_fix: str
    search_tool: str
    title_of: Callable[[Chunk], str] = _title_or_id


VIEWS: dict[str, CollectionView] = {
    "literature": CollectionView(
        collection="literature",
        source_root=LITERATURE_DIR.name,
        keep_source_root=False,
        read_handles=True,
        missing_index_fix="'boepie sync' or 'boepie index build --collection literature'",
        search_tool="search_literature",
    ),
    "docs": CollectionView(
        collection="docs",
        source_root=DOCS_DIR.name,
        keep_source_root=False,
        read_handles=True,
        missing_index_fix="'boepie sync' or 'boepie index build --collection docs'",
        search_tool="search_docs",
        # A docs page carries a real `title` in its frontmatter and a
        # surrogate id that means nothing to a reader, so it titles hits the
        # same way every other collection does.
        title_of=_docs_title,
    ),
    "context": CollectionView(
        collection="context",
        # Kept in the rendered path (unlike the corpus roots) because these are
        # the paths the agent feeds to its own file tools.
        source_root=".boepie",
        keep_source_root=True,
        read_handles=False,  # no read_context tool: the source line is the handle
        missing_index_fix="'boepie context apply'",
        search_tool="search_context",
    ),
    "notes": CollectionView(
        collection="notes",
        source_root=NOTES_DIR.name,
        keep_source_root=False,
        read_handles=True,
        missing_index_fix="'boepie index build --collection notes'",
        search_tool="search_notes",
    ),
}


def one_line(text: object) -> str:
    """Collapse ``text`` to a single line with no markdown emphasis.

    Errors bubbling up from the engine are multi-sentence and backticked; the
    spec allows neither in a payload, and we cannot edit their source here.
    """
    return _WHITESPACE.sub(" ", str(text).replace("`", "'")).strip()


def snippet_text(text: str, snippet: Snippet) -> str | None:
    """Render a chunk body under the ``snippet`` policy (spec section 4).

    ``none`` omits the body entirely, ``short`` collapses it to one line of at
    most ``SHORT_SNIPPET_CHARS`` characters, ``full`` returns it verbatim.
    """
    if snippet == "none":
        return None
    if snippet == "full":
        return text.strip() or None
    collapsed = _WHITESPACE.sub(" ", text).strip()
    if not collapsed:
        return None
    if len(collapsed) <= SHORT_SNIPPET_CHARS:
        return collapsed
    return collapsed[:SHORT_SNIPPET_CHARS].rstrip() + " ..."


def relative_source(source_path: str, root_name: str, *, keep_root: bool = False) -> str:
    """Cut an indexed chunk's absolute ``source_path`` down to a relative one.

    An index is built on one machine and queried on another, so the recorded
    absolute path is both long and meaningless to the caller - and it leaks the
    builder's home directory into every hit. Anchoring on the collection root's
    directory *name* survives that move, where ``Path.relative_to`` would not.
    ``keep_root`` retains the anchor itself, for collections whose source paths
    the caller is expected to open (the ``.boepie/`` bundle).

    Falls back to the path as given when the anchor is not present.
    """
    parts = Path(source_path).parts
    for position in range(len(parts) - 1, -1, -1):
        if parts[position] != root_name:
            continue
        tail = parts[position:] if keep_root else parts[position + 1 :]
        if tail:
            return Path(*tail).as_posix()
    return source_path


def _indent(body: str) -> str:
    return "\n".join(_INDENT + line for line in body.splitlines())


def _score_detail(result: SearchResult) -> str:
    """Raw per-leg scores for CLI/debug rendering, only the legs that ran.

    Compact and clearly labelled, e.g. ``bm25=3.42 cos=0.71 rrf=0.033``
    (context, BM25-only, shows just ``bm25=3.42``). Never shown to the
    agent - see ``format_hits``'s ``score_detail`` parameter.
    """
    parts = []
    if result.bm25_score is not None:
        parts.append(f"bm25={result.bm25_score:.2f}")
    if result.dense_score is not None:
        parts.append(f"cos={result.dense_score:.2f}")
    parts.append(f"rrf={result.score:.3f}")
    return " ".join(parts)


def format_hits(
    question: str,
    collection: str,
    results: list[SearchResult],
    *,
    snippet: Snippet,
    title_of: Callable[[Chunk], str],
    source_root: str,
    keep_source_root: bool = False,
    read_handles: bool = True,
    score_detail: bool = False,
) -> str:
    """Render ranked search results as output family F2.

    ``read_handles=False`` drops the ``read:`` line for collections with no
    ``read_*`` tool - there the ``source:`` line is the handle, and the caller
    opens the file itself.

    The agent-facing (MCP) rendering carries no numeric score: rank order,
    already spelled out by the ``[1] [2] [3]`` numbering, is the relevance
    signal a small model can act on - raw BM25/cosine/RRF values are on
    different, uninterpretable scales and invite miscomparison across tools.
    ``score_detail=True`` (CLI text output only) appends the raw per-leg
    breakdown to the hit line for a maintainer tuning retrieval.
    """
    lines = [f'{len(results)} hits for "{question}" in {collection}']
    for rank, result in enumerate(results, 1):
        lines.extend(
            _hit_block(
                rank,
                result,
                title=title_of(result.chunk),
                source_root=source_root,
                keep_source_root=keep_source_root,
                read_handles=read_handles,
                snippet=snippet,
                score_detail=score_detail,
            )
        )
    return "\n".join(lines)


def _hit_block(
    rank: int,
    result: SearchResult,
    *,
    title: str,
    source_root: str,
    keep_source_root: bool,
    read_handles: bool,
    snippet: Snippet,
    score_detail: bool,
    label: str | None = None,
) -> list[str]:
    """One hit's lines, shared by the single- and multi-collection renderings
    so the two cannot drift. ``label`` names the hit's collection, and is set
    only when the surrounding list mixes several.
    """
    chunk = result.chunk
    # A page whose only heading restates its title (common in the context
    # bundle) would otherwise print the same string twice per hit.
    section = f" #{chunk.section}" if chunk.section and chunk.section != title else ""
    suffix = f"  {_score_detail(result)}" if score_detail else ""
    prefix = f"{label}  " if label else ""
    lines = ["", f"[{rank}] {prefix}{title}{section}{suffix}"]
    if read_handles:
        lines.append(
            f"{_INDENT}read: document_id={chunk.document_id} "
            f"chunk_index={chunk.chunk_index}"
        )
    source = relative_source(chunk.source_path, source_root, keep_root=keep_source_root)
    lines.append(
        f"{_INDENT}source: {source} (chars {chunk.char_start}-{chunk.char_end})"
    )
    body = snippet_text(chunk.text, snippet)
    if body:
        lines.append(_indent(body))
    return lines


def format_merged_hits(
    question: str,
    ranked: Sequence[tuple[str, SearchResult]],
    *,
    collections: Sequence[str],
    snippet: Snippet,
    score_detail: bool = False,
) -> str:
    """Render hits drawn from several collections as one ranked list.

    ``ranked`` is already in final order, each entry paired with the
    collection it came from; every hit is labelled, because "which corpus is
    this from" is the first thing a mixed list has to answer. Ordering across
    collections is by RRF score, which is derived from ranks rather than from
    any one backend's raw scale - the only cross-collection comparison
    available that is not meaningless.
    """
    lines = [f'{len(ranked)} hits for "{question}" in {",".join(collections)}']
    for rank, (collection, result) in enumerate(ranked, 1):
        view = VIEWS[collection]
        lines.extend(
            _hit_block(
                rank,
                result,
                title=view.title_of(result.chunk),
                source_root=view.source_root,
                keep_source_root=view.keep_source_root,
                read_handles=view.read_handles,
                snippet=snippet,
                score_detail=score_detail,
                label=collection,
            )
        )
    return "\n".join(lines)


def format_span(
    span: DocumentSpan, *, source_root: str, keep_source_root: bool = False
) -> str:
    """Render one stitched document span as output family F4."""
    source = relative_source(span.source_path, source_root, keep_root=keep_source_root)
    lines = [
        f"document_id={span.document_id} chunks={span.chunk_start}-{span.chunk_end} "
        f"chars={span.char_start}-{span.char_end}",
        f"source: {source}",
    ]
    if span.sections:
        lines.append(f"sections: {', '.join(span.sections)}")
    lines.append("")
    lines.append(span.text.strip())
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Searching with a lexical fallback
# ---------------------------------------------------------------------------

# What the dense leg raises when the embedding backend is unreachable, per
# provider (verified by reading each SDK's transport-error handling):
# - ollama.AsyncClient.embed() catches httpx.ConnectError internally and
#   re-raises it as a bare builtin ConnectionError (ollama/_client.py).
# - openai.AsyncOpenAI wraps the same httpx.ConnectError as
#   openai.APIConnectionError (openai/_base_client.py).
# Caught narrowly so genuine bugs in the dense leg still surface as errors.
BACKEND_UNREACHABLE_ERRORS: tuple[type[Exception], ...] = (
    ConnectionError,
    APIConnectionError,
)

BACKEND_UNREACHABLE_NOTE = "Note: embedding backend unreachable, lexical-only results."


@dataclass(frozen=True)
class SearchOutcome:
    """Either ranked results (optionally with a degradation note) or an error."""

    results: list[SearchResult] = field(default_factory=list)
    note: str | None = None
    error: str | None = None


async def search_with_lexical_fallback(
    question: str,
    *,
    collection: str,
    top_k: int,
    mode: Mode,
    filters: list[Filter] | None,
    missing_index_fix: str,
    index_root: object | None = None,
    embedding: ModelBinding | None = None,
    index_id: str | None = None,
) -> SearchOutcome:
    """Search ``collection``, degrading to BM25 when the dense leg is down.

    An explicit ``mode="dense"`` request still errors: there is no lexical
    fallback that gives that caller what they asked for. ``missing_index_fix``
    is the command named in the error when no index exists for the collection.

    ``index_root``/``embedding``/``index_id`` default to ``None`` so the MCP
    tools get the machine's active index and config; the CLI passes overrides
    (a specific bundle root, a chosen embedding, a named index) through the same
    helper so both routes share one retrieval path.
    """
    try:
        results = await search(
            question, collection=collection, top_k=top_k, filters=filters, mode=mode,
            index_root=index_root, embedding=embedding, index_id=index_id,
        )
    except FileNotFoundError:
        return SearchOutcome(
            error=f"Error: no '{collection}' index found. Run {missing_index_fix}."
        )
    except ValueError as error:
        return SearchOutcome(error=f"Error: {one_line(error)}")
    except BACKEND_UNREACHABLE_ERRORS as error:
        if mode == "dense":
            return SearchOutcome(
                error=(
                    f"Error: embedding backend unreachable ({one_line(error)}). "
                    f"Retry with mode='bm25' or start the backend."
                )
            )
        try:
            results = await search(
                question, collection=collection, top_k=top_k, filters=filters, mode="bm25",
                index_root=index_root, embedding=embedding, index_id=index_id,
            )
        except (FileNotFoundError, ValueError) as fallback_error:
            return SearchOutcome(error=f"Error: {one_line(fallback_error)}")
        return SearchOutcome(results=results, note=BACKEND_UNREACHABLE_NOTE)

    return SearchOutcome(results=results)


def with_note(payload: str, note: str | None) -> str:
    """Prefix ``payload`` with a one-line degradation note, when there is one."""
    return f"{note}\n{payload}" if note else payload


# ---------------------------------------------------------------------------
# Reading spans
# ---------------------------------------------------------------------------


class ReadRequest(BaseModel):
    """One span to expand. Shared by ``read_literature`` and ``read_docs``."""

    document_id: str = Field(description="The document_id from a search hit's 'read:' line.")
    chunk_index: int | None = Field(
        default=None,
        description="The chunk_index from a search hit to centre on. Omit for the whole document.",
    )
    before: int = Field(
        default=1, ge=0, le=20, description="Neighbouring chunks to include before the anchor."
    )
    after: int = Field(
        default=1, ge=0, le=20, description="Neighbouring chunks to include after the anchor."
    )


async def read_spans(
    requests: Sequence[ReadRequest],
    *,
    collection: str,
    source_root: str,
    keep_source_root: bool = False,
    missing_index_fix: str,
    search_tool: str,
    index_root: object | None = None,
    embedding: ModelBinding | None = None,
    index_id: str | None = None,
) -> str:
    """Expand a batch of read requests into F4 span blocks.

    One failed request does not sink the batch: it contributes a one-line error
    in place of its block, naming the ``search_tool`` that produces valid handles.

    ``index_root``/``embedding``/``index_id`` default to ``None`` (the machine's
    active index and config, as the MCP tools use); the CLI passes overrides so
    both routes load the same handle the same way.
    """
    try:
        handle = await get_or_load(
            index_root=index_root, collection=collection,
            embedding=embedding, index_id=index_id,
        )
    except FileNotFoundError:
        return f"Error: no '{collection}' index found. Run {missing_index_fix}."
    except ValueError as error:
        return f"Error: {one_line(error)}"

    blocks: list[str] = []
    for request in requests:
        try:
            # `read_span` resolves a citekey/arXiv id/title alias itself when
            # the literal id misses, so both this and the CLI's `read` get
            # that behaviour from one place.
            span = read_span(
                handle,
                request.document_id,
                chunk_index=request.chunk_index,
                before=request.before,
                after=request.after,
            )
        except KeyError as error:
            blocks.append(
                f"Error: {one_line(error.args[0])} "
                f"Use a document_id and chunk_index from a {search_tool} hit."
            )
            continue
        blocks.append(
            format_span(span, source_root=source_root, keep_source_root=keep_source_root)
        )

    return SPAN_SEPARATOR.join(blocks)
