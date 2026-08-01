"""MCP tools for searching and reading the radio-astronomy paper corpus.

Two tools over the generic ``boepie.rag`` engine, pinned to the ``literature``
collection: ``search_literature`` locates passages and hands back read
handles, ``read_literature`` expands those handles into wider context. The
shared F2/F4 rendering lives in ``boepie.tools._retrieval``.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from boepie.rag.engine import Mode
from boepie.rag.models import Filter
from boepie.tools._retrieval import (
    MODE_DESCRIPTION,
    SNIPPET_DESCRIPTION,
    TOP_K_DESCRIPTION,
    VIEWS,
    ReadRequest,
    Snippet,
    format_hits,
    read_spans,
    search_with_lexical_fallback,
    with_note,
)

_COLLECTION = "literature"
_VIEW = VIEWS[_COLLECTION]


class SearchLiteratureInput(BaseModel):
    question: str = Field(
        description="Natural-language query, e.g. 'how does wsclean handle multiscale cleaning?'."
    )
    top_k: int = Field(default=5, ge=1, le=20, description=TOP_K_DESCRIPTION)
    mode: Mode = Field(default="hybrid", description=MODE_DESCRIPTION)
    snippet: Snippet = Field(default="short", description=SNIPPET_DESCRIPTION)
    year_min: int | None = Field(
        default=None, description="Only consider papers published in this year or later."
    )
    year_max: int | None = Field(
        default=None, description="Only consider papers published in this year or earlier."
    )


async def search_literature(input: SearchLiteratureInput) -> str:
    """Search the radio-astronomy paper corpus for relevant passages.

    Reach for this to ground algorithm and 'why' answers - interferometry,
    calibration, imaging, the methods behind stimela's cabs - in the papers
    rather than in general knowledge.

    Returns ranked hits: paper title, section, score, a copy-pasteable
    read_literature handle, the source path, and a bounded snippet. If the
    embedding backend is down, hybrid degrades to lexical-only; an explicit
    dense request errors instead.
    """
    filters: list[Filter] = []
    if input.year_min is not None:
        filters.append(Filter(field="year", op="gte", value=input.year_min))
    if input.year_max is not None:
        filters.append(Filter(field="year", op="lte", value=input.year_max))

    outcome = await search_with_lexical_fallback(
        input.question,
        collection=_COLLECTION,
        top_k=input.top_k,
        mode=input.mode,
        filters=filters or None,
        missing_index_fix=_VIEW.missing_index_fix,
    )
    if outcome.error:
        return outcome.error

    return with_note(
        format_hits(
            input.question,
            _COLLECTION,
            outcome.results,
            snippet=input.snippet,
            title_of=_VIEW.title_of,
            source_root=_VIEW.source_root,
        ),
        outcome.note,
    )


class ReadLiteratureInput(BaseModel):
    requests: list[ReadRequest] = Field(
        description="One or more read requests. Batch several to expand multiple hits at once.",
        min_length=1,
    )


async def read_literature(input: ReadLiteratureInput) -> str:
    """Read the text surrounding one or more search_literature hits.

    The follow-up to search_literature: pass a hit's document_id and
    chunk_index to pull the neighbouring chunks (widen with before/after), or
    omit chunk_index to read the whole paper. Batch several requests in one
    call.

    Returns, per request, a provenance header (document_id, chunk and character
    range, source, sections) followed by continuous prose.
    """
    return await read_spans(
        input.requests,
        collection=_COLLECTION,
        source_root=_VIEW.source_root,
        missing_index_fix=_VIEW.missing_index_fix,
        search_tool=_VIEW.search_tool,
    )
