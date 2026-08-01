"""MCP tools for searching and reading vendored upstream documentation.

Two tools over the generic ``boepie.rag`` engine, pinned to the ``docs``
collection: ``search_docs`` locates passages and hands back read handles,
``read_docs`` expands those handles into wider context. A parallel pair to the
literature tools, sharing the F2/F4 rendering in ``boepie.tools._retrieval``.
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

_COLLECTION = "docs"
_VIEW = VIEWS[_COLLECTION]


class SearchDocsInput(BaseModel):
    question: str = Field(
        description="Natural-language query, e.g. 'how do I write a stimela recipe?'."
    )
    top_k: int = Field(default=5, ge=1, le=20, description=TOP_K_DESCRIPTION)
    mode: Mode = Field(default="hybrid", description=MODE_DESCRIPTION)
    snippet: Snippet = Field(default="short", description=SNIPPET_DESCRIPTION)
    project: str | None = Field(
        default=None,
        description="Restrict to one project's documentation, e.g. 'stimela', 'quartical'.",
    )


async def search_docs(input: SearchDocsInput) -> str:
    """Search vendored upstream documentation (stimela, quartical, wsclean, ...).

    Reach for this for usage, configuration and syntax questions about the tools
    a recipe drives - as opposed to parameter facts, which come from the cab
    tools, or algorithm rationale, which comes from search_literature.

    Returns ranked hits: project/page id, section, score, a copy-pasteable
    read_docs handle, the source path, and a bounded snippet. If the embedding
    backend is down, hybrid degrades to lexical-only; an explicit dense request
    errors instead.
    """
    filters: list[Filter] = []
    if input.project is not None:
        filters.append(Filter(field="project", op="eq", value=input.project))

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


class ReadDocsInput(BaseModel):
    requests: list[ReadRequest] = Field(
        description="One or more read requests. Batch several to expand multiple hits at once.",
        min_length=1,
    )


async def read_docs(input: ReadDocsInput) -> str:
    """Read the text surrounding one or more search_docs hits.

    The follow-up to search_docs: pass a hit's document_id and chunk_index to
    pull the neighbouring chunks (widen with before/after), or omit chunk_index
    to read the whole page. Batch several requests in one call.

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


__all__ = [
    "ReadDocsInput",
    "ReadRequest",
    "SearchDocsInput",
    "read_docs",
    "search_docs",
]
