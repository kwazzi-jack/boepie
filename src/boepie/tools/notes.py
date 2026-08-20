"""MCP tools for searching and reading user-added notes.

Two tools over the generic ``boepie.rag`` engine, pinned to the ``notes``
collection: content added via ``boepie corpus add notes`` (local text files,
URLs - see ``boepie.corpus.add``), entirely user content, never
boepie-curated. The shared F2/F4 rendering lives in ``boepie.tools._retrieval``.
"""

from __future__ import annotations

from typing import cast

from pydantic import BaseModel, Field

from boepie.config import DEFAULT_MODE, DEFAULT_SNIPPET, DEFAULT_TOP_K
from boepie.rag.engine import Mode
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

_COLLECTION = "notes"
_VIEW = VIEWS[_COLLECTION]


class SearchNotesInput(BaseModel):
    question: str = Field(
        description="Natural-language query, e.g. 'what did I note about substitution syntax?'."
    )
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=20, description=TOP_K_DESCRIPTION)
    mode: Mode = Field(default=cast(Mode, DEFAULT_MODE), description=MODE_DESCRIPTION)
    snippet: Snippet = Field(default=cast(Snippet, DEFAULT_SNIPPET), description=SNIPPET_DESCRIPTION)


async def search_notes(input: SearchNotesInput) -> str:
    """Search notes added via `boepie corpus add notes` (local files, URLs).

    Reach for this for content the user pulled in themselves - not part of
    the curated knowledge bundle, the docs corpus, or the literature corpus.

    Returns ranked hits: note title, score, a copy-pasteable read_notes
    handle, the source path, and a bounded snippet. If the embedding backend
    is down, hybrid degrades to lexical-only; an explicit dense request errors
    instead.
    """
    outcome = await search_with_lexical_fallback(
        input.question,
        collection=_COLLECTION,
        top_k=input.top_k,
        mode=input.mode,
        filters=None,
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


class ReadNotesInput(BaseModel):
    requests: list[ReadRequest] = Field(
        description="One or more read requests. Batch several to expand multiple hits at once.",
        min_length=1,
    )


async def read_notes(input: ReadNotesInput) -> str:
    """Read the text surrounding one or more search_notes hits.

    The follow-up to search_notes: pass a hit's document_id and chunk_index to
    pull the neighbouring chunks (widen with before/after), or omit
    chunk_index to read the whole note. Batch several requests in one call.

    Returns, per request, a provenance header (document_id, chunk and
    character range, source) followed by continuous prose.
    """
    return await read_spans(
        input.requests,
        collection=_COLLECTION,
        source_root=_VIEW.source_root,
        missing_index_fix=_VIEW.missing_index_fix,
        search_tool=_VIEW.search_tool,
    )
