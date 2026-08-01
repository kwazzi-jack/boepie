"""MCP tool for locating relevant pages in the curated ``.boepie/`` bundle.

One tool over the generic ``boepie.rag`` engine, pinned to the ``knowledge``
collection, which is always built lexical-only (BM25, no embedding backend -
see ``rag.engine``'s ``embedding=None`` mode). There is deliberately no
``read_knowledge``: hits are bundle file paths the agent opens with its own
file tools, so each hit's ``source:`` line is the handle.

The index searched is the one *inside* the bundle that governs the working
directory (``find_bundle`` -> ``index_root_for``), resolved per call rather
than at import: the server is long-lived, its cwd is wherever the client
launched it, and a project's bundle may be created after it starts. Set
``BOEPIE_BUNDLE_DIR`` when the two cannot be made to coincide.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from boepie.knowledge import find_bundle, index_root_for
from boepie.rag import search
from boepie.tools._retrieval import (
    SNIPPET_DESCRIPTION,
    TOP_K_DESCRIPTION,
    VIEWS,
    Snippet,
    format_hits,
    one_line,
)

_COLLECTION = "knowledge"
_VIEW = VIEWS[_COLLECTION]


class SearchKnowledgeInput(BaseModel):
    question: str = Field(
        description="Natural-language query, e.g. 'how does substitution work in a recipe?'."
    )
    top_k: int = Field(default=5, ge=1, le=20, description=TOP_K_DESCRIPTION)
    snippet: Snippet = Field(default="short", description=SNIPPET_DESCRIPTION)


async def search_knowledge(input: SearchKnowledgeInput) -> str:
    """Locate pages in the curated .boepie/ knowledge bundle for a question.

    Reach for this for pipeline strategy, recipe-language concepts, and
    "what does this cab do and when do I reach for it" notes. Never for
    parameter values, defaults or choices - those come only from the cab tools,
    which read the installed cult-cargo version and never go stale.

    Returns locations, not answers: each hit's source line is a bundle file
    path to open with your own file-reading tools. BM25 lexical search only, so
    try another phrasing if a query comes back thin.
    """
    bundle_dir = find_bundle()
    if bundle_dir is None:
        return (
            f"Error: no .boepie/ bundle found in {Path.cwd()} or any parent. "
            "Run 'boepie knowledge init'."
        )

    try:
        results = await search(
            input.question,
            collection=_COLLECTION,
            top_k=input.top_k,
            mode="bm25",
            index_root=index_root_for(bundle_dir),
            embedding=None,
        )
    except FileNotFoundError:
        return (
            f"Error: bundle {bundle_dir} has no search index. "
            "Run 'boepie knowledge apply'."
        )
    except ValueError as error:
        return f"Error: {one_line(error)}"

    return format_hits(
        input.question,
        _COLLECTION,
        results,
        snippet=input.snippet,
        title_of=_VIEW.title_of,
        source_root=_VIEW.source_root,
        keep_source_root=_VIEW.keep_source_root,
        read_handles=_VIEW.read_handles,
    )
