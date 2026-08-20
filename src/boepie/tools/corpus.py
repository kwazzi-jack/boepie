# src/boepie/tools/corpus.py
"""MCP tool for browsing a corpus collection without searching it.

`search_*` answers "what is relevant to this question"; this answers "what is
in here at all". The two are different needs, and until now only the first
was reachable from an agent: a corpus document is addressed by an opaque
surrogate `id`, so an agent that had not just run a search had no way to name
one.

Deliberately shallow and grouped by default. A corpus of a few hundred
documents rendered flat is several thousand tokens, which is a real cost
against the budget boepie is measured on - so this reports group structure
first and expands one group at a time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, cast

from pydantic import BaseModel, Field

from boepie.config import DOCS_DIR, LITERATURE_DIR, NOTES_DIR
from boepie.corpus.layout import IndexedDocument, collection_index
from boepie.corpus.schema import KEY_FIELDS

# How many documents to name inside one group before summarising the rest.
# Chosen so a listing of a typical corpus stays comparable in size to one
# `search_*` response rather than dwarfing it.
_MAX_DOCUMENTS_PER_GROUP = 25

_COLLECTION_DIRS = {
    "literature": LITERATURE_DIR,
    "docs": DOCS_DIR,
    "notes": NOTES_DIR,
}

Collection = Literal["literature", "docs", "notes"]


class ListCorpusInput(BaseModel):
    collection: Collection = Field(
        description="Which corpus to list: 'literature', 'docs', or 'notes'."
    )
    group: str | None = Field(
        default=None,
        description=(
            "Restrict to one group, e.g. 'stimela' or 'calibration/gains'. "
            "Omit for the whole collection."
        ),
    )
    detail: Literal["groups", "documents"] = Field(
        default="groups",
        description=(
            "'groups' (default) names each group and how many documents it "
            "holds; 'documents' also lists each document's title and id."
        ),
    )


def _relative_group(document: IndexedDocument, collection_dir: Path) -> str:
    anchor = document.wrapper_dir or document.md_path
    relative = anchor.parent.relative_to(collection_dir)
    return "" if relative == Path(".") else relative.as_posix()


async def list_corpus(input: ListCorpusInput) -> str:
    """List what a corpus collection contains, by group.

    Use this to find a document when you do not have a search hit to work
    from - to see which documentation projects exist before filtering
    search_docs by one, or to check whether a paper is in the corpus at all.
    Pass detail='documents' to get the document_id needed by read_literature,
    read_docs, or read_notes.

    Returns group names with document counts, and (with detail='documents')
    each document's title and id.
    """
    collection = cast(str, input.collection)
    collection_dir = _COLLECTION_DIRS[collection]
    documents = collection_index(
        collection_dir, collection=collection, key_fields=KEY_FIELDS[collection]
    )
    if not documents:
        return (
            f"The '{collection}' collection is empty. "
            f"Add to it with: boepie corpus add {collection} <identifier>"
        )

    grouped: dict[str, list[IndexedDocument]] = {}
    for document in documents:
        group = _relative_group(document, collection_dir)
        if input.group is not None and not (
            group == input.group or group.startswith(f"{input.group}/")
        ):
            continue
        grouped.setdefault(group, []).append(document)

    if not grouped:
        return f"No group '{input.group}' in '{collection}'."

    total = sum(len(members) for members in grouped.values())
    lines = [f"{total} document(s) in {collection}"]

    for group in sorted(grouped):
        members = sorted(
            grouped[group], key=lambda d: str(d.frontmatter.get("title", "")).lower()
        )
        label = group or "(top level)"
        lines.append("")
        lines.append(f"{label}  {len(members)} document(s)")
        if input.detail != "documents":
            continue
        for document in members[:_MAX_DOCUMENTS_PER_GROUP]:
            title = document.frontmatter.get("title") or document.id
            lines.append(f"    {title}  document_id={document.id}")
        if len(members) > _MAX_DOCUMENTS_PER_GROUP:
            remaining = len(members) - _MAX_DOCUMENTS_PER_GROUP
            lines.append(f"    ... {remaining} more (narrow with group=)")

    return "\n".join(lines)
