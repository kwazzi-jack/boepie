# src/boepie/tools/corpus.py
"""MCP tool for browsing a corpus collection without searching it.

`search_*` answers "what is relevant to this question"; this answers "what is
in here at all". The two are different needs, and until now only the first
was reachable from an agent: a corpus document is addressed by an opaque
surrogate `id`, so an agent that had not just run a search had no way to name
one.

Rendered as an indented tree, collapsed to group structure by default and
expanded one group at a time. The collapsing is the part that pays: a
150-page docs corpus is 52 tokens as group structure and 2042 with every
document named. Between the two renderings themselves there is almost
nothing in it - on a flat 18-document corpus the CLI's `corpus tree` costs
583 tokens against this tool's 517, and that gap is box-drawing glyphs
rather than information.

A collection nobody has grouped is the exception, and it is what literature
and notes normally look like: there the group breakdown would be a single
"(top level)" line restating the count above it, so the documents are named
straight away.

Indentation rather than box-drawing glyphs, because those are non-ASCII.
The CLI's `corpus tree` uses them only because rich emits them at runtime -
they are never literals in the source.
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
            "'groups' (default) is the group tree with a document count per "
            "group; 'documents' also names each document and its id."
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

    Returns an indented tree of groups with document counts, and (with
    detail='documents') each document's title and id. A collection with no
    groups names its documents directly, since there is no structure to
    summarise.
    """
    collection = cast(str, input.collection)
    collection_dir = _COLLECTION_DIRS[collection]
    documents = collection_index(
        collection_dir, collection=collection, key_fields=KEY_FIELDS[collection]
    )
    if not documents:
        return (
            f"The '{collection}' collection is empty. "
            f"Add to it with: boepie corpus add --collection {collection} <identifier>"
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

    # A collection nobody has grouped has nothing to say at detail='groups':
    # the single "(top level)" line would only restate the count above it,
    # which is what made a one-note collection report itself twice and name
    # nothing. Name the documents instead - on a flat collection that is the
    # only information there is, and it is what a caller needs to read one.
    is_flat = set(grouped) == {""}
    naming_documents = input.detail == "documents" or is_flat

    lines = [f"{collection}  {total} document(s)"]
    if is_flat:
        lines.extend(_document_lines(grouped[""], collection, depth=1, is_flat=True))
        return "\n".join(lines)

    for group in _tree_order(grouped):
        segments = group.split("/") if group else []
        # Top-level groups sit one level in from the collection line. The
        # unnamed top-level group has no segments of its own but still needs
        # that indent, or it would collide with the collection line above.
        depth = max(len(segments), 1)
        # Only the segment below the parent: the enclosing directories are
        # already on the lines above, and repeating the full path on every
        # line is what a tree exists to avoid.
        label = segments[-1] if segments else "(top level)"
        lines.append(f"{_INDENT * depth}{label}  {_subtree_total(grouped, group)}")
        if naming_documents and grouped.get(group):
            lines.extend(
                _document_lines(grouped[group], collection, depth + 1, is_flat=False)
            )

    return "\n".join(lines)


def _tree_order(grouped: dict[str, list[IndexedDocument]]) -> list[str]:
    """Every node of the group tree, parents before their children.

    Intermediate directories are included even when no document sits
    directly in them. Without that, a corpus filed under `quartical/gains`
    and `quartical/solver` but with nothing loose in `quartical/` renders
    those two children hanging off the collection itself, indented as though
    they belonged to whatever group happened to sort above them - a tree
    that lies about the structure is worse than the flat paths it replaced.
    """
    nodes: set[str] = set()
    for group in grouped:
        if not group:
            nodes.add("")
            continue
        segments = group.split("/")
        for depth in range(1, len(segments) + 1):
            nodes.add("/".join(segments[:depth]))
    # Sorting on the split path, not the string, keeps a child directly under
    # its parent: "a/b" sorts before "a-c" as a string but not as segments.
    return sorted(nodes, key=lambda group: group.split("/") if group else [])


def _subtree_total(grouped: dict[str, list[IndexedDocument]], group: str) -> int:
    """How many documents sit at or below a node.

    A subtree total rather than a direct count, so an intermediate directory
    reports something useful instead of a bare zero.
    """
    prefix = f"{group}/" if group else ""
    return sum(
        len(members)
        for name, members in grouped.items()
        if name == group or (prefix and name.startswith(prefix))
    )


# Two spaces per level. Deliberately not box-drawing glyphs: those are
# non-ASCII (the CLI's `corpus tree` gets away with them because rich emits
# them at runtime rather than them being literals in the source), and
# measured against indentation they buy nothing an agent can use.
_INDENT = "  "


def _document_lines(
    documents: list[IndexedDocument], collection: str, depth: int, is_flat: bool
) -> list[str]:
    """One line per document, title first, capped with a hint at how to go on."""
    ordered = sorted(
        documents, key=lambda d: str(d.frontmatter.get("title", "")).lower()
    )
    lines = [
        f"{_INDENT * depth}{document.frontmatter.get('title') or document.id}"
        f"  document_id={document.id}"
        for document in ordered[:_MAX_DOCUMENTS_PER_GROUP]
    ]
    if len(ordered) > _MAX_DOCUMENTS_PER_GROUP:
        remaining = len(ordered) - _MAX_DOCUMENTS_PER_GROUP
        # There is no group to narrow to on a flat collection, so pointing at
        # one would be useless advice.
        narrowing = (
            f"search_{collection} to find one" if is_flat else "narrow with group="
        )
        lines.append(f"{_INDENT * depth}... {remaining} more ({narrowing})")
    return lines
