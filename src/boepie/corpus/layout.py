# src/boepie/corpus/layout.py
"""The on-disk shape shared by every machine-global corpus collection
(literature/docs/notes): directory-as-group, full-title filenames, and a
recursive file-vs-directory rule that tells a document from a user-created
group by filesystem shape alone - no metadata field, no reserved bucket
name. See the corpus-unification design's "settled design" section for the
full rationale; this module is the one place that rule is implemented.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from boepie.context.frontmatter import read_frontmatter

# A corpus document's frontmatter is heterogeneous across collections
# (citekey/doi are str|None, tags are list[str], ids are str) in a way
# context.frontmatter.Frontmatter's narrower `str | list[str]` value type
# does not cover - kept as its own alias rather than widening the bundle's.
type CorpusFrontmatter = dict[str, Any]

# Filesystem entries that are corpus bookkeeping, not documents or groups -
# skipped outright by iter_documents before classify_child ever sees them.
_SKIPPED_NAME_PREFIXES = (".",)

# Characters illegal (or awkward) in a filename on at least one common
# filesystem, stripped from a title before it becomes one - not lowercased
# or hyphenated, unlike the old slug/citekey convention, since nothing
# parses this filename as a key any more (see `id`-based addressing).
_ILLEGAL_FILENAME_CHARS = re.compile(r'[/\\:*?"<>|\x00-\x1f]')

DocumentShape = Literal["leaf-bare", "leaf-wrapped", "group"]

# A leaf-wrapped document's markdown always lives at this fixed filename
# inside its wrapper directory, e.g. `Foo/content.md`, never `Foo/Foo.md`.
# Matching the wrapper's own name would make classify_child ambiguous: a
# document titled the same as an existing group (e.g. a note titled "My
# Note" landing in a group also named "My Note") would turn that whole
# group into a single leaf, hiding every sibling already in it. A fixed
# name decouples classification from the wrapper's (title-derived, and so
# collision-prone) name entirely.
WRAPPED_DOCUMENT_FILENAME = "content.md"


def classify_child(path: Path) -> DocumentShape:
    """Classify one child of a corpus directory by filesystem shape alone.

    - A `.md` file is a bare leaf document ("leaf-bare").
    - A directory containing `content.md` (`WRAPPED_DOCUMENT_FILENAME`) is a
      leaf document with assets ("leaf-wrapped") - a caller must not recurse
      into it further.
    - Any other directory is a user-created group ("group") - recurse into
      it.

    Callers are expected to pre-filter dotfiles/dot-directories and
    non-markdown files (corpus bookkeeping like `user-papers.json`) before
    calling this - see `iter_documents`.
    """
    if path.is_file():
        if path.suffix == ".md":
            return "leaf-bare"
        raise ValueError(f"not a document or group: {path}")
    if path.is_dir():
        wrapped_md_path = path / WRAPPED_DOCUMENT_FILENAME
        if wrapped_md_path.is_file():
            return "leaf-wrapped"
        return "group"
    raise ValueError(f"neither a file nor a directory: {path}")


@dataclass(frozen=True)
class DocumentLocation:
    """One document found by `iter_documents`."""

    md_path: Path
    # The asset-wrapper directory, when this is a leaf-wrapped document;
    # None for a bare-file leaf.
    wrapper_dir: Path | None


def _is_bookkeeping(path: Path) -> bool:
    if path.name.startswith(_SKIPPED_NAME_PREFIXES):
        return True
    return path.is_file() and path.suffix != ".md"


def _walk(directory: Path) -> Iterator[DocumentLocation]:
    for child in sorted(directory.iterdir()):
        if _is_bookkeeping(child):
            continue
        shape = classify_child(child)
        if shape == "leaf-bare":
            yield DocumentLocation(md_path=child, wrapper_dir=None)
        elif shape == "leaf-wrapped":
            yield DocumentLocation(
                md_path=child / WRAPPED_DOCUMENT_FILENAME, wrapper_dir=child
            )
        else:
            yield from _walk(child)


def iter_documents(
    collection_root: Path, *, collection: str
) -> Iterator[DocumentLocation]:
    """Walk `collection_root` per the recursive group-walking rule, yielding
    one `DocumentLocation` per document at any nesting depth.

    `collection` names the caller's intent ("literature"/"docs"/"notes") for
    error messages and future per-collection exceptions; every collection
    shares this one walking rule today, so it does not otherwise change the
    walk. Yields nothing for a collection root that does not exist yet (a
    fresh machine with nothing fetched/added).
    """
    if not collection_root.is_dir():
        return
    yield from _walk(collection_root)


def _clean_title(title: str) -> str:
    cleaned = _ILLEGAL_FILENAME_CHARS.sub("", title).strip()
    return re.sub(r"\s+", " ", cleaned)


def full_title_filename(title: str) -> str:
    """The on-disk `.md` filename for a document titled `title`: the title
    verbatim, with filesystem-illegal characters stripped and internal
    whitespace collapsed - not lowercased or hyphenated, since the surrogate
    `id` (not this filename) is what every read handle addresses.

    A leading `.` is also stripped: `_is_bookkeeping` treats any dot-prefixed
    name as corpus bookkeeping (like `user-papers.json`) rather than a
    document, so an un-stripped dotfile-derived title (e.g. a note added
    from `.bashrc`, whose fallback title is its own filename) would write
    successfully but then be permanently invisible to every future walk.
    """
    return f"{_clean_title(title).lstrip('.').strip() or 'untitled'}.md"


def title_needs_dot_stripped(title: str) -> bool:
    """True when `full_title_filename(title)` had to strip a leading dot to
    avoid `title` being mistaken for corpus bookkeeping. Callers that want to
    warn about this (see `boepie.corpus.add`) check it before writing."""
    return _clean_title(title).startswith(".")


def unique_document_name(base_name: str, existing_names: set[str]) -> str:
    """`base_name`, or the first `<stem> (n)<suffix>` (n=2, 3, ...) not
    already taken, e.g. "Notes on Substitution (2).md". Uniqueness is
    collection-wide (see the corpus design), so `existing_names` should be
    every document's filename in the collection, not just one group's.

    `WRAPPED_DOCUMENT_FILENAME` ("content.md") is always treated as already
    taken, regardless of `existing_names` - a *bare* document titled
    "content" claiming that exact name would make `classify_child` mistake
    its own parent directory for a wrapped document the next time something
    is added alongside it, hiding every sibling already there.
    """
    if base_name not in existing_names and base_name != WRAPPED_DOCUMENT_FILENAME:
        return base_name

    dot_index = base_name.rfind(".")
    stem, suffix = (
        (base_name[:dot_index], base_name[dot_index:])
        if dot_index > 0
        else (base_name, "")
    )

    counter = 2
    while True:
        candidate = f"{stem} ({counter}){suffix}"
        if candidate not in existing_names:
            return candidate
        counter += 1


_MISSING = object()


def lookup_path(mapping: CorpusFrontmatter, dotted_path: str) -> Any:
    """Follow a dotted path into nested frontmatter, e.g. "bib.citekey".

    Returns `_MISSING` (not None) when any segment is absent, so a field
    explicitly written as null stays distinguishable from one that was never
    written at all. A path that runs into a non-mapping partway down is a
    miss, not an error.
    """
    current: Any = mapping
    for segment in dotted_path.split("."):
        if not isinstance(current, dict) or segment not in current:
            return _MISSING
        current = current[segment]
    return current


def natural_key_of(
    frontmatter: CorpusFrontmatter, *, key_fields: tuple[str, ...]
) -> str:
    """The natural key a reconciler diffs a manifest against, e.g. a
    literature citekey ("bib.citekey"), or a docs project/page pair joined as
    "project/page". Field names are dotted paths into the collection's
    namespaced frontmatter block (see `boepie.corpus.schema.KEY_FIELDS`).

    Raises KeyError naming the first missing field, so a document with
    incomplete frontmatter fails loudly rather than silently sorting under a
    degenerate key. An empty `key_fields` yields an empty string: `notes` has
    no manifest to reconcile against and so needs no natural key.
    """
    values: list[str] = []
    for field in key_fields:
        value = lookup_path(frontmatter, field)
        if value is _MISSING:
            raise KeyError(f"frontmatter missing key field '{field}'")
        values.append(str(value))
    return "/".join(values)


@dataclass(frozen=True)
class IndexedDocument:
    """One on-disk document's location plus the frontmatter fields a
    reconciler needs to diff it against a manifest, without re-walking the
    tree or re-parsing frontmatter itself."""

    id: str
    natural_key: str
    md_path: Path
    wrapper_dir: Path | None
    frontmatter: CorpusFrontmatter

    @property
    def reserved_filename(self) -> str:
        """The top-level name this document occupies in its parent
        directory, comparable against a candidate `full_title_filename`
        result for uniqueness checks. `md_path.name` is always the fixed
        `WRAPPED_DOCUMENT_FILENAME` for a wrapped document (not its
        title-derived name), so the wrapper directory's own name is what a
        new document's candidate filename would actually collide with."""
        if self.wrapper_dir is not None:
            return f"{self.wrapper_dir.name}.md"
        return self.md_path.name


def collection_index(
    collection_root: Path, *, collection: str, key_fields: tuple[str, ...]
) -> list[IndexedDocument]:
    """Every already-migrated document under `collection_root`, keyed and
    id'd. A document whose frontmatter carries no `id` (not yet migrated to
    the corpus layout - see `scripts/migrate_corpus_layout.py`) is skipped
    rather than raising, since this is a normal, expected state for a corpus
    mid-migration; a document whose frontmatter carries an `id` but is
    missing one of `key_fields` is a real corruption and raises via
    `natural_key_of`.
    """
    indexed: list[IndexedDocument] = []
    for location in iter_documents(collection_root, collection=collection):
        frontmatter, _ = read_frontmatter(location.md_path.read_text(encoding="utf-8"))
        document_id = frontmatter.get("id")
        if not document_id:
            continue
        indexed.append(
            IndexedDocument(
                id=str(document_id),
                natural_key=natural_key_of(frontmatter, key_fields=key_fields),
                md_path=location.md_path,
                wrapper_dir=location.wrapper_dir,
                frontmatter=frontmatter,
            )
        )
    return indexed
