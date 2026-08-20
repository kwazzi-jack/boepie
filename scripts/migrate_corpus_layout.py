# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "boepie",
# ]
#
# [tool.uv]
# override-dependencies = [
#     "rich>=14.3.2", # boepie's own pyproject.toml carries this override too -
#                      # stimela pins rich<14, boepie pins rich==14.3.2.
# ]
#
# [tool.uv.sources]
# boepie = { path = "..", editable = true }
# ///

"""
migrate_corpus_layout.py
--------------------------
One-time, per-machine migration of an existing literature/docs/notes corpus
from the old `{name}/{name}.md` + `metadata.json` sidecar layout to the new
corpus layout `boepie.corpus` reads and writes: full-title `.md` filenames, a
surrogate `id` plus `managed_by: boepie | user` in YAML frontmatter instead of a
JSON sidecar (see `boepie.corpus.layout`/`boepie.corpus.document`).

IMPORTANT: run this only once Phase 4's loader update has landed.
`LiteratureLoader`/`DocsLoader`/`NotesLoader` do not understand the new
layout at all until then - a migrated document is invisible to today's
`corpus_dir.iterdir()`-based walk, not degraded, just silently absent from
`boepie index build` in the interim.

Idempotent: `boepie.corpus.document.read_document` succeeding on a `.md` file
(its frontmatter already carries an `id`) is the skip signal, so
interrupting and resuming this script is safe - a document is only deleted
from its old location once its new one has been written successfully.

Literature and notes share one old shape (`{name}/{name}.md` +
`metadata.json`, one document per directory) and migrate the same way.
Docs is structurally different: `{project}/{docname}.md` files directly
under a project directory (docname may be nested, e.g.
`changelogs/1.0.md`), with a single `metadata.json` describing the whole
project rather than one sidecar per page - each page becomes its own
document, flattened directly under `{project}/` using its derived title as
the filename (nested docname subdirectories are removed once emptied).

`scripts/corpus_to_md.py` (BYO-PDF) is out of scope: it keeps writing the
old format independently - see CLAUDE.md.

Usage:
    uv run scripts/migrate_corpus_layout.py literature ~/.local/share/boepie/literature-corpus
    uv run scripts/migrate_corpus_layout.py docs docs-corpus/
    uv run scripts/migrate_corpus_layout.py notes ~/.local/share/boepie/notes
    uv run scripts/migrate_corpus_layout.py literature ~/.local/share/boepie/literature-corpus --dry-run
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import click
from rich.console import Console

from boepie.corpus.document import read_document, write_leaf_document
from boepie.corpus.ids import unique_id
from boepie.corpus.layout import collection_index, full_title_filename, unique_document_name
from boepie.corpus.schema import KEY_FIELDS, Source

console = Console()

_TITLE_HEADING_RE = re.compile(r"^#\s+(.+)$", re.M)



def _title_from_markdown(markdown: str, fallback: str) -> str:
    match = _TITLE_HEADING_RE.search(markdown)
    return match.group(1).strip() if match else fallback


def _prune(fields: dict[str, Any]) -> dict[str, Any]:
    """Drop keys the old sidecar never carried, rather than writing empty
    strings that would validate but assert something false."""
    return {key: value for key, value in fields.items() if value not in (None, "")}


def _source_block(*, origin: str, via: str, source_format: str) -> dict[str, Any]:
    """The nested `source` block, built through the real schema model so a
    migrated document cannot describe a shape production would not write."""
    return Source(
        origin=origin, via=via, format=source_format,
    ).model_dump(mode="json", by_alias=True, exclude_none=True)


def _is_already_migrated(md_path: Path) -> bool:
    try:
        read_document(md_path)
    except ValueError:
        return False
    return True


def _seed_uniqueness_state(collection_dir: Path, *, collection: str) -> tuple[set[str], set[str]]:
    """Existing ids/filenames from documents already migrated (e.g. by an
    earlier, interrupted run of this script) - so a resumed run cannot
    generate an id or filename that collides with one it already wrote."""
    already_migrated = collection_index(collection_dir, collection=collection, key_fields=KEY_FIELDS[collection])
    existing_ids = {document.id for document in already_migrated}
    existing_filenames = {document.reserved_filename for document in already_migrated}
    return existing_ids, existing_filenames


def _migrate_document_per_directory(
    collection_dir: Path, *, collection: str, dry_run: bool,
) -> int:
    """Migrates literature's or notes' `{name}/{name}.md` + `metadata.json`
    (one document per directory) to the new layout."""
    existing_ids, existing_filenames = _seed_uniqueness_state(collection_dir, collection=collection)
    migrated = 0

    for doc_dir in sorted(path for path in collection_dir.iterdir() if path.is_dir()):
        old_md_path = doc_dir / f"{doc_dir.name}.md"
        if not old_md_path.is_file() or _is_already_migrated(old_md_path):
            continue

        metadata_path = doc_dir / "metadata.json"
        metadata: dict[str, Any] = (
            json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
        )
        markdown = old_md_path.read_text(encoding="utf-8")
        title = str(metadata.get("title") or _title_from_markdown(markdown, doc_dir.name))

        # The old sidecar spelled provenance two ways - `source`/`source_url`
        # on a paper, `source`/`source_path` on a note - which the nested
        # `source` block now unifies. Anything the sidecar never recorded is
        # written as its honest unknown rather than a plausible-looking guess.
        if collection == "literature":
            frontmatter_fields: dict[str, Any] = {
                "title": title,
                "managed_by": "boepie",
                "source": _source_block(
                    origin=metadata.get("source_url")
                    or (
                        f"arxiv:{metadata['arxiv_id']}"
                        if metadata.get("arxiv_id")
                        else str(old_md_path)
                    ),
                    via=metadata.get("source") or "verbatim",
                    source_format="html" if metadata.get("source_url") else "markdown",
                ),
                "bib": _prune(
                    {
                        "citekey": metadata.get("citekey", doc_dir.name),
                        "authors": metadata.get("author", ""),
                        "year": metadata.get("year", ""),
                        "doi": metadata.get("doi"),
                        "arxiv_id": metadata.get("arxiv_id"),
                    }
                ),
            }
        else:
            frontmatter_fields = {
                "title": title,
                "managed_by": "user",
                "source": _source_block(
                    origin=metadata.get("source_path") or str(old_md_path),
                    via="html" if metadata.get("source") == "url" else "verbatim",
                    source_format="html" if metadata.get("source") == "url" else "markdown",
                ),
            }

        asset_paths = [
            path for path in doc_dir.rglob("*")
            if path.is_file() and path not in (old_md_path, metadata_path)
        ]
        assets = {
            str(path.relative_to(doc_dir)): path.read_bytes() for path in asset_paths
        } or None

        filename = unique_document_name(full_title_filename(title), existing_filenames)
        document_id = unique_id(existing_ids)

        console.print(f"  [green]migrate[/green] {doc_dir.name}/ -> {filename}")
        if not dry_run:
            write_leaf_document(
                collection_dir / filename, document_id=document_id,
                frontmatter_fields=frontmatter_fields, body=markdown, assets=assets,
            )
            # Only remove the old directory once the new document is safely
            # written - an interrupted migration leaves it in place,
            # unmigrated, so a rerun retries it instead of losing it.
            shutil.rmtree(doc_dir)

        existing_ids.add(document_id)
        existing_filenames.add(filename)
        migrated += 1

    return migrated


@dataclass(frozen=True)
class _PendingDocsPage:
    old_path: Path
    docname: str
    title: str
    markdown: str


@dataclass(frozen=True)
class _PlannedDocsPage:
    page: _PendingDocsPage
    target_path: Path
    document_id: str
    frontmatter_fields: dict[str, Any]


def _migrate_docs(collection_dir: Path, *, dry_run: bool) -> int:
    """Migrates `{project}/{docname}.md` (+ one `metadata.json` per project)
    to the new layout: each page becomes its own document, flattened
    directly under `{project}/`."""
    existing_ids, existing_filenames = _seed_uniqueness_state(collection_dir, collection="docs")
    migrated = 0

    for project_dir in sorted(path for path in collection_dir.iterdir() if path.is_dir()):
        metadata_path = project_dir / "metadata.json"
        project_metadata: dict[str, Any] = (
            json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
        )

        old_page_paths = [
            path for path in sorted(project_dir.rglob("*.md"))
            if not _is_already_migrated(path)
        ]
        if not old_page_paths:
            continue

        # Read every unmigrated page into memory, and decide every target
        # path, before writing or deleting anything. Flattening moves pages
        # into the same directory their old nested paths already lived
        # under, so a page's derived filename can coincide with a
        # *different* page's still-unmigrated raw path (e.g. two sections
        # both have an untitled "index.md", or share a heading like
        # "Overview") in two ways: an early write could clobber a later
        # page's content before it was ever read, or an early page's freshly
        # written file could later get deleted by a different page's own
        # old-path cleanup, if that old path happens to equal the first
        # page's new target. Planning every target up front, before any
        # write or delete, rules out both.
        pending = [
            _PendingDocsPage(
                old_path=page_path,
                docname=(docname := page_path.relative_to(project_dir).with_suffix("").as_posix()),
                title=_title_from_markdown(markdown := page_path.read_text(encoding="utf-8"), docname),
                markdown=markdown,
            )
            for page_path in old_page_paths
        ]

        planned: list[_PlannedDocsPage] = []
        for page in pending:
            filename = unique_document_name(full_title_filename(page.title), existing_filenames)
            existing_filenames.add(filename)
            document_id = unique_id(existing_ids)
            existing_ids.add(document_id)
            planned.append(
                _PlannedDocsPage(
                    page=page,
                    target_path=project_dir / filename,
                    document_id=document_id,
                    frontmatter_fields={
                        "title": page.title,
                        "managed_by": "boepie",
                        "source": _source_block(
                            origin=project_metadata.get("base_url", ""),
                            via=project_metadata.get("discovery", "sphinx"),
                            source_format="html",
                        ),
                        "docs": _prune(
                            {
                                "project": project_dir.name,
                                "page": page.docname,
                                "base_url": project_metadata.get("base_url", ""),
                                "version": project_metadata.get("version"),
                                "crawl": {
                                    "discovery": project_metadata.get(
                                        "discovery", "sphinx"
                                    )
                                },
                            }
                        ),
                    },
                )
            )

        target_paths = {item.target_path for item in planned}
        for item in planned:
            console.print(
                f"  [green]migrate[/green] {project_dir.name}/{item.page.docname} -> {item.target_path.name}"
            )
            if not dry_run:
                write_leaf_document(
                    item.target_path, document_id=item.document_id,
                    frontmatter_fields=item.frontmatter_fields, body=item.page.markdown,
                )
            migrated += 1

        if not dry_run:
            # Deleted only after every page in this project has been
            # written: an old path that coincides with some *other* page's
            # new target (planned above) must never be unlinked, or it
            # would delete that page's just-written content.
            for item in planned:
                if item.page.old_path not in target_paths:
                    item.page.old_path.unlink()

            if metadata_path.is_file():
                metadata_path.unlink()
            # Clean up subdirectories left empty by nested docnames
            # (e.g. changelogs/1.0.md -> changelogs/ once emptied),
            # deepest first so a parent empties only after its children do.
            for subdir in sorted(
                (path for path in project_dir.rglob("*") if path.is_dir()),
                key=lambda path: len(path.parts), reverse=True,
            ):
                if not any(subdir.iterdir()):
                    subdir.rmdir()

    return migrated


@click.command()
@click.argument("collection", type=click.Choice(sorted(KEY_FIELDS)))
@click.argument("collection_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--dry-run", is_flag=True, help="Report what would migrate without writing or deleting anything.")
def migrate_corpus_layout(collection: str, collection_dir: Path, dry_run: bool) -> None:
    """Migrate COLLECTION_DIR (a literature/docs/notes corpus root) from the
    old `{name}/{name}.md` + `metadata.json` layout to boepie.corpus's new
    layout. Safe to interrupt and rerun - already-migrated documents are
    skipped."""
    console.print(f"[bold]{collection}[/bold]: scanning {collection_dir}...")

    if collection == "docs":
        migrated = _migrate_docs(collection_dir, dry_run=dry_run)
    else:
        migrated = _migrate_document_per_directory(collection_dir, collection=collection, dry_run=dry_run)

    verb = "would migrate" if dry_run else "migrated"
    console.print(f"[green]Done[/green]: {verb} {migrated} document(s) in {collection_dir}")
    if migrated and not dry_run:
        console.print("Next: uv run boepie index build --collection " + collection)


if __name__ == "__main__":
    migrate_corpus_layout()
