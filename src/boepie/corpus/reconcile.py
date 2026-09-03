"""Manifest-diff reconciliation for the literature and docs collections:
fetch/convert whatever a manifest lists into the new corpus layout, skip
what is already present, re-fetch a `--force`d item in place, and delete an
on-disk `managed_by: boepie` document whose manifest entry is gone.

Structurally this is the corpus analogue of `context.bundle`'s
convergence (`_copy_managed_files`/`_delete_orphaned_managed_files`), but a
genuinely different algorithm - a directory-diff there (copy pre-rendered
files from a resolved source dir) versus a manifest-diff here (fetch+convert
whatever the manifest lists, nothing pre-rendered to diff against) - so it
is not built by sharing code with `context.bundle`, only by mirroring its
`--force`-target validation shape (`_normalize_force_path`/
`_resolve_force_paths`) under `normalize_force_path`/`resolve_force_paths`
below. Notes have no manifest and are always `managed_by: user`, so there is no
`sync_notes` - see `boepie.corpus.add`, which writes them directly.
"""

from __future__ import annotations

import shutil
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import httpx

from boepie.corpus.document import write_leaf_document
from boepie.corpus.ids import unique_id
from boepie.corpus.layout import (
    IndexedDocument,
    collection_index,
    full_title_filename,
    lookup_path,
    unique_document_name,
)
from boepie.corpus.schema import (
    KEY_FIELDS,
    Source,
    docs_blocks,
    literature_blocks,
)
from boepie.docs.fetch import (
    PageResult,
    fetch_version,
    iter_project_pages,
    probe_discovery_mode,
)
from boepie.docs.manifest import DocsProject
from boepie.literature.fetch import FetchResult, fetch_paper
from boepie.literature.manifest import ArxivPaper

# Identifies corpus-reconciliation traffic to arxiv.org/ar5iv/upstream docs
# sites, distinct from boepie.literature.fetch's and boepie.docs.fetch's own
# User-Agent strings (those modules' HTTP helpers are called directly here,
# but the httpx.Client itself is opened by this module).
_USER_AGENT = "boepie-corpus-reconcile (+https://github.com/kwazzi-jack/boepie)"

_LITERATURE_KEY_FIELDS = KEY_FIELDS["literature"]
_DOCS_KEY_FIELDS = KEY_FIELDS["docs"]


# ---------------------------------------------------------------------------
# --force target resolution
# ---------------------------------------------------------------------------


def normalize_force_path(raw_path: str | Path) -> Path:
    """Collection-root-relative `Path` for a `--force` target.

    Unlike `context.bundle._normalize_force_path` (which strips a leading
    `.boepie/` off a search hit's path), a corpus document has no single
    reserved root directory name to strip - a search_literature/search_docs
    hit's `source:` line is already collection-relative. Kept as its own
    function, mirroring that one's shape, so a future prefix convention has
    somewhere to land without changing every caller.
    """
    return Path(raw_path)


def resolve_force_paths(
    force_paths: Iterable[str | Path],
    documents: list[IndexedDocument],
    *,
    collection_dir: Path,
) -> frozenset[str]:
    """Validate every `--force` target before any network request is made
    (mirroring `context.bundle._resolve_force_paths`'s atomic-or-nothing
    contract) and translate each into the natural key `sync_literature`/
    `sync_docs_project` diff against, so a caller can force-refresh a
    manifest entry using the collection-relative filesystem path a search
    hit actually showed them.

    Takes an already-computed `collection_index(...)` result rather than
    walking the collection itself - every caller already needs that same
    index for its own existing/skip/refetch decisions, so computing it once
    and passing it in here avoids re-walking and re-parsing frontmatter for
    the whole collection a second time on every sync.
    """
    resolved_collection_dir = collection_dir.resolve()
    indexed_by_relative_path: dict[Path, IndexedDocument] = {
        document.md_path.resolve().relative_to(resolved_collection_dir): document
        for document in documents
    }

    natural_keys: set[str] = set()
    for raw_path in force_paths:
        relative_path = normalize_force_path(raw_path)
        resolved_target = (collection_dir / relative_path).resolve()
        if (
            resolved_target != resolved_collection_dir
            and not resolved_target.is_relative_to(resolved_collection_dir)
        ):
            raise ValueError(
                f"--force target '{relative_path}' escapes the collection directory"
            )

        document = indexed_by_relative_path.get(relative_path)
        if document is None:
            raise ValueError(
                f"no such corpus document to force-refresh: {relative_path}"
            )
        natural_keys.add(document.natural_key)
    return frozenset(natural_keys)


def _delete_document(document: IndexedDocument) -> None:
    if document.wrapper_dir is not None:
        shutil.rmtree(document.wrapper_dir)
    else:
        document.md_path.unlink()


# ---------------------------------------------------------------------------
# Literature
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LiteratureSyncResult:
    citekey: str
    action: Literal["added", "skipped", "refetched", "deleted", "unavailable"]
    id: str | None = None


def _literature_frontmatter_fields(
    paper: ArxivPaper, fetch_result: FetchResult
) -> dict[str, Any]:
    """The frontmatter one fetched paper carries, in the same shape
    `corpus add -l` writes (`boepie.corpus.add._write`) - only
    `managed_by` differs, which is the whole point of the distinction.

    No `sha256`: the arXiv route has no fixed source byte sequence to hash
    (arxiv.org and ar5iv re-render the same paper differently), so duplicate
    detection leans on `bib.arxiv_id` here exactly as `add`'s arXiv path does.
    """
    source = Source(
        origin=f"arxiv:{paper.arxiv_id}",
        # Which of the two renderers actually served this markdown.
        via="ar5iv" if fetch_result.source == "ar5iv" else "arxiv-html",
        format="html",
    )
    return {
        "title": paper.title,
        "managed_by": "boepie",
        "source": source.model_dump(mode="json", by_alias=True, exclude_none=True),
        **literature_blocks(
            citekey=paper.citekey,
            authors=paper.authors,
            year=paper.year,
            doi=paper.doi,
            arxiv_id=paper.arxiv_id,
        ),
    }


def sync_literature(
    collection_dir: Path,
    manifest: list[ArxivPaper],
    *,
    force_paths: Iterable[str | Path] = (),
    delay: float = 1.0,
    on_progress: (
        Callable[[ArxivPaper | None, LiteratureSyncResult], None] | None
    ) = None,
) -> list[LiteratureSyncResult]:
    """Converge `collection_dir` with `manifest`: add a paper missing on
    disk, skip one already present unless named in `force_paths`, re-fetch a
    forced one in place (same id, same path), and delete any on-disk
    `managed_by: boepie` paper whose citekey has left the manifest. A
    `managed_by: user` paper sharing a citekey is never touched at any step -
    not added over, not force-refetched, not deleted as an orphan.

    `on_progress` receives `None` for the paper argument on a
    manifest-departure deletion, since no `ArxivPaper` exists for it any
    more by definition.
    """
    existing = collection_index(
        collection_dir, collection="literature", key_fields=_LITERATURE_KEY_FIELDS
    )
    forced_citekeys = resolve_force_paths(force_paths, existing, collection_dir=collection_dir)

    existing_by_citekey = {document.natural_key: document for document in existing}
    existing_ids = {document.id for document in existing}
    existing_filenames = {document.reserved_filename for document in existing}
    manifest_citekeys = {paper.citekey for paper in manifest}

    results: list[LiteratureSyncResult] = []
    with httpx.Client(headers={"User-Agent": _USER_AGENT}) as client:
        for paper in manifest:
            existing_doc = existing_by_citekey.get(paper.citekey)
            if (
                existing_doc is not None
                and existing_doc.frontmatter.get("managed_by") == "user"
            ):
                continue

            force = paper.citekey in forced_citekeys
            if existing_doc is not None and not force:
                result = LiteratureSyncResult(
                    citekey=paper.citekey, action="skipped", id=existing_doc.id
                )
                results.append(result)
                if on_progress is not None:
                    on_progress(paper, result)
                continue

            fetch_result = fetch_paper(client, paper.citekey, paper.arxiv_id)
            if fetch_result.markdown is None:
                result = LiteratureSyncResult(
                    citekey=paper.citekey, action="unavailable"
                )
                results.append(result)
                if on_progress is not None:
                    on_progress(paper, result)
                time.sleep(delay)
                continue

            if existing_doc is not None:
                document_id, target_path = existing_doc.id, existing_doc.md_path
                action: Literal["added", "refetched"] = "refetched"
            else:
                filename = unique_document_name(
                    full_title_filename(paper.title), existing_filenames
                )
                existing_filenames.add(filename)
                document_id = unique_id(existing_ids)
                existing_ids.add(document_id)
                target_path, action = collection_dir / filename, "added"

            write_leaf_document(
                target_path,
                document_id=document_id,
                frontmatter_fields=_literature_frontmatter_fields(paper, fetch_result),
                body=fetch_result.markdown,
            )
            result = LiteratureSyncResult(
                citekey=paper.citekey, action=action, id=document_id
            )
            results.append(result)
            if on_progress is not None:
                on_progress(paper, result)
            time.sleep(delay)

    for document in existing:
        if document.natural_key in manifest_citekeys:
            continue
        if document.frontmatter.get("managed_by") != "boepie":
            continue
        _delete_document(document)
        result = LiteratureSyncResult(
            citekey=document.natural_key, action="deleted", id=document.id
        )
        results.append(result)
        if on_progress is not None:
            on_progress(None, result)

    return results


# ---------------------------------------------------------------------------
# Docs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DocsPageSyncResult:
    project: str
    page: str
    action: Literal["added", "skipped", "refetched", "deleted", "failed"]
    id: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class DocsSyncResult:
    project: str
    added: int
    skipped: int
    refetched: int
    deleted: int
    failures: list[PageResult]


def _title_from_markdown(markdown: str, fallback: str) -> str:
    """Shared with `corpus add` (see `boepie.corpus.intake`) so a fetched
    page and an added one derive their titles the same way - emphasis and
    link syntax stripped, so no markdown reaches the title field."""
    from boepie.corpus.intake import title_from_markdown

    return title_from_markdown(markdown, fallback)


def sync_docs_project(
    collection_dir: Path,
    project: DocsProject,
    *,
    force_paths: Iterable[str | Path] = (),
    delay: float = 0.2,
    timeout: int = 30,
    on_page: Callable[[DocsPageSyncResult], None] | None = None,
    existing_documents: list[IndexedDocument] | None = None,
    existing_ids: set[str] | None = None,
    existing_filenames: set[str] | None = None,
) -> DocsSyncResult:
    """Converge one docs project's pages with what its site currently
    serves: add a page missing on disk, skip one already present unless
    forced, re-fetch a forced one in place, and delete any on-disk
    `managed_by: boepie` page for this project no longer served - the per-page
    analogue of `sync_literature`'s per-citekey orphan handling.
    `managed_by: user` pages are never touched at any step.

    `existing_documents`/`existing_ids`/`existing_filenames`, when given,
    are an already-computed `collection_index(...)` result and its derived
    id/filename sets. `sync_docs` passes these (as the same mutable set
    objects across every project's call, so a page this function adds is
    visible to the next project's uniqueness check within the same sync)
    so a multi-project sync walks and parses the whole docs collection once
    rather than once per project. Computed fresh here when this function is
    called standalone for a single project.
    """
    if existing_documents is None:
        existing_documents = collection_index(
            collection_dir, collection="docs", key_fields=_DOCS_KEY_FIELDS
        )
    if existing_ids is None:
        existing_ids = {document.id for document in existing_documents}
    if existing_filenames is None:
        existing_filenames = {document.reserved_filename for document in existing_documents}

    forced_natural_keys = resolve_force_paths(
        force_paths, existing_documents, collection_dir=collection_dir
    )

    existing_by_key = {
        document.natural_key: document
        for document in existing_documents
        if lookup_path(document.frontmatter, "docs.project") == project.project
    }

    project_dir = collection_dir / project.project
    added = skipped = refetched = deleted = 0
    failures: list[PageResult] = []
    seen_natural_keys: set[str] = set()

    with httpx.Client(headers={"User-Agent": _USER_AGENT}) as client:
        # iter_project_pages re-probes discovery internally; done again here
        # (one cheap extra request in the sphinx case) so this function has
        # the mode/version to record in frontmatter without iter_project_pages
        # having to thread per-project metadata back out of a page generator.
        discovery = project.discovery or probe_discovery_mode(
            client, project.base_url, timeout
        )
        version = (
            fetch_version(client, project.base_url, timeout)
            if discovery == "sphinx"
            else None
        )

        for page in iter_project_pages(client, project, delay=delay, timeout=timeout):
            natural_key = f"{project.project}/{page.docname}"
            seen_natural_keys.add(natural_key)
            existing_doc = existing_by_key.get(natural_key)

            if page.markdown is None:
                failures.append(
                    PageResult(docname=page.docname, ok=False, error=page.error)
                )
                if on_page is not None:
                    on_page(
                        DocsPageSyncResult(
                            project=project.project,
                            page=page.docname,
                            action="failed",
                            error=page.error,
                        )
                    )
                continue

            if (
                existing_doc is not None
                and existing_doc.frontmatter.get("managed_by") == "user"
            ):
                continue

            force = natural_key in forced_natural_keys
            if existing_doc is not None and not force:
                skipped += 1
                if on_page is not None:
                    on_page(
                        DocsPageSyncResult(
                            project=project.project,
                            page=page.docname,
                            action="skipped",
                            id=existing_doc.id,
                        )
                    )
                continue

            title = _title_from_markdown(page.markdown, page.docname)
            source = Source(
                origin=page.page_url or project.base_url,
                via="sphinx" if discovery == "sphinx" else "crawl",
                format="html",
            )
            fields = {
                "title": title,
                "managed_by": "boepie",
                "source": source.model_dump(
                    mode="json", by_alias=True, exclude_none=True
                ),
                **docs_blocks(
                    project=project.project,
                    page=page.docname,
                    base_url=project.base_url,
                    version=version,
                    discovery=discovery,
                    exclude=project.exclude,
                    path_prefix=project.path_prefix,
                ),
            }

            if existing_doc is not None:
                document_id, target_path = existing_doc.id, existing_doc.md_path
                action: Literal["added", "refetched"] = "refetched"
                refetched += 1
            else:
                filename = unique_document_name(
                    full_title_filename(title), existing_filenames
                )
                existing_filenames.add(filename)
                document_id = unique_id(existing_ids)
                existing_ids.add(document_id)
                target_path, action = project_dir / filename, "added"
                added += 1

            write_leaf_document(
                target_path,
                document_id=document_id,
                frontmatter_fields=fields,
                body=page.markdown,
            )
            if on_page is not None:
                on_page(
                    DocsPageSyncResult(
                        project=project.project,
                        page=page.docname,
                        action=action,
                        id=document_id,
                    )
                )

    for natural_key, document in existing_by_key.items():
        if natural_key in seen_natural_keys:
            continue
        if document.frontmatter.get("managed_by") != "boepie":
            continue
        _delete_document(document)
        deleted += 1
        if on_page is not None:
            on_page(
                DocsPageSyncResult(
                    project=project.project,
                    page=str(document.frontmatter.get("page", "")),
                    action="deleted",
                    id=document.id,
                )
            )

    return DocsSyncResult(
        project=project.project,
        added=added,
        skipped=skipped,
        refetched=refetched,
        deleted=deleted,
        failures=failures,
    )


def _delete_orphaned_project_pages(project_dir: Path) -> int:
    """Deletes every `managed_by: boepie` page under `project_dir` - used when
    the whole project has left the manifest, per the confirmed design
    decision that a removed `DocsProject` has every page under it
    auto-deleted rather than left for manual cleanup."""
    deleted = 0
    for document in collection_index(
        project_dir, collection="docs", key_fields=_DOCS_KEY_FIELDS
    ):
        if document.frontmatter.get("managed_by") != "boepie":
            continue
        _delete_document(document)
        deleted += 1
    return deleted


def sync_docs(
    collection_dir: Path,
    manifest: list[DocsProject],
    *,
    force_paths: Iterable[str | Path] = (),
    delay: float = 0.2,
    timeout: int = 30,
    on_progress: Callable[[DocsProject | None, DocsSyncResult], None] | None = None,
) -> list[DocsSyncResult]:
    """Converges every project in `manifest` (via `sync_docs_project`), then
    deletes every `managed_by: boepie` page under a project directory that has
    left the manifest entirely.

    `on_progress` receives `None` for the project argument on a
    manifest-departure whole-project deletion, since no `DocsProject` exists
    for it any more by definition.

    Walks and parses the docs collection's frontmatter once for the whole
    batch, not once per project: `existing_ids`/`existing_filenames` are
    built here and passed into every `sync_docs_project` call as the same
    mutable sets, so a page one project adds is already accounted for in the
    next project's collection-wide uniqueness check, matching what a fresh
    per-project walk would have seen anyway.
    """
    results: list[DocsSyncResult] = []
    manifest_project_names = {project.project for project in manifest}

    existing_documents = collection_index(
        collection_dir, collection="docs", key_fields=_DOCS_KEY_FIELDS
    )
    existing_ids = {document.id for document in existing_documents}
    existing_filenames = {document.reserved_filename for document in existing_documents}

    for project in manifest:
        result = sync_docs_project(
            collection_dir,
            project,
            force_paths=force_paths,
            delay=delay,
            timeout=timeout,
            existing_documents=existing_documents,
            existing_ids=existing_ids,
            existing_filenames=existing_filenames,
        )
        results.append(result)
        if on_progress is not None:
            on_progress(project, result)

    if collection_dir.is_dir():
        for project_dir in sorted(
            path for path in collection_dir.iterdir() if path.is_dir()
        ):
            if project_dir.name in manifest_project_names:
                continue
            deleted = _delete_orphaned_project_pages(project_dir)
            if deleted == 0:
                continue
            result = DocsSyncResult(
                project=project_dir.name,
                added=0,
                skipped=0,
                refetched=0,
                deleted=deleted,
                failures=[],
            )
            results.append(result)
            if on_progress is not None:
                on_progress(None, result)

    return results
