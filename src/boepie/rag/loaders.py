"""Source-specific document loaders.

A ``Loader`` is the only part of the retrieval stack that knows about a
particular corpus layout. The engine (chunking, embedding, store, search)
consumes ``Document`` objects and is otherwise source-agnostic, so adding a
new source (e.g. plain-text docs) means writing one more loader here.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Protocol, runtime_checkable

from boepie.config import DOCS_DIR, LITERATURE_DIR
from boepie.rag.chunking import resolve_image_refs
from boepie.rag.models import Document


@runtime_checkable
class Loader(Protocol):
    """Yields the documents that make up one named collection."""

    name: str

    def iter_documents(self) -> Iterable[Document]: ...


@runtime_checkable
class SourceDescribing(Protocol):
    """A loader that can say where its corpus came from.

    Optional extension to ``Loader``. ``engine.build`` records the returned
    mapping under the manifest's ``sources``, so a shipped index tarball
    carries its own provenance - which upstream site and version a docs page
    was scraped from, which bibliography a literature citekey came out of -
    rather than relying on a table kept next to the client. ``build`` always
    adds the ids it actually indexed, so what a loader returns here is the
    corpus-level context around that list, not the list itself.
    """

    def describe_sources(self) -> dict[str, Any]: ...


class LiteratureLoader:
    """Loads the markdown literature corpus under ``stimela-corpus/``.

    Each document is a directory ``{citekey}/`` holding ``{citekey}.md``,
    a ``metadata.json`` (citekey/title/author/year/doi/...) and an
    ``images/`` subdirectory. Image references in the markdown are resolved
    to real files and attached to the document metadata.
    """

    name = "literature"

    def __init__(self, corpus_dir: Path | str = LITERATURE_DIR) -> None:
        self.corpus_dir = Path(corpus_dir)

    def iter_documents(self) -> Iterable[Document]:
        for doc_dir in sorted(p for p in self.corpus_dir.iterdir() if p.is_dir()):
            md_path = doc_dir / f"{doc_dir.name}.md"
            if not md_path.exists():
                continue

            text = md_path.read_text(encoding="utf-8")
            metadata = self._read_metadata(doc_dir)
            metadata["images"] = resolve_image_refs(text, doc_dir)

            yield Document(
                id=doc_dir.name,
                text=text,
                source_path=str(md_path),
                base_path=str(doc_dir),
                metadata=metadata,
            )

    def describe_sources(self) -> dict[str, Any]:
        """Which bibliography this corpus was extracted from.

        ``corpus.bib`` is the BibTeX file the PDFs were collected against, and
        it is the only place the corpus's citekeys carry titles/DOIs as a set
        rather than one per-paper ``metadata.json``. Its entry count is
        reported alongside the indexed citekeys precisely so the two can
        disagree visibly: a bib entry whose PDF was never converted leaves no
        other trace.

        ``content`` carries the file verbatim. Nothing at query time reads it -
        it is there so a shipped index tarball is a complete record of what it
        was built from, recoverable with::

            jq -r '.sources.bibtex.content' manifest.json > corpus.bib
        """
        sources: dict[str, Any] = {"corpus_dir": self.corpus_dir.name}

        bib_path = self.corpus_dir / "corpus.bib"
        if bib_path.is_file():
            raw = bib_path.read_bytes()
            text = raw.decode("utf-8", "replace")
            sources["bibtex"] = {
                "path": bib_path.name,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "entry_count": len(re.findall(r"^@\w+\s*\{", text, re.M)),
                "content": text,
            }
        return sources

    @staticmethod
    def _read_metadata(doc_dir: Path) -> dict:
        meta_path = doc_dir / "metadata.json"
        if not meta_path.exists():
            return {"citekey": doc_dir.name}
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        # Normalise the BibTeX-brace-wrapped title for display.
        if "title" in meta:
            meta["title"] = meta["title"].replace("{", "").replace("}", "").strip()
        return meta


# ---------------------------------------------------------------------------


class DocsLoader:
    """Loads the upstream documentation corpus under ``docs-corpus/``.

    Each project is a directory ``{project}/`` holding ``{project}/**/*.md``
    pages and an optional ``{project}/metadata.json``
    (project/version/base_url/...). Document IDs are ``{project}/{page}``,
    where ``page`` is the extension-less path relative to the project
    directory. Metadata from metadata.json is merged into every page's
    metadata, along with a ``page`` field set to that same relative name.

    Pages are found recursively because upstream docs sites nest them
    (stimela publishes ``fundamentals/params``, ``reference/cabdefs``), and
    that nesting is worth preserving: ``base_url`` joined with ``page`` plus
    ``.html`` reconstructs the page's real URL.
    """

    name = "docs"

    def __init__(self, corpus_dir: Path | str = DOCS_DIR) -> None:
        self.corpus_dir = Path(corpus_dir)

    def iter_documents(self) -> Iterable[Document]:
        for project_dir in sorted(p for p in self.corpus_dir.iterdir() if p.is_dir()):
            project_metadata = self._read_metadata(project_dir)

            for page_path in sorted(p for p in project_dir.rglob("*.md")):
                text = page_path.read_text(encoding="utf-8")
                page_name = page_path.relative_to(project_dir).with_suffix("").as_posix()
                document_id = f"{project_dir.name}/{page_name}"

                metadata: dict[str, Any] = project_metadata.copy()
                metadata["page"] = page_name

                yield Document(
                    id=document_id,
                    text=text,
                    source_path=str(page_path),
                    base_path=str(project_dir),
                    metadata=metadata,
                )

    def describe_sources(self) -> dict[str, Any]:
        """Which docs site, at which version, each project was scraped from.

        These pages are converted copies of someone else's rendered docs, so
        the fact worth keeping is where and when: ``base_url`` plus a page name
        reconstructs the live URL, and ``version``/``retrieved_at`` say which
        snapshot of upstream the copy corresponds to. Written by
        ``scripts/docs_to_md.py`` into each project's ``metadata.json``.
        """
        projects = [
            self._read_metadata(project_dir)
            for project_dir in sorted(p for p in self.corpus_dir.iterdir() if p.is_dir())
        ]
        return {"corpus_dir": self.corpus_dir.name, "projects": projects}

    @staticmethod
    def _read_metadata(project_dir: Path) -> dict[str, Any]:
        metadata_path = project_dir / "metadata.json"
        if not metadata_path.exists():
            return {"project": project_dir.name}
        return json.loads(metadata_path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------


class KnowledgeLoader:
    """Loads the curated knowledge bundle from ``.boepie/``.

    Each markdown file under the bundle directory (except log.md and anything
    under a dot-directory such as the derived `.index/`) becomes one
    document. Document IDs are the POSIX relative path without the suffix
    (e.g. "concepts/substitution" for ".boepie/concepts/substitution.md").
    Frontmatter is parsed using boepie.knowledge.frontmatter helpers into
    Document.metadata. The body text (with frontmatter stripped) is the
    document text. index.md has no frontmatter (OKF reserved): it yields
    empty metadata and the full file as body text.
    """

    name = "knowledge"

    def __init__(self, bundle_dir: Path | str) -> None:
        self.bundle_dir = Path(bundle_dir)

    def describe_sources(self) -> dict[str, Any]:
        """Which bundle snapshot this index was built over.

        The bundle's own ``manifest.json`` already versions the content
        (``content_version``) and the tooling that applied it; copying it here
        ties a built index to the exact bundle revision it indexed.
        """
        sources: dict[str, Any] = {"bundle_dir": self.bundle_dir.name}

        manifest_path = self.bundle_dir / "manifest.json"
        if manifest_path.is_file():
            sources["bundle_manifest"] = json.loads(manifest_path.read_text(encoding="utf-8"))
        return sources

    def iter_documents(self) -> Iterable[Document]:
        from boepie.knowledge.frontmatter import read_frontmatter

        md_paths = sorted(self.bundle_dir.rglob("*.md"))
        for md_path in md_paths:
            # log.md is append-only bundle history; index.md is the navigation
            # entry point the agent is told to read first anyway (its escalation
            # table isn't a search answer). Neither belongs in the search corpus.
            if md_path.name in ("log.md", "index.md"):
                continue

            relative_path = md_path.relative_to(self.bundle_dir)
            # Skip the bundle's own dot-directories - `.index/` is the search
            # index built from this very corpus, so indexing it would feed
            # derived state back in.
            if any(part.startswith(".") for part in relative_path.parts):
                continue

            # Document ID is the POSIX relative path without suffix.
            document_id = relative_path.with_suffix("").as_posix()
            text = md_path.read_text(encoding="utf-8")

            # Parse frontmatter and extract body.
            frontmatter, body = read_frontmatter(text)

            yield Document(
                id=document_id,
                text=body,
                source_path=str(md_path),
                base_path=str(self.bundle_dir),
                metadata=frontmatter,
            )
