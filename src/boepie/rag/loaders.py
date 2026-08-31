"""Source-specific document loaders.

A ``Loader`` is the only part of the retrieval stack that knows about a
particular corpus layout. The engine (chunking, embedding, store, search)
consumes ``Document`` objects and is otherwise source-agnostic, so adding a
new source (e.g. plain-text docs) means writing one more loader here.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Protocol, runtime_checkable

from boepie.config import DOCS_DIR, LITERATURE_DIR, NOTES_DIR
from boepie.corpus.document import read_document
from boepie.corpus.layout import DocumentLocation
from boepie.corpus.layout import iter_documents as iter_corpus_documents
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


# Bumped when the digest below stops meaning what it used to, so an index
# built under the old rule is treated as unverifiable rather than as stale.
_REVISION_FORMAT = "1"


def body_digest(text: str) -> str:
    """The fingerprint of one document's indexed body."""
    return hashlib.sha256(f"{_REVISION_FORMAT}\x00{text}".encode()).hexdigest()[:16]


@dataclass(frozen=True)
class CorpusRevision:
    """What an index was built over, in enough detail to detect staleness.

    `path` is where the documents were read from, absolute, and it is what
    makes the check self-contained: the index says which directory to ask, so
    nothing has to reconstruct that from configuration. An index built on
    another machine names a directory that is not here, and is then simply not
    checked - which is the right answer for a prebuilt index fetched from a
    release, where there is no local corpus to be inconsistent with.

    `documents` maps each indexed document's id to a digest of the body that
    was indexed. A map rather than a list because the two halves of staleness
    are not the same thing:

    - a document that has **changed or gone** since the build makes the index
      *wrong* - it serves text that is no longer there, under a `source_path`
      an agent is invited to open with its own file tools;
    - a document that is **new** since the build makes it merely *incomplete*,
      which is the normal state between `corpus add` and `index build` and
      must not be an error, or the documented staging workflow would break
      search between every add and the rebuild that follows it.
    """

    path: str
    documents: dict[str, str] = field(default_factory=dict)


@runtime_checkable
class Revisioned(Protocol):
    """A loader whose corpus can be fingerprinted, and re-fingerprinted later.

    Optional extension to ``Loader``, and deliberately optional: a loader
    reading something that cannot be walked again (a fixture, a stream) simply
    does not implement it, and its index is then recorded as unverifiable
    rather than as fresh.

    ``corpus_path`` exists separately from ``corpus_revision`` so a build can
    record the documents it *actually indexed* without walking the corpus a
    second time - a second walk would also open a window in which the corpus
    could change between the two, and record a state the index does not have.
    """

    def corpus_path(self) -> Path: ...

    def corpus_revision(self) -> CorpusRevision | None: ...


def _revision_of(loader: Revisioned) -> CorpusRevision | None:
    """Fingerprint whatever `loader` currently yields.

    ``None`` when there is nothing there to compare against. An absent or
    empty corpus is not evidence that an index is stale - it is the ordinary
    shape of a machine holding a fetched index and no corpus of its own - and
    refusing to serve a working index over it would be the worse failure.
    """
    source_dir = loader.corpus_path()
    if not source_dir.is_dir():
        return None
    documents = {
        document.id: body_digest(document.text)
        for document in loader.iter_documents()
    }
    if not documents:
        return None
    return CorpusRevision(path=str(source_dir.resolve()), documents=documents)


class _CorpusLoader:
    """Shared walk over a machine-global corpus collection.

    `literature`, `docs` and `notes` share one on-disk shape (see
    `boepie.corpus.layout`): a `.md` file is a document, a directory holding
    `content.md` is a document with assets, and any other directory is a
    user-created group to recurse into. One walk covers all three; what
    differs per collection is only the frontmatter block each carries, which
    passes through to `Document.metadata` untouched for filters to reach with
    dotted paths.

    A document whose frontmatter carries no `id` is skipped rather than
    raising: that is the pre-migration signal (see
    `scripts/migrate_corpus_layout.py`), and a half-migrated corpus should
    build what it can rather than abort.
    """

    name: str

    def __init__(self, corpus_dir: Path | str) -> None:
        self.corpus_dir = Path(corpus_dir)

    def iter_documents(self) -> Iterable[Document]:
        for location in iter_corpus_documents(self.corpus_dir, collection=self.name):
            try:
                document = read_document(location.md_path)
            except ValueError:
                continue

            metadata: dict[str, Any] = dict(document.frontmatter)
            # Assets live inside a wrapped document's own directory; a bare
            # leaf has nowhere to put them, so it resolves to none rather
            # than reaching into its enclosing group and picking up a
            # sibling's images.
            metadata["images"] = resolve_image_refs(document.body, document.wrapper_dir)
            # The group path this document sits under, recorded explicitly
            # rather than left to be re-derived from `source_path`: that path
            # is absolute and machine-specific (it is why `relative_source`
            # exists), so an index built on one machine could not be filtered
            # by group on another. "" is the collection root.
            metadata["group"] = self._group_of(location)

            yield Document(
                id=document.id,
                text=document.body,
                source_path=str(document.md_path),
                base_path=str(document.wrapper_dir) if document.wrapper_dir else None,
                metadata=metadata,
            )

    def _group_of(self, location: DocumentLocation) -> str:
        """`location`'s group as a `/`-joined path relative to the corpus root.

        A wrapped document's own directory is the document, not a group, so
        the anchor is the wrapper when there is one and the file otherwise.
        """
        anchor = location.wrapper_dir or location.md_path
        try:
            relative = anchor.parent.relative_to(self.corpus_dir)
        except ValueError:
            return ""
        return "" if relative == Path(".") else relative.as_posix()

    def describe_sources(self) -> dict[str, Any]:
        return {"corpus_dir": self.corpus_dir.name}

    def corpus_path(self) -> Path:
        return self.corpus_dir

    def corpus_revision(self) -> CorpusRevision | None:
        return _revision_of(self)


class LiteratureLoader(_CorpusLoader):
    """Loads the paper corpus under ``LITERATURE_DIR``.

    Every document here was fetched and converted on this machine (see
    `boepie.literature.fetch`); boepie never ships converted paper text, so
    there is no prebuilt index to describe and no bibliography file to
    fingerprint - the packaged manifest naming which arXiv ids to pull is the
    only thing boepie itself publishes.
    """

    name = "literature"

    def __init__(self, corpus_dir: Path | str = LITERATURE_DIR) -> None:
        super().__init__(corpus_dir)

    def describe_sources(self) -> dict[str, Any]:
        """Which papers the packaged manifest names, alongside the corpus dir.

        Reported so a built index carries a record of what boepie's own
        default corpus was at build time, which the indexed ids alone do not
        say: a manifest entry whose paper failed to convert (no HTML rendering
        at either arXiv or ar5iv) leaves no other trace.
        """
        from boepie.literature.manifest import load_default_manifest

        papers = load_default_manifest()
        return {
            "corpus_dir": self.corpus_dir.name,
            "default_manifest": {
                "entry_count": len(papers),
                "citekeys": sorted(paper.citekey for paper in papers),
            },
        }


class NotesLoader(_CorpusLoader):
    """Loads the user-added notes corpus under ``NOTES_DIR``.

    Structurally the base case: notes carry the base frontmatter and nothing
    else, are always ``managed_by: user``, and have no manifest to reconcile
    against. Kept as its own collection rather than folded into the others so
    a user's own material never competes with curated sources in a search.
    """

    name = "notes"

    def __init__(self, corpus_dir: Path | str = NOTES_DIR) -> None:
        super().__init__(corpus_dir)


class DocsLoader(_CorpusLoader):
    """Loads the upstream documentation corpus under ``DOCS_DIR``.

    Pages carry a ``docs`` frontmatter block (project/page/base_url/version),
    so ``base_url`` joined with ``page`` plus ``.html`` still reconstructs the
    live URL. Grouping by project is a directory group like any other, not a
    special case in the walk.
    """

    name = "docs"

    def __init__(self, corpus_dir: Path | str = DOCS_DIR) -> None:
        super().__init__(corpus_dir)

    def describe_sources(self) -> dict[str, Any]:
        """Which docs site, at which version, each project was scraped from.

        These pages are converted copies of someone else's rendered docs, so
        the fact worth keeping is where and when. Collected from the pages'
        own frontmatter rather than a side file, so it describes what was
        actually indexed rather than what a manifest intended.
        """
        projects: dict[str, dict[str, Any]] = {}
        for document in self.iter_documents():
            docs_block = document.metadata.get("docs")
            if not isinstance(docs_block, dict):
                continue
            project_name = str(docs_block.get("project", ""))
            entry = projects.setdefault(
                project_name,
                {
                    "project": project_name,
                    "base_url": docs_block.get("base_url"),
                    "version": docs_block.get("version"),
                    "page_count": 0,
                },
            )
            entry["page_count"] += 1
        return {
            "corpus_dir": self.corpus_dir.name,
            "projects": [projects[name] for name in sorted(projects)],
        }


# ---------------------------------------------------------------------------


class ContextLoader:
    """Loads the curated context bundle from ``.boepie/``.

    Each markdown file under the bundle directory (except apply-log.md and
    anything under a dot-directory such as the derived `.index/`) becomes one
    document. Document IDs are the POSIX relative path without the suffix
    (e.g. "concepts/substitution" for ".boepie/concepts/substitution.md").
    Frontmatter is parsed using boepie.context.frontmatter helpers into
    Document.metadata. The body text (with frontmatter stripped) is the
    document text. index.md has no frontmatter (OKF reserved): it yields
    empty metadata and the full file as body text.
    """

    name = "context"

    def __init__(self, bundle_dir: Path | str) -> None:
        self.bundle_dir = Path(bundle_dir)

    def corpus_path(self) -> Path:
        return self.bundle_dir

    def corpus_revision(self) -> CorpusRevision | None:
        return _revision_of(self)

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
        from boepie.context.frontmatter import read_frontmatter

        md_paths = sorted(self.bundle_dir.rglob("*.md"))
        for md_path in md_paths:
            # apply-log.md is append-only bundle history; index.md is the
            # navigation entry point the agent is told to read first anyway
            # (its escalation table isn't a search answer). Neither belongs
            # in the search corpus.
            if md_path.name in ("apply-log.md", "index.md"):
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


# ---------------------------------------------------------------------------
# Re-reading a corpus an index already named
# ---------------------------------------------------------------------------


_LOADERS: dict[str, type] = {
    "literature": LiteratureLoader,
    "docs": DocsLoader,
    "notes": NotesLoader,
    "context": ContextLoader,
}


def loader_for(collection: str, source_dir: Path | str) -> Loader | None:
    """The loader that reads `collection` out of `source_dir`.

    Used to re-derive a corpus revision at query time from the directory the
    index recorded, rather than from configuration: an index and the corpus it
    was built over can legitimately be decoupled (a test builds a fixture
    corpus into a temporary index root), and asking the configured directory
    instead would compare an index against a corpus it was never built from.

    ``None`` for a collection boepie has no loader for, which is honest: an
    index over something else entirely cannot be verified, and pretending
    otherwise would fail every such index rather than skip it.
    """
    loader_class = _LOADERS.get(collection)
    return None if loader_class is None else loader_class(source_dir)
