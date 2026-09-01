# src/boepie/corpus/schema.py
"""The frontmatter schema every corpus document carries, declared once.

One base model (`CorpusDocument`) holds what is true of any document in any
collection: its surrogate `id`, its `title`, the `managed_by` guard that
decides whether `corpus fetch` may touch it, and a nested `source` block
recording where the bytes came from and how they were converted. Each
collection extends that base with exactly one namespaced block of its own -
`bib` for literature, `docs` for documentation pages, nothing at all for
notes, which is the base case.

Nesting is deliberate. The flat layout this replaces spelled one concept
three ways (`fetched_via`/`fetched_from` on literature, `source_kind`/
`source_path` on notes) and collided with the provenance guard, which was
itself named `source`. Namespacing the blocks makes the guard's name honest
and gives every collection one vocabulary.

`context.frontmatter`'s codec needs no changes to carry this: `yaml.safe_load`
and `safe_dump` handle nested mappings natively. What this module adds is
validation at the boundary - `dump_frontmatter` on the way to disk,
`parse_frontmatter` on the way back.
"""

from __future__ import annotations

from datetime import datetime, timezone
from collections.abc import Iterable
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# How a document's markdown was produced from its source bytes. Distinct from
# `format`: a PDF and a DOCX both arrive via "mineru", and an arXiv paper's
# HTML arrives via "arxiv-html" or "ar5iv" depending on which renderer served
# it.
type ConversionVia = Literal[
    "verbatim", "html", "mineru", "arxiv-html", "ar5iv", "sphinx", "crawl"
]

# The shape of the source bytes themselves, before conversion.
type SourceFormat = Literal[
    "pdf", "docx", "pptx", "xlsx", "html", "markdown", "text", "code"
]

# Who owns a document: whether `corpus fetch` may refetch or delete it.
# `boepie` documents are reconciled against a packaged manifest; `user`
# documents are never touched by any reconciler at any step.
type ManagedBy = Literal["boepie", "user"]


def utc_now() -> datetime:
    """Timezone-aware ingestion timestamp. A module-level function rather
    than an inline lambda so tests can monkeypatch it."""
    return datetime.now(timezone.utc)


class Source(BaseModel):
    """Where a document came from and how it got here.

    Replaces the old `fetched_via`/`fetched_from` (literature) and
    `source_kind`/`source_path` (notes) pairs, which named one concept two
    ways. `from` is a Python keyword, hence the alias.
    """

    model_config = ConfigDict(populate_by_name=True)

    origin: str = Field(
        alias="from",
        description="url, filesystem path, arxiv:ID, or doi:DOI.",
    )
    via: ConversionVia
    format: SourceFormat
    # Absent for documents whose source is not a fixed byte sequence (a live
    # site crawl re-renders per page), present for everything ingested from a
    # file or a single fetched resource.
    sha256: str | None = None
    at: datetime = Field(default_factory=utc_now)
    # Filename of the retained source bytes inside this document's wrapper
    # directory, when `corpus.keep_original` was on at ingestion time.
    original: str | None = None


class CorpusDocument(BaseModel):
    """The frontmatter fields every corpus document carries."""

    model_config = ConfigDict(populate_by_name=True)

    id: str
    title: str
    managed_by: ManagedBy
    source: Source


class NoteDocument(CorpusDocument):
    """A note adds nothing to the base. Notes are addressed by `id` and have
    no manifest to reconcile against, so they need no natural key of their
    own - the reason the old `slug` field is gone."""


class Bibliography(BaseModel):
    """Bibliographic facts about a paper. None of it is the paper's own text,
    which is why the default manifest can ship these fields when the converted
    Markdown may never be redistributed (see `boepie.literature.fetch`)."""

    citekey: str
    authors: str | None = None
    year: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    # ADS bibcode. The astronomy-native identifier, and the only one a
    # pre-arXiv paper reliably has - recorded so such a paper can still be
    # deduplicated, never resolved (that needs the ADS API and a key).
    bibcode: str | None = None


class LiteratureDocument(CorpusDocument):
    bib: Bibliography


class CrawlScope(BaseModel):
    """How a docs project's pages were discovered.

    Carried on every page rather than in a project-level side file: it is the
    one thing a page cannot otherwise reconstruct about its own project, so
    recording it here keeps a re-crawl from needing the original flags typed
    again, without reintroducing a second source of truth.
    """

    discovery: str | None = None
    exclude: list[str] = Field(default_factory=list)
    path_prefix: str | None = None


class DocsPage(BaseModel):
    project: str
    page: str
    base_url: str | None = None
    version: str | None = None
    crawl: CrawlScope | None = None


class DocsDocument(CorpusDocument):
    docs: DocsPage


DOCUMENT_MODELS: dict[str, type[CorpusDocument]] = {
    "literature": LiteratureDocument,
    "docs": DocsDocument,
    "notes": NoteDocument,
}

# Which frontmatter paths a reconciler diffs a manifest against, per
# collection. Dotted because the fields now live inside their collection's
# namespaced block; `notes` has none, since it has no manifest.
KEY_FIELDS: dict[str, tuple[str, ...]] = {
    "literature": ("bib.citekey",),
    "docs": ("docs.project", "docs.page"),
    "notes": (),
}


def dump_frontmatter(document: CorpusDocument) -> dict[str, Any]:
    """Serialise a document model into the mapping written to disk.

    `mode="json"` renders the timestamp as an ISO string rather than a
    `datetime` object, which `yaml.safe_dump` would refuse. `exclude_none`
    keeps optional fields out of the file entirely instead of writing
    `null`s - a note ingested from a local file has no `original`, and
    saying so by omission is both smaller and honest.
    """
    return document.model_dump(mode="json", by_alias=True, exclude_none=True)


def literature_blocks(
    *, citekey: str, authors: str = "", year: str = "",
    doi: str | None = None, arxiv_id: str | None = None,
    bibcode: str | None = None,
) -> dict[str, Any]:
    """The `bib:` block for one paper, with empty fields omitted.

    Shared by `corpus.add` and `corpus.reconcile` so a paper you add and the
    same paper `corpus fetch` pulls from the packaged manifest land in
    identical shape - the two used to build their frontmatter independently,
    and fetch's copy was left behind on the flat pre-`bib:` layout.
    """
    bib: dict[str, Any] = {"citekey": citekey}
    if authors:
        bib["authors"] = authors
    if year:
        bib["year"] = year
    if doi:
        bib["doi"] = doi
    if arxiv_id:
        bib["arxiv_id"] = arxiv_id
    if bibcode:
        bib["bibcode"] = bibcode
    return {"bib": bib}


def docs_blocks(
    *, project: str, page: str, base_url: str | None = None,
    version: str | None = None, discovery: str | None = None,
    exclude: Iterable[str] = (), path_prefix: str | None = None,
) -> dict[str, Any]:
    """The `docs:` block for one documentation page, crawl scope included."""
    crawl = CrawlScope(
        discovery=discovery, exclude=list(exclude), path_prefix=path_prefix
    )
    page_model = DocsPage(
        project=project, page=page, base_url=base_url, version=version, crawl=crawl,
    )
    return {"docs": page_model.model_dump(mode="json", exclude_none=True)}


def parse_frontmatter(collection: str, frontmatter: dict[str, Any]) -> CorpusDocument:
    """Validate an on-disk frontmatter mapping against `collection`'s model.

    Raises `pydantic.ValidationError` naming the offending field, which is
    what a corrupted or hand-broken document should produce - loudly, not by
    silently sorting under a degenerate key.
    """
    model = DOCUMENT_MODELS.get(collection)
    if model is None:
        raise ValueError(f"unknown collection '{collection}'")
    return model.model_validate(frontmatter)
