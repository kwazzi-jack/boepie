# src/boepie/corpus/add.py
"""Orchestrates `boepie corpus add`: identifier in, document on disk.

`add` has exactly one contract in every collection - it ingests what you hand
it, immediately, and writes a `managed_by: user` document. It never stages
anything for a later `fetch`. `fetch` is the other half of that split: it
reconciles boepie's own packaged manifest and, by the guarantee in
`boepie.corpus.reconcile`, never touches a `managed_by: user` document at any
step. So there is no user manifest here and nothing for one to drift from -
the documents are the record.

Each collection is the shared core (see `boepie.corpus.intake`) plus its own
resolvers and its own frontmatter block:

    notes       the base case: local files and URLs, nothing added
    literature  + arXiv ids in any spelling, DOIs, and .bib files
    docs        + a site URL, which crawls the whole site rather than one page
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import httpx

from boepie.corpus.document import write_leaf_document
from boepie.corpus.intake import (
    Converted,
    IntakeError,
    convert_local_file,
    convert_url,
    looks_like_url,
)
from boepie.corpus.ids import unique_id
from boepie.corpus.layout import (
    collection_index,
    full_title_filename,
    lookup_path,
    title_needs_dot_stripped,
    unique_document_name,
)
from boepie.corpus.inputs import ResolvedInputs, resolve_inputs
from boepie.corpus.schema import KEY_FIELDS, Source, literature_blocks

# `skipped` is not a soft failure: it is boepie declining to take something it
# was pointed at, which a folder walk makes routine (an image among the PDFs,
# a format whose converter is not installed). It must never set the exit code,
# or a directory containing one file boepie cannot read would fail the command.
type AddStatus = Literal["added", "duplicate", "failed", "skipped"]


@dataclass(frozen=True)
class AddOptions:
    """Everything the CLI can vary about one `corpus add` invocation."""

    title: str | None = None
    group: str | None = None
    citekey: str | None = None
    project: str | None = None
    keep_original: bool = False
    mineru_device_mode: str = "auto"
    mineru_backend: str = "pipeline"
    mineru_model_source: str = "auto"
    # Extra suffixes a folder walk accepts (corpus.extra_file_types).
    extra_file_types: tuple[str, ...] = ()
    # Politeness delay between pages of a crawled docs site.
    delay: float = 0.2


@dataclass(frozen=True)
class AddOutcome:
    """What became of one identifier in a batch.

    A batch never aborts on a single failure: one unreachable URL among
    twenty should not cost the other nineteen, so every identifier produces
    an outcome and the caller reports them together.
    """

    identifier: str
    status: AddStatus
    document_id: str | None = None
    title: str | None = None
    path: Path | None = None
    via: str | None = None
    detail: str | None = None
    # A non-fatal remark about how the document was written, e.g. a title
    # whose leading dot had to be stripped. The CLI decides whether to show
    # it; the outcome always carries it.
    notice: str | None = None


@dataclass
class _CorpusState:
    """The uniqueness bookkeeping a batch has to keep consistent as it goes.

    Loaded once per batch rather than per document: `collection_index` walks
    and parses the whole collection, so doing it per item would make adding
    fifty papers quadratic. Every write updates these in place so the second
    document in a batch sees the first.
    """

    collection: str
    ids: set[str] = field(default_factory=set)
    filenames: set[str] = field(default_factory=set)
    checksums: dict[str, str] = field(default_factory=dict)
    natural_keys: dict[str, str] = field(default_factory=dict)
    # arXiv id / DOI -> document id. A paper reached by two different routes
    # (a .bib entry and a bare arXiv id, say) derives two different citekeys
    # and has no source checksum to compare, so neither the natural key nor
    # the checksum catches the duplicate. Its bibliographic identity does.
    bib_identities: dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(cls, collection_dir: Path, collection: str) -> _CorpusState:
        state = cls(collection=collection)
        documents = collection_index(
            collection_dir, collection=collection, key_fields=KEY_FIELDS[collection]
        )
        for document in documents:
            state.ids.add(document.id)
            state.filenames.add(document.reserved_filename)
            checksum = lookup_path(document.frontmatter, "source.sha256")
            if isinstance(checksum, str):
                state.checksums[checksum] = document.id
            if document.natural_key:
                state.natural_keys[document.natural_key] = document.id
            for path in ("bib.arxiv_id", "bib.doi"):
                identity = lookup_path(document.frontmatter, path)
                if isinstance(identity, str) and identity:
                    state.bib_identities[identity.lower()] = document.id
        return state


def _write(
    collection_dir: Path,
    state: _CorpusState,
    *,
    converted: Converted,
    title: str,
    options: AddOptions,
    blocks: dict[str, Any],
    group: str | None = None,
) -> tuple[str, Path]:
    """Write one converted source as a corpus document, keeping `state` current.

    `group` overrides `options.group` for this one document, which a folder
    walk needs: every file in a batch shares one `AddOptions` but lands in the
    group mirroring its own subdirectory. `add_docs` used to express the same
    thing by rebuilding a throwaway `AddOptions` per page.
    """
    document_id = unique_id(state.ids)
    state.ids.add(document_id)

    filename = unique_document_name(full_title_filename(title), state.filenames)
    state.filenames.add(filename)

    source = Source(
        origin=converted.origin,
        via=converted.via,
        format=converted.format,
        sha256=converted.sha256,
        original=converted.original_name,
    )
    frontmatter: dict[str, Any] = {
        "title": title,
        "managed_by": "user",
        "source": source.model_dump(mode="json", by_alias=True, exclude_none=True),
        **blocks,
    }

    assets: dict[str, bytes] | None = None
    if converted.original_bytes is not None and converted.original_name is not None:
        assets = {converted.original_name: converted.original_bytes}

    placement = options.group if group is None else group
    target_dir = collection_dir / placement if placement else collection_dir
    document = write_leaf_document(
        target_dir / filename,
        document_id=document_id,
        frontmatter_fields=frontmatter,
        body=converted.markdown,
        assets=assets,
    )
    if converted.sha256:
        state.checksums[converted.sha256] = document_id
    return document_id, document.md_path


def _group_for(options: AddOptions, walked: str) -> str | None:
    """Where one walked file lands, given `--group` and its own subdirectory.

    `--group` is a prefix rather than an override once a directory is being
    walked: collapsing every file into one flat group would undo the very
    thing the mirrored structure is for, which is keeping same-named files
    (a `README.md` per subdirectory) apart.
    """
    if not walked:
        return None
    return f"{options.group}/{walked}" if options.group else walked


def _skipped_outcomes(resolved: ResolvedInputs) -> list[AddOutcome]:
    """Files the walk declined, reported rather than dropped in silence."""
    return [
        AddOutcome(identifier=skip.identifier, status="skipped", detail=skip.reason)
        for skip in resolved.skipped
    ]


def _notice_for(title: str) -> str | None:
    """A remark worth surfacing about how `title` had to be written.

    A dot-prefixed title (from a dotfile like `.bashrc`) would produce a
    filename the corpus walk treats as bookkeeping and skips forever, so the
    dot is stripped - which means the file on disk is not named what the
    title says, and that is worth saying out loud once.
    """
    if title_needs_dot_stripped(title):
        return (
            f"title '{title}' looked like a dotfile name; the leading dot was "
            f"stripped from the filename so it stays visible to search"
        )
    return None


def _duplicate_by_checksum(state: _CorpusState, converted: Converted) -> str | None:
    if converted.sha256 is None:
        return None
    return state.checksums.get(converted.sha256)


def _convert_identifier(identifier: str, options: AddOptions) -> Converted:
    """The shared core: a local path or an http(s) URL becomes Markdown."""
    if looks_like_url(identifier):
        return convert_url(identifier, keep_original=options.keep_original)

    path = Path(identifier).expanduser()
    if path.is_file():
        return convert_local_file(
            path,
            keep_original=options.keep_original,
            mineru_device_mode=options.mineru_device_mode,
            mineru_backend=options.mineru_backend,
            mineru_model_source=options.mineru_model_source,
        )

    raise IntakeError(
        f"'{identifier}' is not an existing file or an http(s) URL, and no "
        f"resolver recognised it."
    )


# ---------------------------------------------------------------------------
# notes: the base case
# ---------------------------------------------------------------------------


def add_notes(
    collection_dir: Path, identifiers: Sequence[str], options: AddOptions
) -> list[AddOutcome]:
    """Add local files or URLs to the notes corpus.

    Notes extend the base schema with nothing: no natural key, no namespaced
    block. They are addressed by `id` and reconciled against nothing, which is
    why the old collection-wide `slug` is gone.
    """
    resolved = resolve_inputs(identifiers, extra_file_types=options.extra_file_types)
    state = _CorpusState.load(collection_dir, "notes")
    outcomes: list[AddOutcome] = _skipped_outcomes(resolved)

    for item in resolved.items:
        identifier = item.identifier
        try:
            converted = _convert_identifier(identifier, options)
        except IntakeError as error:
            outcomes.append(
                AddOutcome(identifier=identifier, status="failed", detail=str(error))
            )
            continue

        existing = _duplicate_by_checksum(state, converted)
        if existing is not None:
            outcomes.append(
                AddOutcome(
                    identifier=identifier, status="duplicate", document_id=existing,
                    detail="identical content is already in this collection",
                )
            )
            continue

        title = options.title or converted.suggested_title or identifier
        document_id, path = _write(
            collection_dir, state, converted=converted, title=title,
            options=options, blocks={}, group=_group_for(options, item.group),
        )
        outcomes.append(
            AddOutcome(
                identifier=identifier, status="added", document_id=document_id,
                title=title, path=path, via=converted.via,
                notice=_notice_for(title),
            )
        )
    return outcomes


# ---------------------------------------------------------------------------
# literature: + arXiv ids, DOIs, .bib files
# ---------------------------------------------------------------------------


def _resolve_citekey(state: _CorpusState, base: str, override: str | None) -> str:
    from boepie.literature.manifest import unique_citekey

    if override is not None:
        return override
    return unique_citekey(base, set(state.natural_keys))


def _add_arxiv_paper(
    collection_dir: Path, state: _CorpusState, arxiv_id: str, identifier: str,
    options: AddOptions, *, entry: Any = None, group: str | None = None,
) -> AddOutcome:
    """Look a paper up on arXiv, fetch its HTML, and convert it locally.

    Nothing is downloaded from a boepie release and no paper text is ever
    redistributed by boepie: the conversion happens on the machine that will
    read it (see `boepie.literature.fetch`).
    """
    from boepie.literature.fetch import fetch_paper, lookup_arxiv_metadata
    from boepie.literature.manifest import derive_citekey

    already = state.bib_identities.get(arxiv_id.lower())
    if already is not None:
        return AddOutcome(
            identifier=identifier, status="duplicate", document_id=already,
            detail=f"arXiv:{arxiv_id} is already in this collection",
        )

    try:
        metadata = lookup_arxiv_metadata(arxiv_id)
    except httpx.HTTPError as error:
        return AddOutcome(
            identifier=identifier, status="failed",
            detail=f"could not reach arXiv: {error}",
        )
    if metadata is None:
        return AddOutcome(
            identifier=identifier, status="failed",
            detail=f"arXiv has no entry for '{arxiv_id}'.",
        )

    # A .bib entry already carries the citekey the user cites this paper by -
    # keeping it is the main reason to add a library that way rather than by
    # bare id. Only derive one when there is nothing to preserve.
    base_citekey = (
        entry.citekey
        if entry is not None and entry.citekey
        else derive_citekey(metadata["authors"], metadata["year"], metadata["title"])
    )
    citekey = _resolve_citekey(state, base_citekey, options.citekey)
    if citekey in state.natural_keys:
        return AddOutcome(
            identifier=identifier, status="duplicate",
            document_id=state.natural_keys[citekey],
            detail=f"citekey '{citekey}' is already in this collection",
        )

    with httpx.Client(
        headers={"User-Agent": "boepie-literature-fetch"}, follow_redirects=True
    ) as client:
        result = fetch_paper(client, citekey, arxiv_id)

    if result.markdown is None:
        return AddOutcome(
            identifier=identifier, status="failed",
            detail=(
                f"arXiv:{arxiv_id} has no HTML rendering at arxiv.org or ar5iv "
                f"(common for pre-2007 submissions). Supply the PDF instead: "
                f"boepie corpus add literature <file.pdf>"
            ),
        )

    converted = Converted(
        markdown=result.markdown,
        via="ar5iv" if result.source == "ar5iv" else "arxiv-html",
        format="html",
        origin=f"arxiv:{arxiv_id}",
        suggested_title=metadata["title"],
    )
    title = options.title or metadata["title"]
    document_id, path = _write(
        collection_dir, state, converted=converted, title=title, options=options,
        blocks=literature_blocks(
            citekey=citekey, authors=metadata["authors"],
            year=metadata["year"], arxiv_id=arxiv_id,
            doi=entry.doi if entry is not None else None,
        ),
        group=_group_for(options, group or ""),
    )
    state.natural_keys[citekey] = document_id
    state.bib_identities[arxiv_id.lower()] = document_id
    if entry is not None and entry.doi:
        state.bib_identities[entry.doi.lower()] = document_id
    return AddOutcome(
        identifier=identifier, status="added", document_id=document_id,
        title=title, path=path, via=converted.via, notice=_notice_for(title),
    )


def add_literature(
    collection_dir: Path, identifiers: Sequence[str], options: AddOptions
) -> list[AddOutcome]:
    """Add papers by arXiv id, DOI, .bib file, local file, or URL.

    Order matters: a `.bib` is expanded first (it names many papers), then
    arXiv ids, then DOIs, and only then does the shared file/URL core run. An
    arXiv id is checked before the filesystem so `2409.19750` is read as a
    paper rather than as a missing file.
    """
    from boepie.literature.identifiers import (
        arxiv_id_if_reference,
        looks_like_bibtex,
        normalize_doi,
        parse_bibtex_file,
        resolve_doi_to_arxiv,
    )

    resolved = resolve_inputs(identifiers, extra_file_types=options.extra_file_types)
    state = _CorpusState.load(collection_dir, "literature")
    outcomes: list[AddOutcome] = _skipped_outcomes(resolved)

    # A .bib names many papers, so it expands into the queue rather than
    # becoming one document. Expanded entries carry their own citekey, so an
    # explicit --citekey would be ambiguous across them.
    queue: list[tuple[str, Any, str]] = []
    for item in resolved.items:
        identifier = item.identifier
        if looks_like_bibtex(identifier) and Path(identifier).expanduser().is_file():
            try:
                entries = parse_bibtex_file(Path(identifier).expanduser())
            except OSError as error:
                outcomes.append(
                    AddOutcome(identifier=identifier, status="failed", detail=str(error))
                )
                continue
            if not entries:
                outcomes.append(
                    AddOutcome(
                        identifier=identifier, status="failed",
                        detail="no usable entries found in this BibTeX file.",
                    )
                )
                continue
            queue.extend((identifier, entry, item.group) for entry in entries)
        else:
            queue.append((identifier, None, item.group))

    for identifier, entry, walked_group in queue:
        # A bib entry resolves to whatever it points at: an arXiv id, then a
        # DOI, then a local PDF named by its `file` field.
        if entry is not None:
            label = f"{identifier}:{entry.citekey}"
            if entry.arxiv_id:
                outcomes.append(
                    _add_arxiv_paper(
                        collection_dir, state, entry.arxiv_id, label, options,
                        entry=entry, group=walked_group,
                    )
                )
                continue
            resolved = resolve_doi_to_arxiv(entry.doi) if entry.doi else None
            if resolved:
                outcomes.append(
                    _add_arxiv_paper(
                        collection_dir, state, resolved, label, options, entry=entry,
                        group=walked_group,
                    )
                )
                continue
            if entry.file_path and Path(entry.file_path).expanduser().is_file():
                outcomes.append(
                    _add_literature_file(
                        collection_dir, state, Path(entry.file_path).expanduser(),
                        label, options, entry=entry, group=walked_group,
                    )
                )
                continue
            outcomes.append(
                AddOutcome(
                    identifier=label, status="failed",
                    detail=(
                        "no arXiv id, resolvable DOI, or readable file for this "
                        "entry - add its PDF directly."
                    ),
                )
            )
            continue

        # Only when the whole argument is an arXiv reference, and only when no
        # such file exists. A filename carrying a date-shaped run of digits
        # ('notes-2409.19750.md') otherwise reads as an arXiv id, and both a
        # real file and a mistyped one were answered by silently fetching a
        # stranger's paper of that number.
        arxiv_id = arxiv_id_if_reference(identifier)
        if arxiv_id is not None and not Path(identifier).expanduser().is_file():
            outcomes.append(
                _add_arxiv_paper(
                    collection_dir, state, arxiv_id, identifier, options,
                    group=walked_group,
                )
            )
            continue

        doi = normalize_doi(identifier)
        if doi is not None and not Path(identifier).expanduser().is_file():
            resolved = resolve_doi_to_arxiv(doi)
            if resolved is None:
                outcomes.append(
                    AddOutcome(
                        identifier=identifier, status="failed",
                        detail=(
                            f"arXiv has no preprint for DOI {doi}. Supply the "
                            f"publisher PDF instead, if its licence allows."
                        ),
                    )
                )
                continue
            outcomes.append(
                _add_arxiv_paper(
                    collection_dir, state, resolved, identifier, options,
                    group=walked_group,
                )
            )
            continue

        path = Path(identifier).expanduser()
        if path.is_file():
            outcomes.append(
                _add_literature_file(
                    collection_dir, state, path, identifier, options,
                    group=walked_group,
                )
            )
            continue

        try:
            converted = _convert_identifier(identifier, options)
        except IntakeError as error:
            outcomes.append(
                AddOutcome(identifier=identifier, status="failed", detail=str(error))
            )
            continue
        outcomes.append(
            _finish_literature(
                collection_dir, state, converted, identifier, options, entry=None,
                group=walked_group,
            )
        )

    return outcomes


def _add_literature_file(
    collection_dir: Path, state: _CorpusState, path: Path, identifier: str,
    options: AddOptions, *, entry: Any = None, group: str | None = None,
) -> AddOutcome:
    try:
        converted = convert_local_file(
            path,
            keep_original=options.keep_original,
            mineru_device_mode=options.mineru_device_mode,
            mineru_backend=options.mineru_backend,
            mineru_model_source=options.mineru_model_source,
        )
    except IntakeError as error:
        return AddOutcome(identifier=identifier, status="failed", detail=str(error))
    return _finish_literature(
        collection_dir, state, converted, identifier, options, entry=entry,
        group=group,
    )


def _finish_literature(
    collection_dir: Path, state: _CorpusState, converted: Converted, identifier: str,
    options: AddOptions, *, entry: Any, group: str | None = None,
) -> AddOutcome:
    """Write a paper that came from a file or a plain URL rather than arXiv.

    A local PDF carries no bibliography of its own, so its citekey is derived
    from whatever is available: a `.bib` entry when one pointed here, else the
    document title. That is deliberately the user's problem to correct with
    `--citekey` - guessing harder would be less predictable, not more.
    """
    from boepie.literature.manifest import derive_citekey

    existing = _duplicate_by_checksum(state, converted)
    if existing is not None:
        return AddOutcome(
            identifier=identifier, status="duplicate", document_id=existing,
            detail="identical content is already in this collection",
        )

    if entry is not None:
        title = options.title or entry.title
        base_citekey = entry.citekey
        authors, year, doi, arxiv_id = (
            entry.authors, entry.year, entry.doi, entry.arxiv_id
        )
    else:
        title = options.title or converted.suggested_title or Path(identifier).stem
        base_citekey = derive_citekey("", "", title)
        authors, year, doi, arxiv_id = "", "", None, None

    for identity in (arxiv_id, doi):
        already = state.bib_identities.get(identity.lower()) if identity else None
        if already is not None:
            return AddOutcome(
                identifier=identifier, status="duplicate", document_id=already,
                detail=f"'{identity}' is already in this collection",
            )

    citekey = _resolve_citekey(state, base_citekey, options.citekey)
    if citekey in state.natural_keys:
        return AddOutcome(
            identifier=identifier, status="duplicate",
            document_id=state.natural_keys[citekey],
            detail=f"citekey '{citekey}' is already in this collection",
        )

    document_id, path = _write(
        collection_dir, state, converted=converted, title=title, options=options,
        blocks=literature_blocks(
            citekey=citekey, authors=authors, year=year, doi=doi,
            arxiv_id=arxiv_id,
        ),
        group=_group_for(options, group or ""),
    )
    state.natural_keys[citekey] = document_id
    for identity in (arxiv_id, doi):
        if identity:
            state.bib_identities[identity.lower()] = document_id
    return AddOutcome(
        identifier=identifier, status="added", document_id=document_id, title=title,
        path=path, via=converted.via, notice=_notice_for(title),
    )


# ---------------------------------------------------------------------------
# docs: + a site URL crawls the whole site
# ---------------------------------------------------------------------------


def add_docs(
    collection_dir: Path,
    identifiers: Sequence[str],
    options: AddOptions,
    *,
    on_page: Callable[[str], None] | None = None,
) -> list[AddOutcome]:
    """Add documentation, either a whole site or a single file.

    A URL here means the whole site, not one page - that is the difference
    between `add docs <url>` and `add notes <url>`, and the reason these stay
    separate subcommands instead of one command behind a `--collection` flag.
    """
    from boepie.docs.fetch import (
        fetch_version,
        iter_project_pages,
        probe_discovery_mode,
    )
    from boepie.docs.manifest import DocsProject
    from boepie.corpus.intake import title_from_markdown

    # docs does not mirror a walked directory onto groups, unlike notes and
    # literature. A docs page's `docs.project` *is* the group it lives in and
    # is always exactly one level - that is what `search_docs(project=...)`
    # filters on and what `corpus move` keeps in step. Nesting groups under a
    # project would leave the field naming a directory the page is not in. So
    # discovery recurses, and every page found lands in the one named project.
    resolved = resolve_inputs(identifiers, extra_file_types=options.extra_file_types)
    state = _CorpusState.load(collection_dir, "docs")
    outcomes: list[AddOutcome] = _skipped_outcomes(resolved)

    for item in resolved.items:
        identifier = item.identifier
        project_name = options.project
        if project_name is None:
            outcomes.append(
                AddOutcome(
                    identifier=identifier, status="failed",
                    detail="--project is required: it names the group these pages live in.",
                )
            )
            continue

        if not looks_like_url(identifier):
            path = Path(identifier).expanduser()
            if not path.is_file():
                outcomes.append(
                    AddOutcome(
                        identifier=identifier, status="failed",
                        detail=f"'{identifier}' is not an existing file or an http(s) URL.",
                    )
                )
                continue
            try:
                converted = convert_local_file(
                    path,
                    keep_original=options.keep_original,
                    mineru_device_mode=options.mineru_device_mode,
                    mineru_backend=options.mineru_backend,
                    mineru_model_source=options.mineru_model_source,
                )
            except IntakeError as error:
                outcomes.append(
                    AddOutcome(identifier=identifier, status="failed", detail=str(error))
                )
                continue

            existing = _duplicate_by_checksum(state, converted)
            if existing is not None:
                outcomes.append(
                    AddOutcome(
                        identifier=identifier, status="duplicate", document_id=existing,
                        detail="identical content is already in this collection",
                    )
                )
                continue

            title = options.title or converted.suggested_title or path.stem
            page_name = path.stem
            document_id, written = _write(
                collection_dir, state, converted=converted, title=title,
                options=AddOptions(**{**options.__dict__, "group": project_name}),
                blocks={"docs": {"project": project_name, "page": page_name}},
            )
            state.natural_keys[f"{project_name}/{page_name}"] = document_id
            outcomes.append(
                AddOutcome(
                    identifier=identifier, status="added", document_id=document_id,
                    title=title, path=written, via=converted.via,
                )
            )
            continue

        project = DocsProject(project=project_name, base_url=identifier)
        added = 0
        failed = 0
        with httpx.Client(headers={"User-Agent": "boepie-docs-fetch"}) as client:
            discovery = probe_discovery_mode(client, identifier, 30)
            version = fetch_version(client, identifier, 30) if discovery == "sphinx" else None

            for page in iter_project_pages(client, project, delay=options.delay):
                if page.markdown is None:
                    failed += 1
                    continue
                natural_key = f"{project_name}/{page.docname}"
                if natural_key in state.natural_keys:
                    continue

                title = title_from_markdown(page.markdown, page.docname)
                converted = Converted(
                    markdown=page.markdown,
                    via="sphinx" if discovery == "sphinx" else "crawl",
                    format="html",
                    origin=page.url if hasattr(page, "url") else identifier,
                    suggested_title=title,
                )
                document_id, _written = _write(
                    collection_dir, state, converted=converted, title=title,
                    options=AddOptions(**{**options.__dict__, "group": project_name}),
                    blocks={
                        "docs": {
                            "project": project_name,
                            "page": page.docname,
                            "base_url": identifier,
                            "version": version,
                            "crawl": {"discovery": discovery},
                        }
                    },
                )
                state.natural_keys[natural_key] = document_id
                added += 1
                if on_page is not None:
                    on_page(page.docname)

        detail = f"{added} page(s)" + (f", {failed} failed" if failed else "")
        outcomes.append(
            AddOutcome(
                identifier=identifier,
                status="added" if added else "failed",
                title=project_name,
                via="sphinx" if discovery == "sphinx" else "crawl",
                detail=detail if added else "no pages could be fetched from this site.",
            )
        )
    return outcomes


ADDERS: dict[str, Callable[[Path, Sequence[str], AddOptions], list[AddOutcome]]] = {
    "literature": add_literature,
    "docs": add_docs,
    "notes": add_notes,
}
