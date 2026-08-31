# boepie/cli.py
"""Command-line interface for boepie."""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import logging
import os
import shutil
import tarfile
import textwrap
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import tomlkit
# rich_click is a drop-in for click that renders --help through rich, so every
# `click.option`/`click.argument` below is the real click decorator and only
# the help formatting changes. Imported under the name `click` because that is
# what it is: swapping the alias back is the whole uninstall.
import rich_click as click
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.tree import Tree

from boepie import __version__, settings
from boepie import _display as display
from boepie._display import CliError, console
from boepie.config import (
    CORPUS_EXTRA_FILE_TYPES,
    CORPUS_KEEP_ORIGINAL,
    CORPUS_WARN_ON_DOTFILE_TITLE,
    DEFAULT_MODE,
    DEFAULT_SNIPPET,
    DEFAULT_TOP_K,
    DOCS_DIR,
    EMBEDDING_BINDING,
    EMBEDDING_MODEL,
    INDEX_DIR,
    LITERATURE_DIR,
    LITERATURE_FETCH_DELAY,
    MINERU_BACKEND,
    MINERU_BATCH_SIZE,
    MINERU_DEVICE_MODE,
    MINERU_MODEL_SOURCE,
    NOTES_DIR,
)
from boepie.context import (
    append_agents_pointer,
    apply_bundle,
    bundle_status,
    fetch_content,
    find_bundle,
    index_root_for,
    init_bundle,
    list_source_local_files,
    reset_bundle,
    resolve_content_source,
)
from boepie.corpus import collection_index, sync_docs, sync_literature
from boepie.corpus.add import (
    AddOptions,
    AddOutcome,
    add_docs,
    add_literature,
    add_notes,
)
from boepie.corpus.document import move_leaf_document, read_document
from boepie.corpus.inputs import InputError
from boepie.corpus.intake import IntakeError
from boepie.corpus.layout import (
    full_title_filename,
    lookup_path,
    unique_document_name,
)
from boepie.corpus.schema import KEY_FIELDS as _CORPUS_KEY_FIELDS
from boepie.docs import DocsProject
from boepie.docs import load_manifest as load_docs_manifest
from boepie.literature import ArxivPaper
from boepie.literature import load_manifest as load_literature_manifest
from boepie.rag import (
    ContextLoader,
    DocsLoader,
    EmptyCollectionError,
    StaleIndexError,
    index_freshness,
    LiteratureLoader,
    ModelBinding,
    NotesLoader,
    build,
    default_embedding_binding,
    embedding_options,
    index_id_for,
    search,
)
from boepie.rag import read as rag_read
from boepie.rag.models import Filter, SearchResult
from boepie.release import (
    download_verified_asset,
)
from boepie.tools._retrieval import (
    VIEWS,
    format_hits,
    format_merged_hits,
    format_span,
    one_line,
    relative_source,
    search_with_lexical_fallback,
    with_note,
)

class _PlainErrorFormatter(click.RichHelpFormatter):
    """rich-click's help formatter, with boepie's own aborts left plain.

    rich-click renders every ClickException as a bordered panel and word-wraps
    the message inside it, bypassing `ClickException.show` entirely. boepie's
    abort messages routinely end by naming the command that fixes the problem,
    and a wrap splits that across a border into something that cannot be
    copied - the exact regression `CliError.show` exists to prevent. Usage
    errors rich-click raises itself (unknown option, missing argument) keep
    the panel, where it is genuinely clearer.
    """

    def write_error(self, error: click.ClickException) -> None:
        if isinstance(error, CliError):
            error.show()
            return
        super().write_error(error)


# Set on the context class rather than on the group: rich-click builds the
# error formatter from whichever context is current, which for any failure
# below the top level is the subcommand's own.
click.RichContext.formatter_class = _PlainErrorFormatter

# rich-click's own defaults already name only the eight standard ANSI colours,
# which is boepie's rule for the same reason (see `_display.THEME`): they
# resolve against the user's own palette and stay legible on a light
# background. Three deviations:
# - `max_width`, because help text set to the full width of a wide terminal
#   is a paragraph a screen and a half long and unreadable as prose.
# - `style_option`/`style_argument` plain cyan rather than bold cyan, which
#   `_display` reserves for a *suggested command* - the one thing in boepie's
#   output a reader is meant to copy and run.
# - `style_metavar` dim rather than bold yellow: yellow is the warning colour
#   everywhere else, and a help screen full of it reads as a page of alarms.
_HELP_CONFIG = click.RichHelpConfiguration(
    max_width=100,
    style_option="cyan",
    style_argument="cyan",
    style_metavar="dim",
)


# Maps a collection name to the loader that builds it.
# Context loader requires bundle_dir passed to __init__, so it's not here.
_LOADERS = {"literature": LiteratureLoader, "docs": DocsLoader, "notes": NotesLoader}

# Threshold for hint search results, on the *raw BM25* score of the top hit
# (hint is BM25-only; see `_hint_search`). Placeholder value - the dummy
# content in this repo has no real distribution to calibrate against, so this
# needs tuning once a real corpus exists. See design/phase-2.md item 3.
_HINT_MIN_SCORE = 1.0

# The one collection whose index is per-project rather than machine-global:
# it is built from a `.boepie/` bundle, so it lives inside that bundle (see
# `boepie.context.index_root_for`). Two projects sharing INDEX_DIR for it
# would silently clobber each other's index.
_CONTEXT_COLLECTION = "context"

# Which collections each verb can address. `context` is the odd one out
# everywhere: it is per-project (its index lives inside the `.boepie/` bundle
# rather than the machine-global store), BM25-only, and has no corpus on disk
# and no `read_*` counterpart - so it is buildable and searchable but never
# fetchable, listable or readable.
_CORPUS_COLLECTIONS = ("literature", "docs", "notes")
_BUILD_COLLECTIONS = (*_CORPUS_COLLECTIONS, _CONTEXT_COLLECTION)
_SEARCH_COLLECTIONS = _BUILD_COLLECTIONS
# No read_context tool: a context hit's `source:` line is its handle, and the
# agent opens that file itself.
_READ_COLLECTIONS = _CORPUS_COLLECTIONS
# Only these two have a packaged manifest for `fetch` to reconcile against.
_FETCH_COLLECTIONS = ("literature", "docs")

# The token that selects everything a given command can address.
_ALL = "all"


class CollectionList(click.ParamType):
    """A comma-separated list of collection names, or `all`.

    Comma rather than a repeatable flag because the collection names are a
    closed set of short identifiers that can never themselves contain a
    comma - the case where an in-band separator costs nothing and saves the
    caller three flags. Resolves to the declared order regardless of how the
    list was typed, so output ordering never depends on the spelling.
    """

    name = "collections"

    def __init__(self, choices: tuple[str, ...]) -> None:
        self.choices = choices

    def get_metavar(self, *_args: object, **_kwargs: object) -> str:
        return f"[{'|'.join((*self.choices, _ALL))}]"

    def convert(self, value, param, ctx) -> tuple[str, ...]:
        if isinstance(value, tuple):
            return value
        selected: list[str] = []
        for part in str(value).split(","):
            name = part.strip()
            if not name:
                continue
            if name == _ALL:
                selected.extend(self.choices)
            elif name in self.choices:
                selected.append(name)
            else:
                self.fail(
                    f"'{name}' is not one of {', '.join(self.choices)}, or '{_ALL}'.",
                    param,
                    ctx,
                )
        if not selected:
            self.fail("name at least one collection.", param, ctx)
        chosen = set(selected)
        return tuple(name for name in self.choices if name in chosen)


# Every `status` command reports the same rows for each of several things, so
# they are laid out as a label column with the values lined up beside it: a
# reader scans down one column of numbers instead of hunting for them inside
# differently worded sentences. The label carries the severity colour, which
# makes that same scan answer "is anything wrong" without reading a word.
_STATUS_LABEL_WIDTH = 11


def _status_label(label: str) -> str:
    """A status row's label, padded so every row's value starts in one column.

    `_line` puts a single space after a `lead`, so padding to one less than
    the column width lands the value exactly on it.
    """
    return f"{label}:".ljust(_STATUS_LABEL_WIDTH - 1)


# Where a status row's value starts, for the continuation lines of a wrapped
# value and for any extra line that belongs to the row above it.
_STATUS_VALUE_INDENT = " " * (2 + _STATUS_LABEL_WIDTH)


def _wrap_into_value_column(names: list[str]) -> list[str]:
    """`names` as a comma-separated list, wrapped to the value column's width.

    Wrapped here rather than left to rich because these lists run long - 17
    citekeys is three terminal lines - and rich restarts each continuation at
    column zero, where it collides with the next heading and the block stops
    reading as one value.
    """
    width = max(console.width - (2 + _STATUS_LABEL_WIDTH), 24)
    return textwrap.wrap(", ".join(names), width=width)


def _status_items(names: list[str]) -> None:
    """The names behind a status count, below the row that counted them.

    Dim, because the count and the label above are the message; the names are
    the reference detail.
    """
    for line in _wrap_into_value_column(names):
        display.muted(line, indent=_STATUS_VALUE_INDENT)


def _status_list(label: str, names: list[str]) -> None:
    """A status row whose value *is* the list, starting on the label's line.

    The alternative - a count row with the names underneath - restates itself
    whenever the count is one, which is the common case for index ids.
    """
    lines = _wrap_into_value_column(names)
    display.muted(lines[0], lead=_status_label(label), indent="  ")
    for line in lines[1:]:
        display.muted(line, indent=_STATUS_VALUE_INDENT)


def _index_root_for_collection(collection: str) -> Path:
    """Where `collection`'s index lives: inside the bundle governing the cwd
    for `context`, the machine-global store for every other collection."""
    if collection != _CONTEXT_COLLECTION:
        return INDEX_DIR
    bundle_dir = find_bundle()
    if bundle_dir is None:
        raise _no_bundle_error()
    return index_root_for(bundle_dir)


def _set_verbosity(verbose: bool) -> None:
    """Quiet by default; --verbose opts back in for watching a slow build.

    bm25s logs an INFO line per index operation, so pin its level explicitly
    alongside root's.
    """
    level = logging.INFO if verbose else logging.WARNING
    logging.getLogger("bm25s").setLevel(level)
    logging.getLogger().setLevel(level)


def _run(coro):
    """Run an async CLI action, turning failures into a short message instead
    of a raw asyncio traceback (Ctrl-C and connection errors both land here)."""
    try:
        return asyncio.run(coro)
    except KeyboardInterrupt:
        raise CliError("cancelled.") from None
    except (EmptyCollectionError, StaleIndexError):
        # Both are ValueError subclasses that a caller sweeping several
        # collections needs to tell apart from an ordinary failure - one to
        # skip on, one to stop on - and flattening either into CliError here
        # would hide it.
        raise
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        raise CliError(str(error)) from error


@click.group()
@click.rich_config(help_config=_HELP_CONFIG)
@click.version_option(version=__version__, prog_name="boepie")
def cli() -> None:
    """Boepie - MCP server for AI-assisted stimela pipeline creation."""


@cli.command()
def serve() -> None:
    """Start the boepie MCP server (stdio transport)."""
    from boepie.server import mcp

    mcp.run("stdio")


# ---------------------------------------------------------------------------
# Index management: build, fetch, status, list
# ---------------------------------------------------------------------------


@cli.group()
def index() -> None:
    """Manage search indices (build, fetch, status, list)."""


@index.command("build")
@click.option(
    "--collection",
    "collections",
    default=_ALL,
    show_default=True,
    type=CollectionList(_BUILD_COLLECTIONS),
    help="Comma-separated collections to build, or 'all'.",
)
@embedding_options
@click.option(
    "--embedding-concurrency",
    default=None,
    type=int,
    help="Max concurrent embedding requests (default: 4). Lower this if you're hitting API rate limits.",
)
@click.option(
    "--index-name",
    default=None,
    help="Override the auto-derived index id (default: <binding>-<model>).",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    help="Show per-batch progress logging (useful for slow builds).",
)
def index_build(
    collections: tuple[str, ...],
    resolve_embedding,
    embedding_concurrency: int | None,
    index_name: str | None,
    verbose: bool,
) -> None:
    """Build the search index for one or more collections.

    With no --collection this builds every collection that has something to
    index, reporting the ones it skipped rather than failing on them: an
    empty corpus is a normal state, not an error, when you did not name it.
    Naming a collection explicitly does make an empty one an error - you
    asked for that index specifically.

    Needs nothing running by default: fastembed runs a small ONNX model
    locally on CPU (one-time model download, then fully offline). Use
    --embedding-binding=ollama for a local Ollama daemon, or
    --embedding-binding=openai for an OpenAI API key or a local
    OpenAI-compatible server (vLLM/SGLang/TGI) via --embedding-host=<url>.
    """
    _set_verbosity(verbose)
    if index_name is not None and len(collections) > 1:
        raise CliError(
            "--index-name names one index, so it cannot be combined with "
            "several collections. Build them one at a time, or drop the flag."
        )
    # An explicit single collection is a request for that index; anything
    # broader is a sweep, where "nothing to index" is a skip, not a failure.
    sweeping = len(collections) > 1

    built = 0
    for collection in collections:
        try:
            _build_one(
                collection,
                resolve_embedding=resolve_embedding,
                embedding_concurrency=embedding_concurrency,
                index_name=index_name,
            )
        except EmptyCollectionError as error:
            if not sweeping:
                raise CliError(str(error)) from error
            display.muted(f"nothing to index in '{collection}'", indent="  ")
            continue
        except _NoBundleError:
            if not sweeping:
                raise _no_bundle_error() from None
            display.muted("no .boepie/ bundle here, skipping 'context'", indent="  ")
            continue
        built += 1

    if sweeping:
        display.heading(
            f"of {len(collections)} collection(s) indexed.", lead=f"\n{built}"
        )


class _NoBundleError(Exception):
    """`context` was selected but no `.boepie/` bundle governs the cwd."""


def _build_one(
    collection: str,
    *,
    resolve_embedding,
    embedding_concurrency: int | None,
    index_name: str | None,
) -> None:
    """Build one collection's index, with a progress bar over its chunks."""
    if collection == _CONTEXT_COLLECTION:
        # Per-project and BM25-only: its index belongs inside the bundle it
        # was built from, not the machine-global store.
        bundle_dir = find_bundle()
        if bundle_dir is None:
            raise _NoBundleError
        _build_context_index(bundle_dir)
        return

    loader = _LOADERS[collection]()
    embedding = resolve_embedding(max_async=embedding_concurrency)

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        # total is unknown until build() finishes chunking every document and
        # reports the real count via its first on_progress(0, total) call -
        # until then this renders as an indeterminate spinner.
        task_id = progress.add_task(
            f"Indexing '{collection}' ({embedding.kind}:{embedding.model} embeddings)",
            total=None,
        )

        def on_progress(done: int, total: int) -> None:
            progress.update(task_id, completed=done, total=total)

        manifest = _run(
            build(
                loader,
                index_root=INDEX_DIR,
                embedding=embedding,
                index_id=index_name,
                on_progress=on_progress,
            )
        )
    display.success(
        f"{manifest.count} chunks into '{collection}/{manifest.index_id}' "
        f"(embedding={manifest.embedding_kind}:{manifest.embedding_model}).",
        lead="Indexed",
    )


@index.command("fetch")
@click.option("--collection", default="docs", show_default=True)
@click.option(
    "--tag",
    default="latest",
    show_default=True,
    help="GitHub release tag to fetch from.",
)
@click.option(
    "--index-name",
    default=None,
    help="Which built index id to fetch (default: derived from the active embedding config).",
)
@click.option(
    "--force", is_flag=True, help="Re-download and overwrite even if already present."
)
@click.option(
    "--embedding-binding",
    default=EMBEDDING_BINDING,
    show_default=True,
    type=click.Choice(["fastembed", "ollama", "openai"]),
)
@click.option("--embedding-model", default=EMBEDDING_MODEL, show_default=True)
def index_fetch(
    collection: str,
    tag: str,
    index_name: str | None,
    force: bool,
    embedding_binding: str,
    embedding_model: str,
) -> None:
    """Download a prebuilt index from a GitHub release.

    End-user path: no LLM needed, just an embedding backend matching the
    fetched index (checked lazily by `boepie search` / `search_literature`,
    warned about eagerly here too). The `context` collection is not
    fetchable: it is derived from a project's own bundle, never shipped.
    Neither is `literature`: boepie does not redistribute paper text, even in
    built-index form - run `boepie corpus fetch --collection literature` then
    `boepie index build --collection literature` instead.
    """
    if collection == _CONTEXT_COLLECTION:
        raise CliError(
            "the context index is built from a project's own .boepie/ bundle, "
            "not published as a release asset. Run 'boepie context apply'."
        )
    if collection == "literature":
        raise CliError(
            "boepie does not publish a prebuilt literature index (that would "
            "redistribute paper text). Run 'boepie corpus fetch --collection "
            "literature' then 'boepie index build --collection literature' instead."
        )
    if collection == "notes":
        raise CliError(
            "notes are entirely user content and never published. Run "
            "'boepie corpus add notes <identifier>' then "
            "'boepie index build --collection notes' instead."
        )

    embedding = ModelBinding(
        kind=embedding_binding, model=embedding_model
    )  # pyright: ignore[reportArgumentType]
    resolved_name = index_name or index_id_for(embedding)
    asset = f"{collection}-{resolved_name}.tar.gz"

    dest = INDEX_DIR / collection / resolved_name
    if dest.exists() and not force:
        display.warning(
            f"Index '{collection}/{resolved_name}' already present at "
            f"{dest} - skipped fetching tag={tag} (use --force to re-download)."
        )
        return

    with console.status(f"Downloading {asset} from boepie release {tag}..."):
        try:
            data = download_verified_asset(tag, asset)
        except ValueError as error:
            display.error(str(error))
            raise SystemExit(1) from error

    (INDEX_DIR / collection).mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        tar.extractall(INDEX_DIR / collection, filter="data")

    manifest_path = dest / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("embedding_kind") != embedding.kind
            or manifest.get("embedding_model") != embedding.model
        ):
            display.warning(
                f"fetched index was built with embedding "
                f"'{manifest.get('embedding_kind')}:{manifest.get('embedding_model')}' but "
                f"the active config is '{embedding.kind}:{embedding.model}'. Querying will "
                f"fail until these match - see BOEPIE_EMBEDDING_BINDING/BOEPIE_EMBEDDING_MODEL."
            )

    latest_path = INDEX_DIR / collection / "latest.json"
    latest_path.write_text(
        json.dumps({"index_id": resolved_name}, indent=2), encoding="utf-8"
    )

    display.success(
        f"'{collection}/{resolved_name}' (tag={tag}) into {dest}.", lead="Fetched"
    )


# How each freshness state reads, and how loudly. Only "stale" is a fault;
# the other three are facts about what can be checked, so they stay dim.
_FRESHNESS_WORDING: dict[str, str] = {
    "in step": "in step with its corpus",
    "corpus absent": "corpus not on this machine (nothing to check against)",
    "unrecorded": "not recorded (built before this was tracked)",
}


def _report_freshness(collection: str, manifest: dict[str, Any]) -> None:
    """Say whether an index still matches the corpus it was built over.

    The failure this reports used to be undiagnosable from `status`: an index
    built over an older corpus answers queries perfectly happily, with
    plausible scores, pointing at text that has since changed. One line here
    is what turns that into a thing you can see before it misleads you.
    """
    freshness = index_freshness(manifest.get("built_from"), collection)
    if freshness.state != "stale":
        display.muted(
            _FRESHNESS_WORDING[freshness.state],
            lead=_status_label("corpus"),
            indent="  ",
        )
        return

    counts = ", ".join(
        part
        for part in (
            f"{freshness.changed} changed" if freshness.changed else "",
            f"{freshness.gone} gone" if freshness.gone else "",
        )
        if part
    )
    display.warning(
        f"stale - of {freshness.document_count} document(s), {counts}",
        lead=_status_label("corpus"),
        indent="  ",
    )
    # The fix goes on its own line in the value column rather than trailing
    # the row: rich would break `boepie index build --collection notes` across
    # a line end, which is the exact wrap CliError exists to avoid elsewhere.
    display.next_step(
        f"boepie index build --collection {collection}",
        indent=_STATUS_VALUE_INDENT,
    )


@index.command("status")
def index_status() -> None:
    """Report the status of all indices under the index directory.

    For each collection, shows the active index id and whether its embedding
    config matches the active embedding environment.
    """
    if not INDEX_DIR.exists():
        display.warning(f"No index directory yet at {INDEX_DIR}")
        return

    collections_with_indices: dict[str, list[str]] = {}
    for collection_dir in INDEX_DIR.iterdir():
        if not collection_dir.is_dir():
            continue
        index_ids: list[str] = []
        for item in collection_dir.iterdir():
            if item.is_dir() and item.name != ".":
                index_ids.append(item.name)
        if index_ids:
            collections_with_indices[collection_dir.name] = sorted(index_ids)

    if not collections_with_indices:
        display.warning("No indices built or fetched yet.")
        return

    for position, (collection_name, index_ids) in enumerate(
        sorted(collections_with_indices.items())
    ):
        latest_link = INDEX_DIR / collection_name / "latest.json"
        if latest_link.exists():
            latest_data = json.loads(latest_link.read_text(encoding="utf-8"))
            active_id = latest_data.get("index_id", "unknown")
        else:
            active_id = "none"

        if position:
            console.print()
        console.print(
            display.collection_root(collection_name, INDEX_DIR / collection_name),
            soft_wrap=True,
        )

        # An index with nothing pointing at it cannot be searched, so "none"
        # is the one value here that is a problem rather than a fact.
        report = display.muted if active_id != "none" else display.warning
        report(active_id, lead=_status_label("active"), indent="  ")

        if active_id != "none" and active_id in index_ids:
            manifest_path = INDEX_DIR / collection_name / active_id / "manifest.json"
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                embedding_kind = manifest.get("embedding_kind")
                embedding_model = manifest.get("embedding_model")
                display.muted(
                    f"{embedding_kind}:{embedding_model}",
                    lead=_status_label("embedding"),
                    indent="  ",
                )
                _report_freshness(collection_name, manifest)

        # Only what you could switch *to*: repeating the active id under
        # "available" is the one-element case saying nothing twice.
        alternatives = [index_id for index_id in index_ids if index_id != active_id]
        if alternatives:
            _status_list("others", alternatives)


@index.command("list")
def index_list() -> None:
    """Enumerate all built and fetched indices on this machine.

    Shows collection names and their available index ids.
    """
    if not INDEX_DIR.exists():
        display.warning(f"No index directory yet at {INDEX_DIR}")
        return

    listed: dict[str, list[str]] = {}
    for collection_dir in sorted(INDEX_DIR.iterdir()):
        if not collection_dir.is_dir():
            continue
        index_ids = sorted(
            item.name
            for item in collection_dir.iterdir()
            if item.is_dir() and item.name not in (".", "..")
        )
        if index_ids:
            listed[collection_dir.name] = index_ids

    if not listed:
        display.warning("No indices found.")
        return

    # Aligned on the longest collection name rather than on the fixed status
    # column: here the label is the name, not one of a known set of rows.
    label_width = max(len(name) for name in listed) + 2
    for collection_name, index_ids in listed.items():
        display.muted(
            ", ".join(index_ids), lead=f"{collection_name}:".ljust(label_width - 1)
        )


# ---------------------------------------------------------------------------
# Corpus: add, fetch, status, list - literature/docs/notes, built on this
# machine, unified around boepie.corpus's shared layout (see boepie.corpus
# for the on-disk shape: directory-as-group, full-title filenames, a
# surrogate `id`, `managed_by: boepie | user` provenance).
# ---------------------------------------------------------------------------
#
# No literature Markdown or built index is ever published by boepie (see
# boepie.literature.fetch): `corpus fetch --collection literature` pulls each
# manifest paper's HTML straight from arxiv.org/ar5iv.labs.arxiv.org and
# converts it locally, so the only thing boepie itself ships is the small
# bibliographic manifest. Papers with no arXiv presence (pre-arXiv-era, or
# never preprinted) fall to the BYO-PDF path: `corpus add literature
# <file.pdf>` against a copy you supply, converted with MinerU.


def _corpus_collection_dir(collection: str) -> Path:
    """`collection`'s on-disk root, looked up by name at call time (not
    precomputed into a module-level dict) so a test's
    `monkeypatch.setattr(cli, "LITERATURE_DIR", ...)`-style override - or,
    in principle, any other runtime change to these globals - is honoured
    here exactly as it already is by every other command that references
    LITERATURE_DIR/DOCS_DIR/NOTES_DIR directly."""
    return {"literature": LITERATURE_DIR, "docs": DOCS_DIR, "notes": NOTES_DIR}[
        collection
    ]


@cli.group()
def corpus() -> None:
    """Manage the literature/docs/notes corpora (add, fetch, status, list)."""


# `add` writes documents immediately and always as `managed_by: user`; it
# never stages anything for a later `fetch`. The three subcommands share one
# core (`boepie.corpus.add`) and differ only in which identifier resolvers run
# first and which frontmatter block they write. Kept as subcommands rather
# than one command behind `--collection` because the same input means
# different things per collection: a URL is one page for notes, a whole site
# for docs.


def _add_options(function):
    """Options every `corpus add` subcommand shares."""
    function = click.option(
        "--title",
        default=None,
        help="Override the derived title. With several identifiers this "
        "applies to each, so it is usually only useful for one.",
    )(function)
    function = click.option(
        "--group",
        default=None,
        metavar="PATH",
        help="Place the document inside a group, e.g. calibration/subtopic.",
    )(function)
    function = click.option(
        "--keep-original/--no-keep-original",
        default=None,
        help="Retain the source bytes alongside the Markdown "
        f"(default: corpus.keep_original).",
    )(function)
    return function


def _build_add_options(**overrides) -> AddOptions:
    """Merge CLI overrides onto the configured defaults."""
    keep_original = overrides.pop("keep_original", None)
    return AddOptions(
        keep_original=CORPUS_KEEP_ORIGINAL if keep_original is None else keep_original,
        extra_file_types=tuple(CORPUS_EXTRA_FILE_TYPES),
        mineru_device_mode=MINERU_DEVICE_MODE,
        mineru_backend=MINERU_BACKEND,
        mineru_model_source=MINERU_MODEL_SOURCE,
        mineru_batch_size=MINERU_BATCH_SIZE,
        **overrides,
    )


def _converting_with_mineru(documents: int, number: int, total: int) -> None:
    """Say what MinerU is about to spend minutes on.

    Per run rather than per document, because MinerU writes nothing until a
    whole run has finished - there is no per-document moment to report.
    """
    run = f" (run {number} of {total})" if total > 1 else ""
    display.info(
        f"{documents} document(s) with MinerU{run} - this takes a few minutes",
        lead="converting",
    )


def _add_and_report(collection: str, adder: Callable[[], list[AddOutcome]]) -> None:
    """Run one `corpus add` and report it.

    Both exceptions mean the batch as typed cannot be carried out at all - an
    argument naming nothing, or a converter that is not installed - rather
    than one identifier having failed, so they are caught here and given
    click's own error wording instead of a traceback.
    """
    try:
        outcomes = adder()
    except (InputError, IntakeError) as error:
        raise CliError(str(error)) from error
    _report_add(collection, outcomes)


def _report_add(collection: str, outcomes: list[AddOutcome]) -> None:
    """One line per identifier, then a single summary and next step.

    Printed per batch rather than per item: adding is meant to be staged like
    commits, several at a time, with one index build at the end.
    """
    added = [outcome for outcome in outcomes if outcome.status == "added"]
    duplicates = [outcome for outcome in outcomes if outcome.status == "duplicate"]
    failures = [outcome for outcome in outcomes if outcome.status == "failed"]
    skipped = [outcome for outcome in outcomes if outcome.status == "skipped"]

    for outcome in outcomes:
        if outcome.status == "added":
            via = f" via {outcome.via}" if outcome.via else ""
            detail = f" ({outcome.detail})" if outcome.detail else ""
            display.success(
                (
                    f"{outcome.title} (id={outcome.document_id}{via}){detail}"
                    if outcome.document_id
                    else f"{outcome.title}{detail}"
                ),
                lead="added",
            )
            if outcome.notice and CORPUS_WARN_ON_DOTFILE_TITLE:
                display.warning(
                    f"{outcome.notice}. Pass --title to control this, or set "
                    f"corpus.warn_on_dotfile_title=false.",
                    lead="note:",
                    indent="  ",
                )
        elif outcome.status == "duplicate":
            display.warning(
                f"{outcome.identifier} - {outcome.detail} (id={outcome.document_id})",
                lead="duplicate",
            )
        elif outcome.status == "skipped":
            # Quieter than a duplicate: nothing is wrong, and a folder walk can
            # produce a great many of these at once.
            display.muted(f"{outcome.identifier} - {outcome.detail}", lead="skipped")
        else:
            display.error(f"{outcome.identifier} - {outcome.detail}", lead="failed")

    # Skips are counted only when there are any - on a hand-typed batch the
    # count is always zero and would just be one more number to read past.
    tail = f", {len(skipped)} skipped" if skipped else ""
    display.heading(
        f"{len(duplicates)} already present, {len(failures)} failed{tail}.",
        lead=f"{len(added)} added,",
        indent="\n",
    )
    if added:
        display.next_step(
            f"boepie index build --collection {collection}",
            note="(once you have finished adding)",
        )
    # Failures only. A skip is a deliberate decision not to take something, so
    # a folder holding one unreadable file still exits 0.
    if failures:
        raise SystemExit(1)


@corpus.group("add")
def corpus_add() -> None:
    """Add documents to a corpus collection, immediately.

    Everything `add` writes is yours (`managed_by: user`) and is never
    touched by `corpus fetch`, which only reconciles boepie's own packaged
    manifest. Every subcommand accepts several identifiers at once; run
    `boepie index build` once when you have finished adding.
    """


@corpus_add.command("literature")
@click.argument("identifiers", nargs=-1, required=True)
@click.option("--citekey", default=None, help="Override the derived citekey.")
@_add_options
def corpus_add_literature(
    identifiers: tuple[str, ...],
    citekey: str | None,
    title: str | None,
    group: str | None,
    keep_original: bool | None,
) -> None:
    """Add papers by arXiv id, DOI, .bib file, PDF, or URL.

    An arXiv id is understood in any of its spellings - bare, versioned,
    `arXiv:`-prefixed, or as an abs/pdf URL - and its HTML is fetched and
    converted on this machine. A DOI is resolved to an arXiv preprint where
    one exists. A `.bib` file expands into all of its entries, following each
    one's arXiv id, DOI, or `file` path in turn: exporting from Zotero and
    adding the `.bib` is the best-supported way to bring in your own library.
    PDFs and other documents are converted with MinerU.
    """
    options = _build_add_options(
        title=title, group=group, keep_original=keep_original, citekey=citekey
    )
    _add_and_report(
        "literature",
        lambda: add_literature(
            LITERATURE_DIR, identifiers, options, on_batch=_converting_with_mineru
        ),
    )


@corpus_add.command("docs")
@click.argument("identifiers", nargs=-1, required=True)
@click.option(
    "--project",
    required=True,
    help="Project name: the group these pages live under, and what "
    "search_docs filters on.",
)
@_add_options
def corpus_add_docs(
    identifiers: tuple[str, ...],
    project: str,
    title: str | None,
    group: str | None,
    keep_original: bool | None,
) -> None:
    """Add documentation from a site URL or a local file.

    A URL crawls the whole site, not just the page you name - that is what
    separates this from `add notes`, which converts a single page. Sphinx
    sites are detected and read through their own object inventory; anything
    else is crawled generically.
    """
    options = _build_add_options(
        title=title, group=group, keep_original=keep_original, project=project
    )
    _add_and_report(
        "docs",
        lambda: add_docs(
            DOCS_DIR, identifiers, options, on_batch=_converting_with_mineru
        ),
    )


@corpus_add.command("notes")
@click.argument("identifiers", nargs=-1, required=True)
@_add_options
def corpus_add_notes(
    identifiers: tuple[str, ...],
    title: str | None,
    group: str | None,
    keep_original: bool | None,
) -> None:
    """Add your own files or web pages to the notes corpus.

    The base case: local files of any supported format (Markdown, text,
    source code, PDF, DOCX, PPTX, XLSX) and http(s) URLs, which are converted
    to Markdown one page at a time. Notes are machine-global, separate from a
    project's `.boepie/` bundle.
    """
    options = _build_add_options(title=title, group=group, keep_original=keep_original)
    _add_and_report(
        "notes",
        lambda: add_notes(
            NOTES_DIR, identifiers, options, on_batch=_converting_with_mineru
        ),
    )


@corpus.command("remove")
@click.option(
    "--collection", required=True, type=click.Choice(["literature", "docs", "notes"])
)
@click.argument("document_ids", nargs=-1, required=True)
@click.option("--yes", is_flag=True, help="Skip the confirmation prompt.")
def corpus_remove(collection: str, document_ids: tuple[str, ...], yes: bool) -> None:
    """Delete documents from a collection by id.

    The only way out of a corpus: with no user manifest to edit, removing an
    entry and re-running `fetch` is no longer a deletion path. A
    `managed_by: boepie` document can be removed too, but `corpus fetch` will
    restore it while its manifest entry stands.
    """
    collection_dir = _corpus_collection_dir(collection)
    documents = {
        document.id: document
        for document in collection_index(
            collection_dir,
            collection=collection,
            key_fields=_CORPUS_KEY_FIELDS[collection],
        )
    }

    targets = []
    for document_id in document_ids:
        document = documents.get(document_id)
        if document is None:
            raise _no_such_document_error(document_id, (collection,))
        targets.append(document)

    for document in targets:
        title = document.frontmatter.get("title", document.id)
        display.info(f"{title} (id={document.id})", indent="  ")
    if not yes:
        click.confirm(f"Delete {len(targets)} document(s)?", abort=True)

    for document in targets:
        if document.wrapper_dir is not None:
            shutil.rmtree(document.wrapper_dir)
        else:
            document.md_path.unlink()

    display.success(f"{len(targets)} document(s).", lead="Removed")
    display.next_step(f"boepie index build --collection {collection}")


@corpus.command("fetch")
@click.option(
    "--collection",
    "collections",
    default=",".join(_FETCH_COLLECTIONS),
    show_default=True,
    type=CollectionList(_CORPUS_COLLECTIONS),
    help="Comma-separated collections to reconcile, or 'all'.",
)
@click.option(
    "--force",
    "force_targets",
    multiple=True,
    metavar="PATH",
    help="Re-fetch/regenerate this boepie-managed document even though it's "
    "unchanged (collection-relative path; repeatable).",
)
@click.option(
    "--delay",
    default=None,
    type=float,
    help="Seconds between fetches (default: a collection-specific politeness delay).",
)
@click.option("-v", "--verbose", is_flag=True, help="Show progress per item.")
def corpus_fetch(
    collections: tuple[str, ...],
    force_targets: tuple[str, ...],
    delay: float | None,
    verbose: bool,
) -> None:
    """Converge a manifest-backed corpus with what's on disk: add anything
    missing, skip anything already present, re-fetch anything named by
    --force, and delete anything boepie-managed whose manifest entry is gone.

    Runs entirely on this machine (arXiv HTML for literature, each site's own
    pages for docs) - no marker/OCR pass, nothing downloaded from a boepie
    release. `managed_by: user` documents are never touched. Run `boepie index
    build --collection <collection>` afterward to index what changed.
    """
    _set_verbosity(verbose)
    for collection in collections:
        _corpus_fetch_one(collection, force_targets, delay, verbose)


def _corpus_fetch_one(
    collection: str, force_targets: tuple[str, ...], delay: float | None, verbose: bool
) -> None:
    if collection == "notes":
        # Accepted rather than rejected as an invalid choice: "notes is not
        # one of literature, docs" says nothing about why, and the reason is
        # worth stating - notes exist only because you added them.
        display.warning(
            "Notes have no packaged manifest to reconcile against - every note "
            "is one you added.",
            lead="Nothing to fetch.",
        )
        display.info("Add one with: boepie corpus add notes <file-or-url>")
        return
    try:
        if collection == "literature":
            _corpus_fetch_literature(force_targets, delay, verbose)
        else:
            _corpus_fetch_docs(force_targets, delay, verbose)
    except ValueError as error:
        raise CliError(str(error)) from error
    except KeyboardInterrupt:
        display.warning(
            f"Documents already written are kept. Re-run the same command to "
            f"carry on from where it stopped.",
            lead="Interrupted.",
        )
        raise SystemExit(130) from None


def _corpus_fetch_literature(
    force_targets: tuple[str, ...], delay: float | None, verbose: bool
) -> None:
    papers = load_literature_manifest(LITERATURE_DIR)
    if not papers:
        display.warning("No papers in the literature manifest.")
        return

    with _fetch_progress(
        f"Fetching {len(papers)} paper(s) from arXiv", len(papers), verbose
    ) as advance:

        def on_progress(paper: ArxivPaper | None, result) -> None:
            advance()
            if not verbose:
                return
            if result.action == "unavailable":
                display.warning(
                    f"{result.citekey} (arXiv:{paper.arxiv_id if paper else '?'})",
                    lead="unavailable",
                )
            else:
                display.success(result.citekey, lead=result.action)

        results = sync_literature(
            LITERATURE_DIR,
            papers,
            force_paths=force_targets,
            delay=delay if delay is not None else LITERATURE_FETCH_DELAY,
            on_progress=on_progress,
        )

    added = sum(1 for r in results if r.action == "added")
    refetched = sum(1 for r in results if r.action == "refetched")
    skipped = sum(1 for r in results if r.action == "skipped")
    deleted = sum(1 for r in results if r.action == "deleted")
    unavailable = [r for r in results if r.action == "unavailable"]

    console.print(
        display.collection_root("literature", LITERATURE_DIR), soft_wrap=True
    )
    display.success(
        f"{added} added, {refetched} refetched, "
        f"{skipped} skipped, {deleted} deleted",
        lead=_status_label("fetched"),
        indent="  ",
    )
    if unavailable:
        display.warning(
            f"{len(unavailable)} paper(s) have no HTML rendering at arxiv.org "
            f"or ar5iv",
            lead=_status_label("no HTML"),
            indent="  ",
        )
        _status_items([result.citekey for result in unavailable])
        # In the value column, so it reads as part of the row above rather
        # than as the command's own closing advice - which the Next: line is.
        display.info(
            "supply the PDF: boepie corpus add literature <file.pdf>",
            indent=_STATUS_VALUE_INDENT,
        )
    display.next_step("boepie index build --collection literature", indent="  ")


def _corpus_fetch_docs(
    force_targets: tuple[str, ...], delay: float | None, verbose: bool
) -> None:
    projects = load_docs_manifest(DOCS_DIR)
    if not projects:
        display.warning("No projects in the docs manifest.")
        return

    with _fetch_progress(
        f"Fetching {len(projects)} docs project(s)", len(projects), verbose
    ) as advance:

        def on_progress(project: DocsProject | None, result) -> None:
            advance()
            if not verbose:
                return
            display.heading(
                f"{result.added} added, {result.refetched} refetched, "
                f"{result.skipped} skipped, {result.deleted} deleted "
                f"({len(result.failures)} failure(s)).",
                lead=f"{result.project}:",
            )

        results = sync_docs(
            DOCS_DIR,
            projects,
            force_paths=force_targets,
            delay=delay if delay is not None else 0.2,
            on_progress=on_progress,
        )

    total_added = sum(r.added for r in results)
    total_refetched = sum(r.refetched for r in results)
    total_skipped = sum(r.skipped for r in results)
    total_deleted = sum(r.deleted for r in results)
    total_failures = sum(len(r.failures) for r in results)

    console.print(display.collection_root("docs", DOCS_DIR), soft_wrap=True)
    display.success(
        f"{total_added} added, {total_refetched} refetched, "
        f"{total_skipped} skipped, {total_deleted} deleted "
        f"across {len(results)} project(s)",
        lead=_status_label("fetched"),
        indent="  ",
    )
    if total_failures:
        display.warning(
            f"{total_failures} page(s) could not be fetched",
            lead=_status_label("failures"),
            indent="  ",
        )
    display.next_step("boepie index build --collection docs", indent="  ")


@contextlib.contextmanager
def _fetch_progress(description: str, total: int, verbose: bool):
    """Yield an `advance()` to call once per fetched item.

    Suppressed under --verbose, which prints a line per item instead: a live
    progress bar and a stream of prints fight over the same terminal rows.
    """
    if verbose:
        yield lambda: None
        return
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task(description, total=total)
        yield lambda: progress.advance(task_id)


def _no_bundle_error() -> CliError:
    """No `.boepie/` bundle governs the working directory."""
    return CliError(
        f"no .boepie/ bundle found in {Path.cwd()} or any parent. "
        f"Run 'boepie context init'."
    )


def _no_such_document_error(document_id: str, collections: tuple[str, ...]) -> CliError:
    """That id is not in the corpus.

    One message wherever it is raised. The three call sites used to suggest
    three different ways to go looking (`corpus list`, `corpus tree`,
    `boepie search`), which made one failure read as three problems. The
    comma-separated selector means a single suggestion covers both the
    one-collection and the swept case.
    """
    where = ",".join(collections)
    return CliError(
        f"no document with id '{document_id}' in {where}. "
        f"Run 'boepie corpus list --collection {where}' to see what is there."
    )


def _corpus_documents(collection: str):
    try:
        return collection_index(
            _corpus_collection_dir(collection),
            collection=collection,
            key_fields=_CORPUS_KEY_FIELDS[collection],
        )
    except KeyError as error:
        raise CliError(
            f"a document in '{collection}' predates the current frontmatter "
            f"schema ({one_line(error.args[0])}). Run "
            f"'uv run scripts/migrate_corpus_layout.py' to bring the corpus "
            f"up to date."
        ) from error


def _managed_counts(documents) -> tuple[int, int]:
    """(boepie-managed, yours) - the split that decides what `fetch` may touch."""
    boepie_managed = sum(
        1
        for document in documents
        if document.frontmatter.get("managed_by") == "boepie"
    )
    return boepie_managed, len(documents) - boepie_managed


@corpus.command("status")
@click.option(
    "--collection",
    "collections",
    default=_ALL,
    show_default=True,
    type=CollectionList(_CORPUS_COLLECTIONS),
    help="Comma-separated collections, or 'all'.",
)
def corpus_status(collections: tuple[str, ...]) -> None:
    """Report what is in each collection and what `fetch` would change.

    Advisory only, like `context status`: never fetches or writes anything.
    Every collection reports the same three things - how much is boepie's,
    how much is yours, and what is out of step with the packaged manifest.
    """
    for position, collection in enumerate(collections):
        _corpus_status_one(collection, first=position == 0)


def _corpus_status_one(collection: str, *, first: bool) -> None:
    collection_dir = _corpus_collection_dir(collection)
    documents = _corpus_documents(collection)
    boepie_managed, user_managed = _managed_counts(documents)

    # A blank line between collections: three of these run together
    # otherwise, and the heading is the only thing separating them.
    if not first:
        console.print()
    # soft_wrap, or rich breaks a long corpus path mid-token across two lines.
    console.print(display.collection_root(collection, collection_dir), soft_wrap=True)
    display.muted(
        f"{len(documents)} total, {boepie_managed} boepie-managed, "
        f"{user_managed} yours",
        lead=_status_label("documents"),
        indent="  ",
    )

    if collection == "notes":
        # No manifest to diff against: notes are always yours.
        if not documents:
            display.next_step("boepie corpus add notes <file-or-url>", indent="  ")
        return

    if collection == "literature":
        entries = {
            paper.citekey: paper for paper in load_literature_manifest(LITERATURE_DIR)
        }
        present = {
            document.natural_key
            for document in documents
            if document.frontmatter.get("managed_by") == "boepie"
        }
        missing = sorted(set(entries) - present)
        orphaned = sorted(
            document.natural_key
            for document in documents
            if document.frontmatter.get("managed_by") == "boepie"
            and document.natural_key not in entries
        )
        label = "paper"
    else:
        projects = {project.project for project in load_docs_manifest(DOCS_DIR)}
        fetched_projects = {
            str(lookup_path(document.frontmatter, "docs.project"))
            for document in documents
            if document.frontmatter.get("managed_by") == "boepie"
        }
        missing = sorted(projects - fetched_projects)
        orphaned = sorted(fetched_projects - projects - {"None"})
        label = "project"

    if missing:
        display.warning(
            f"{len(missing)} {label}(s) in the manifest not fetched yet",
            lead=_status_label("missing"),
            indent="  ",
        )
        _status_items(missing)
    if orphaned:
        display.warning(
            f"{len(orphaned)} {label}(s) no longer in the manifest "
            f"(next fetch deletes them)",
            lead=_status_label("orphaned"),
            indent="  ",
        )
        _status_items(orphaned)
    if not missing and not orphaned:
        display.success(
            "in step with the packaged manifest",
            lead=_status_label("manifest"),
            indent="  ",
        )
    else:
        display.next_step(f"boepie corpus fetch --collection {collection}", indent="  ")


@corpus.command("list")
@click.option(
    "--collection",
    "collections",
    default=_ALL,
    show_default=True,
    type=CollectionList(_CORPUS_COLLECTIONS),
    help="Comma-separated collections, or 'all'.",
)
def corpus_list(collections: tuple[str, ...]) -> None:
    """Enumerate every document currently on disk, per collection."""
    for collection in collections:
        if len(collections) > 1:
            display.heading(collection, indent="\n")
        _corpus_list_one(collection)


def _corpus_list_one(collection: str) -> None:
    documents = _corpus_documents(collection)
    if not documents:
        display.warning(
            f"Add one with: boepie corpus add {collection} <identifier>",
            lead=f"No documents in '{collection}'.",
        )
        return

    # Title first: it is the only field a person recognises. The id follows
    # because it is what `read_*` and `corpus remove` take.
    for document in sorted(
        documents, key=lambda d: str(d.frontmatter.get("title", "")).lower()
    ):
        title = document.frontmatter.get("title") or document.id
        managed_by = document.frontmatter.get("managed_by", "?")
        display.document_entry(str(title), document.id, managed_by)
    display.info(f"{len(documents)} document(s).", indent="\n")


@corpus.command("move")
@click.option(
    "--collection", required=True, type=click.Choice(["literature", "docs", "notes"])
)
@click.argument("document_id")
@click.option(
    "--group",
    default=None,
    metavar="PATH",
    help="New group, e.g. calibration/gains. Pass '' to move to the top level.",
)
@click.option("--title", default=None, help="New title, which also renames the file.")
def corpus_move(
    collection: str, document_id: str, group: str | None, title: str | None
) -> None:
    """Move or rename a document without breaking its read handles.

    A document is addressed by its `id`, never by its path, so regrouping and
    retitling are both safe: every `read_literature`/`read_docs`/`read_notes`
    handle, and every search hit already in an agent's context, stays valid.
    Rebuild the index afterwards so the recorded source paths match again.
    """
    if group is None and title is None:
        raise CliError("nothing to do: pass --group, --title, or both.")

    collection_dir = _corpus_collection_dir(collection)
    documents = _corpus_documents(collection)
    document = next((d for d in documents if d.id == document_id), None)
    if document is None:
        raise _no_such_document_error(document_id, (collection,))

    source = read_document(document.md_path)
    new_title = title or str(source.frontmatter.get("title", document_id))

    # Uniqueness is collection-wide, and this document's own current name must
    # not count against it or a pure regroup would gratuitously suffix itself.
    taken = {other.reserved_filename for other in documents if other.id != document_id}
    filename = unique_document_name(full_title_filename(new_title), taken)

    if group is None:
        anchor_path = source.wrapper_dir or source.md_path
        target_dir = anchor_path.parent
    else:
        target_dir = collection_dir / group if group else collection_dir

    updates: dict[str, object] = {}
    if title is not None:
        updates["title"] = new_title

    # A docs page's `project` is both its natural key and what `search_docs`
    # filters on, and by convention it is the top-level group it lives in.
    # Letting the two disagree would make the page unfilterable, so the block
    # follows the move.
    if collection == "docs" and group is not None:
        new_project = (group.split("/", 1)[0] if group else "") or None
        docs_block = dict(source.frontmatter.get("docs") or {})
        if new_project and docs_block.get("project") != new_project:
            docs_block["project"] = new_project
            updates["docs"] = docs_block
            display.muted(
                f"docs.project updated to '{new_project}' to match the new group."
            )

    moved = move_leaf_document(
        source, target_md_path=target_dir / filename, frontmatter_updates=updates
    )

    display.success(
        f"{new_title} (id={document_id}) -> "
        f"{moved.md_path.relative_to(collection_dir)}",
        lead="Moved",
    )
    display.next_step(
        f"boepie index build --collection {collection}",
        before="Read handles are unchanged.",
    )


@corpus.command("tree")
@click.option(
    "--collection",
    "collections",
    default=_ALL,
    show_default=True,
    type=CollectionList(_CORPUS_COLLECTIONS),
    help="Comma-separated collections, or 'all'.",
)
def corpus_tree(collections: tuple[str, ...]) -> None:
    """Show each collection's group structure as a tree.

    The corpus is addressed by opaque surrogate ids, which are stable across
    renames but say nothing about what a document is. This is how you find
    out what is actually in there without running a search.
    """
    for collection in collections:
        _corpus_tree_one(collection)


def _corpus_tree_one(collection: str) -> None:
    collection_dir = _corpus_collection_dir(collection)
    documents = _corpus_documents(collection)
    if not documents:
        display.warning(
            f"Add one with: boepie corpus add {collection} <identifier>",
            lead=f"No documents in '{collection}'.",
        )
        return

    tree = Tree(display.collection_root(collection, collection_dir))
    branches: dict[str, Tree] = {}

    def branch_for(relative_group: Path) -> Tree:
        """Create (and cache) the branch for a group path, parents first.

        `Path(".")` is the collection root itself, which is the tree, not a
        group inside it - the base case that stops the recursion from adding
        an empty branch above every top-level group.
        """
        key = relative_group.as_posix()
        if key == ".":
            return tree
        if key not in branches:
            parent = branch_for(relative_group.parent)
            branches[key] = parent.add(display.group_leaf(relative_group.name))
        return branches[key]

    for document in sorted(documents, key=lambda d: d.md_path.as_posix()):
        anchor = document.wrapper_dir or document.md_path
        relative_group = anchor.parent.relative_to(collection_dir)
        parent = branch_for(relative_group)
        title = document.frontmatter.get("title") or document.id
        managed_by = document.frontmatter.get("managed_by", "?")
        parent.add(display.document_leaf(str(title), document.id, managed_by))

    console.print(tree)
    display.info(f"{len(documents)} document(s).", indent="\n")


# ---------------------------------------------------------------------------
# Query: search / read - thin frontends over the retrieval tool functions
# ---------------------------------------------------------------------------
#
# `search` and `read` route through the exact helpers the search_*/read_* MCP
# tools use (`search_with_lexical_fallback`, `rag.read`) and render with the
# same `format_hits`/`format_span`, so terminal and server output cannot drift.
# They add only what the tools cannot: embedding/index overrides for pointing
# at a specific dev index, and a `--json` mode. The context collection is
# BM25-only (embedding=None, mode='bm25'); its hits are bundle paths and it has
# no read counterpart.


def _emit_outcome_error(error: str) -> None:
    """Turn a SearchOutcome error string (already 'Error: ...') into a
    ClickException, without click re-prefixing a second 'Error:'."""
    raise CliError(error.removeprefix("Error: "))


def _hits_as_json(
    question: str,
    collections: tuple[str, ...],
    ranked: list[tuple[str, SearchResult]],
    note: str | None,
) -> str:
    """Serialise ranked hits to JSON, mirroring the F2 fields format_hits shows.

    Every hit names its own collection, so a merged multi-collection result is
    unambiguous and a single-collection one stays self-describing.
    """
    payload = {
        "collections": list(collections),
        "question": question,
        "note": note,
        "hits": [
            {
                "rank": rank,
                "collection": collection,
                "rrf_score": result.score,
                "bm25_score": result.bm25_score,
                "dense_score": result.dense_score,
                "dense_rank": result.dense_rank,
                "bm25_rank": result.bm25_rank,
                "title": VIEWS[collection].title_of(result.chunk),
                "document_id": result.chunk.document_id,
                "chunk_index": result.chunk.chunk_index,
                "section": result.chunk.section,
                "source": relative_source(
                    result.chunk.source_path,
                    VIEWS[collection].source_root,
                    keep_root=VIEWS[collection].keep_source_root,
                ),
                "char_start": result.chunk.char_start,
                "char_end": result.chunk.char_end,
                "read_handle": VIEWS[collection].read_handles,
                "text": result.chunk.text,
            }
            for rank, (collection, result) in enumerate(ranked, 1)
        ],
    }
    return json.dumps(payload, indent=2)


@cli.command("search")
@click.argument("question")
@click.option(
    "--collection",
    "collections",
    default=_ALL,
    show_default=True,
    type=CollectionList(_SEARCH_COLLECTIONS),
    help="Comma-separated collections to search, or 'all'.",
)
@click.option("--top-k", default=DEFAULT_TOP_K, show_default=True, type=int)
@click.option(
    "--mode",
    default=DEFAULT_MODE,
    show_default=True,
    type=click.Choice(["hybrid", "dense", "bm25"]),
    help="Ignored for context (BM25-only).",
)
@click.option(
    "--snippet",
    default=DEFAULT_SNIPPET,
    show_default=True,
    type=click.Choice(["none", "short", "full"]),
    help="How much of each hit's text to show.",
)
@click.option(
    "--year-min",
    type=int,
    default=None,
    help="Literature only: chunks from this year onward.",
)
@click.option(
    "--year-max",
    type=int,
    default=None,
    help="Literature only: chunks up to this year.",
)
@click.option(
    "--group",
    default=None,
    metavar="PATTERN",
    help="Restrict to documents filed under a group, shell-style: "
    "'quartical', 'calibration/*', '**/gains'. Quote it - your shell "
    "expands an unquoted '*' against the working directory first.",
)
@click.option(
    "--project",
    default=None,
    help="Docs only: alias for --group, since a docs page's project is the "
    "group it lives in.",
)
@click.option(
    "--index-name",
    default=None,
    help="Query a specific built index id instead of the latest.",
)
@embedding_options
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    help="Emit machine-readable JSON instead of text.",
)
@click.option("-v", "--verbose", is_flag=True, help="Show debug/progress logging.")
def search_cli(
    question: str,
    collections: tuple[str, ...],
    top_k: int,
    mode: str,
    snippet: str,
    year_min: int | None,
    year_max: int | None,
    group: str | None,
    project: str | None,
    index_name: str | None,
    resolve_embedding,
    as_json: bool,
    verbose: bool,
) -> None:
    """Search one or more collections and print ranked hits.

    With no --collection this searches everything that has an index, merging
    the results into a single ranked list with each hit labelled by the
    collection it came from; a collection with no index yet is skipped rather
    than failing the run. Naming exactly one collection reproduces the MCP
    tools' output byte for byte.

    Routes through the same retrieval path as search_literature/search_docs/
    search_context, so results match the server. Only needs the embedding
    backend configured (no LLM); context is BM25-only and works with none.
    """
    _set_verbosity(verbose)
    if index_name is not None and len(collections) > 1:
        raise CliError(
            "--index-name names one index, so it cannot be combined with "
            "several collections. Search them one at a time, or drop the flag."
        )
    if group is not None and project is not None:
        raise CliError("--project is an alias for --group; pass one or the other.")
    # A docs page is filed under its project, so the two select the same thing.
    group = group if group is not None else project
    sweeping = len(collections) > 1

    ranked: list[tuple[str, SearchResult]] = []
    notes: list[str] = []
    for collection in collections:
        outcome = _run(
            search_with_lexical_fallback(
                question,
                collection=collection,
                top_k=top_k,
                mode="bm25" if collection == _CONTEXT_COLLECTION else mode,
                filters=_search_filters(collection, year_min, year_max, group),
                missing_index_fix=VIEWS[collection].missing_index_fix,
                index_root=_index_root_for_collection(collection),
                # No dense leg in the bundle's lexical-only context index.
                embedding=(
                    None if collection == _CONTEXT_COLLECTION else resolve_embedding()
                ),
                index_id=index_name,
            )
        )
        if outcome.error:
            # A collection you have not indexed is fatal only when it is the
            # one you asked for; in a sweep it is simply not part of the
            # answer. Anything else - a stale index, an embedding mismatch -
            # stops the search even in a sweep, because silently dropping a
            # collection the user believes was searched is the failure this
            # whole check exists to prevent.
            if not (sweeping and outcome.missing_index):
                _emit_outcome_error(outcome.error)
            continue
        if outcome.note:
            notes.append(f"{collection}: {outcome.note}" if sweeping else outcome.note)
        ranked.extend((collection, result) for result in outcome.results)

    # RRF scores come from ranks, not from any backend's raw scale, so they
    # are the one cross-collection comparison that is not meaningless.
    ranked.sort(key=lambda pair: pair[1].score, reverse=True)
    ranked = ranked[:top_k]
    note = "\n".join(notes) or None

    if as_json:
        click.echo(_hits_as_json(question, collections, ranked, note))
        return

    if len(collections) == 1:
        collection = collections[0]
        view = VIEWS[collection]
        payload = format_hits(
            question,
            collection,
            [result for _, result in ranked],
            snippet=snippet,
            title_of=view.title_of,
            source_root=view.source_root,
            keep_source_root=view.keep_source_root,
            read_handles=view.read_handles,
            score_detail=True,
        )
    else:
        payload = format_merged_hits(
            question,
            ranked,
            collections=collections,
            snippet=snippet,
            score_detail=True,
        )
    display.hits(with_note(payload, note))


def _search_filters(
    collection: str, year_min: int | None, year_max: int | None, group: str | None
) -> list[Filter] | None:
    """The filters that apply to `collection`, on the dotted frontmatter paths
    the schema declares - a flat `year`/`project` matches nothing, silently,
    because `Filter.predicate` answers a missing field with False.

    `group` filters on the `group` metadata every corpus loader records, which
    is one mechanism rather than two: a docs page's project *is* its group, so
    `--project stimela` and `--group stimela` resolve to the same predicate
    instead of a `docs.project` filter that only docs could honour.
    """
    filters: list[Filter] = []
    if collection == "literature":
        if year_min is not None:
            filters.append(Filter(field="bib.year", op="gte", value=year_min))
        if year_max is not None:
            filters.append(Filter(field="bib.year", op="lte", value=year_max))
    if group is not None:
        filters.append(Filter(field="group", op="glob", value=group))
    return filters or None


@cli.command("read")
@click.argument("document_id")
@click.option(
    "--collection",
    "collections",
    default=_ALL,
    show_default=True,
    type=CollectionList(_READ_COLLECTIONS),
    help="Comma-separated collections to look in, or 'all'. context has no "
    "read: open its source path directly.",
)
@click.option(
    "--chunk-index",
    type=int,
    default=None,
    help="Chunk to centre on (from a search hit). Omit to read the whole document.",
)
@click.option(
    "--before",
    type=int,
    default=1,
    show_default=True,
    help="Neighbouring chunks to include before the anchor.",
)
@click.option(
    "--after",
    type=int,
    default=1,
    show_default=True,
    help="Neighbouring chunks to include after the anchor.",
)
@click.option(
    "--index-name",
    default=None,
    help="Read from a specific built index id instead of the latest.",
)
@embedding_options
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    help="Emit machine-readable JSON instead of text.",
)
@click.option("-v", "--verbose", is_flag=True, help="Show debug/progress logging.")
def read_cli(
    document_id: str,
    collections: tuple[str, ...],
    chunk_index: int | None,
    before: int,
    after: int,
    index_name: str | None,
    resolve_embedding,
    as_json: bool,
    verbose: bool,
) -> None:
    """Expand a search hit into wider context (same output as read_docs/read_literature).

    Pass a hit's document_id and --chunk-index to pull the neighbouring chunks
    (widen with --before/--after), or omit --chunk-index to read the whole
    document. A document id is a surrogate that says nothing about which
    collection it belongs to, so with no --collection this looks in each in
    turn and reads whichever holds it. Loads the same handle the MCP read_*
    tools load.
    """
    _set_verbosity(verbose)
    span = None
    collection = collections[0]
    for candidate in collections:
        try:
            span = _run(
                _read_span(
                    document_id,
                    collection=candidate,
                    chunk_index=chunk_index,
                    before=before,
                    after=after,
                    index_name=index_name,
                    embedding=resolve_embedding(),
                    view=VIEWS[candidate],
                )
            )
        except StaleIndexError as error:
            # Not "look in the next one": a stale index would otherwise be
            # reported as an unknown document id, which is the silent failure
            # `built_from` exists to prevent.
            raise CliError(str(error)) from error
        except CliError:
            # "not indexed here" and "no such document here" are both just
            # "look in the next one" when the caller did not name a collection.
            if len(collections) == 1:
                raise
            continue
        collection = candidate
        break

    if span is None:
        raise _no_such_document_error(document_id, collections)
    view = VIEWS[collection]

    if as_json:
        click.echo(
            json.dumps(
                {
                    "document_id": span.document_id,
                    "chunk_start": span.chunk_start,
                    "chunk_end": span.chunk_end,
                    "char_start": span.char_start,
                    "char_end": span.char_end,
                    "source": relative_source(
                        span.source_path,
                        view.source_root,
                        keep_root=view.keep_source_root,
                    ),
                    "sections": span.sections,
                    "text": span.text,
                },
                indent=2,
            )
        )
        return

    display.span(
        format_span(
            span, source_root=view.source_root, keep_source_root=view.keep_source_root
        )
    )


async def _read_span(
    document_id: str,
    *,
    collection: str,
    chunk_index: int | None,
    before: int,
    after: int,
    index_name: str | None,
    embedding: ModelBinding,
    view,
):
    """Load a document span the way rag.read (and thus read_docs) does, turning
    the engine's typed failures into short CLI messages."""
    try:
        return await rag_read(
            document_id,
            chunk_index=chunk_index,
            before=before,
            after=after,
            collection=collection,
            index_root=_index_root_for_collection(collection),
            embedding=embedding,
            index_id=index_name,
        )
    except FileNotFoundError:
        raise CliError(
            f"no '{collection}' index found. Run {view.missing_index_fix}."
        ) from None
    except StaleIndexError:
        # Never softened into CliError: `read` tries each collection in turn
        # and treats a CliError as "look in the next one", which would report
        # a stale index as an unknown document id.
        raise
        raise CliError(one_line(error)) from error
    except KeyError as error:
        raise CliError(
            f"{one_line(error.args[0])} Use a document_id and chunk_index from a "
            f"'boepie search --collection {collection}' hit."
        ) from error


# ---------------------------------------------------------------------------
# Context bundle: init, fetch, apply, status, reset
# ---------------------------------------------------------------------------


def _build_context_index(bundle_dir: Path) -> None:
    """Build the bundle's own BM25 index and report the count."""
    manifest = _run(
        build(
            ContextLoader(bundle_dir),
            embedding=None,
            index_root=index_root_for(bundle_dir),
        )
    )
    display.success(
        f"{manifest.count} chunks into "
        f"'{bundle_dir.name}/.index/context/bm25' (BM25 only).",
        lead="Indexed",
    )


def _note_legacy_global_index() -> None:
    """Point out (never delete) a knowledge index left in the old global store.

    Machines that ran an earlier boepie still carry
    `INDEX_DIR/knowledge/` (the literal pre-rename directory name - this
    detector intentionally does not track the `knowledge` -> `context`
    collection rename, since it identifies an older, unrelated legacy
    artefact from before the index moved inside the bundle at all), which
    nothing reads any more now that the index lives inside the bundle it was
    built from.
    """
    legacy_dir = INDEX_DIR / "knowledge"
    if legacy_dir.exists():
        display.warning(
            f"unused legacy knowledge index at {legacy_dir} "
            f"(superseded by the per-bundle one); safe to delete.",
            lead="Note:",
        )


@cli.group()
def context() -> None:
    """Manage the `.boepie/` context bundle (fetch, init, apply, status, reset)."""


@context.command()
@click.option(
    "--tag",
    default="latest",
    show_default=True,
    help="GitHub release tag to fetch from.",
)
def fetch(tag: str) -> None:
    """Bring the local content cache up to date with a GitHub release.

    The cache is machine-global, shared by all `.boepie/` bundles. Cached
    content is preferred by `apply` over packaged seeds. Checks the
    release's `.sha256` sidecar first and skips the (much larger) tarball
    download when the cache already matches it.
    """
    with console.status(f"Checking knowledge-content.tar.gz from release {tag}..."):
        try:
            result = fetch_content(tag=tag)
        except ValueError as error:
            display.error(str(error))
            raise SystemExit(1) from error

    content_dir = result.content_dir
    manifest_path = content_dir / "content-manifest.json"
    content_version = (
        json.loads(manifest_path.read_text(encoding="utf-8")).get("content_version")
        if manifest_path.exists()
        else None
    )
    version_note = f", content_version={content_version}" if content_version else ""
    if result.changed:
        display.success(
            f"content (tag={tag}{version_note}) into {content_dir}.", lead="Fetched"
        )
    else:
        display.success(f"(tag={tag}{version_note}).", lead="Content up to date")


@context.command()
@click.option(
    "--directory",
    type=click.Path(exists=True, file_okay=False, path_type=str),
    default=".",
    show_default=True,
    help="Target directory where .boepie/ will be created.",
)
@click.option("--skills", is_flag=True, help="(not implemented yet)")
@click.option("--hooks", is_flag=True, help="(not implemented yet)")
def init(directory: str, skills: bool, hooks: bool) -> None:
    """Initialize the `.boepie/` context bundle.

    Creates the bundle from the resolved content source (cached content when
    available, else packaged seeds), appends the bundle pointer to AGENTS.md,
    and builds the BM25 search index into the bundle's own `.index/`
    (git-ignored, so the committable bundle carries no derived state).
    """
    if skills:
        display.warning("--skills not implemented yet")
    if hooks:
        display.warning("--hooks not implemented yet")

    target_dir = Path(directory).resolve()
    try:
        init_bundle(target_dir)
    except FileExistsError as error:
        raise CliError(str(error)) from error

    agents_md = target_dir / "AGENTS.md"
    append_agents_pointer(agents_md)

    bundle_dir = target_dir / ".boepie"
    display.success(f"bundle at {bundle_dir}", lead="Initialized")

    _build_context_index(bundle_dir)
    _note_legacy_global_index()


@context.command()
@click.option(
    "--directory",
    type=click.Path(exists=True, file_okay=False, path_type=str),
    default=".",
    show_default=True,
    help="Target directory containing .boepie/.",
)
@click.option(
    "--force",
    "force_targets",
    multiple=True,
    metavar="PATH",
    help=(
        "Revert a managed_by: user file back to boepie-managed "
        "(bundle-root-relative path, e.g. concepts/my-notes.md; "
        "a leading .boepie/ is stripped if present; repeatable)."
    ),
)
def apply(directory: str, force_targets: tuple[str, ...]) -> None:
    """Converge the bundle with the resolved content source.

    Rewrites every `managed_by: boepie` file from the resolved source (cached
    content when available, else packaged seeds), deletes orphaned boepie-managed
    files, preserves every `managed_by: user` file byte-for-byte, and rebuilds the
    bundle's own BM25 search index under `.boepie/.index/`. Pass --force with
    one or more bundle-relative paths to revert specific `managed_by: user`
    files back to boepie-managed instead (see `context reset` to discard
    every local file at once).
    """
    target_dir = Path(directory).resolve()
    try:
        source_dir = resolve_content_source()
        apply_bundle(target_dir, source_dir, force_paths=force_targets)
    except (FileNotFoundError, ValueError) as error:
        raise CliError(str(error)) from error

    bundle_dir = target_dir / ".boepie"
    display.success(f"bundle at {bundle_dir}", lead="Applied")

    _build_context_index(bundle_dir)
    _note_legacy_global_index()


@context.command()
@click.option(
    "--directory",
    type=click.Path(exists=True, file_okay=False, path_type=str),
    default=".",
    show_default=True,
    help="Target directory containing .boepie/.",
)
def status(directory: str) -> None:
    """Report the bundle's state relative to installed versions."""
    target_dir = Path(directory).resolve()
    try:
        status_result = bundle_status(target_dir)
    except FileNotFoundError as error:
        raise CliError(str(error)) from error

    report = display.success if status_result.state == "current" else display.warning
    report(status_result.detail, lead=f"{status_result.state}:")

    bundle_dir = target_dir / ".boepie"
    if not index_root_for(bundle_dir).exists():
        display.warning(
            f"run `boepie context apply` to build {index_root_for(bundle_dir)}",
            lead="no search index:",
        )
    _note_legacy_global_index()


@context.command("reset")
@click.option(
    "--directory",
    type=click.Path(exists=True, file_okay=False, path_type=str),
    default=".",
    show_default=True,
    help="Target directory containing .boepie/.",
)
@click.option("--yes", is_flag=True, help="Skip the confirmation prompt.")
def context_reset(directory: str, yes: bool) -> None:
    """Delete `.boepie/` and rebuild it from scratch.

    Discards every `managed_by: user` file outright, including ones with no
    upstream counterpart to revert to (unlike `context apply --force`, which
    only reverts a named file when boepie still has something to revert it
    to). Prompts for confirmation naming every local file that would be lost
    unless --yes is passed or there is nothing to lose.
    """
    target_dir = Path(directory).resolve()
    bundle_dir = target_dir / ".boepie"
    if not bundle_dir.exists():
        raise CliError(f"no bundle at {bundle_dir}. Run 'boepie context init' first.")

    local_paths = list_source_local_files(bundle_dir)
    if local_paths and not yes:
        for relative_path in local_paths:
            display.info(str(relative_path), indent="  ")
        confirmed = click.confirm(
            f"This permanently deletes {len(local_paths)} local file(s) listed above "
            "and rebuilds .boepie/ from scratch. Continue?",
            default=False,
        )
        if not confirmed:
            raise CliError("reset cancelled; no changes made.")

    try:
        reset_bundle(target_dir)
    except FileNotFoundError as error:
        raise CliError(str(error)) from error

    display.success(f"bundle at {bundle_dir}", lead="Reset")

    _build_context_index(bundle_dir)
    _note_legacy_global_index()


# ---------------------------------------------------------------------------
# Sync: composite bootstrap (context fetch -> index fetch -> context apply/init)
# ---------------------------------------------------------------------------


def _build_literature_index() -> None:
    """Build the literature collection's index and report the count.

    Hybrid (BM25 + dense), unlike the context bundle's BM25-only index: the
    embedding config comes from `default_embedding_binding()` (the active
    BOEPIE_EMBEDDING_* config), matching what `index build --collection
    literature` would use with no overrides.
    """
    manifest = _run(
        build(
            LiteratureLoader(),
            index_root=INDEX_DIR,
            embedding=default_embedding_binding(),
        )
    )
    display.success(
        f"{manifest.count} chunks into "
        f"'literature/{manifest.index_id}' (embedding={manifest.embedding_kind}:{manifest.embedding_model}).",
        lead="Indexed",
    )


# Collections `sync` fetches a prebuilt index for from a release, unless
# restricted to `--only context`. `literature` is deliberately absent: it is
# fetched from arXiv and built locally instead (see `sync` below) - boepie
# does not redistribute paper text, even in built-index form.
_SYNC_RELEASE_INDEX_COLLECTIONS = ("docs",)


@contextlib.contextmanager
def _quiet(enabled: bool):
    """Swallow everything `console` prints inside the block when `enabled`.

    Used by `sync`'s default (non-verbose) run so its component steps -
    `context fetch`, `index fetch`, `apply`/`init` - stay silent and sync
    can report a single summary line instead of each step's own output.
    """
    if not enabled:
        yield
        return
    with console.capture():
        yield


def _sync_network_step(
    ctx: click.Context,
    command: click.Command,
    label: str,
    quiet: bool,
    **params: object,
) -> None:
    """Run one network step of `sync` and turn a failure into a warning
    instead of aborting, so the offline convergence step that follows still
    runs against whatever content or indices are already cached or packaged.

    `command` is invoked exactly as its own CLI entry point would be (missing
    options fall back to that command's own defaults via `ctx.invoke`), so
    this adds no fetch logic of its own - only the warn-and-continue wrapper.
    The warning itself is printed outside `_quiet` so it is visible even in
    the default, non-verbose run.
    """
    try:
        with _quiet(quiet):
            ctx.invoke(command, **params)
    except (SystemExit, httpx.HTTPError) as error:
        display.warning(f"{label} failed: {error}", lead="Warning:")


@cli.command()
@click.option(
    "--only",
    type=click.Choice(["context", "indices"]),
    default=None,
    help="Restrict sync to just the context bundle or just the indices (default: both).",
)
@click.option(
    "--tag",
    default="latest",
    show_default=True,
    help="GitHub release tag to fetch from.",
)
@click.option(
    "--directory",
    type=click.Path(exists=True, file_okay=False, path_type=str),
    default=".",
    show_default=True,
    help="Target directory for the context bundle.",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    help="Show each step's own output instead of a one-line summary.",
)
@click.pass_context
def sync(
    ctx: click.Context, only: str | None, tag: str, directory: str, verbose: bool
) -> None:
    """Bring the context bundle and default indices up to date in one step.

    Composite of `context fetch` -> `index fetch` (docs) + `corpus fetch
    --collection literature` + a local literature index build -> `context
    apply` (or `init` on a first run, when no `.boepie/` exists yet under
    --directory). Adds nothing of its own beyond calling those steps: `index
    fetch` already skips a collection that is present, `corpus fetch`
    already skips a paper already converted, and `apply`/`init` already
    rebuild the BM25 index. Unlike `docs`, the literature index is never
    downloaded from a release - it is fetched from arXiv and built locally,
    tag/network issues notwithstanding (see `boepie.literature.fetch`).

    Each network step (`context fetch`, `index fetch`, `corpus fetch
    --collection literature`) warns and continues on failure instead of
    aborting, so the final offline convergence step still runs against
    whatever content or indices are already cached, packaged or previously
    fetched - the overall exit code stays 0 as long as that local step
    succeeds. By default only a one-line summary is printed; pass --verbose
    to see each step's own message.
    """
    sync_context = only != "indices"
    sync_indices = only != "context"
    quiet = not verbose

    if sync_context:
        _sync_network_step(ctx, fetch, f"context fetch --tag {tag}", quiet, tag=tag)

    if sync_indices:
        for collection in _SYNC_RELEASE_INDEX_COLLECTIONS:
            _sync_network_step(
                ctx,
                index_fetch,
                f"index fetch --collection {collection} --tag {tag}",
                quiet,
                collection=collection,
                tag=tag,
            )

        _sync_network_step(
            ctx,
            corpus_fetch,
            "corpus fetch --collection literature",
            quiet,
            collections=("literature",),
            force_targets=(),
            delay=None,
            verbose=False,
        )
        if LITERATURE_DIR.exists():
            with _quiet(quiet):
                _build_literature_index()

    if sync_context:
        target_dir = Path(directory).resolve()
        with _quiet(quiet):
            if (target_dir / ".boepie").exists():
                ctx.invoke(apply, directory=directory)
            else:
                ctx.invoke(init, directory=directory, skills=False, hooks=False)

    if quiet:
        display.success(f"(tag={tag}).", lead="Synced")


# ---------------------------------------------------------------------------
# Hint: BM25 lookup for hook injection
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("prompt")
@click.option(
    "--collection",
    "collections",
    default=_ALL,
    show_default=True,
    type=CollectionList(_SEARCH_COLLECTIONS),
    help="Comma-separated collections to draw hints from, or 'all'.",
)
def hint(prompt: str, collections: tuple[str, ...]) -> None:
    """Print BM25 coordinates for hook injection.

    Runs a BM25-only search over each selected collection - the `.boepie/`
    bundle and the machine-global corpora alike - and prints at most 3
    results overall as plain-text coordinates (path#section: snippet). Exits
    silently (0) when there are no hits or the top score is below the
    configured threshold.

    BM25-only by design: this fires on every prompt, so it must stay cheap
    and must never reach for an embedding backend.
    """
    _run(_hint_search(prompt, collections))


async def _hint_search(prompt: str, collections: tuple[str, ...]) -> None:
    """Async helper for hint command.

    Every failure mode here is silent: a missing bundle, a collection with no
    index, a collection with no hits. This runs on every prompt, so it must
    never interrupt and never explain itself.
    """
    scored: list[tuple[float, SearchResult]] = []
    for collection in collections:
        if collection == _CONTEXT_COLLECTION:
            # A hook runs from the project directory, so the bundle governing
            # the cwd is the one to search.
            bundle_dir = find_bundle()
            if bundle_dir is None:
                continue
            index_root = index_root_for(bundle_dir)
        else:
            index_root = INDEX_DIR

        try:
            results = await search(
                prompt,
                collection=collection,
                mode="bm25",
                top_k=3,
                index_root=index_root,
                embedding=None,
            )
        except (FileNotFoundError, ValueError):
            continue

        for result in results:
            # mode='bm25' populates bm25_score; guard against None anyway
            # (fail closed) since this must never spam on an unexpected shape.
            score = result.bm25_score
            if score is None or score < _HINT_MIN_SCORE:
                continue
            scored.append((score, result))

    if not scored:
        return

    scored.sort(key=lambda pair: pair[0], reverse=True)
    for _, result in scored[:3]:
        chunk = result.chunk
        snippet = chunk.text.strip()
        if len(snippet) > 120:
            snippet = snippet[:120]
        section_part = f"#{chunk.section}" if chunk.section else ""
        display.hint(f"{chunk.document_id}{section_part}", snippet)


# ---------------------------------------------------------------------------
# Config: the user-editable ~/.config/boepie/config.toml
# ---------------------------------------------------------------------------


def _check_known_key(key: str) -> None:
    if key not in settings.known_keys():
        raise CliError(
            f"unknown config key '{key}'. Known keys: {', '.join(sorted(settings.known_keys()))}"
        )


_MISSING_FILE_HINT = (
    "No config file yet - boepie is running on built-in defaults. "
    "Run 'boepie config create' to write one."
)


@cli.group()
def config() -> None:
    """Manage the user config file (~/.config/boepie/config.toml)."""


@config.command("create")
@click.option(
    "-f",
    "--force",
    is_flag=True,
    help="Overwrite an existing config file, discarding whatever it holds.",
)
def config_create_cmd(force: bool) -> None:
    """Write a fresh config file with every setting at its built-in default.

    The file is a full reference: each key is present, commented with what it
    does, so it can be edited directly instead of discovered through
    `config set`.
    """
    try:
        created_path = settings.create(force=force)
    except FileExistsError as error:
        raise CliError(
            f"{error.args[0]} already exists. Edit it directly, or pass --force "
            "to replace it with a fresh default file (your current settings "
            "would be lost)."
        ) from error

    display.success(
        f"{len(settings.known_keys())} settings at their defaults in:", lead="Created"
    )
    display.path(created_path)


@config.command("path")
def config_path_cmd() -> None:
    """Print the config file's path (it may not exist yet)."""
    if not settings.config_file_exists():
        display.warning(_MISSING_FILE_HINT, lead="Warning:")
    display.path(settings.config_path())


@config.command("show")
@click.option(
    "--sources/--no-sources",
    default=True,
    show_default=True,
    help="Annotate each value with the layer it came from.",
)
def config_show(sources: bool) -> None:
    """Print every setting's resolved value and where it came from.

    Resolution is env var > config file > built-in default, so a value here
    does not imply a config file exists - `--sources` (on by default) says
    which layer actually supplied each one.
    """
    resolved = settings.resolve_settings()
    file_exists = settings.config_file_exists()

    lines: list[str] = []
    if sources:
        lines.append("# Resolved config: env var > config file > built-in default.")
        lines.append(f"# Config file: {settings.config_path()}")
        if not file_exists:
            lines.append(f"# {_MISSING_FILE_HINT}")

    current_section = ""
    for setting in resolved:
        section, _, name = setting.key.partition(".")
        if section != current_section:
            lines.append("")
            lines.append(f"[{section}]")
            current_section = section
        rendered = tomlkit.item(setting.value).as_string()
        annotation = ""
        if sources:
            origin = setting.env_var if setting.source == "env" else setting.source
            annotation = f"  # {origin}"
        lines.append(f"{name} = {rendered}{annotation}")

    display.toml("\n".join(lines).strip())


@config.command("get")
@click.argument("key")
@click.option(
    "--source", is_flag=True, help="Print which layer supplied the value too."
)
def config_get(key: str, source: bool) -> None:
    """Print one setting's resolved value, e.g. `boepie config get embedding.binding`."""
    _check_known_key(key)
    if not source:
        display.plain(str(settings.get(key)))
        return

    setting = next(item for item in settings.resolve_settings() if item.key == key)
    origin = setting.env_var if setting.source == "env" else setting.source
    display.info(f"{setting.value} ({origin})")


@config.command("set")
@click.argument("key")
@click.argument("value")
def config_set(key: str, value: str) -> None:
    """Set one setting and write it to the config file, e.g.
    `boepie config set literature.prefer_pdf true`."""
    _check_known_key(key)
    try:
        parsed = settings.parse_value(key, value)
    except settings.ConfigError as error:
        raise CliError(str(error)) from error

    created = not settings.config_file_exists()
    settings.set_value(key, parsed)

    display.success(f"{key} = {parsed!r} in:", lead="Set")
    display.path(settings.config_path())
    if created:
        display.info(
            "Created that file with this key only. "
            "'boepie config create' would instead write every key at its default."
        )

    # The write has already succeeded; an env var shadowing it is worth
    # saying, but not worth failing the command over if resolving the other
    # keys happens to trip on something unrelated.
    env_var = settings.env_var_for(key)
    if env_var in os.environ:
        display.warning(
            f"{env_var} is set in your environment and "
            f"overrides the file, so {key} still resolves to "
            f"{settings.get(key)!r}.",
            lead="Note:",
        )
