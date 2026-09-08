# boepie/cli.py
"""Command-line interface for boepie."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import shutil
import sys
import textwrap
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import httpx
import tomlkit

# rich_click is a drop-in for click that renders --help through rich, so every
# `click.option`/`click.argument` below is the real click decorator and only
# the help formatting changes. Imported under the name `click` because that is
# what it is: swapping the alias back is the whole uninstall.
import rich_click as click
from rich.tree import Tree

from boepie import __version__, settings
from boepie import _display as display
from boepie._display import Cancelled, CliError, PlainMessage, console
from boepie.config import (
    CORPUS_EXTRA_FILE_TYPES,
    CORPUS_KEEP_ORIGINAL,
    CORPUS_WARN_ON_DOTFILE_TITLE,
    DEFAULT_MODE,
    DEFAULT_SNIPPET,
    DEFAULT_TOP_K,
    DOCS_DIR,
    INDEX_DIR,
    LITERATURE_DIR,
    LITERATURE_FETCH_DELAY,
    MINERU_BACKEND,
    MINERU_BATCH_SIZE,
    MINERU_DEVICE_MODE,
    MINERU_MODEL_SOURCE,
    NOTES_DIR,
)
from boepie.assets import context_content_dir
from boepie.context import (
    append_agents_pointer,
    apply_bundle,
    bundle_status,
    find_bundle,
    index_root_for,
    init_bundle,
    list_source_local_files,
    reset_bundle,
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
from boepie.corpus.review import ReviewUnavailable
from boepie.corpus.schema import KEY_FIELDS as _CORPUS_KEY_FIELDS
from boepie.docs import DocsProject
from boepie.docs import load_manifest as load_docs_manifest
from boepie.literature import ArxivPaper
from boepie.literature import load_manifest as load_literature_manifest
from boepie.mcp_config import (
    DEFAULT_TARGETS,
    TARGET_NAMES,
    McpConfigError,
    TargetResult,
    apply_target,
    inspect_target,
    manual_definition,
    server_command,
    target_named,
)
from boepie.rag import (
    ContextLoader,
    DocsLoader,
    EmptyCollectionError,
    StaleIndexError,
    index_command,
    index_freshness,
    LiteratureLoader,
    ModelBinding,
    NotesLoader,
    build,
    embedding_options,
    search,
)
from boepie.rag import read as rag_read
from boepie.rag.models import Filter, SearchResult
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
        # Every abort boepie words itself, not `CliError` alone: `Cancelled`
        # was written later and got the panel back until it shared a base.
        if isinstance(error, PlainMessage):
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


def _loader_for(collection: str):
    """The loader for `collection`, reading the directory *this module* resolved.

    Each loader also has its own `boepie.config` default, and constructing one
    with no argument used to take it. That is the same path in ordinary use,
    but it meant the CLI had two answers to "where does this corpus live" -
    and the one a caller could redirect was not the one the build used.
    Looked up inside the function rather than in a module-level table so the
    current value is read at call time.
    """
    directories = {
        "literature": LITERATURE_DIR,
        "docs": DOCS_DIR,
        "notes": NOTES_DIR,
    }
    return _LOADERS[collection](directories[collection])


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
        # `Cancelled`, not `CliError`: a stopped command is not a failed one,
        # and the composites that carry on past a failure must not carry on
        # past this.
        raise Cancelled("stopped before it finished.") from None
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
@click.option(
    "-q",
    "--quiet",
    is_flag=True,
    help="Print errors only. Suppresses the report, never a problem.",
)
@click.option(
    "--no-progress",
    is_flag=True,
    help="Never draw a progress bar, even on a terminal.",
)
def cli(quiet: bool, no_progress: bool) -> None:
    """Boepie - MCP server for AI-assisted stimela pipeline creation."""
    # Both options sit on the group rather than on each command because they
    # describe how boepie talks, not what it does: someone who wants quiet
    # wants it everywhere, and having to remember which subcommands accept it
    # would make it useless. Applied to `_display` once, read by every
    # primitive from then on.
    display.set_verbosity(quiet=quiet, progress=not no_progress)


@cli.command()
def serve() -> None:
    """Start the boepie MCP server (stdio transport)."""
    from boepie.server import mcp

    mcp.run("stdio")


# ---------------------------------------------------------------------------
# Indexing: a verb on the noun that owns the index, not a noun of its own
# ---------------------------------------------------------------------------
#
# The two index families share no storage, no scope and no retrieval stack:
# a corpus collection indexes to `INDEX_DIR/<collection>/<id>/`, machine-
# global and hybrid BM25+dense; the context bundle indexes to
# `<bundle>/.index/context/bm25/`, per-project and BM25-only. Grouping them
# under one `index` noun put a command boundary where there is no boundary in
# the code, and `index status` proved it by enumerating INDEX_DIR alone - the
# noun that claimed to own every index could not see one of them.
#
# So indexing is a verb each owner carries. `boepie corpus index -l` names its
# scope once, in the same spelling `corpus sync -l` uses, instead of repeating
# it as `--collection literature` under a different noun.


class _NoBundleError(Exception):
    """`context` was selected but no `.boepie/` bundle governs the cwd."""


# How each freshness state reads, and how loudly. Only "stale" is a fault;
# the other three are facts about what can be checked, so they stay dim.
#
# Every one of these names a *condition of the index*, because that is the
# question `status` answers. "in step with its corpus" said the same thing
# in a shape that reads as a relationship between two things rather than as
# a verdict on one, so a reader had to work out which of the two was wrong.
# `current` needs no unpacking.
_FRESHNESS_WORDING: dict[str, str] = {
    "in step": "current",
    "corpus absent": "unverifiable - its corpus is not on this machine",
    "unrecorded": "unverifiable - built before boepie recorded what it read",
}


def _index_rows(collection: str, index_root: Path) -> bool:
    """The `index:`/`embedding:` rows for one collection, under its heading.

    Printed by `corpus status` and `context status` rather than by an
    `index status` of its own: an index belongs to the thing it was built
    from, so its state is one more fact about that thing. Returns whether
    anything is wrong, so `--check-only` can exit on it.

    The failure this reports used to be undiagnosable: an index built over an
    older corpus answers queries perfectly happily, with plausible scores,
    pointing at text that has since changed. One row is what turns that into
    something you can see before it misleads you.
    """
    latest_path = index_root / collection / "latest.json"
    if not latest_path.is_file():
        display.warning("not built yet", lead=_status_label("index"), indent="  ")
        display.next_step(f"boepie {index_command(collection)}")
        return True

    try:
        active_id = json.loads(latest_path.read_text(encoding="utf-8"))["index_id"]
        manifest_path = index_root / collection / str(active_id) / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        display.warning(
            "unreadable - rebuild it", lead=_status_label("index"), indent="  "
        )
        return True

    freshness = index_freshness(manifest.get("built_from"), collection)
    if freshness.state == "stale":
        counts = ", ".join(
            part
            for part in (
                f"{freshness.changed} changed" if freshness.changed else "",
                f"{freshness.gone} gone" if freshness.gone else "",
            )
            if part
        )
        display.warning(
            f"stale - {counts} of {freshness.document_count} documents",
            lead=_status_label("index"),
            indent="  ",
        )
        display.next_step(f"boepie {index_command(collection)}")
        wrong = True
    else:
        display.muted(
            _FRESHNESS_WORDING[freshness.state]
            + (
                f" ({freshness.document_count} documents)"
                if freshness.state == "in step"
                else ""
            ),
            lead=_status_label("index"),
            indent="  ",
        )
        wrong = False

    kind = manifest.get("embedding_kind")
    display.muted(
        f"{kind}:{manifest.get('embedding_model')} ({active_id})"
        if kind
        else f"none - BM25 only ({active_id})",
        lead=_status_label("embedding"),
        indent="  ",
    )
    return wrong


def _build_corpus_index(
    collection: str,
    *,
    resolve_embedding,
    embedding_concurrency: int | None,
    index_name: str | None,
) -> None:
    """Build one corpus collection's index, with a progress bar over its chunks."""
    loader = _loader_for(collection)
    embedding = resolve_embedding(max_async=embedding_concurrency)

    # total is unknown until build() finishes chunking every document and
    # reports the real count via its first on_progress(0, total) call - until
    # then the bar renders as an indeterminate spinner.
    started = time.monotonic()
    with display.progress_bar(f"Indexing '{collection}'", total=None) as advance:
        manifest = _run(
            build(
                loader,
                index_root=INDEX_DIR,
                embedding=embedding,
                index_id=index_name,
                on_progress=lambda done, total: advance(done, total),
            )
        )
    display.operation(
        "Indexed",
        f"{_plural(manifest.count, 'chunk')} into {collection}/{manifest.index_id}",
        elapsed=time.monotonic() - started,
    )


# ---------------------------------------------------------------------------
# Corpus: init, sync, add, status, list - literature/docs/notes, built on this
# machine, unified around boepie.corpus's shared layout (see boepie.corpus
# for the on-disk shape: directory-as-group, full-title filenames, a
# surrogate `id`, `managed_by: boepie | user` provenance).
# ---------------------------------------------------------------------------
#
# No literature Markdown or built index is ever published by boepie (see
# boepie.literature.fetch): `corpus sync --collection literature` pulls each
# manifest paper's HTML straight from arxiv.org/ar5iv.labs.arxiv.org and
# converts it locally, so the only thing boepie itself ships is the small
# bibliographic manifest. Papers with no arXiv presence (pre-arXiv-era, or
# never preprinted) fall to the BYO-PDF path: `corpus add -l <file.pdf>`
# against a copy you supply, converted with MinerU.


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


def _corpus_population_command(collection: str) -> str:
    """The command that puts documents into `collection`.

    `notes` has no packaged manifest and so no `fetch` leg - every note is
    one you added - which is why this is a lookup rather than one string with
    the collection interpolated into it.
    """
    if collection == "notes":
        return "boepie corpus add -n <file-or-url>"
    return f"boepie corpus sync --collection {collection}"


@cli.group()
def corpus() -> None:
    """Manage the literature/docs/notes corpora (init, sync, add, status, list)."""


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
        "(default: corpus.keep_original).",
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
        arxiv_delay=LITERATURE_FETCH_DELAY,
        **overrides,
    )


def _converting_with_mineru(documents: int, number: int, total: int) -> None:
    """Say what MinerU is about to spend minutes on.

    Per run rather than per document, because MinerU writes nothing until a
    whole run has finished - there is no per-document moment to report.
    """
    run = f" (run {number} of {total})" if total > 1 else ""
    display.info(
        f"{_plural(documents, 'document')} with MinerU{run} - this takes a few minutes",
        lead="converting",
    )


def _can_review() -> bool:
    """Whether there is a terminal to open the review buffer in.

    Both ends, not just one: a piped `boepie corpus add ... | tee` still has
    a terminal on stdin, and opening an editor whose output nobody sees is
    worse than saying there is nowhere to review.
    """
    return sys.stdin.isatty() and sys.stdout.isatty()


# Each collection's shorthand flag, for naming it back in an error. The
# options themselves are declared literally on the command.
_ADD_SHORTHANDS = {"literature": "-l", "docs": "-d", "notes": "-n"}

# Which options mean anything for which destination. `--project` is a docs
# natural key and the other three are all about establishing a paper's
# identity, so each is a usage error elsewhere rather than a flag that is
# quietly ignored.
_COLLECTION_ONLY_OPTIONS = {
    "--project": "docs",
    "--citekey": "literature",
    "--identifier": "literature",
    "--yes": "literature",
}


def _add_destination(collection: str | None, shorthand: str | None) -> str:
    """The one collection this `corpus add` writes to.

    `--collection` here names a destination, not a selection, so it is a
    single `Choice` rather than the `CollectionList` every other command
    takes: a document is written to exactly one collection, and `all` would
    mean nothing.
    """
    if collection is not None and shorthand is not None and collection != shorthand:
        raise CliError(
            f"--collection {collection} and {_ADD_SHORTHANDS[shorthand]} name "
            f"different collections. Pass one of them."
        )
    chosen = collection or shorthand
    if chosen is None:
        raise CliError(
            "name a collection: --collection literature|docs|notes, or -l/-d/-n."
        )
    return chosen


def _reject_foreign_add_options(
    collection: str,
    *,
    project: str | None,
    citekey: str | None,
    identifier: str | None,
    accept_review: bool,
) -> None:
    """Refuse an option that means nothing for the collection being written.

    Louder than ignoring it: `--citekey` on a notes add is someone expecting
    a citekey to come out the other end, and silence would let them find out
    from the corpus instead.
    """
    supplied = {
        "--project": project is not None,
        "--citekey": citekey is not None,
        "--identifier": identifier is not None,
        "--yes": accept_review,
    }
    for flag, given in supplied.items():
        owner = _COLLECTION_ONLY_OPTIONS[flag]
        if given and owner != collection:
            raise CliError(f"{flag} applies to {owner} only, not to {collection}.")


def _add_and_report(collection: str, adder: Callable[[], list[AddOutcome]]) -> None:
    """Run one `corpus add` and report it.

    Both exceptions mean the batch as typed cannot be carried out at all - an
    argument naming nothing, or a converter that is not installed - rather
    than one identifier having failed, so they are caught here and given
    click's own error wording instead of a traceback.
    """
    try:
        outcomes = adder()
    except (InputError, IntakeError, ReviewUnavailable) as error:
        raise CliError(str(error)) from error
    _report_add(collection, outcomes)


def _report_add(collection: str, outcomes: list[AddOutcome]) -> None:
    """One line per identifier, then a single summary and next step.

    Printed per batch rather than per item: adding is meant to be staged like
    commits, several at a time, with one corpus index at the end.
    """
    added = [outcome for outcome in outcomes if outcome.status == "added"]
    duplicates = [outcome for outcome in outcomes if outcome.status == "duplicate"]
    failures = [outcome for outcome in outcomes if outcome.status == "failed"]
    skipped = [outcome for outcome in outcomes if outcome.status == "skipped"]

    for outcome in outcomes:
        if outcome.status == "added":
            via = f" via {outcome.via}" if outcome.via else ""
            detail = f" ({outcome.detail})" if outcome.detail else ""
            display.detail(
                "+",
                (
                    f"{outcome.title} (id={outcome.document_id}{via}){detail}"
                    if outcome.document_id
                    else f"{outcome.title}{detail}"
                ),
            )
            if outcome.notice and CORPUS_WARN_ON_DOTFILE_TITLE:
                display.note(
                    f"{outcome.notice}. Pass --title to control this, or set "
                    f"corpus.warn_on_dotfile_title=false."
                )
        elif outcome.status == "duplicate":
            display.detail(
                "=",
                f"{outcome.identifier} - {outcome.detail} (id={outcome.document_id})",
            )
        elif outcome.status == "skipped":
            # Quieter than a duplicate: nothing is wrong, and a folder walk can
            # produce a great many of these at once.
            display.detail("=", f"{outcome.identifier} - {outcome.detail}")
        else:
            display.note(f"{outcome.identifier} - {outcome.detail}")

    # The summary comes last here, unlike everywhere else, because the detail
    # lines above it are the per-document record and this counts them. Only
    # the outcomes that occurred are named: on a hand-typed batch the other
    # three counts are always zero and would be three numbers to read past.
    counted = [
        f"{len(added)} added" if added else "",
        f"{len(duplicates)} already present" if duplicates else "",
        f"{len(skipped)} skipped" if skipped else "",
        f"{len(failures)} failed" if failures else "",
    ]
    summary = ", ".join(part for part in counted if part) or "nothing to do"
    display.operation("Added", f"{summary} in {collection}")
    if added:
        # The review buffer can send a paper to notes, so the collections
        # named here are the ones actually written to, not the one the
        # command was called on.
        landed = {outcome.collection or collection for outcome in added}
        written = [name for name in _CORPUS_COLLECTIONS if name in landed]
        display.next_step(
            f"boepie corpus index --collection {','.join(written)}",
            note="(once you have finished adding)",
        )
    # Failures only. A skip is a deliberate decision not to take something, so
    # a folder holding one unreadable file still exits 0.
    if failures:
        raise SystemExit(1)


@corpus.command("add")
@click.argument("identifiers", nargs=-1, required=True)
@click.option(
    "--collection",
    type=click.Choice(_CORPUS_COLLECTIONS),
    default=None,
    help="Which collection to write to.",
)
# The shorthands are one option repeated, click's feature-switch form, so
# they cannot disagree with each other - only with an explicit --collection,
# which is checked below.
@click.option(
    "-l",
    "--literature",
    "shorthand",
    flag_value="literature",
    help="Shorthand for --collection literature.",
)
@click.option(
    "-d",
    "--docs",
    "shorthand",
    flag_value="docs",
    help="Shorthand for --collection docs.",
)
@click.option(
    "-n",
    "--notes",
    "shorthand",
    flag_value="notes",
    help="Shorthand for --collection notes.",
)
@click.option(
    "--project",
    default=None,
    help="docs only, and required there: the group these pages live under, "
    "and what search_docs filters on.",
)
@click.option(
    "--citekey", default=None, help="literature only: override the derived citekey."
)
@click.option(
    "--identifier",
    default=None,
    help="literature only: an arXiv id, DOI or ADS bibcode for a document "
    "that does not state one on its own first page. Names one paper, so it "
    "cannot be combined with several inputs.",
)
@click.option(
    "--yes",
    "accept_review",
    is_flag=True,
    help="literature only: skip the review buffer and take the likeliest "
    "identifier boepie found for each document. What scripts use.",
)
@_add_options
def corpus_add(
    identifiers: tuple[str, ...],
    collection: str | None,
    shorthand: str | None,
    project: str | None,
    citekey: str | None,
    identifier: str | None,
    accept_review: bool,
    title: str | None,
    group: str | None,
    keep_original: bool | None,
) -> None:
    """Add documents to a corpus collection, immediately.

    Everything `add` writes is yours (`managed_by: user`) and is never
    touched by `corpus sync`, which only reconciles boepie's own packaged
    manifest. Several identifiers at once; run `boepie corpus index` once when
    you have finished adding.

    \b
    literature (-l)  arXiv ids in any spelling, DOIs, .bib files, PDFs, URLs.
                     A .bib expands into all of its entries, following each
                     one's arXiv id, DOI or `file` path - exporting from
                     Zotero is the best-supported way to bring in a library,
                     since it keeps the citekeys you already cite by. A
                     converted PDF is put in front of you in an editor first:
                     boepie ranks the identifiers on its first page, you pick.
    docs (-d)        A site URL, which crawls the whole site rather than the
                     one page you name; Sphinx sites are read through their
                     own object inventory. Needs --project.
    notes (-n)       The base case: local files of any supported format and
                     http(s) URLs, converted one page at a time. Machine-
                     global, separate from a project's `.boepie/` bundle.
    """
    collection = _add_destination(collection, shorthand)
    _reject_foreign_add_options(
        collection,
        project=project,
        citekey=citekey,
        identifier=identifier,
        accept_review=accept_review,
    )
    if collection == "docs" and project is None:
        raise CliError(
            "--project is required for docs: it is the group the pages are "
            "filed under and what search_docs filters on."
        )
    if identifier is not None and len(identifiers) > 1:
        raise CliError(
            "--identifier names one paper, so it cannot be combined with "
            "several inputs. Add them one at a time, or drop the flag and let "
            "each document be read from its own first page."
        )

    overrides: dict[str, object] = {}
    if collection == "docs":
        overrides["project"] = project
    elif collection == "literature":
        overrides.update(
            citekey=citekey,
            identifier=identifier,
            accept_review=accept_review,
            can_review=_can_review(),
        )
    options = _build_add_options(
        title=title, group=group, keep_original=keep_original, **overrides
    )

    if collection == "literature":
        _add_and_report(
            collection,
            lambda: add_literature(
                LITERATURE_DIR,
                identifiers,
                options,
                notes_dir=NOTES_DIR,
                on_batch=_converting_with_mineru,
            ),
        )
    elif collection == "docs":
        _add_and_report(
            collection,
            lambda: add_docs(
                DOCS_DIR, identifiers, options, on_batch=_converting_with_mineru
            ),
        )
    else:
        _add_and_report(
            collection,
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
    `managed_by: boepie` document can be removed too, but `corpus sync` will
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
        click.confirm(f"Delete {_plural(len(targets), 'document')}?", abort=True)

    for document in targets:
        if document.wrapper_dir is not None:
            shutil.rmtree(document.wrapper_dir)
        else:
            document.md_path.unlink()

    display.operation("Removed", f"{len(targets)} documents from {collection}")
    display.next_step(f"boepie corpus index --collection {collection}")


@corpus.command("init")
@click.option(
    "--collection",
    "collections",
    default=_ALL,
    show_default=True,
    type=CollectionList(_CORPUS_COLLECTIONS),
    help="Comma-separated collections to create, or 'all'.",
)
def corpus_init(collections: tuple[str, ...]) -> None:
    """Create this machine's corpus directories, empty.

    The scaffold half of the corpus: fast, offline, and idempotent, with
    nothing fetched. `corpus sync` fills them.

    **It exists so that nothing else has to create them by accident.** Until
    now the only thing that made a corpus directory was writing the first
    document into it (`corpus.document`'s `mkdir(parents=True)`), so a corpus
    came into being as a side effect of a write. A mistyped
    `BOEPIE_LITERATURE_DIR` silently produced a second corpus at the wrong
    path instead of an error, and every command had to answer "is there a
    corpus here" for itself.

    Machine-global, unlike `context init`: these three directories are shared
    by every workspace, so the second project on a machine finds them already
    made and is told so rather than being given its own.
    """
    started = time.monotonic()
    created: list[str] = []
    present = 0
    for collection in collections:
        collection_dir = _corpus_collection_dir(collection)
        if collection_dir.is_dir():
            present += 1
            continue
        collection_dir.mkdir(parents=True, exist_ok=True)
        created.append(str(collection_dir))

    # One operation line and the paths beneath it, rather than three lines
    # each carrying a long absolute path. The three are normally siblings
    # under one data root, so repeating it would be most of the output.
    if created:
        summary = _plural(len(created), "corpus directory", "corpus directories")
        if present:
            summary += f", {present} already there"
        display.operation("Created", summary, elapsed=time.monotonic() - started)
        display.details("+", created)
    else:
        display.operation(
            "Checked",
            _plural(present, "corpus directory", "corpus directories"),
            elapsed=time.monotonic() - started,
            style="muted",
        )


def _require_corpus(collections: tuple[str, ...]) -> None:
    """Refuse to work against a corpus this machine has not created.

    The one readiness check, so every command fails the same way instead of
    quietly scaffolding. `sync` used to create the `.boepie/` bundle itself
    when there was none, which made a fresh workspace look like a working one
    - the command reported success against state it had just invented.
    """
    absent = [
        name for name in collections if not _corpus_collection_dir(name).is_dir()
    ]
    if not absent:
        return
    raise CliError(
        f"no corpus on this machine for {', '.join(absent)}. "
        f"Run {display.command('boepie init')} first."
    )


@corpus.command("sync")
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
def corpus_sync(
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
    sync --collection <collection>` afterward to index what changed.
    """
    _set_verbosity(verbose)
    _require_corpus(collections)
    for collection in collections:
        _corpus_fetch_one(collection, force_targets, delay, verbose)


def _corpus_fetch_one(
    collection: str, force_targets: tuple[str, ...], delay: float | None, verbose: bool
) -> None:
    if collection == "notes":
        # Accepted rather than rejected as an invalid choice: "notes is not
        # one of literature, docs" says nothing about why, and the reason is
        # worth stating - notes exist only because you added them.
        display.operation(
            "Skipped",
            "notes - no packaged manifest to reconcile against, every note is "
            "one you added",
            style="muted",
        )
        display.next_step("boepie corpus add -n <file-or-url>")
        return
    try:
        if collection == "literature":
            _corpus_fetch_literature(force_targets, delay, verbose)
        else:
            _corpus_fetch_docs(force_targets, delay, verbose)
    except ValueError as error:
        raise CliError(str(error)) from error
    except KeyboardInterrupt:
        display.note(
            "documents already written are kept. Re-run the same command to "
            "carry on from where it stopped."
        )
        raise Cancelled(
            f"stopped during the {collection} fetch."
        ) from None


def _corpus_fetch_literature(
    force_targets: tuple[str, ...], delay: float | None, verbose: bool
) -> None:
    papers = load_literature_manifest(LITERATURE_DIR)
    if not papers:
        display.note("No papers in the literature manifest.")
        return

    started = time.monotonic()
    with _fetch_progress(
        f"Fetching {len(papers)} papers from arXiv", len(papers), verbose
    ) as advance:

        def on_progress(paper: ArxivPaper | None, result) -> None:
            advance()
            if not verbose:
                return
            if result.action == "unavailable":
                display.note(
                    f"no HTML for {result.citekey} "
                    f"(arXiv:{paper.arxiv_id if paper else '?'})"
                )
            else:
                display.detail(_ACTION_MARKERS.get(result.action, "="), result.citekey)

        results = sync_literature(
            LITERATURE_DIR,
            papers,
            force_paths=force_targets,
            delay=delay if delay is not None else LITERATURE_FETCH_DELAY,
            on_progress=on_progress,
        )

    added = [r for r in results if r.action == "added"]
    refetched = [r for r in results if r.action == "refetched"]
    skipped = sum(1 for r in results if r.action == "skipped")
    deleted = [r for r in results if r.action == "deleted"]
    unavailable = [r for r in results if r.action == "unavailable"]
    yours = [r for r in results if r.action == "yours"]

    _report_fetch(
        "papers",
        elapsed=time.monotonic() - started,
        added=added,
        refetched=refetched,
        deleted=deleted,
        skipped=skipped,
        verbose=verbose,
    )
    if yours:
        # Not a failure and not a skip: fetch is doing what it promises by
        # leaving these alone. Said out loud because the alternative is a
        # manifest entry that never appears in the corpus and never explains
        # itself.
        display.operation(
            "Kept",
            f"{len(yours)} papers that are yours, not boepie's",
            style="muted",
        )
        display.details(
            "=", [result.citekey for result in yours], limit=_detail_limit(verbose)
        )
    if unavailable:
        display.note(
            f"{len(unavailable)} papers have no HTML rendering at arxiv.org or ar5iv"
        )
        display.details(
            "!",
            [result.citekey for result in unavailable],
            limit=_detail_limit(verbose),
        )
        display.hint("supply the PDF yourself with `boepie corpus add -l <file.pdf>`")
    display.next_step("boepie corpus index --collection literature")


def _corpus_fetch_docs(
    force_targets: tuple[str, ...], delay: float | None, verbose: bool
) -> None:
    projects = load_docs_manifest(DOCS_DIR)
    if not projects:
        display.note("No projects in the docs manifest.")
        return

    started = time.monotonic()
    with _fetch_progress(
        f"Fetching {len(projects)} docs projects", len(projects), verbose
    ) as advance:

        def on_progress(project: DocsProject | None, result) -> None:
            advance()
            if not verbose:
                return
            display.detail(
                "+" if result.added else "=",
                f"{result.project} - {result.added} added, {result.refetched} "
                f"refetched, {result.skipped} unchanged, {result.deleted} deleted",
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
    total_yours = sum(r.yours for r in results)

    # Pages, not projects: a docs result is per-project but the unit a reader
    # counts in is pages, so the counts are summed and the projects named in
    # the detail lines under them.
    elapsed = time.monotonic() - started
    changes = [
        f"{total_added} added" if total_added else "",
        f"{total_refetched} refetched" if total_refetched else "",
        f"{total_deleted} deleted" if total_deleted else "",
    ]
    summary = ", ".join(part for part in changes if part)
    if summary:
        unchanged = f", {total_skipped} unchanged" if total_skipped else ""
        display.operation(
            "Fetched",
            f"{summary}{unchanged} across {len(results)} docs projects",
            elapsed=elapsed,
        )
        display.details(
            "+",
            [
                f"{r.project} - {r.added + r.refetched} pages"
                for r in results
                if r.added or r.refetched
            ],
            limit=_detail_limit(verbose),
        )
    else:
        display.operation(
            "Checked",
            f"{total_skipped} pages across {len(results)} docs projects, all current",
            elapsed=elapsed,
            style="muted",
        )
    if total_yours:
        yours_projects = sorted(r.project for r in results if r.yours)
        display.operation(
            "Kept",
            f"{total_yours} pages that are yours, not boepie's "
            f"({', '.join(yours_projects)})",
            style="muted",
        )
    if total_failures:
        display.note(f"{total_failures} pages could not be fetched")
    display.next_step("boepie corpus index --collection docs")


# Which marker a reconciliation action earns in a detail line.
_ACTION_MARKERS = {"added": "+", "refetched": "~", "deleted": "-", "skipped": "="}


def _detail_limit(verbose: bool) -> int | None:
    """How many item lines an operation may print. None (all of them) under
    --verbose, which is what that flag is for."""
    return None if verbose else display.DETAIL_LIMIT


def _report_fetch(
    noun: str,
    *,
    elapsed: float,
    added: list,
    refetched: list,
    deleted: list,
    skipped: int,
    verbose: bool,
) -> None:
    """One line for a reconciliation, then the documents that actually moved.

    `Fetched` when something changed and `Checked` when nothing did - the
    distinction a reader is really after, and one the old
    "0 added, 0 refetched, 17 skipped, 0 deleted" made them compute for
    themselves out of four numbers, three of which are zero on any ordinary
    run. The counts are still there, but only for the actions that happened.
    """
    changes = [
        f"{len(added)} added" if added else "",
        f"{len(refetched)} refetched" if refetched else "",
        f"{len(deleted)} deleted" if deleted else "",
    ]
    summary = ", ".join(part for part in changes if part)
    if not summary:
        display.operation(
            "Checked", f"{skipped} {noun}, all current", elapsed=elapsed, style="muted"
        )
        return

    unchanged = f", {skipped} unchanged" if skipped else ""
    display.operation("Fetched", f"{summary}{unchanged}", elapsed=elapsed)
    limit = _detail_limit(verbose)
    display.details("+", [_fetch_label(result) for result in added], limit=limit)
    display.details("~", [_fetch_label(result) for result in refetched], limit=limit)
    display.details("-", [_fetch_label(result) for result in deleted], limit=limit)


def _fetch_label(result: object) -> str:
    """What to call one reconciled document in a detail line: a paper's
    citekey, a docs project's name."""
    return str(getattr(result, "citekey", None) or getattr(result, "project", "?"))


@contextlib.contextmanager
def _fetch_progress(description: str, total: int, verbose: bool):
    """Yield an `advance()` to call once per fetched item.

    Suppressed under --verbose, which prints a line per item instead: a live
    progress bar and a stream of prints fight over the same terminal rows.
    """
    if verbose:
        yield lambda: None
        return
    with display.progress_bar(description, total) as advance:
        yield advance


def _plural(count: int, noun: str, plural: str = "") -> str:
    """`1 document`, `2 documents` - the count and its noun, agreeing.

    `document(s)` is the shape that avoids the decision, and it reads as
    machine output in a report whose whole point is that a person can scan
    it. Nine sites spelled it that way; this is the one place that now does
    not have to.
    """
    return f"{count} {noun if count == 1 else (plural or noun + 's')}"


def _relative_to(location: Path, directory: Path) -> str:
    """`location` written relative to `directory` when it sits inside it.

    An absolute path is usually the longest thing on a line and is one token,
    so rich breaks it mid-word across two or three rows - unreadable, and
    impossible to copy back out. Inside a workspace the leading directory is
    also the least informative part: the reader is standing in it. Falls back
    to the absolute path when the location is genuinely elsewhere, where the
    prefix is the whole point.
    """
    try:
        return str(location.relative_to(directory))
    except ValueError:
        return str(location)


def _no_bundle_error() -> CliError:
    """No `.boepie/` bundle governs the working directory."""
    return CliError(
        f"no .boepie/ bundle found in {Path.cwd()} or any parent. "
        f"Run {display.command('boepie context init')}."
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
        f"Run {display.command(f'boepie corpus list --collection {where}')} "
        f"to see what is there."
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
            f"{display.command('uv run scripts/migrate_corpus_layout.py')} "
            f"to bring the corpus "
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


@corpus.command("index")
@click.option(
    "--collection",
    "collections",
    default=_ALL,
    show_default=True,
    type=CollectionList(_CORPUS_COLLECTIONS),
    help="Comma-separated collections to index, or 'all'.",
)
@click.option(
    "-l", "--literature", "shorthand", flag_value="literature",
    help="Index the literature corpus.",
)
@click.option(
    "-d", "--docs", "shorthand", flag_value="docs", help="Index the docs corpus."
)
@click.option(
    "-n", "--notes", "shorthand", flag_value="notes", help="Index the notes corpus."
)
@embedding_options
@click.option(
    "--embedding-concurrency",
    default=None,
    type=int,
    help="Max concurrent embedding requests (default: 4). Lower this if you're "
    "hitting API rate limits.",
)
@click.option(
    "--index-name",
    default=None,
    help="Override the auto-derived index id (default: <binding>-<model>).",
)
@click.option(
    "--check-only",
    "check_only",
    is_flag=True,
    help="Report each index's state and path and build nothing. "
    "Exits non-zero if any selected index is missing or stale.",
)
@click.option("-v", "--verbose", is_flag=True, help="Show per-batch progress logging.")
@click.pass_context
def corpus_index(
    ctx: click.Context,
    collections: tuple[str, ...],
    shorthand: str | None,
    resolve_embedding,
    embedding_concurrency: int | None,
    index_name: str | None,
    check_only: bool,
    verbose: bool,
) -> None:
    """Build the search index over one or more corpus collections.

    A verb on `corpus` rather than a noun of its own, so the scope is named
    once and in one spelling: `corpus sync -l` then `corpus index -l`, not
    `corpus index --collection literature`.

    With no selection this indexes every corpus collection that has something
    to index, reporting the ones it skipped rather than failing on them - an
    empty corpus is a normal state when you did not name it. Naming one
    explicitly does make an empty one an error: you asked for that index.

    Needs nothing running by default: fastembed runs a small ONNX model
    locally on CPU (one-time model download, then fully offline). Use
    --embedding-binding=ollama for a local Ollama daemon, or
    --embedding-binding=openai for an OpenAI API key or a local
    OpenAI-compatible server (vLLM/SGLang/TGI) via --embedding-host=<url>.
    """
    _set_verbosity(verbose)
    selected = _index_selection(ctx, collections, shorthand)

    if check_only:
        _check_indices(selected, INDEX_DIR)
        return

    if index_name is not None and len(selected) > 1:
        raise CliError(
            "--index-name names one index, so it cannot be combined with "
            "several collections. Build them one at a time, or drop the flag."
        )
    _require_corpus(selected)

    # An explicit single collection is a request for that index; anything
    # broader is a sweep, where "nothing to index" is a skip, not a failure.
    sweeping = len(selected) > 1
    built = 0
    for collection in selected:
        try:
            _build_corpus_index(
                collection,
                resolve_embedding=resolve_embedding,
                embedding_concurrency=embedding_concurrency,
                index_name=index_name,
            )
        except EmptyCollectionError as error:
            if not sweeping:
                raise CliError(str(error)) from error
            display.operation(
                "Skipped", f"{collection} index - no documents", style="muted"
            )
            continue
        built += 1

    if sweeping:
        # Coloured by what happened, like every other line: green when
        # something was built, dim when nothing was. It was unconditionally
        # dim, so one run could print `Indexed 122 chunks ...` in green and
        # `Indexed 2 of 3 collections` in grey two lines later - the same verb
        # in two colours for no reason a reader could see.
        display.operation(
            "Indexed",
            f"{built} of {_plural(len(selected), 'collection')}",
            style="success" if built else "muted",
        )


def _index_selection(
    ctx: click.Context, collections: tuple[str, ...], shorthand: str | None
) -> tuple[str, ...]:
    """`--collection` or a `-l`/`-d`/`-n` shorthand, never both.

    The same rule `corpus add` follows, for the same reason: two ways of
    saying which collection can disagree, and silence about that would let
    the wrong index be built.
    """
    if shorthand is None:
        return collections
    source = ctx.get_parameter_source("collections")
    if source is not None and source.name != "DEFAULT":
        raise CliError(
            f"--collection and {_ADD_SHORTHANDS[shorthand]} both name a "
            f"collection. Use one."
        )
    return (shorthand,)


def _report_index_states(
    targets: list[tuple[str, Path]], *, first: bool = True
) -> list[str]:
    """One heading and state block per (collection, index root); the unusable
    ones come back.

    Reporting and deciding are split because a check must describe
    *everything* before it fails: `boepie index --check-only` covers the
    corpus and the bundle, and raising at the end of the first leg would mean
    never looking at the second - which is the one an `index status` could
    never see in the first place.
    """
    wrong: list[str] = []
    with display.following_steps(False):
        for position, (collection, index_root) in enumerate(targets):
            if position or not first:
                console.print()
            console.print(
                display.collection_root(collection, index_root / collection),
                soft_wrap=True,
            )
            if _index_rows(collection, index_root):
                wrong.append(collection)
    return wrong


def _check_indices(collections: tuple[str, ...], index_root: Path) -> None:
    """Report each selected corpus index's state, building nothing.

    The read half of the same command, as `register --check-only` is - the
    analysis is identical, only the applying is skipped.
    """
    wrong = _report_index_states([(name, index_root) for name in collections])
    if wrong:
        raise _unusable_indices_error(wrong)


def _unusable_indices_error(wrong: list[str]) -> CliError:
    """The one refusal for indices that cannot be searched, naming the commands
    that rebuild exactly those.

    The corpus collections collapse into a single `--collection a,b,c`, since
    one command builds them all; the bundle needs its own, because its index
    is not a corpus collection at all. Two commands at most, never one per
    collection.
    """
    corpus_names = [name for name in wrong if name != _CONTEXT_COLLECTION]
    commands = []
    if corpus_names:
        commands.append(f"boepie corpus index --collection {','.join(corpus_names)}")
    if _CONTEXT_COLLECTION in wrong:
        commands.append("boepie context index")
    return CliError(
        f"{_plural(len(wrong), 'index', 'indices')} not usable: "
        f"{', '.join(wrong)}. Run "
        + " then ".join(display.command(name) for name in commands)
        + "."
    )


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

    **A corpus with no directory yet is reported as absent, not as empty.**
    On a machine where nothing has been fetched the old output described
    three directories that do not exist, counted zero documents in each, and
    then listed all 17 manifest citekeys and 3 projects as `missing:` - a
    wall of text about a corpus that has no place on this machine at all.
    Diffing against the packaged manifest only means something once there is
    something to diff. `context status` already refuses this way when there
    is no bundle; when every selected collection is absent this does the
    same, and a mixed machine gets one row per absent collection instead.
    """
    if all(not _corpus_collection_dir(name).is_dir() for name in collections):
        selected = ", ".join(collections)
        # `boepie init` whichever collection was asked for: it is the one
        # command that creates a corpus directory, and it creates all three.
        # It used to name `setup`, which populates literature and docs and
        # never notes - so `--collection notes` was told to run something
        # that would not have created it.
        raise CliError(
            f"no corpus on this machine - {selected} "
            f"{'has' if len(collections) == 1 else 'have'} no directory to "
            f"report on. Run {display.command('boepie init')} to create one."
        )
    for position, collection in enumerate(collections):
        _corpus_status_one(collection, first=position == 0)


def _corpus_status_one(collection: str, *, first: bool) -> None:
    collection_dir = _corpus_collection_dir(collection)

    # A blank line between collections: three of these run together
    # otherwise, and the heading is the only thing separating them.
    if not first:
        console.print()
    # soft_wrap, or rich breaks a long corpus path mid-token across two lines.
    console.print(display.collection_root(collection, collection_dir), soft_wrap=True)

    if not collection_dir.is_dir():
        display.warning(
            "not created yet",
            lead=_status_label("corpus"),
            indent="  ",
        )
        display.next_step("boepie init")
        return

    documents = _corpus_documents(collection)
    boepie_managed, user_managed = _managed_counts(documents)
    display.muted(
        f"{len(documents)} total, {boepie_managed} boepie-managed, "
        f"{user_managed} yours",
        lead=_status_label("documents"),
        indent="  ",
    )
    # The index, with the corpus it was built from, rather than under an
    # `index status` of its own: an index belongs to the thing it indexes,
    # and a reader asking "is literature in good shape" wants both answers
    # in one place. After the document count, which is what it is a claim
    # about - and only when there is something to index, since "not built
    # yet" against an empty collection sends the reader in a circle.
    if documents:
        _index_rows(collection, INDEX_DIR)

    if collection == "notes":
        # No manifest to diff against: notes are always yours.
        if not documents:
            display.next_step(_corpus_population_command(collection))
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
        yours = {
            document.natural_key
            for document in documents
            if document.frontmatter.get("managed_by") == "user"
        }
        missing = sorted(set(entries) - present - yours)
        claimed = sorted(set(entries) & yours)
        orphaned = sorted(
            document.natural_key
            for document in documents
            if document.frontmatter.get("managed_by") == "boepie"
            and document.natural_key not in entries
        )
        label = "paper"
    else:
        projects = {project.project for project in load_docs_manifest(DOCS_DIR)}
        by_management = {
            managed: {
                str(lookup_path(document.frontmatter, "docs.project"))
                for document in documents
                if document.frontmatter.get("managed_by") == managed
            }
            for managed in ("boepie", "user")
        }
        fetched_projects = by_management["boepie"]
        missing = sorted(projects - fetched_projects - by_management["user"])
        claimed = sorted(projects & by_management["user"] - fetched_projects)
        orphaned = sorted(fetched_projects - projects - {"None"})
        label = "project"

    if missing:
        display.warning(
            f"{_plural(len(missing), label)} in the manifest not fetched yet",
            lead=_status_label("missing"),
            indent="  ",
        )
        _status_items(missing)
    if claimed:
        # Separated from `missing` because the fix is different, and because
        # a fetch cannot resolve it: these are entries the manifest names
        # that this machine holds as `managed_by: user`, which `reconcile`
        # never touches at any step. Calling them "not fetched yet" sent the
        # reader round a loop - status says fetch, fetch does nothing, status
        # says fetch.
        display.muted(
            f"{_plural(len(claimed), label)} in the manifest are yours here, so "
            f"fetch leaves them alone",
            lead=_status_label("yours"),
            indent="  ",
        )
        _status_items(claimed)
    if orphaned:
        display.warning(
            f"{_plural(len(orphaned), label)} no longer in the manifest "
            f"(next fetch deletes them)",
            lead=_status_label("orphaned"),
            indent="  ",
        )
        _status_items(orphaned)
    if not missing and not orphaned:
        display.muted(
            "current with the one boepie ships",
            lead=_status_label("manifest"),
            indent="  ",
        )
    else:
        display.next_step(_corpus_population_command(collection))
    if claimed:
        # In the value column and kept short, like the "no HTML" advice
        # above: rich would otherwise break the command across a line end,
        # which is the one thing a line naming a command must not do.
        display.info(
            f"hand back: "
            f"{display.command(f'boepie corpus remove --collection {collection} <id>')}",
            indent=_STATUS_VALUE_INDENT,
        )


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
            f"Add one with: "
            f"{display.command(f'boepie corpus add --collection {collection} <identifier>')}",
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
    display.info(f"{_plural(len(documents), 'document')}.", indent="\n")


def _reject_unusable_citekey(citekey: str) -> None:
    """A citekey is typed into a `.bib` and into `read_literature`, so it has
    to survive both: no whitespace, no separators, not empty."""
    if not citekey.strip():
        raise CliError("--citekey cannot be empty.")
    bad = [
        character for character in citekey if character.isspace() or character in "/\\,"
    ]
    if bad:
        raise CliError(
            "a citekey is a single token you can type into a .bib entry and "
            "into read_literature, so it cannot contain whitespace, a slash "
            "or a comma."
        )


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
@click.option(
    "--citekey",
    default=None,
    help="literature only: a new citekey, which is also a read handle. Must "
    "not already be taken in the collection.",
)
def corpus_move(
    collection: str,
    document_id: str,
    group: str | None,
    title: str | None,
    citekey: str | None,
) -> None:
    """Move, rename or re-key a document without breaking its read handles.

    A document is addressed by its `id`, never by its path, so regrouping and
    retitling are both safe: every `read_literature`/`read_docs`/`read_notes`
    handle, and every search hit already in an agent's context, stays valid.
    Rebuild the index afterwards so the recorded source paths match again.
    """
    if group is None and title is None and citekey is None:
        raise CliError("nothing to do: pass --group, --title, --citekey, or any two.")
    if citekey is not None and collection != "literature":
        raise CliError(
            f"--citekey applies to literature only, not to {collection}: only "
            f"literature documents carry a bib block."
        )

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

    # The citekey is literature's natural key: duplicate detection compares
    # it, and `read_literature` accepts it as an alias for the surrogate id.
    # Two documents sharing one would make the alias ambiguous, and an
    # ambiguous alias is dropped rather than guessed - so the handle would
    # stop working for both.
    if citekey is not None:
        _reject_unusable_citekey(citekey)
        taken_by = next(
            (
                other
                for other in documents
                if other.id != document_id and other.natural_key == citekey
            ),
            None,
        )
        if taken_by is not None:
            raise CliError(
                f"citekey '{citekey}' is already used by id={taken_by.id}. "
                f"Citekeys are how a paper is cited and read, so they have to "
                f"stay unique within the collection."
            )
        bib_block = dict(source.frontmatter.get("bib") or {})
        bib_block["citekey"] = citekey
        updates["bib"] = bib_block

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

    display.operation(
        "Moved",
        f"{new_title} (id={document_id}) -> "
        f"{moved.md_path.relative_to(collection_dir)}",
    )
    if citekey is not None:
        display.muted(f"citekey is now '{citekey}'.")
    display.next_step(
        f"boepie corpus index --collection {collection}",
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
            f"Add one with: "
            f"{display.command(f'boepie corpus add --collection {collection} <identifier>')}",
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
    display.info(f"{_plural(len(documents), 'document')}.", indent="\n")


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
    except ValueError as error:
        raise CliError(one_line(error)) from error
    except KeyError as error:
        raise CliError(
            f"{one_line(error.args[0])} Use a document_id and chunk_index from a "
            f"{display.command(f'boepie search --collection {collection}')} hit."
        ) from error


# ---------------------------------------------------------------------------
# Context bundle: init, fetch, apply, status, reset
# ---------------------------------------------------------------------------


def _build_context_index(bundle_dir: Path) -> None:
    """Build the bundle's own BM25 index and report the count.

    No progress bar: BM25 needs no embedding backend, so this is under a
    second even on a full bundle. A bar that appears and vanishes inside one
    frame is worse than none.
    """
    started = time.monotonic()
    manifest = _run(
        build(
            ContextLoader(bundle_dir),
            embedding=None,
            index_root=index_root_for(bundle_dir),
        )
    )
    display.operation(
        "Indexed",
        f"{_plural(manifest.count, 'chunk')} into {bundle_dir.name}/.index (BM25 only)",
        elapsed=time.monotonic() - started,
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
        display.note(
            f"unused legacy knowledge index at {legacy_dir} "
            f"(superseded by the per-bundle one); safe to delete."
        )


@cli.group()
def context() -> None:
    """Manage the `.boepie/` context bundle (init, sync, status, reset)."""


@context.command("init")
@click.option(
    "--directory",
    type=click.Path(exists=True, file_okay=False, path_type=str),
    default=".",
    show_default=True,
    help="Target directory where .boepie/ will be created.",
)
@click.option("--skills", is_flag=True, help="(not implemented yet)")
@click.option("--hooks", is_flag=True, help="(not implemented yet)")
def context_init(directory: str, skills: bool, hooks: bool) -> None:
    """Initialize the `.boepie/` context bundle.

    Creates the bundle from the content this boepie ships (in the venv it is
    installed in - nothing is downloaded), appends the pointer to AGENTS.md,
    and builds the BM25 search index into the bundle's own `.index/`
    (git-ignored, so the committable bundle carries no derived state).
    """
    if skills:
        display.note("--skills not implemented yet")
    if hooks:
        display.note("--hooks not implemented yet")

    target_dir = Path(directory).resolve()
    started = time.monotonic()
    try:
        init_bundle(target_dir)
    except FileExistsError as error:
        raise CliError(str(error)) from error

    agents_md = target_dir / "AGENTS.md"
    append_agents_pointer(agents_md)

    bundle_dir = target_dir / ".boepie"
    display.operation(
        "Initialized",
        f"bundle at {_relative_to(bundle_dir, target_dir)}",
        elapsed=time.monotonic() - started,
    )

    _note_legacy_global_index()
    display.next_step("boepie context index")


@context.command("sync")
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
def context_sync(directory: str, force_targets: tuple[str, ...]) -> None:
    """Converge the bundle with the content this boepie ships.

    Rewrites every `managed_by: boepie` file from the content in the venv
    boepie is installed in, deletes orphaned boepie-managed files, preserves
    every `managed_by: user` file byte-for-byte, and rebuilds the bundle's
    own BM25 search index under `.boepie/.index/`. Pass --force with
    one or more bundle-relative paths to revert specific `managed_by: user`
    files back to boepie-managed instead (see `context reset` to discard
    every local file at once).
    """
    target_dir = Path(directory).resolve()
    started = time.monotonic()
    try:
        apply_bundle(target_dir, context_content_dir(), force_paths=force_targets)
    except (FileNotFoundError, ValueError) as error:
        raise CliError(str(error)) from error

    bundle_dir = target_dir / ".boepie"
    display.operation(
        "Applied",
        f"bundle at {_relative_to(bundle_dir, target_dir)}",
        elapsed=time.monotonic() - started,
    )

    _note_legacy_global_index()
    # Not indexed here. Converging the bundle and indexing it are two steps
    # with two owners, the same way `corpus sync` leaves `corpus index` to
    # follow it - and context was the only noun that did both, so it was the
    # one place the model lied.
    display.next_step("boepie context index")


@context.command("index")
@click.option(
    "--directory",
    type=click.Path(exists=True, file_okay=False, path_type=str),
    default=".",
    show_default=True,
    help="Target directory containing .boepie/.",
)
@click.option(
    "--check-only",
    "check_only",
    is_flag=True,
    help="Report the index's state and path and build nothing. "
    "Exits non-zero if it is missing or stale.",
)
def context_index(directory: str, check_only: bool) -> None:
    """Build the bundle's own BM25 search index.

    Per-project and BM25-only, so it lands inside the bundle it was built
    from (`<bundle>/.index/`) rather than in the machine-global store: two
    projects must not share one index of two different bundles. That is also
    why this is `context index` and not a collection of `corpus index` -
    they share no storage, no scope and no retrieval stack.
    """
    bundle_dir = _require_bundle(Path(directory).resolve())
    if check_only:
        console.print(
            display.collection_root(
                _CONTEXT_COLLECTION, index_root_for(bundle_dir) / _CONTEXT_COLLECTION
            ),
            soft_wrap=True,
        )
        with display.following_steps(False):
            unusable = _index_rows(_CONTEXT_COLLECTION, index_root_for(bundle_dir))
        if unusable:
            raise CliError(
                f"the bundle's index is not usable. Run "
                f"{display.command('boepie context index')}."
            )
        return
    _build_context_index(bundle_dir)


def _require_bundle(target_dir: Path) -> Path:
    """The `.boepie/` governing `target_dir`, or the one refusal for its absence."""
    bundle_dir = target_dir / ".boepie"
    if not bundle_dir.is_dir():
        raise _no_bundle_error()
    return bundle_dir


@context.command()
@click.option(
    "--directory",
    type=click.Path(exists=True, file_okay=False, path_type=str),
    default=".",
    show_default=True,
    help="Target directory containing .boepie/.",
)
def status(directory: str) -> None:
    """Report the bundle's state relative to installed versions, and its index."""
    target_dir = Path(directory).resolve()
    try:
        status_result = bundle_status(target_dir)
    except (FileNotFoundError, ValueError) as error:
        raise CliError(str(error)) from error

    report = display.success if status_result.state == "current" else display.warning
    report(status_result.detail, lead=f"{status_result.state}:")

    # The bundle's index is one more fact about the bundle, so it is reported
    # here rather than by an `index status` of its own - which could not see
    # it anyway, having only ever enumerated the machine-global INDEX_DIR.
    bundle_dir = target_dir / ".boepie"
    _index_rows(_CONTEXT_COLLECTION, index_root_for(bundle_dir))
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
    upstream counterpart to revert to (unlike `context sync --force`, which
    only reverts a named file when boepie still has something to revert it
    to). Prompts for confirmation naming every local file that would be lost
    unless --yes is passed or there is nothing to lose.
    """
    target_dir = Path(directory).resolve()
    bundle_dir = target_dir / ".boepie"
    if not bundle_dir.exists():
        raise CliError(
            f"no bundle at {bundle_dir}. "
            f"Run {display.command('boepie context init')} first."
        )

    local_paths = list_source_local_files(bundle_dir)
    if local_paths and not yes:
        for relative_path in local_paths:
            display.info(str(relative_path), indent="  ")
        confirmed = click.confirm(
            f"This permanently deletes {_plural(len(local_paths), 'local file')} listed above "
            "and rebuilds .boepie/ from scratch. Continue?",
            default=False,
        )
        if not confirmed:
            raise CliError("reset cancelled; no changes made.")

    try:
        reset_bundle(target_dir)
    except FileNotFoundError as error:
        raise CliError(str(error)) from error

    display.operation("Reset", f"bundle at {_relative_to(bundle_dir, target_dir)}")

    _build_context_index(bundle_dir)
    _note_legacy_global_index()


# ---------------------------------------------------------------------------
# Sync: composite convergence (corpus sync -> context sync), which
# apply/init)
# ---------------------------------------------------------------------------


# The collections `sync` converges: fetched into the corpus, then indexed
# here. There is no prebuilt index to download for either - boepie publishes
# none, so every index on a machine is built on it. For literature that has
# always been true (converted paper text is not boepie's to redistribute);
# docs joined it when index assets were dropped altogether.
_SYNC_COLLECTIONS = ("literature", "docs")


def _sync_network_step(
    ctx: click.Context,
    command: click.Command,
    label: str,
    **params: object,
) -> None:
    """Run the network step of `sync`/`setup` - the corpus fetch, now the only
    one - and turn a failure into a warning instead of aborting, so the
    convergence steps that follow still run against whatever was previously
    fetched and against the content in the venv.

    `command` is invoked exactly as its own CLI entry point would be (missing
    options fall back to that command's own defaults via `ctx.invoke`), so
    this adds no fetch logic of its own - only the warn-and-continue wrapper.
    There is no quieting here any more. `sync` used to run every step inside
    `console.capture()` and print one summary line, which hid its slowest leg
    and - as a side effect nobody decided - disabled the progress bars too,
    since a captured console is not a terminal. `-q` on the top-level group
    is the one place that turns output down now.
    """
    try:
        ctx.invoke(command, **params)
    except Cancelled:
        # Deliberately not swallowed. This wrapper exists to carry on past a
        # fetch that *failed* - an unreachable arXiv should still leave the
        # previously fetched corpus indexed - and a Ctrl-C is the one thing
        # that must stop the whole run instead. It used to arrive here as
        # `SystemExit(130)` and be reported as `{label} failed: 130`, after
        # which every later phase ran and the command exited 0.
        raise
    except (SystemExit, httpx.HTTPError) as error:
        display.note(f"{label} failed: {error}")


def _report_index_drift(target_dir: Path) -> None:
    """After converging, say what that did to each index - and nothing more.

    `sync` used to rebuild them itself. Reporting instead is the same split
    every noun already follows (`corpus sync` -> `corpus index`), and it is
    the more honest of the two: the reader learns that two papers arrived and
    that the literature index no longer covers them, rather than waiting out
    a rebuild nobody announced.

    The analysis is `_index_plan`'s, the one `setup` already used to decide -
    so this reports exactly what the `boepie index` it names would do, rather
    than a second opinion about it.
    """
    drifted: list[str] = []
    # Every corpus collection, not just the two `sync` fetches. `notes` is
    # where documents you added yourself land, so it is precisely the case
    # this report exists for - sync did not put them there, but it is the
    # command that tells you the index has not caught up.
    for collection in _CORPUS_COLLECTIONS:
        action, reason = _index_plan(collection)
        if action == "empty":
            continue
        if action == "keep":
            display.operation(
                "Checked", f"{collection} index - {reason}", style="muted"
            )
            continue
        drifted.append(collection)
        display.operation(
            "Checked",
            f"{collection} index - "
            f"{'not built yet' if action == 'build' else reason}",
            style="warning",
        )

    bundle_index = index_root_for(target_dir / ".boepie") / _CONTEXT_COLLECTION
    if not (bundle_index / "latest.json").is_file():
        drifted.append(_CONTEXT_COLLECTION)
        display.operation("Checked", "context index - not built yet", style="warning")
    else:
        # `context sync` rewrites every boepie-managed file from the venv, so
        # the bundle's index is behind whenever anything actually changed.
        # `index_freshness` reads the recorded digests rather than guessing.
        try:
            manifest = json.loads(
                (bundle_index / "manifest.json").read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            manifest = {}
        freshness = index_freshness(manifest.get("built_from"), _CONTEXT_COLLECTION)
        if freshness.state == "stale" or freshness.added:
            drifted.append(_CONTEXT_COLLECTION)
            display.operation(
                "Checked", "context index - behind the bundle", style="warning"
            )
        else:
            display.operation("Checked", "context index - current", style="muted")

    if drifted:
        display.next_step("boepie index")


@cli.command()
@click.option(
    "--directory",
    type=click.Path(exists=True, file_okay=False, path_type=str),
    default=".",
    show_default=True,
    help="Workspace holding the .boepie/ bundle to converge.",
)
@click.option("-v", "--verbose", is_flag=True, help="Show progress per item.")
@click.pass_context
def sync(ctx: click.Context, directory: str, verbose: bool) -> None:
    """Converge every collection with its source: the corpora and the bundle.

    The populate half of the pair. `boepie init` scaffolds - fast, offline,
    idempotent - and this is the slow leg: `corpus sync` pulls each paper
    from arXiv and crawls each documentation site, and `context sync`
    converges the bundle with the content in this venv. The first run on a
    machine takes minutes; both fetches are resumable, so later runs pick up
    only what is new.

    **It scaffolds nothing.** An uninitialised workspace is refused, naming
    `boepie init`. It used to create the `.boepie/` bundle itself when there
    was none, which made a fresh directory look like a working one - the
    command reported success against state it had just invented, and there
    was no point at which anyone had said "set this up here".

    **There is no `--only`.** It named one noun at a time, which is what the
    per-noun commands already are: `boepie sync --only corpus` was a second
    spelling of `boepie corpus sync`, with a second place for the two to
    disagree. This command means both.

    **It builds no index.** Converging content and rebuilding the index over
    it are two different costs - a fetch is seconds, an embed is minutes -
    and folding them together meant a run that only wanted to check for new
    papers spent that time without being asked. What it does instead is
    *say* what the convergence did to each index, which is information the
    silent rebuild never gave, and name `boepie index`.
    """
    target_dir = Path(directory).resolve()
    _require_initialised(target_dir)

    # `corpus sync` closes by advising `corpus index`, once per collection.
    # This command reports index drift as a whole at the end and names
    # `boepie index` once, so the per-collection advice would be the same
    # thing said four times over.
    with display.following_steps(False):
        _sync_network_step(
            ctx,
            corpus_sync,
            f"corpus sync --collection {','.join(_SYNC_COLLECTIONS)}",
            collections=_SYNC_COLLECTIONS,
            force_targets=(),
            delay=None,
            verbose=verbose,
        )

    with display.following_steps(False):
        ctx.invoke(context_sync, directory=directory)

    _report_index_drift(target_dir)


def _suggest_registration(target_dir: Path) -> None:
    """Close the chain by naming `boepie register`, if nothing is registered.

    `init` -> `sync` -> `index` -> `register` is the order `setup` runs them,
    and each step naming the next is what makes that order discoverable
    without reading `setup`'s source. Silent once a config exists: advice to
    do something already done is noise, and registration is the one genuinely
    optional step, so nagging about it would be wrong.
    """
    for name in DEFAULT_TARGETS:
        relative = target_named(name).relative_path
        if relative and (target_dir / relative).is_file():
            return
    display.next_step("boepie register")


def _require_initialised(target_dir: Path, *, corpus: bool = True) -> None:
    """Refuse a workspace nothing has initialised, naming `boepie init`.

    The whole point of splitting `init` from `sync`: scaffolding is one
    explicit act, so everything else can assume it happened and say so when
    it did not. Before this, four commands each answered "is this ready" for
    themselves and answered differently - `sync` created a bundle, `corpus
    sync` created its directories by writing into them, `setup._index_plan`
    had four answers and `corpus status` a fifth - and none of them could be
    told to stop.
    """
    if not (target_dir / ".boepie").is_dir():
        raise CliError(
            f"no context bundle at {target_dir}. "
            f"Run {display.command('boepie init')} first."
        )
    if corpus:
        _require_corpus(_SYNC_COLLECTIONS)




# ---------------------------------------------------------------------------
# Setup: one command from a fresh install to a working workspace
# ---------------------------------------------------------------------------


def _active_index_manifest(collection: str) -> dict[str, Any] | None:
    """The manifest of the index `search` would currently use, if there is one.

    Read off disk rather than rebuilt from configuration, and tolerant of a
    truncated or foreign file: setup's job here is to decide whether to
    build, and "I could not read it" is a reason to build, not to abort.
    """
    latest = INDEX_DIR / collection / "latest.json"
    if not latest.is_file():
        return None
    try:
        index_id = json.loads(latest.read_text(encoding="utf-8")).get("index_id")
        manifest_path = INDEX_DIR / collection / str(index_id) / "manifest.json"
        loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return None
    return loaded if isinstance(loaded, dict) else None


# What setup does about one collection's index, and why. `empty` is a
# separate answer from `keep` so the reason can be stated before a build is
# announced rather than after one is abandoned.
type IndexAction = Literal["build", "rebuild", "keep", "empty"]


def _index_plan(collection: str) -> tuple[IndexAction, str]:
    """Whether this collection's index needs building, and why.

    Wider than what *serving* refuses. `index_freshness` calls an index with
    new documents beside it `in step`, because incomplete is not wrong and
    refusing to serve it would break the ordinary `corpus add` -> `index
    build` gap. Setup is the command that closes that gap, so here an
    addition counts too.
    """
    try:
        documents = len(_corpus_documents(collection))
    except (OSError, ValueError, CliError):
        documents = 0
    if not documents:
        return "empty", "no documents"

    manifest = _active_index_manifest(collection)
    if manifest is None:
        return "build", f"{_plural(documents, 'document')}, no index yet"

    freshness = index_freshness(manifest.get("built_from"), collection)
    if freshness.state == "stale":
        counts = ", ".join(
            part
            for part in (
                f"{freshness.changed} changed" if freshness.changed else "",
                f"{freshness.gone} gone" if freshness.gone else "",
            )
            if part
        )
        return "rebuild", f"{counts} of {_plural(freshness.document_count, 'document')}"
    if freshness.added:
        return (
            "rebuild",
            f"{freshness.added} of "
            f"{_plural(freshness.document_count, 'document')} added",
        )
    if freshness.state == "unrecorded":
        return "rebuild", "built before the freshness check existed"
    if freshness.state == "corpus absent":
        return "keep", "corpus not on this machine"
    return "keep", f"{_plural(freshness.document_count, 'document')}, current"


def _report_mcp_targets(
    results: list[TargetResult], directory: Path, command: list[str]
) -> None:
    """One line for the whole registration, then the files it wrote.

    Collapsed rather than one line per agent because the interesting facts
    are which agents can now launch boepie and which files carry that - and
    two agents routinely share one file, which a per-agent listing has to
    explain away ("same file as claude") instead of just showing two names
    against one path. A failure still gets its own diagnostic: it is the one
    case where the agent's name is the point.
    """
    registered = [result.name for result in results if result.status == "written"]
    current = [result.name for result in results if result.status == "current"]
    working = sorted(registered + current)
    if working:
        aside = "" if registered else " - already current"
        display.operation("Registered", f"boepie with {', '.join(working)}{aside}")
        written = sorted(
            {
                _relative_to(result.path, directory)
                for result in results
                if result.status == "written" and result.path is not None
            }
        )
        display.details("+", written)

    # An agent that was *asked for* and is not installed is still reported.
    # This is not the "here is every agent boepie knows and did not touch"
    # block that used to close every run - that was help text. This is the
    # answer to a question the user asked by naming the agent, and without it
    # the report would claim boepie set up something it did not.
    for result in results:
        if result.status == "skipped":
            display.operation(
                "Skipped", f"{result.name} - {result.detail}", style="muted"
            )

    failed = [result for result in results if result.status == "failed"]
    for result in failed:
        display.note(f"could not register {result.name}: {result.detail}")
    if failed:
        # The paste-in definition used to close *every* run, alongside a list
        # of every agent boepie knows and did not touch - help text in the
        # middle of a report. Here it is the answer to a question the reader
        # now actually has, so it appears only when boepie could not do the
        # job itself.
        display.hint("register it by hand with:")
        console.print(manual_definition(command))


_DIRECTORY_OPTION = click.option(
    "--directory",
    type=click.Path(exists=True, file_okay=False, path_type=str),
    default=".",
    show_default=True,
    help="The workspace to act on.",
)
_AGENTS_OPTION = click.option(
    "--agents",
    default=",".join(DEFAULT_TARGETS),
    show_default=True,
    type=CollectionList(TARGET_NAMES),
    help="Comma-separated agents, or 'all'. The default is every agent whose "
    "config stays inside the workspace; codex and gemini change user-level "
    "config through their own CLIs.",
)
_FORCE_OPTION = click.option(
    "--force",
    is_flag=True,
    help="Replace an existing boepie entry in an agent's config.",
)


@cli.command()
@_DIRECTORY_OPTION
@click.pass_context
def init(ctx: click.Context, directory: str) -> None:
    """Scaffold a workspace: the context bundle and this machine's corpora.

    Fast, offline and idempotent - nothing is fetched, nothing is embedded
    and no agent is touched. `boepie sync` fills what this creates, `boepie
    register` points an agent at it, and `boepie setup` runs all three.

    \b
    context   The `.boepie/` bundle in --directory, created if absent. An
              existing one is reported and left alone: converging it is
              `context sync`'s job, not a scaffold's.
    corpus    The machine-global literature, docs and notes directories.
              **Shared by every workspace**, so the second project on a
              machine finds them already made rather than getting its own.
    """
    target_dir = Path(directory).resolve()

    # The whole block, because `context init` closes by naming `context
    # index` - which is a step this command takes two lines later.
    with display.following_steps(False):
        if (target_dir / ".boepie").is_dir():
            display.operation(
                "Checked",
                f"bundle at {_relative_to(target_dir / '.boepie', target_dir)}",
                style="muted",
            )
        else:
            ctx.invoke(context_init, directory=directory, skills=False, hooks=False)

        ctx.invoke(corpus_init, collections=_CORPUS_COLLECTIONS)
        # The bundle ships with content, so unlike the empty corpus
        # directories it has something to index the moment it exists.
        # `context init` leaves that to its own verb; the composite closes
        # the gap.
        ctx.invoke(context_index, directory=directory, check_only=False)
    display.next_step("boepie sync")


@cli.command()
@_DIRECTORY_OPTION
@_AGENTS_OPTION
@_FORCE_OPTION
@click.option(
    "--check-only",
    "check_only",
    is_flag=True,
    help="Report what each agent's config holds and change nothing. "
    "Exits non-zero if any selected agent is unregistered or stale.",
)
def register(
    directory: str, agents: tuple[str, ...], force: bool, check_only: bool
) -> None:
    """Point an agent at this boepie, or check that one already is.

    Its own command rather than a phase of `init` because it is the one part
    of setting up a workspace that is genuinely optional - you may want the
    bundle and the corpora without changing any agent's configuration - and
    because "is my agent pointed at the right boepie" is a question worth
    being able to ask on its own.

    The command written into each config is an absolute path to **this**
    installation's console script, and that is the whole point of the step:
    boepie's pipeline tools drive stimela's configuration chain in-process,
    so the server has to run in the venv stimela is installed in. A boepie
    installed as an isolated tool starts cleanly and then sees zero cabs,
    which is exactly the failure `--check` is for - the agent lists no
    boepie tools and explains nothing.
    """
    target_dir = Path(directory).resolve()
    command = server_command()

    # The venv, first, because "is this the boepie that lives beside stimela"
    # is the single thing most likely to be wrong and every line below is
    # worthless if it is - and because this is the command that *commits* to
    # that venv by writing its path into every agent config. `init` and
    # `sync` print no such line: neither changes a registration, so naming
    # the environment there would be ceremony. uv draws the same
    # distinction, printing `Using CPython 3.14.3` when it creates a venv and
    # nothing when it reuses one.
    display.using(f"boepie {__version__} ({command[0]})")

    if check_only:
        _report_registrations(target_dir, agents, command)
        return
    _register_agents(target_dir, agents, command, force=force)


# How each inspected state reads, and whether it means something is wrong.
# `opaque` is not a fault: codex and gemini can be asked whether they know
# the server but not what they would launch, and calling that `current`
# would be the same unfounded claim `--check` exists to catch.
_REGISTRATION_WORDING: dict[str, tuple[str, str]] = {
    "current": ("registered, and points at this boepie", "success"),
    "opaque": ("registered - its CLI does not say with which boepie", "muted"),
    "absent": ("not installed", "muted"),
    "missing": ("not registered", "warning"),
    "stale": ("registered, but points somewhere else", "warning"),
}
_REGISTRATION_FAULTS = frozenset({"missing", "stale"})


def _report_registrations(
    target_dir: Path, agents: tuple[str, ...], command: list[str]
) -> None:
    """Describe each agent's registration - state, not a sequence of steps.

    So it is the aligned `label:` column the status commands use, not the
    operation lines: nothing was done here, and writing `Checked` against every
    row would claim an action where there was only a look.
    """
    faults: list[str] = []
    stale = False
    for position, name in enumerate(agents):
        inspection = inspect_target(name, target_dir, command)
        wording, style = _REGISTRATION_WORDING[inspection.state]
        if position:
            console.print()
        console.print(
            display.collection_root(name, inspection.path or "(no file)"),
            soft_wrap=True,
        )
        writer = {
            "success": display.success,
            "warning": display.warning,
            "muted": display.muted,
        }[style]
        sentence = wording + (f" - {inspection.detail}" if inspection.detail else "")
        # Wrapped into the value column rather than left to rich, which
        # restarts every continuation at column zero where it collides with
        # the next heading and the block stops reading as one value.
        head, *rest = _wrap_into_value_column([sentence])
        writer(head, lead=_status_label("state"), indent="  ")
        for line in rest:
            display.info(line, indent=_STATUS_VALUE_INDENT)
        if inspection.registered_command:
            # The command it would actually launch, so a stale entry names
            # the venv it points at instead of only saying that it is wrong.
            display.muted(
                " ".join(inspection.registered_command),
                lead=_status_label("launches"),
                indent="  ",
            )
        if inspection.state in _REGISTRATION_FAULTS:
            faults.append(name)
            stale = stale or inspection.state == "stale"

    if faults:
        # `--force` when anything is *stale*: `register` on its own leaves an
        # entry that is already there, so naming the bare command would send
        # the reader to something that reports `skipped` and changes nothing.
        fix = "boepie register --force" if stale else "boepie register"
        # Non-zero so `--check` is usable as a precondition in a script, the
        # way `black --check` or `terraform plan -detailed-exitcode` are.
        raise CliError(
            f"{_plural(len(faults), 'agent')} not pointed at this boepie: "
            f"{', '.join(faults)}. Run {display.command(fix)}."
        )


@cli.command("index")
@_DIRECTORY_OPTION
@embedding_options
@click.option(
    "--embedding-concurrency",
    default=None,
    type=int,
    help="Max concurrent embedding requests (default: 4). Lower this if you're "
    "hitting API rate limits.",
)
@click.option(
    "--check-only",
    "check_only",
    is_flag=True,
    help="Report every index's state and path and build nothing. "
    "Exits non-zero if any is missing or stale.",
)
@click.option("-v", "--verbose", is_flag=True, help="Show per-batch progress logging.")
@click.pass_context
def index(
    ctx: click.Context,
    directory: str,
    resolve_embedding,
    embedding_concurrency: int | None,
    check_only: bool,
    verbose: bool,
) -> None:
    """Rebuild every index: `corpus index` then `context index`.

    The composite for the two indexing verbs, as `boepie sync` is for the two
    converging ones. It adds nothing of its own - name a collection with
    `boepie corpus index -l` when you want one.

    **`boepie sync` is not this command.** Sync converges first and then
    rebuilds only what changed; this rebuilds unconditionally and fetches
    nothing, which is what a change of embedding backend or model needs -
    the corpus has not moved, but every vector in it has to be recomputed.

    The embedding options reach the corpus leg only. The bundle's index is
    BM25-only by nature (no backend, no model), so there is nothing for them
    to apply to there.
    """
    # `embedding_options` is a decorator that turns four `--embedding-*`
    # options into a `resolve_embedding` callable, and it wraps `corpus_index`
    # too - so that leg wants the four values, not an already-resolved
    # binding, which would collide with the one it builds for itself.
    # Resolving here and taking the pieces back off works however this
    # command was reached; reading `ctx.params` did not, because a
    # programmatic `ctx.invoke` builds a fresh context whose params are empty.
    binding = resolve_embedding()
    embedding_params = {
        "embedding_binding": binding.kind,
        "embedding_model": binding.model,
        "embedding_host": binding.host,
        "embedding_dim": binding.dim,
    }
    if check_only:
        bundle_dir = _require_bundle(Path(directory).resolve())
        wrong = _report_index_states(
            [(name, INDEX_DIR) for name in _CORPUS_COLLECTIONS]
        ) + _report_index_states(
            [(_CONTEXT_COLLECTION, index_root_for(bundle_dir))], first=False
        )
        if wrong:
            raise _unusable_indices_error(wrong)
        return

    ctx.invoke(
        corpus_index,
        collections=_CORPUS_COLLECTIONS,
        shorthand=None,
        embedding_concurrency=embedding_concurrency,
        index_name=None,
        check_only=False,
        verbose=verbose,
        **embedding_params,
    )
    console.print()
    ctx.invoke(context_index, directory=directory, check_only=False)
    _suggest_registration(Path(directory).resolve())


@cli.command()
@_DIRECTORY_OPTION
@_AGENTS_OPTION
@_FORCE_OPTION
@click.option("-v", "--verbose", is_flag=True, help="Show progress per item.")
@click.pass_context
def setup(
    ctx: click.Context,
    directory: str,
    agents: tuple[str, ...],
    force: bool,
    verbose: bool,
) -> None:
    """Scaffold a workspace, fill it, and register it: init, sync, register.

    The one command between installing boepie and having an agent that can
    use it, and safe to repeat - every step converges rather than starting
    over. It adds nothing of its own, so run the three separately whenever
    you want one without the others: `init` alone touches no network, and
    `register` alone changes no content.
    """
    ctx.invoke(init, directory=directory)
    with display.following_steps(False):
        ctx.invoke(sync, directory=directory, verbose=verbose)
        # `sync` reports index drift and stops; `setup` is the command that
        # closes that gap, the same way it closes the one between `corpus
        # sync` and `corpus index`.
        ctx.invoke(index, directory=directory, check_only=False, verbose=verbose)
    ctx.invoke(
        register, directory=directory, agents=agents, force=force, check_only=False
    )



def _register_agents(
    target_dir: Path,
    agents: tuple[str, ...],
    command: list[str],
    *,
    force: bool,
) -> None:
    """Write the MCP launch config for each agent asked for.

    Scaffolding rather than convergence - fast, offline, per-workspace - so
    it belongs to `init` and never runs in `sync`, which changes no
    registration.
    """
    # Two agents can read one file - Claude Code and Copilot CLI both take
    # `.mcp.json` - so the second is reported as already covered rather than
    # written a second time.
    written_by: dict[Path, str] = {}
    results: list[TargetResult] = []
    for name in agents:
        try:
            result = apply_target(name, target_dir, command, force=force)
        except McpConfigError as error:
            result = TargetResult(name, "failed", None, str(error))
        # Keyed on the target's own file rather than on what the result
        # carries: a registration made through an agent's CLI reports the
        # command it ran, not the path that command wrote, and the next
        # agent reading that same file still needs to be told why it had
        # nothing to do.
        shared = target_named(name).relative_path
        path = target_dir / shared if shared else None
        if path is not None and path in written_by and result.status != "failed":
            result = TargetResult(name, "current", path, "")
        elif path is not None and result.status in ("written", "current"):
            written_by[path] = name
        # An agent's own CLI reports the command it ran, not the file it
        # wrote, so the result carries no path and the detail lines would
        # silently omit the very file that was created. The target declares
        # where it writes, so fill it in here.
        if result.path is None and path is not None and result.status == "written":
            result = TargetResult(result.name, result.status, path, result.detail)
        results.append(result)
    _report_mcp_targets(results, target_dir, command)

    # Nothing is said about the agents that were not asked for. Listing every
    # target boepie knows, with a JSON definition to paste, made the tail of
    # a successful run longer than the run itself - and it is help text, not
    # a report of what happened. `--agents` and `--help` carry it instead.


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
        display.hint_coordinate(f"{chunk.document_id}{section_part}", snippet)


# ---------------------------------------------------------------------------
# Config: the user-editable ~/.config/boepie/config.toml
# ---------------------------------------------------------------------------


def _check_known_key(key: str) -> None:
    """Refuse a key `config get`/`set` cannot honour, saying which kind it is.

    A key in a deferred section is declared in the schema but read by
    nothing, so setting it would change no behaviour at all. That is a
    different problem from a typo and gets a different sentence: "not
    implemented" is actionable, "unknown key" would send someone looking for
    a spelling mistake that is not there.
    """
    if settings.deferred_key(key):
        raise CliError(
            f"'{key}' is declared for future development and is not implemented "
            f"yet - nothing reads it, so setting it would have no effect."
        )
    if key not in settings.known_keys():
        raise CliError(
            f"unknown config key '{key}'. Known keys: {', '.join(sorted(settings.known_keys()))}"
        )


# A diagnostic rather than a remark, because running on built-in defaults is
# not what someone reading `config show` expects to be told in passing: a
# commented line inside the TOML block reads as part of the config, not as an
# answer to "why is none of this in my file". Lowercase and without the
# command, which `next_step` supplies in boepie's usual shape.
_MISSING_FILE_HINT = "no config file yet - boepie is running on built-in defaults"


def _warn_no_config_file() -> None:
    """Say that nothing on disk backs what was just printed.

    Always *after* the output it is about, on stderr. Both callers print a
    payload first - `config show` a whole file's worth - and a terminal
    leaves the reader at the bottom, so a diagnostic that goes first is the
    one part of the run they have to scroll back for.
    """
    display.note(_MISSING_FILE_HINT, stderr=True)
    display.next_step("boepie config init", stderr=True)


@cli.group()
def config() -> None:
    """Manage the user config file (~/.config/boepie/config.toml)."""


@config.command("init")
@click.option(
    "-f",
    "--force",
    is_flag=True,
    help="Overwrite an existing config file, discarding whatever it holds.",
)
def config_init_cmd(force: bool) -> None:
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
    display.path(settings.config_path())
    if not settings.config_file_exists():
        _warn_no_config_file()


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

    lines: list[str] = []
    if sources:
        lines.append("# Resolved config: env var > config file > built-in default.")
        lines.append(f"# Config file: {settings.config_path()}")

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

    if not settings.config_file_exists():
        # After the payload, not before it: this is forty-odd lines of TOML,
        # so a warning at the top is scrolled off by its own output and the
        # terminal leaves the reader at the bottom. Last is where it is read.
        #
        # On stderr, and outside the `--sources` guard. This is a fact about
        # the run rather than an annotation on a value, so `--no-sources`
        # must not hide it - and stdout here is valid TOML by design (see
        # `display.toml`), so a warning printed into it would land in
        # whatever file the output was redirected to.
        _warn_no_config_file()


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
            f"{display.command('boepie config init')} would instead write "
            f"every key at its default."
        )

    # The write has already succeeded; an env var shadowing it is worth
    # saying, but not worth failing the command over if resolving the other
    # keys happens to trip on something unrelated.
    env_var = settings.env_var_for(key)
    if env_var in os.environ:
        display.note(
            f"{env_var} is set in your environment and overrides the file, "
            f"so {key} still resolves to {settings.get(key)!r}."
        )
