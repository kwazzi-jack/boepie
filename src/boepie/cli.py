"""Command-line interface for boepie."""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import logging
import tarfile
from pathlib import Path

import click
import httpx
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from boepie import __version__
from boepie.config import (
    DEFAULT_TOP_K,
    EMBEDDING_BINDING,
    EMBEDDING_MODEL,
    INDEX_DIR,
)
from boepie.knowledge import (
    ContentFetchResult,
    append_agents_pointer,
    apply_bundle,
    bundle_status,
    ensure_gitignore,
    fetch_content,
    find_bundle,
    index_root_for,
    init_bundle,
    resolve_content_source,
)
from boepie.release import (
    download_verified_asset,
    release_asset_url,
)
from boepie.rag import (
    DocsLoader,
    KnowledgeLoader,
    LiteratureLoader,
    ModelBinding,
    build,
    embedding_options,
    index_id_for,
    search,
)
from boepie.rag import read as rag_read
from boepie.rag.models import Filter
from boepie.tools._retrieval import (
    VIEWS,
    format_hits,
    format_span,
    one_line,
    relative_source,
    search_with_lexical_fallback,
    with_note,
)

console = Console()

# Maps a collection name to the loader that builds it.
# Knowledge loader requires bundle_dir passed to __init__, so it's not here.
_LOADERS = {"literature": LiteratureLoader, "docs": DocsLoader}

# Threshold for hint search results, on the *raw BM25* score of the top hit
# (hint is BM25-only; see `_hint_search`). Placeholder value - the dummy
# content in this repo has no real distribution to calibrate against, so this
# needs tuning once a real corpus exists. See design/phase-2.md item 3.
_HINT_MIN_SCORE = 1.0

# The one collection whose index is per-project rather than machine-global:
# it is built from a `.boepie/` bundle, so it lives inside that bundle (see
# `boepie.knowledge.index_root_for`). Two projects sharing INDEX_DIR for it
# would silently clobber each other's index.
_KNOWLEDGE_COLLECTION = "knowledge"


def _index_root_for_collection(collection: str) -> Path:
    """Where `collection`'s index lives: inside the bundle governing the cwd
    for `knowledge`, the machine-global store for every other collection."""
    if collection != _KNOWLEDGE_COLLECTION:
        return INDEX_DIR
    bundle_dir = find_bundle()
    if bundle_dir is None:
        raise click.ClickException(
            f"no .boepie/ bundle found in {Path.cwd()} or any parent. "
            f"Run 'boepie knowledge init'."
        )
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
        raise click.ClickException("Cancelled.") from None
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        raise click.ClickException(str(error)) from error


@click.group()
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
    "--collection", default="literature", show_default=True,
    type=click.Choice(sorted(_LOADERS)), help="Which collection to build.",
)
@embedding_options
@click.option("--embedding-concurrency", default=None, type=int, help="Max concurrent embedding requests (default: 4). Lower this if you're hitting API rate limits.")
@click.option("--index-name", default=None, help="Override the auto-derived index id (default: <binding>-<model>).")
@click.option("-v", "--verbose", is_flag=True, help="Show per-batch progress logging (useful for slow builds).")
def index_build(
    collection: str,
    resolve_embedding,
    embedding_concurrency: int | None,
    index_name: str | None,
    verbose: bool,
) -> None:
    """Build the search index for a collection.

    Dev-only, but needs nothing running by default: fastembed runs a small
    ONNX model locally on CPU (one-time model download, then fully offline).
    Use --embedding-binding=ollama for a local Ollama daemon, or
    --embedding-binding=openai for an OpenAI API key or a local
    OpenAI-compatible server (vLLM/SGLang/TGI) via --embedding-host=<url>.
    """
    _set_verbosity(verbose)
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
                loader, index_root=INDEX_DIR, embedding=embedding,
                index_id=index_name, on_progress=on_progress,
            )
        )
    console.print(
        f"[green]Indexed[/green] {manifest.count} chunks into '{collection}/{manifest.index_id}' "
        f"(embedding={manifest.embedding_kind}:{manifest.embedding_model})."
    )


@index.command("fetch")
@click.option("--collection", default="literature", show_default=True)
@click.option("--tag", default="latest", show_default=True, help="GitHub release tag to fetch from.")
@click.option("--index-name", default=None, help="Which built index id to fetch (default: derived from the active embedding config).")
@click.option("--force", is_flag=True, help="Re-download and overwrite even if already present.")
@click.option("--embedding-binding", default=EMBEDDING_BINDING, show_default=True, type=click.Choice(["fastembed", "ollama", "openai"]))
@click.option("--embedding-model", default=EMBEDDING_MODEL, show_default=True)
def index_fetch(
    collection: str, tag: str, index_name: str | None, force: bool,
    embedding_binding: str, embedding_model: str,
) -> None:
    """Download a prebuilt index from a GitHub release.

    End-user path: no LLM needed, just an embedding backend matching the
    fetched index (checked lazily by `boepie search` / `search_literature`,
    warned about eagerly here too). The `knowledge` collection is not
    fetchable: it is derived from a project's own bundle, never shipped.
    """
    if collection == _KNOWLEDGE_COLLECTION:
        raise click.ClickException(
            "the knowledge index is built from a project's own .boepie/ bundle, "
            "not published as a release asset. Run 'boepie knowledge apply'."
        )

    embedding = ModelBinding(kind=embedding_binding, model=embedding_model)
    resolved_name = index_name or index_id_for(embedding)
    asset = f"{collection}-{resolved_name}.tar.gz"
    url = release_asset_url(tag, asset)

    dest = INDEX_DIR / collection / resolved_name
    if dest.exists() and not force:
        console.print(
            f"[yellow]Index '{collection}/{resolved_name}' already present at "
            f"{dest} - skipped fetching tag={tag} (use --force to re-download).[/yellow]"
        )
        return

    with console.status(f"Downloading {asset} from boepie release {tag}..."):
        try:
            data = download_verified_asset(tag, asset)
        except ValueError as error:
            console.print(f"[red]{str(error)}[/red]")
            raise SystemExit(1) from error

    (INDEX_DIR / collection).mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        tar.extractall(INDEX_DIR / collection, filter="data")

    manifest_path = dest / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("embedding_kind") != embedding.kind or manifest.get("embedding_model") != embedding.model:
            console.print(
                f"[yellow]Warning:[/yellow] fetched index was built with embedding "
                f"'{manifest.get('embedding_kind')}:{manifest.get('embedding_model')}' but "
                f"the active config is '{embedding.kind}:{embedding.model}'. Querying will "
                f"fail until these match - see BOEPIE_EMBEDDING_BINDING/BOEPIE_EMBEDDING_MODEL."
            )

    latest_path = INDEX_DIR / collection / "latest.json"
    latest_path.write_text(json.dumps({"index_id": resolved_name}, indent=2), encoding="utf-8")

    console.print(
        f"[green]Fetched[/green] '{collection}/{resolved_name}' (tag={tag}) into {dest}."
    )


@index.command("status")
def index_status() -> None:
    """Report the status of all indices under the index directory.

    For each collection, shows the active index id and whether its embedding
    config matches the active embedding environment.
    """
    if not INDEX_DIR.exists():
        console.print(f"[yellow]No index directory yet at {INDEX_DIR}[/yellow]")
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
        console.print("[yellow]No indices built or fetched yet.[/yellow]")
        return

    for collection_name, index_ids in sorted(collections_with_indices.items()):
        latest_link = INDEX_DIR / collection_name / "latest.json"
        if latest_link.exists():
            latest_data = json.loads(latest_link.read_text(encoding="utf-8"))
            active_id = latest_data.get("index_id", "unknown")
        else:
            active_id = "none"

        console.print(f"[bold]{collection_name}[/bold]: active={active_id}, available=[{', '.join(index_ids)}]")

        if active_id != "none" and active_id in index_ids:
            manifest_path = INDEX_DIR / collection_name / active_id / "manifest.json"
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                embedding_kind = manifest.get("embedding_kind")
                embedding_model = manifest.get("embedding_model")
                console.print(
                    f"  [dim]embedding: {embedding_kind}:{embedding_model}[/dim]"
                )


@index.command("list")
def index_list() -> None:
    """Enumerate all built and fetched indices on this machine.

    Shows collection names and their available index ids.
    """
    if not INDEX_DIR.exists():
        console.print(f"[yellow]No index directory yet at {INDEX_DIR}[/yellow]")
        return

    found_any = False
    for collection_dir in sorted(INDEX_DIR.iterdir()):
        if not collection_dir.is_dir():
            continue
        index_ids: list[str] = []
        for item in sorted(collection_dir.iterdir()):
            if item.is_dir() and item.name not in (".", ".."):
                index_ids.append(item.name)
        if index_ids:
            console.print(f"{collection_dir.name}: {', '.join(index_ids)}")
            found_any = True

    if not found_any:
        console.print("[yellow]No indices found.[/yellow]")


# ---------------------------------------------------------------------------
# Query: search / read - thin frontends over the retrieval tool functions
# ---------------------------------------------------------------------------
#
# `search` and `read` route through the exact helpers the search_*/read_* MCP
# tools use (`search_with_lexical_fallback`, `rag.read`) and render with the
# same `format_hits`/`format_span`, so terminal and server output cannot drift.
# They add only what the tools cannot: embedding/index overrides for pointing
# at a specific dev index, and a `--json` mode. The knowledge collection is
# BM25-only (embedding=None, mode='bm25'); its hits are bundle paths and it has
# no read counterpart.

_SEARCH_COLLECTIONS = ("literature", "docs", "knowledge")
_READ_COLLECTIONS = ("literature", "docs")


def _emit_outcome_error(error: str) -> None:
    """Turn a SearchOutcome error string (already 'Error: ...') into a
    ClickException, without click re-prefixing a second 'Error:'."""
    raise click.ClickException(error.removeprefix("Error: "))


def _hits_as_json(question: str, collection: str, view, results, note: str | None) -> str:
    """Serialise ranked hits to JSON, mirroring the F2 fields format_hits shows."""
    payload = {
        "collection": collection,
        "question": question,
        "note": note,
        "hits": [
            {
                "rank": rank,
                "rrf_score": result.score,
                "bm25_score": result.bm25_score,
                "dense_score": result.dense_score,
                "dense_rank": result.dense_rank,
                "bm25_rank": result.bm25_rank,
                "title": view.title_of(result.chunk),
                "document_id": result.chunk.document_id,
                "chunk_index": result.chunk.chunk_index,
                "section": result.chunk.section,
                "source": relative_source(
                    result.chunk.source_path, view.source_root,
                    keep_root=view.keep_source_root,
                ),
                "char_start": result.chunk.char_start,
                "char_end": result.chunk.char_end,
                "read_handle": view.read_handles,
                "text": result.chunk.text,
            }
            for rank, result in enumerate(results, 1)
        ],
    }
    return json.dumps(payload, indent=2)


@cli.command("search")
@click.argument("question")
@click.option(
    "--collection", default="literature", show_default=True,
    type=click.Choice(_SEARCH_COLLECTIONS),
)
@click.option("--top-k", default=DEFAULT_TOP_K, show_default=True, type=int)
@click.option(
    "--mode", default="hybrid", show_default=True,
    type=click.Choice(["hybrid", "dense", "bm25"]),
    help="Ignored for knowledge (BM25-only).",
)
@click.option("--snippet", default="short", show_default=True, type=click.Choice(["none", "short", "full"]), help="How much of each hit's text to show.")
@click.option("--year-min", type=int, default=None, help="Literature only: chunks from this year onward.")
@click.option("--year-max", type=int, default=None, help="Literature only: chunks up to this year.")
@click.option("--project", default=None, help="Docs only: restrict to one project, e.g. 'stimela'.")
@click.option("--index-name", default=None, help="Query a specific built index id instead of the latest.")
@embedding_options
@click.option("--json", "as_json", is_flag=True, help="Emit machine-readable JSON instead of text.")
@click.option("-v", "--verbose", is_flag=True, help="Show debug/progress logging.")
def search_cli(
    question: str, collection: str, top_k: int, mode: str, snippet: str,
    year_min: int | None, year_max: int | None, project: str | None,
    index_name: str | None, resolve_embedding, as_json: bool, verbose: bool,
) -> None:
    """Search a collection and print ranked hits (same output as the MCP tools).

    Routes through the same retrieval path as search_literature/search_docs/
    search_knowledge, so results match the server. Only needs the embedding
    backend configured (no LLM); knowledge is BM25-only and works with none.
    """
    _set_verbosity(verbose)
    view = VIEWS[collection]

    filters: list[Filter] = []
    if collection == "literature":
        if year_min is not None:
            filters.append(Filter(field="year", op="gte", value=year_min))
        if year_max is not None:
            filters.append(Filter(field="year", op="lte", value=year_max))
    if collection == "docs" and project is not None:
        filters.append(Filter(field="project", op="eq", value=project))

    if collection == _KNOWLEDGE_COLLECTION:
        embedding: ModelBinding | None = None
        mode = "bm25"  # no dense leg in the bundle's lexical-only index
    else:
        embedding = resolve_embedding()

    outcome = _run(
        search_with_lexical_fallback(
            question,
            collection=collection,
            top_k=top_k,
            mode=mode,
            filters=filters or None,
            missing_index_fix=view.missing_index_fix,
            index_root=_index_root_for_collection(collection),
            embedding=embedding,
            index_id=index_name,
        )
    )
    if outcome.error:
        _emit_outcome_error(outcome.error)

    if as_json:
        click.echo(_hits_as_json(question, collection, view, outcome.results, outcome.note))
        return

    payload = with_note(
        format_hits(
            question,
            collection,
            outcome.results,
            snippet=snippet,
            title_of=view.title_of,
            source_root=view.source_root,
            keep_source_root=view.keep_source_root,
            read_handles=view.read_handles,
            score_detail=True,
        ),
        outcome.note,
    )
    # markup=False so the "[1]" rank markers survive rich (which would read them
    # as style tags); colour still strips automatically when piped.
    console.print(payload, markup=False, highlight=False)


@cli.command("read")
@click.argument("document_id")
@click.option(
    "--collection", default="literature", show_default=True,
    type=click.Choice(_READ_COLLECTIONS),
    help="knowledge has no read: open its source path directly.",
)
@click.option("--chunk-index", type=int, default=None, help="Chunk to centre on (from a search hit). Omit to read the whole document.")
@click.option("--before", type=int, default=1, show_default=True, help="Neighbouring chunks to include before the anchor.")
@click.option("--after", type=int, default=1, show_default=True, help="Neighbouring chunks to include after the anchor.")
@click.option("--index-name", default=None, help="Read from a specific built index id instead of the latest.")
@embedding_options
@click.option("--json", "as_json", is_flag=True, help="Emit machine-readable JSON instead of text.")
@click.option("-v", "--verbose", is_flag=True, help="Show debug/progress logging.")
def read_cli(
    document_id: str, collection: str, chunk_index: int | None,
    before: int, after: int, index_name: str | None, resolve_embedding,
    as_json: bool, verbose: bool,
) -> None:
    """Expand a search hit into wider context (same output as read_docs/read_literature).

    Pass a hit's document_id and --chunk-index to pull the neighbouring chunks
    (widen with --before/--after), or omit --chunk-index to read the whole
    document. Loads the same handle the MCP read_* tools load.
    """
    _set_verbosity(verbose)
    view = VIEWS[collection]
    span = _run(
        _read_span(
            document_id, collection=collection, chunk_index=chunk_index,
            before=before, after=after, index_name=index_name,
            embedding=resolve_embedding(), view=view,
        )
    )

    if as_json:
        click.echo(json.dumps({
            "document_id": span.document_id,
            "chunk_start": span.chunk_start,
            "chunk_end": span.chunk_end,
            "char_start": span.char_start,
            "char_end": span.char_end,
            "source": relative_source(span.source_path, view.source_root, keep_root=view.keep_source_root),
            "sections": span.sections,
            "text": span.text,
        }, indent=2))
        return

    console.print(
        format_span(span, source_root=view.source_root, keep_source_root=view.keep_source_root),
        markup=False, highlight=False,
    )


async def _read_span(
    document_id: str, *, collection: str, chunk_index: int | None,
    before: int, after: int, index_name: str | None,
    embedding: ModelBinding, view,
):
    """Load a document span the way rag.read (and thus read_docs) does, turning
    the engine's typed failures into short CLI messages."""
    try:
        return await rag_read(
            document_id, chunk_index=chunk_index, before=before, after=after,
            collection=collection, index_root=_index_root_for_collection(collection),
            embedding=embedding, index_id=index_name,
        )
    except FileNotFoundError:
        raise click.ClickException(
            f"no '{collection}' index found. Run {view.missing_index_fix}."
        ) from None
    except ValueError as error:
        raise click.ClickException(one_line(error)) from error
    except KeyError as error:
        raise click.ClickException(
            f"{one_line(error.args[0])} Use a document_id and chunk_index from a "
            f"'boepie search --collection {collection}' hit."
        ) from error


# ---------------------------------------------------------------------------
# Knowledge bundle: init, fetch, apply, status
# ---------------------------------------------------------------------------

def _build_knowledge_index(bundle_dir: Path) -> None:
    """Build the bundle's own BM25 index and report the count."""
    manifest = _run(
        build(
            KnowledgeLoader(bundle_dir),
            embedding=None,
            index_root=index_root_for(bundle_dir),
        )
    )
    console.print(
        f"[green]Indexed[/green] {manifest.count} chunks into "
        f"'{bundle_dir.name}/.index/knowledge/bm25' (BM25 only)."
    )


def _note_legacy_global_index() -> None:
    """Point out (never delete) a knowledge index left in the old global store.

    Machines that ran an earlier boepie still carry
    `INDEX_DIR/knowledge/`, which nothing reads any more now that the index
    lives inside the bundle it was built from.
    """
    legacy_dir = INDEX_DIR / _KNOWLEDGE_COLLECTION
    if legacy_dir.exists():
        console.print(
            f"[yellow]Note:[/yellow] unused legacy knowledge index at {legacy_dir} "
            f"(superseded by the per-bundle one); safe to delete."
        )


@cli.group()
def knowledge() -> None:
    """Manage the `.boepie/` knowledge bundle (init, fetch, apply, status)."""


@knowledge.command()
@click.option("--tag", default="latest", show_default=True, help="GitHub release tag to fetch from.")
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
            console.print(f"[red]{str(error)}[/red]")
            raise SystemExit(1) from error

    content_dir = result.content_dir
    manifest_path = content_dir / "content-manifest.json"
    content_version = (
        json.loads(manifest_path.read_text(encoding="utf-8")).get("content_version")
        if manifest_path.exists() else None
    )
    version_note = f", content_version={content_version}" if content_version else ""
    if result.changed:
        console.print(f"[green]Fetched[/green] content (tag={tag}{version_note}) into {content_dir}.")
    else:
        console.print(f"[green]Content up to date[/green] (tag={tag}{version_note}).")


@knowledge.command()
@click.option(
    "--directory", type=click.Path(exists=True, file_okay=False, path_type=str),
    default=".", show_default=True,
    help="Target directory where .boepie/ will be created.",
)
@click.option("--skills", is_flag=True, help="(not implemented yet)")
@click.option("--hooks", is_flag=True, help="(not implemented yet)")
def init(directory: str, skills: bool, hooks: bool) -> None:
    """Initialize the `.boepie/` knowledge bundle.

    Creates the bundle from the resolved content source (cached content when
    available, else packaged seeds), appends the bundle pointer to AGENTS.md,
    and builds the BM25 search index into the bundle's own `.index/`
    (git-ignored, so the committable bundle carries no derived state).
    """
    if skills:
        console.print("[yellow]--skills not implemented yet[/yellow]")
    if hooks:
        console.print("[yellow]--hooks not implemented yet[/yellow]")

    target_dir = Path(directory).resolve()
    try:
        manifest = init_bundle(target_dir)
    except FileExistsError as error:
        raise click.ClickException(str(error)) from error

    agents_md = target_dir / "AGENTS.md"
    append_agents_pointer(agents_md)

    bundle_dir = target_dir / ".boepie"
    console.print(f"[green]Initialized[/green] bundle at {bundle_dir}")

    _build_knowledge_index(bundle_dir)
    _note_legacy_global_index()


@knowledge.command()
@click.option(
    "--directory", type=click.Path(exists=True, file_okay=False, path_type=str),
    default=".", show_default=True,
    help="Target directory containing .boepie/.",
)
def apply(directory: str) -> None:
    """Converge the bundle with the resolved content source.

    Rewrites every `managed: boepie` file from the resolved source (cached
    content when available, else packaged seeds), deletes orphaned boepie-managed
    files, preserves every `managed: human` file byte-for-byte, and rebuilds the
    bundle's own BM25 search index under `.boepie/.index/`.
    """
    target_dir = Path(directory).resolve()
    try:
        source_dir = resolve_content_source()
        manifest = apply_bundle(target_dir, source_dir)
    except FileNotFoundError as error:
        raise click.ClickException(str(error)) from error

    bundle_dir = target_dir / ".boepie"
    console.print(f"[green]Applied[/green] bundle at {bundle_dir}")

    _build_knowledge_index(bundle_dir)
    _note_legacy_global_index()


@knowledge.command()
@click.option(
    "--directory", type=click.Path(exists=True, file_okay=False, path_type=str),
    default=".", show_default=True,
    help="Target directory containing .boepie/.",
)
def status(directory: str) -> None:
    """Report the bundle's state relative to installed versions."""
    target_dir = Path(directory).resolve()
    try:
        status_result = bundle_status(target_dir)
    except FileNotFoundError as error:
        raise click.ClickException(str(error)) from error

    color = "green" if status_result.state == "current" else "yellow"
    state_label = f"[{color}]{status_result.state}[/{color}]"
    console.print(f"{state_label}: {status_result.detail}")

    bundle_dir = target_dir / ".boepie"
    if not index_root_for(bundle_dir).exists():
        console.print(
            f"[yellow]no search index[/yellow]: run `boepie knowledge apply` to build "
            f"{index_root_for(bundle_dir)}"
        )
    _note_legacy_global_index()


# ---------------------------------------------------------------------------
# Sync: composite bootstrap (knowledge fetch -> index fetch -> knowledge apply/init)
# ---------------------------------------------------------------------------

# Collections `sync` fetches indices for, unless restricted to `--only knowledge`.
# Reuses `_LOADERS` (in its declared order) - the same source of truth `index
# build`/`index fetch` draw from - instead of a separate hardcoded list.
_SYNC_INDEX_COLLECTIONS = tuple(_LOADERS)


@contextlib.contextmanager
def _quiet(enabled: bool):
    """Swallow everything `console` prints inside the block when `enabled`.

    Used by `sync`'s default (non-verbose) run so its component steps -
    `knowledge fetch`, `index fetch`, `apply`/`init` - stay silent and sync
    can report a single summary line instead of each step's own output.
    """
    if not enabled:
        yield
        return
    with console.capture():
        yield


def _sync_network_step(
    ctx: click.Context, command: click.Command, label: str, quiet: bool, **params: object,
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
        console.print(f"[yellow]Warning:[/yellow] {label} failed: {error}")


@cli.command()
@click.option(
    "--only", type=click.Choice(["knowledge", "indices"]), default=None,
    help="Restrict sync to just the knowledge bundle or just the indices (default: both).",
)
@click.option("--tag", default="latest", show_default=True, help="GitHub release tag to fetch from.")
@click.option(
    "--directory", type=click.Path(exists=True, file_okay=False, path_type=str),
    default=".", show_default=True,
    help="Target directory for the knowledge bundle.",
)
@click.option("-v", "--verbose", is_flag=True, help="Show each step's own output instead of a one-line summary.")
@click.pass_context
def sync(ctx: click.Context, only: str | None, tag: str, directory: str, verbose: bool) -> None:
    """Bring the knowledge bundle and default indices up to date in one step.

    Composite of `knowledge fetch` -> `index fetch` for the default
    collections -> `knowledge apply` (or `init` on a first run, when no
    `.boepie/` exists yet under --directory). Adds nothing of its own beyond
    calling those steps: `index fetch` already skips a collection that is
    present, and `apply`/`init` already rebuild the BM25 index.

    Each network step (`knowledge fetch`, `index fetch`) warns and continues
    on failure instead of aborting, so the final offline convergence step
    still runs against whatever content or indices are already cached or
    packaged - the overall exit code stays 0 as long as that local step
    succeeds. By default only a one-line summary is printed; pass --verbose
    to see each step's own message.
    """
    sync_knowledge = only != "indices"
    sync_indices = only != "knowledge"
    quiet = not verbose

    if sync_knowledge:
        _sync_network_step(ctx, fetch, f"knowledge fetch --tag {tag}", quiet, tag=tag)

    if sync_indices:
        for collection in _SYNC_INDEX_COLLECTIONS:
            _sync_network_step(
                ctx, index_fetch, f"index fetch --collection {collection} --tag {tag}", quiet,
                collection=collection, tag=tag,
            )

    if sync_knowledge:
        target_dir = Path(directory).resolve()
        with _quiet(quiet):
            if (target_dir / ".boepie").exists():
                ctx.invoke(apply, directory=directory)
            else:
                ctx.invoke(init, directory=directory, skills=False, hooks=False)

    if quiet:
        console.print(f"[green]Synced[/green] (tag={tag}).")


# ---------------------------------------------------------------------------
# Hint: BM25 lookup for hook injection
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("prompt")
def hint(prompt: str) -> None:
    """Search the knowledge bundle and print BM25 coordinates for hook injection.

    Runs a BM25-only search over the `.boepie/` bundle and prints at most 3 results
    as plain-text coordinates (path#section: snippet). Exits silently (0) when there
    are no hits or the top score is below the configured threshold.
    """
    _run(_hint_search(prompt))


async def _hint_search(prompt: str) -> None:
    """Async helper for hint command.

    Discovers the bundle governing the cwd (a hook runs from the project
    directory) and searches its own index. A missing bundle is as silent as a
    missing index: this runs on every prompt, so it must never interrupt.
    """
    bundle_dir = find_bundle()
    if bundle_dir is None:
        return

    try:
        results = await search(
            prompt,
            collection=_KNOWLEDGE_COLLECTION,
            mode="bm25",
            top_k=3,
            index_root=index_root_for(bundle_dir),
            embedding=None,
        )
    except FileNotFoundError:
        # No index yet; silently exit.
        return

    if not results:
        return

    # hint runs mode='bm25', so bm25_score is populated; guard against None
    # anyway (fail closed) since this hook fires on every prompt and must
    # never spam on an unexpected shape.
    top_score = results[0].bm25_score
    if top_score is None or top_score < _HINT_MIN_SCORE:
        return

    for result in results:
        chunk = result.chunk
        snippet = chunk.text.strip()
        if len(snippet) > 120:
            snippet = snippet[:120]
        section_part = f"#{chunk.section}" if chunk.section else ""
        console.print(f"{chunk.document_id}{section_part}: {snippet}")
