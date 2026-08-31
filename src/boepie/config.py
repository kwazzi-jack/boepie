# boepie/config.py
"""Central configuration for boepie.

Paths, the embedding model binding, retrieval defaults, and literature/mineru/
ingestion/sync preferences. Everything below is a flat, typed constant other
modules import; nothing else in boepie should reach for `boepie.settings` or
`os.environ` to read a setting.

Two kinds of setting live here, and they resolve differently:

- **Config-file-backed** (most of them): declared once in
  `boepie.settings.BoepieSettings`, which resolves env var > the user's
  `config.toml` > the field's default and validates the result. The
  constants below just name the resolved values.
- **Paths** (LITERATURE_DIR, INDEX_DIR, ...) plus a few tuning constants:
  env-var-only, read straight from `os.environ` here. They are per-machine
  plumbing rather than preferences, so they stay out of the config file and
  out of `boepie config show`.

An invalid value in either layer raises `settings.ConfigError` at import,
naming the offending key and the file to fix - loudly and immediately,
rather than surfacing later as a confusing error deep in the retrieval
engine.
"""

from __future__ import annotations

import os
from pathlib import Path

from platformdirs import user_cache_dir, user_data_dir

from boepie import settings

_SETTINGS = settings.load()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Project root (two levels up from this file: src/boepie/config.py)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Where the literature corpus lives: a document per bare-file or
# same-named-wrapper-directory leaf (see boepie.corpus.layout's recursive
# group-walking rule), each carrying a surrogate `id` and `managed_by: boepie |
# local` in its YAML frontmatter (boepie.corpus.document) rather than a
# {citekey}/metadata.json sidecar. Populated by `boepie corpus fetch
# --collection literature` (arXiv HTML, converted locally) and `boepie corpus
# add literature <arxiv_id>`, and optionally supplemented with
# `scripts/corpus_to_md.py` (BYO-PDF, also local, and not yet migrated to
# this layout - see scripts/migrate_corpus_layout.py). Unlike DOCS_DIR this is
# machine-global user data, not a repo-relative dev path: no literature
# Markdown is shipped or redistributed by boepie, so every machine builds its
# own copy of whichever papers its manifest names.
LITERATURE_DIR: Path = Path(
    os.environ.get("BOEPIE_LITERATURE_DIR", str(Path(user_data_dir("boepie")) / "literature-corpus"))
)

# Where the upstream docs corpus lives. One writer only: `boepie corpus add
# docs` / `corpus fetch --collection docs` (`boepie.corpus.reconcile.sync_docs`,
# the corpus layout - a document per leaf with a surrogate `id` and
# `managed_by: boepie | user` in frontmatter, see boepie.corpus.layout). A
# second, dev-time writer producing `{project}/{docname}.md` + `metadata.json`
# was removed on 2026-08-20: `DocsLoader` had moved onto the corpus layout and
# could no longer read anything it wrote.
#
# The default is still repo-relative, which is a leftover from that dev-time
# build and the reason a fresh clone's `docs-corpus/` may hold the old,
# unreadable layout - re-run `corpus fetch --collection docs` to replace it.
DOCS_DIR: Path = Path(
    os.environ.get("BOEPIE_DOCS_DIR", str(_PROJECT_ROOT / "docs-corpus"))
)

# Where user-added notes live: same corpus layout as LITERATURE_DIR/DOCS_DIR
# (a document per leaf, surrogate `id` plus `managed_by: user` always - a note
# is never boepie-managed - in frontmatter), populated by `boepie corpus add
# notes <identifier>` (local text files and URLs for now - see
# boepie.corpus.add). Machine-global user data, like LITERATURE_DIR: nothing
# under here is boepie-curated, boepie-managed, or ever shipped/redistributed
# by boepie itself.
NOTES_DIR: Path = Path(
    os.environ.get("BOEPIE_NOTES_DIR", str(Path(user_data_dir("boepie")) / "notes"))
)

# Where built/fetched indices live, one subdirectory per named collection.
# Defaults to a platform-appropriate user data directory (not repo-relative)
# so an installed (not cloned) boepie has somewhere sensible to keep a
# fetched index across runs without redownloading it.
#
# The `index` component is load-bearing: LITERATURE_DIR, NOTES_DIR and
# CONTENT_DIR all sit directly under the same user data directory, so an
# INDEX_DIR pointing at that root makes every corpus a sibling of the real
# index collections - and `index status`/`index list`, which enumerate
# INDEX_DIR's subdirectories, then report `notes/` and `content/` as
# collections and the document directories inside them as available index
# ids. Anything that enumerates collections needs this to name only indices.
INDEX_DIR: Path = Path(
    os.environ.get("BOEPIE_INDEX_DIR", str(Path(user_data_dir("boepie")) / "index"))
)

# Explicit pointer at the `.boepie/` bundle directory to use, overriding the
# upward search from the cwd that `boepie.context.find_bundle` otherwise
# does. The escape hatch for an MCP server whose client launched it from
# somewhere other than the project directory it is meant to serve.
#
# Unlike the paths above this is resolved per call, not at import: the server
# is long-lived and both the cwd and the bundle can change under it.
BUNDLE_DIR_ENV_VAR = "BOEPIE_BUNDLE_DIR"


def bundle_dir_override() -> Path | None:
    """The bundle directory `BOEPIE_BUNDLE_DIR` names, or None when unset."""
    raw_path = os.environ.get(BUNDLE_DIR_ENV_VAR)
    return Path(raw_path).expanduser() if raw_path else None


# Machine-global cache for curated context-bundle content fetched via
# `boepie context fetch` (the knowledge-content.tar.gz release asset
# extracts here). Distinct from INDEX_DIR: this holds source markdown that
# `context init`/`apply` copy from, not a built search index.
CONTENT_DIR: Path = Path(
    os.environ.get("BOEPIE_CONTENT_DIR", str(Path(user_data_dir("boepie")) / "content"))
)

# Where the fastembed backend caches its downloaded ONNX model files.
# fastembed's own default is a tempdir (cleared on reboot on most systems,
# forcing a redownload); pointing it at a platform cache dir instead means
# the one-time download really only happens once.
EMBEDDING_CACHE_DIR: Path = Path(
    os.environ.get("BOEPIE_EMBEDDING_CACHE_DIR", str(Path(user_cache_dir("boepie")) / "embedding-models"))
)

# ---------------------------------------------------------------------------
# Embedding binding (build-time AND query-time)
# ---------------------------------------------------------------------------

# "fastembed", "ollama", or "openai". Default is fastembed: a small ONNX
# model run locally on CPU via the fastembed package, needing no running
# service, API key, or GPU - just a one-time model download cached locally.
# "ollama"/"openai" remain available for anyone already running local LLM
# infra or who wants a hosted API's quality.
EMBEDDING_BINDING: str = _SETTINGS.embedding.binding
EMBEDDING_MODEL: str = _SETTINGS.embedding.model
EMBEDDING_HOST: str = _SETTINGS.embedding.host

# Embedding dimension, recorded in the manifest and used to shape empty
# matrices. Defaults to BAAI/bge-small-en-v1.5's known dimension; set it
# alongside any change to EMBEDDING_MODEL.
EMBEDDING_DIM: int = _SETTINGS.embedding.dim

# ---------------------------------------------------------------------------
# Chunking defaults (characters)
# ---------------------------------------------------------------------------

PROSE_CHUNK_SIZE: int = 1500
PROSE_CHUNK_OVERLAP: int = 200

# ---------------------------------------------------------------------------
# Retrieval defaults
# ---------------------------------------------------------------------------

DEFAULT_TOP_K: int = _SETTINGS.retrieval.default_top_k

# 'hybrid' (dense + lexical), 'dense', or 'bm25' - the search_*/read_* MCP
# tools' and `boepie search`'s default when a caller doesn't pass `mode`.
DEFAULT_MODE: str = _SETTINGS.retrieval.default_mode

# 'none', 'short', or 'full' - same defaulting role as DEFAULT_MODE, for
# how much of each hit's chunk body to render.
DEFAULT_SNIPPET: str = _SETTINGS.retrieval.default_snippet

# Number of candidates pulled from each retriever before fusion.
FUSION_CANDIDATES: int = 50

# Reciprocal Rank Fusion constant: score = sum 1 / (RRF_K + rank).
RRF_K: int = 60

# ---------------------------------------------------------------------------
# Literature: arXiv-vs-PDF preference
# ---------------------------------------------------------------------------

# When True, `corpus fetch --collection literature`/`corpus add literature`
# resolve each paper's DOI via Unpaywall and prefer a legitimately
# open-access published PDF (converted locally with MinerU) over the arXiv
# HTML rendering, on the reasoning that the published version is the
# corrected version of record. False (the default) keeps the lightweight
# arXiv-HTML-only path: no OCR, no GPU, no API key, matching CLAUDE.md's
# documented default-path guarantee.
LITERATURE_PREFER_PDF: bool = _SETTINGS.literature.prefer_pdf

# Seconds between paper fetches - politeness towards arxiv.org/ar5iv/Unpaywall,
# not an API requirement. `boepie corpus fetch --collection literature --delay`
# overrides per run.
LITERATURE_FETCH_DELAY: float = _SETTINGS.literature.fetch_delay

# ---------------------------------------------------------------------------
# Corpus: shared warnings across literature/docs/notes
# ---------------------------------------------------------------------------

# `boepie corpus add notes` prints a warning when a note's derived title
# would have produced a filename starting with '.' (e.g. added from a
# dotfile like .bashrc with no --title override) before
# boepie.corpus.layout.full_title_filename strips the leading dot - such a
# filename would otherwise be treated as corpus bookkeeping and become
# permanently invisible to search/list. Set False to suppress.
CORPUS_WARN_ON_DOTFILE_TITLE: bool = _SETTINGS.corpus.warn_on_dotfile_title

# Retain each added document's source bytes next to its converted Markdown,
# inside the document's own wrapper directory. Off by default: PDFs are large
# and most corpora are never re-converted. `source.sha256` is recorded either
# way, which answers "is this the same document" without the bytes.
CORPUS_KEEP_ORIGINAL: bool = _SETTINGS.corpus.keep_original

# Extra suffixes a folder walk accepts, on top of the formats boepie already
# converts (`corpus.intake.SUPPORTED_SUFFIXES`). Additive: a walk is an
# accept-list, because `detect_format`'s catch-all plus a latin-1 encoding
# fallback would otherwise turn any binary it met into a document.
CORPUS_EXTRA_FILE_TYPES: list[str] = list(_SETTINGS.corpus.extra_file_types)

# ---------------------------------------------------------------------------
# MinerU: PDF/image/DOCX/PPTX/XLSX conversion (used when LITERATURE_PREFER_PDF
# is set, and by `corpus add notes` for any identifier that resolves to one
# of those formats rather than plain text/HTML)
# ---------------------------------------------------------------------------

# 'auto', 'cpu', 'cuda', 'mps', 'npu', 'musa', or 'mlu' - passed through to
# MinerU's own MINERU_DEVICE_MODE env var at conversion time. 'auto' lets
# MinerU's own detection (cuda -> mps -> specialized hardware -> cpu) decide.
MINERU_DEVICE_MODE: str = _SETTINGS.mineru.device_mode

# 'pipeline' (fast, no LLM, the default), 'vlm' (vision-language model pass -
# needed for image/figure captioning, higher precision on complex layouts),
# or 'hybrid' (both). Maps to MinerU's own backend selection.
MINERU_BACKEND: str = _SETTINGS.mineru.backend

# 'huggingface', 'modelscope', or 'auto' - passed through to MinerU's own
# MINERU_MODEL_SOURCE env var. 'auto' means the variable is left unset:
# MinerU rejects the literal string and asks for the variable to be absent.
MINERU_MODEL_SOURCE: str = _SETTINGS.mineru.model_source
MINERU_BATCH_SIZE: int = _SETTINGS.mineru.batch_size

# ---------------------------------------------------------------------------
# Ingestion: `corpus add notes`'s collection classification
# ---------------------------------------------------------------------------

# Whether `corpus add notes` may use MCP sampling (an agent-hosted LLM call,
# no separate API key) to classify an ambiguous identifier into a collection,
# when the connected client supports it. Not every MCP host implements
# sampling, so this is an enhancement over the heuristic classifier, not a
# hard requirement - `corpus add notes` still works with it False/unsupported.
# Anticipates a future bare ambiguous-identifier entry point (the
# per-collection `corpus add literature|docs|notes` subcommands shipped in
# this phase do not need it, since the collection is already explicit);
# declared but not yet consumed anywhere.
INGESTION_USE_MCP_SAMPLING: bool = _SETTINGS.ingestion.use_mcp_sampling

# Which collection an ambiguous `corpus add` identifier files under when
# neither the heuristic classifier nor sampling can decide. Same
# not-yet-consumed status as INGESTION_USE_MCP_SAMPLING above.
#
# 'notes' rather than 'docs' because the fallback fires on classification
# *failure*, and notes is the only collection that can actually take a bare
# identifier: `corpus.add.add_notes` turns a path or URL into one document,
# whereas `corpus.add.add_docs` wants a project name and a base URL and
# yields a whole crawled site. Notes is also always `managed_by: user`, so a
# misfile never lands a foreign document in a manifest-reconciled corpus,
# and promoting out of notes is the designed direction of travel.
INGESTION_DEFAULT_COLLECTION: str = _SETTINGS.ingestion.default_collection

# ---------------------------------------------------------------------------
# Pipeline: which stimela libraries the cab and recipe tools see
# ---------------------------------------------------------------------------

# stimela sources merged into the resolved config the cab and recipe tools
# read, in stimela's own spelling (`cultcargo::`, `otherlib.recipes::`, a
# plain path). Loading them is a one-off ~10s cost per process, so this list
# is deliberately for *installed libraries* only: a recipe file the user is
# working on is layered on per call instead, via a recipe tool's
# `recipe_file` argument.
PIPELINE_SOURCES: list[str] = list(_SETTINGS.pipeline.sources)

# Whether to scan the installed environment for stimela libraries on top of
# PIPELINE_SOURCES. On by default: `stimela doc otherlib.recipes::thing` is a
# runtime lookup against whatever the user typed, so a server reading only a
# configured list is blind to a library already installed in the same venv -
# and an agent cannot be expected to invent that `module::path` spelling.
#
# Env-var-only rather than a `config show` key: for everyday use there is
# nothing to tune, since discovery is what makes the common case need no
# configuration at all. The off switch exists for reproducibility - pinning
# exactly which libraries an agent can see, which matters when comparing
# runs - not as a preference.
PIPELINE_DISCOVER: bool = os.environ.get(
    "BOEPIE_PIPELINE_DISCOVER", "1"
).strip().lower() not in ("0", "false", "no")

# Where scabha caches parsed configs. Pointed at stimela's own cache
# directory rather than scabha's `~/.cache/configuratt` default so boepie
# and the `stimela` CLI share one cache instead of each paying the parse.
# `stimela.main` sets exactly this; nothing else does, so an in-process
# caller has to set it itself.
STIMELA_CONFIG_CACHE_DIR: Path = Path(
    os.environ.get("BOEPIE_STIMELA_CONFIG_CACHE_DIR", str(Path.home() / ".cache" / "stimela-configs"))
)

# ---------------------------------------------------------------------------
# Sync: staleness checks (soft nudges only - never OS-level scheduling, never
# a self-update)
# ---------------------------------------------------------------------------

# Whether a stale `sync` (see SYNC_CHECK_INTERVAL_DAYS) runs automatically
# (quietly, on `boepie serve` startup or a CLI invocation) versus only
# printing a one-line nudge for the user to run it themselves.
SYNC_AUTO_SYNC: bool = _SETTINGS.sync.auto_sync

# How many days since the last `sync` before it's considered stale.
SYNC_CHECK_INTERVAL_DAYS: int = _SETTINGS.sync.check_interval_days

# Whether to check the installed boepie version against the latest GitHub
# release and print an upgrade nudge - prompt-only, boepie never self-updates.
SYNC_CHECK_BOEPIE_VERSION: bool = _SETTINGS.sync.check_boepie_version

# ---------------------------------------------------------------------------
# Custom instructions
# ---------------------------------------------------------------------------

# Free-text user preferences. Surfaced through its own doorway (an MCP
# resource, or a note appended where the agent already looks, e.g.
# `.boepie/index.md`) - never spliced into the MCP `instructions=` block in
# server.py, which stays boepie-authored only.
INSTRUCTIONS_CUSTOM: str = _SETTINGS.instructions.custom
