"""The `.boepie/` OKF knowledge bundle: init, apply, status, and discovery.

The bundle (design doc section 3) is a small, curated set of markdown files
that back the L2 "curated knowledge" layer of the escalation ladder (section
2): stable concepts, playbooks, and cab notes, never volatile parameter
facts. This module owns its on-disk lifecycle -- creating it, converging the
files boepie manages against a content source, and reporting whether it has
drifted from the installed cult-cargo/boepie versions and the cached content
-- but not the CLI commands or the BM25 search index built over it; those are
separate concerns.

Curated content reaches a bundle through two channels (design section 4):
packaged seed files baked into the wheel (offline bootstrap, under
``knowledge/content/``) and a ``knowledge-content.tar.gz`` release asset
fetched into a machine-global cache (``fetch_content``). ``resolve_content_source``
picks whichever is authoritative -- the cache once populated, the seeds
otherwise -- and both ``init_bundle`` and ``apply_bundle`` copy from it.
``apply_bundle`` rewrites every ``managed: boepie`` file from that source and
deletes ones whose source counterpart is gone; a user's ``managed: human``
files are never rewritten or deleted, even when their source counterpart
disappears.
"""

from __future__ import annotations

import io
import json
import shutil
import tarfile
import time
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version as installed_version
from pathlib import Path
from typing import Literal

from boepie import __version__ as _installed_boepie_version
from boepie.config import CONTENT_DIR, bundle_dir_override
from boepie.knowledge.frontmatter import read_frontmatter
from boepie.release import download_verified_asset, fetch_asset_checksum

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BUNDLE_DIRNAME = ".boepie"
_MANIFEST_FILENAME = "manifest.json"
_INDEX_FILENAME = "index.md"
_LOG_FILENAME = "log.md"

# Derived state living inside the bundle: the BM25 search index built over it
# (`<bundle>/.index/knowledge/bm25/`, see `index_root_for`). Binary, rebuilt by
# init/apply, and therefore neither content to converge nor something to
# commit -- hence `_GITIGNORE_FILENAME` below.
_DERIVED_DIRNAME = ".index"
_GITIGNORE_FILENAME = ".gitignore"
_GITIGNORE_LINE = f"{_DERIVED_DIRNAME}/"

# The release asset name `scripts/package_content.py` produces; `fetch_content`
# downloads exactly this name.
_CONTENT_ASSET_NAME = "knowledge-content.tar.gz"

# Written at the root of a fetched content cache (by scripts/package_content.py);
# its absence at CONTENT_DIR means "never fetched, or fetch left it empty",
# so resolve_content_source() falls back to the packaged seeds.
_CONTENT_MANIFEST_FILENAME = "content-manifest.json"

# The cult-cargo distribution name on PyPI/uv, confirmed via `uv pip show
# cult-cargo` (hyphenated; the import name `cultcargo` is not registered).
_CULTCARGO_DISTRIBUTION_NAME = "cult-cargo"

# Sentinel recorded when a version can't be resolved (e.g. cult-cargo not
# installed in this environment) so callers never see a bare None/KeyError.
_UNKNOWN_VERSION = "unknown"

# OKF content-schema version for the seed files this package ships, and the
# content_version recorded whenever the resolved source is the packaged
# seeds (no content-manifest.json of their own). Bumped when the bundle's
# directory layout or required frontmatter fields change, independent of
# boepie's own package version.
_BUNDLE_VERSION = "0.1.0"

_POINTER_LINE = (
    "Stimela knowledge base in `.boepie/`: start at `.boepie/index.md`, "
    "or call `search_knowledge`."
)


def _seed_content_dir() -> Path:
    return Path(__file__).resolve().parent / "content"


def _cultcargo_version() -> str:
    try:
        return installed_version(_CULTCARGO_DISTRIBUTION_NAME)
    except PackageNotFoundError:
        return _UNKNOWN_VERSION


# ---------------------------------------------------------------------------
# Discovery: which bundle governs a directory
# ---------------------------------------------------------------------------


def is_bundle_dir(candidate_dir: Path) -> bool:
    """Whether `candidate_dir` is a real bundle rather than a stray `.boepie/`.

    The manifest is the marker: an empty directory someone created by hand
    must not shadow a genuine bundle further up the tree.
    """
    return (candidate_dir / _MANIFEST_FILENAME).is_file()


def find_bundle(start_dir: Path | None = None) -> Path | None:
    """Locate the `.boepie/` bundle governing `start_dir`, git-`.git`-style.

    Walks up from `start_dir` (default: the cwd, read at call time so a
    long-lived server sees a bundle created after it started) to the
    filesystem root, returning the first `.boepie/` that carries a
    `manifest.json`. `BOEPIE_BUNDLE_DIR`, when set, names a bundle directly
    and is honoured before any walking -- but is still manifest-checked, so a
    mistyped override fails loudly as "no bundle" instead of silently serving
    a different project's.

    Returns None when nothing is found; callers render their own error,
    because "no bundle anywhere" and "bundle with no index" have different
    fixes.
    """
    override_dir = bundle_dir_override()
    if override_dir is not None:
        return override_dir if is_bundle_dir(override_dir) else None

    current_dir = (start_dir if start_dir is not None else Path.cwd()).resolve()
    for directory in (current_dir, *current_dir.parents):
        candidate_dir = directory / _BUNDLE_DIRNAME
        if is_bundle_dir(candidate_dir):
            return candidate_dir
    return None


def index_root_for(bundle_dir: Path) -> Path:
    """The index root for a bundle's own BM25 index.

    Per-project, not machine-global: the index's source is this bundle, so
    two projects must not share (and clobber) one index. Keeps the engine's
    `index_root/<collection>/<index_id>` layout, landing at
    `<bundle>/.index/knowledge/bm25/`.
    """
    return bundle_dir / _DERIVED_DIRNAME


def ensure_gitignore(bundle_dir: Path) -> bool:
    """Idempotently ignore the derived index inside a committable bundle.

    The bundle itself is meant to be committed (design section 7); `.index/`
    is rebuilt binary state that is not. Returns whether anything was written.
    """
    gitignore_path = bundle_dir / _GITIGNORE_FILENAME
    existing_text = gitignore_path.read_text(encoding="utf-8") if gitignore_path.exists() else ""
    if _GITIGNORE_LINE in existing_text.splitlines():
        return False

    separator = "" if not existing_text or existing_text.endswith("\n") else "\n"
    gitignore_path.write_text(existing_text + separator + _GITIGNORE_LINE + "\n", encoding="utf-8")
    return True


# ---------------------------------------------------------------------------
# Content source: packaged seeds vs fetched cache
# ---------------------------------------------------------------------------


def _is_populated_content_cache(candidate_dir: Path) -> bool:
    """A cache counts as populated once it carries its own manifest.

    Guards against a half-extracted or never-fetched CONTENT_DIR being
    mistaken for real content.
    """
    return (candidate_dir / _CONTENT_MANIFEST_FILENAME).is_file()


def resolve_content_source() -> Path:
    """Return the content directory `init_bundle`/`apply_bundle` copy from.

    Prefers the machine-global fetched cache (`CONTENT_DIR`) once it is
    populated; falls back to the packaged seeds shipped in the wheel
    otherwise, so a bundle can always be created/converged offline.
    """
    if _is_populated_content_cache(CONTENT_DIR):
        return CONTENT_DIR
    return _seed_content_dir()


def _content_version_for(source_dir: Path) -> str:
    manifest_path = source_dir / _CONTENT_MANIFEST_FILENAME
    if not manifest_path.exists():
        return _BUNDLE_VERSION
    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    return str(manifest_data.get("content_version", _BUNDLE_VERSION))


@dataclass
class ContentFetchResult:
    content_dir: Path
    changed: bool


def _content_digest_path() -> Path:
    """Sidecar recording the sha256 of the tarball last fetched into
    `CONTENT_DIR`, so a later `fetch_content` can tell the cache is still
    current from the release's small `.sha256` file alone, without
    re-downloading the (much larger) tarball body."""
    return CONTENT_DIR.parent / f".{CONTENT_DIR.name}.sha256"


def fetch_content(tag: str = "latest") -> ContentFetchResult:
    """Bring `CONTENT_DIR` up to date with `knowledge-content.tar.gz` from a GitHub release.

    Checks the release's `.sha256` sidecar first (one small request); if it
    matches the digest recorded from the cache's last successful fetch, the
    tarball download is skipped entirely and `changed=False` is returned.
    Otherwise downloads and checksum-verifies via `boepie.release`, extracting
    into a sibling staging directory that swaps into place only once
    complete, so an interrupted download never leaves a half-extracted cache
    behind; any previously cached content is replaced outright.
    """
    remote_digest = fetch_asset_checksum(tag, _CONTENT_ASSET_NAME)
    digest_path = _content_digest_path()

    if (
        _is_populated_content_cache(CONTENT_DIR)
        and digest_path.is_file()
        and digest_path.read_text(encoding="utf-8").strip() == remote_digest
    ):
        return ContentFetchResult(content_dir=CONTENT_DIR, changed=False)

    archive_bytes = download_verified_asset(tag, _CONTENT_ASSET_NAME, expected_digest=remote_digest)

    staging_dir = CONTENT_DIR.parent / f".{CONTENT_DIR.name}.fetch-staging"
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as tar:
        tar.extractall(staging_dir, filter="data")

    if CONTENT_DIR.exists():
        shutil.rmtree(CONTENT_DIR)
    staging_dir.rename(CONTENT_DIR)
    digest_path.write_text(remote_digest, encoding="utf-8")
    return ContentFetchResult(content_dir=CONTENT_DIR, changed=True)


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


@dataclass
class BundleManifest:
    bundle_version: str
    cultcargo_version: str
    boepie_version: str
    content_version: str
    generated_at: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def _current_manifest(source_dir: Path) -> BundleManifest:
    return BundleManifest(
        bundle_version=_BUNDLE_VERSION,
        cultcargo_version=_cultcargo_version(),
        boepie_version=_installed_boepie_version,
        content_version=_content_version_for(source_dir),
        generated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )


def _write_manifest(bundle_dir: Path, source_dir: Path) -> BundleManifest:
    manifest = _current_manifest(source_dir)
    manifest_path = bundle_dir / _MANIFEST_FILENAME
    manifest_path.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")
    return manifest


def _read_manifest(bundle_dir: Path) -> BundleManifest:
    manifest_path = bundle_dir / _MANIFEST_FILENAME
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No manifest at {manifest_path}. Run `boepie knowledge init` first."
        )
    return BundleManifest(**json.loads(manifest_path.read_text(encoding="utf-8")))


# ---------------------------------------------------------------------------
# log.md -- newest-first update history
# ---------------------------------------------------------------------------


def _prepend_log_entry(log_path: Path, message: str) -> None:
    """Insert a timestamped entry as the first item in the newest-first log.

    Preserves whatever header and older entries already exist; creates the
    file with a header if it is missing entirely.
    """
    existing_text = log_path.read_text(encoding="utf-8") if log_path.exists() else "# Update log\n"
    lines = existing_text.splitlines()

    if lines and lines[0].startswith("#"):
        header_line, remaining_lines = lines[0], lines[1:]
    else:
        header_line, remaining_lines = "# Update log", lines

    while remaining_lines and remaining_lines[0].strip() == "":
        remaining_lines = remaining_lines[1:]

    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    new_entry_line = f"- {timestamp}: {message}"

    rebuilt_lines = [header_line, "", new_entry_line]
    if remaining_lines:
        rebuilt_lines.extend(["", *remaining_lines])

    log_path.write_text("\n".join(rebuilt_lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Convergence helpers shared by init_bundle / apply_bundle
# ---------------------------------------------------------------------------

# Files that live under a content source but are not themselves bundle
# content to copy -- log.md is append-only bundle history (see
# _prepend_log_entry) and content-manifest.json only describes the source
# itself, not a bundle document.
_NON_CONTENT_SOURCE_NAMES = frozenset({_LOG_FILENAME, _CONTENT_MANIFEST_FILENAME})

# Files that live directly under a bundle but are never source-managed
# content -- manifest.json, log.md and .gitignore are bundle-lifecycle
# bookkeeping that no content source ever provides a counterpart for.
_NON_CONTENT_BUNDLE_NAMES = frozenset({_MANIFEST_FILENAME, _LOG_FILENAME, _GITIGNORE_FILENAME})


def _is_derived_state(relative_path: Path) -> bool:
    """Whether a bundle-relative path is inside the derived `.index/` tree.

    Convergence must skip it outright: those files are binary, so merely
    asking whether they are `managed: boepie` (which reads them as UTF-8
    text) would raise.
    """
    return relative_path.parts[0] == _DERIVED_DIRNAME


def _is_human_managed(document_path: Path) -> bool:
    frontmatter, _ = read_frontmatter(document_path.read_text(encoding="utf-8"))
    return frontmatter.get("managed") == "human"


def _is_boepie_managed(document_path: Path, relative_path: Path) -> bool:
    """`index.md` carries no frontmatter (OKF reserved) but is always
    boepie's to manage; every other document must say `managed: boepie`
    explicitly before convergence is allowed to delete it."""
    if relative_path.name == _INDEX_FILENAME:
        return True
    frontmatter, _ = read_frontmatter(document_path.read_text(encoding="utf-8"))
    return frontmatter.get("managed") == "boepie"


def _copy_managed_files(bundle_dir: Path, source_dir: Path) -> list[Path]:
    """Rewrite every non-human bundle file from its counterpart in `source_dir`.

    `index.md` carries no frontmatter and is always rewritten.
    """
    rewritten_paths: list[Path] = []
    for source_path in sorted(source_dir.rglob("*")):
        if source_path.is_dir():
            continue
        relative_path = source_path.relative_to(source_dir)
        if relative_path.name in _NON_CONTENT_SOURCE_NAMES:
            continue

        target_path = bundle_dir / relative_path
        if relative_path.name != _INDEX_FILENAME and target_path.exists() and _is_human_managed(target_path):
            continue

        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_bytes(source_path.read_bytes())
        rewritten_paths.append(relative_path)
    return rewritten_paths


def _delete_orphaned_managed_files(bundle_dir: Path, source_dir: Path) -> list[Path]:
    """Delete `managed: boepie` bundle files whose source counterpart is gone.

    The bundle mirrors the resolved content source; it is not an archive.
    Files without an explicit `managed: boepie` marker -- human-managed or
    otherwise -- are left alone even when orphaned (`bundle_status` flags a
    human orphan as informational, never as stale).
    """
    deleted_paths: list[Path] = []
    for bundle_path in sorted(bundle_dir.rglob("*")):
        if bundle_path.is_dir():
            continue
        relative_path = bundle_path.relative_to(bundle_dir)
        if relative_path.name in _NON_CONTENT_BUNDLE_NAMES or _is_derived_state(relative_path):
            continue
        if (source_dir / relative_path).exists():
            continue
        if not _is_boepie_managed(bundle_path, relative_path):
            continue

        bundle_path.unlink()
        deleted_paths.append(relative_path)
    return deleted_paths


# ---------------------------------------------------------------------------
# init / apply
# ---------------------------------------------------------------------------


def init_bundle(target_dir: Path) -> BundleManifest:
    """Create `.boepie/` under `target_dir` from the resolved content source.

    Raises `FileExistsError` if a bundle is already present -- use
    `apply_bundle` to reconverge an existing one.
    """
    bundle_dir = target_dir / _BUNDLE_DIRNAME
    if bundle_dir.exists():
        raise FileExistsError(
            f"Bundle already exists at {bundle_dir}. Use `boepie knowledge apply` instead."
        )

    source_dir = resolve_content_source()
    bundle_dir.mkdir(parents=True)
    _copy_managed_files(bundle_dir, source_dir)
    ensure_gitignore(bundle_dir)

    manifest = _write_manifest(bundle_dir, source_dir)
    _prepend_log_entry(
        bundle_dir / _LOG_FILENAME,
        f"bundle initialized (boepie {manifest.boepie_version}, "
        f"cult-cargo {manifest.cultcargo_version}, content {manifest.content_version})",
    )
    return manifest


def apply_bundle(target_dir: Path, source_dir: Path) -> BundleManifest:
    """Converge `.boepie/` with `source_dir`: rewrite `managed: boepie` (and
    `index.md`) files, delete boepie-managed files whose source counterpart
    is gone, and never touch `managed: human` files.

    Callers choose `source_dir` explicitly -- normally `resolve_content_source()`
    -- so convergence against an arbitrary directory (e.g. in tests) is also
    possible.
    """
    bundle_dir = target_dir / _BUNDLE_DIRNAME
    if not bundle_dir.exists():
        raise FileNotFoundError(
            f"No bundle at {bundle_dir}. Run `boepie knowledge init` first."
        )

    rewritten_paths = _copy_managed_files(bundle_dir, source_dir)
    deleted_paths = _delete_orphaned_managed_files(bundle_dir, source_dir)
    ensure_gitignore(bundle_dir)

    manifest = _write_manifest(bundle_dir, source_dir)

    log_message = (
        f"bundle applied (boepie {manifest.boepie_version}, "
        f"cult-cargo {manifest.cultcargo_version}, content {manifest.content_version}); "
        f"rewrote {len(rewritten_paths)} file(s)"
    )
    if deleted_paths:
        deleted_list = ", ".join(str(path) for path in deleted_paths)
        log_message += f"; deleted {len(deleted_paths)} orphaned file(s): {deleted_list}"
    _prepend_log_entry(bundle_dir / _LOG_FILENAME, log_message)
    return manifest


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------

BundleState = Literal["current", "stale"]


@dataclass
class BundleStatus:
    state: BundleState
    detail: str
    manifest: BundleManifest
    installed_cultcargo_version: str
    installed_boepie_version: str
    resolved_content_version: str


def bundle_status(target_dir: Path) -> BundleStatus:
    """Three-way comparison: bundle manifest vs installed versions vs the
    currently resolvable content source (cache when populated, else seeds).

    Stays offline (design section 7: `status` never touches the network), so
    it cannot see a newer release sitting upstream that has never been
    fetched -- only drift the bundle can already detect locally: an installed
    cult-cargo/boepie version different from what the bundle was generated
    against, or a content source that has moved on since the last `apply`.
    """
    bundle_dir = target_dir / _BUNDLE_DIRNAME
    manifest = _read_manifest(bundle_dir)

    installed_cultcargo_version = _cultcargo_version()
    installed_boepie_version = _installed_boepie_version
    resolved_content_version = _content_version_for(resolve_content_source())

    mismatches: list[str] = []
    if manifest.bundle_version != _BUNDLE_VERSION:
        mismatches.append(
            f"bundle_version {manifest.bundle_version} != installed {_BUNDLE_VERSION}"
        )
    if manifest.cultcargo_version != installed_cultcargo_version:
        mismatches.append(
            f"cultcargo_version {manifest.cultcargo_version} != installed "
            f"{installed_cultcargo_version}"
        )
    if manifest.boepie_version != installed_boepie_version:
        mismatches.append(
            f"boepie_version {manifest.boepie_version} != installed {installed_boepie_version}"
        )
    if manifest.content_version != resolved_content_version:
        mismatches.append(
            f"content_version {manifest.content_version} != resolved "
            f"{resolved_content_version}: bundle behind cache: run `boepie knowledge apply`"
        )

    if mismatches:
        return BundleStatus(
            state="stale",
            detail="; ".join(mismatches),
            manifest=manifest,
            installed_cultcargo_version=installed_cultcargo_version,
            installed_boepie_version=installed_boepie_version,
            resolved_content_version=resolved_content_version,
        )

    return BundleStatus(
        state="current",
        detail="bundle matches installed versions and resolved content",
        manifest=manifest,
        installed_cultcargo_version=installed_cultcargo_version,
        installed_boepie_version=installed_boepie_version,
        resolved_content_version=resolved_content_version,
    )


# ---------------------------------------------------------------------------
# AGENTS.md pointer
# ---------------------------------------------------------------------------


def append_agents_pointer(agents_md: Path) -> bool:
    """Idempotently append the bundle pointer line to `agents_md`.

    Creates the file if it does not exist. Returns whether anything was
    written (`False` when the pointer is already present).
    """
    existing_text = agents_md.read_text(encoding="utf-8") if agents_md.exists() else ""
    if _POINTER_LINE in existing_text:
        return False

    if not existing_text:
        new_text = _POINTER_LINE + "\n"
    else:
        separator = "" if existing_text.endswith("\n") else "\n"
        new_text = existing_text + separator + "\n" + _POINTER_LINE + "\n"

    agents_md.write_text(new_text, encoding="utf-8")
    return True
