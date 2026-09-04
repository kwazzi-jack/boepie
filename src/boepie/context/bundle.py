"""The `.boepie/` OKF knowledge bundle: init, apply, status, and discovery.

The bundle (design doc section 3) is a small, curated set of markdown files
that back the L2 "curated knowledge" layer of the escalation ladder (section
2): stable concepts, playbooks, and cab notes, never volatile parameter
facts. This module owns its on-disk lifecycle -- creating it, converging the
files boepie manages against a content source, and reporting whether it has
drifted from the installed cult-cargo/boepie versions and the content they
ship -- but not the CLI commands or the BM25 search index built over it; those are
separate concerns.

Curated content reaches a bundle through exactly one channel: the seed
files baked into the wheel, at ``boepie.assets.context_content_dir()``. Both
``init_bundle`` and ``apply_bundle`` copy from there, so the content a bundle
carries is the content the installed boepie was built with, and there is
nothing to fetch, cache or reconcile. (There used to be a second channel - a
``knowledge-content.tar.gz`` release asset extracted into a machine-global
cache that was preferred over the seeds. It could disagree with the installed
boepie, and did: a cache fetched before a frontmatter rename served
pre-rename content that ``apply_bundle`` then read as the user's own files
and refused to regenerate, freezing a bundle at an old revision with no
error to say so.) ``apply_bundle`` rewrites every ``managed_by: boepie``
file from that source and deletes ones whose source counterpart is gone; a user's ``managed_by: user``
files are never rewritten or deleted, even when their source counterpart
disappears -- except when named explicitly via ``apply_bundle``'s
``force_paths`` (a scalpel: revert one file at a time, only when boepie still
has something to revert it to) or discarded wholesale by ``reset_bundle`` (a
blunt instrument: tear the bundle down and rebuild it from nothing).
"""

from __future__ import annotations

import json
import shutil
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version as installed_version
from pathlib import Path
from typing import Literal

from boepie import __version__ as _installed_boepie_version
from boepie.assets import asset_checksum, context_content_dir
from boepie.config import bundle_dir_override
from boepie.context.frontmatter import read_frontmatter

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BUNDLE_DIRNAME = ".boepie"
_MANIFEST_FILENAME = "manifest.json"
_INDEX_FILENAME = "index.md"
_LOG_FILENAME = "apply-log.md"

# Derived state living inside the bundle: the BM25 search index built over it
# (`<bundle>/.index/context/bm25/`, see `index_root_for`). Binary, rebuilt by
# init/apply, and therefore neither content to converge nor something to
# commit -- hence `_GITIGNORE_FILENAME` below.
_DERIVED_DIRNAME = ".index"
_GITIGNORE_FILENAME = ".gitignore"
_GITIGNORE_LINE = f"{_DERIVED_DIRNAME}/"

# The cult-cargo distribution name on PyPI/uv, confirmed via `uv pip show
# cult-cargo` (hyphenated; the import name `cultcargo` is not registered).
_CULTCARGO_DISTRIBUTION_NAME = "cult-cargo"

# Sentinel recorded when a version can't be resolved (e.g. cult-cargo not
# installed in this environment) so callers never see a bare None/KeyError.
_UNKNOWN_VERSION = "unknown"

# OKF content-schema version for the seed files this package ships. Bumped
# when the bundle's directory layout or required frontmatter fields change,
# independent of boepie's own package version. 0.4.0 dropped the manifest's
# `content_version` field for `content_sha256` (see `BundleManifest`).
_BUNDLE_VERSION = "0.4.0"

_POINTER_LINE = (
    "Stimela knowledge base in `.boepie/`: start at `.boepie/index.md`, "
    "or call `search_context`."
)


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
    `<bundle>/.index/context/bm25/`.
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
# Manifest
# ---------------------------------------------------------------------------


@dataclass
class BundleManifest:
    bundle_version: str
    cultcargo_version: str
    boepie_version: str
    # sha256 of the whole content tree this bundle was generated from, which
    # is what lets `bundle_status` say "the content moved under you, run
    # apply". A version string cannot: the content ships in the wheel, so the
    # only hand-maintained number that could track it is boepie's own, and in
    # an editable checkout (where `.boepie/` is regenerated from
    # `src/boepie/context/content/`) that never changes at all.
    content_sha256: str
    generated_at: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def _current_manifest(source_dir: Path) -> BundleManifest:
    return BundleManifest(
        bundle_version=_BUNDLE_VERSION,
        cultcargo_version=_cultcargo_version(),
        boepie_version=_installed_boepie_version,
        content_sha256=asset_checksum(source_dir),
        generated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )


def _write_manifest(bundle_dir: Path, source_dir: Path) -> BundleManifest:
    manifest = _current_manifest(source_dir)
    manifest_path = bundle_dir / _MANIFEST_FILENAME
    manifest_path.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")
    return manifest


def _read_manifest(bundle_dir: Path) -> BundleManifest:
    """The bundle's own manifest, or an error naming the command that fixes it.

    Three ways this file can be unusable, and all three end the same way -
    named, with `boepie context apply` beside them, because `apply` writes
    the manifest rather than reading it and so fixes every one of them.
    - **Unparseable**, from a write interrupted partway. `json`'s own error
      names a column and nothing else: not the file, not the bundle, not the
      fix.
    - **Not an object** at all, which `BundleManifest(**data)` would answer
      with a `TypeError` about argument unpacking.
    - **Written by an older boepie**, so the fields differ (`0.3.0` carried
      `content_version` where `0.4.0` carries `content_sha256`). Rejecting it
      is right - the two are not derivable from each other, and guessing
      would be inventing state - but a bare `TypeError` three frames down
      names neither the cause nor the cure.

    `is_bundle_dir` deliberately does not do these checks: it asks only
    whether a `.boepie/` is a real bundle rather than a stray directory, and
    a bundle with a damaged manifest is still this project's bundle. Failing
    here, where the manifest is actually needed, keeps `find_bundle` from
    walking past a damaged bundle to a different project's.
    """
    manifest_path = bundle_dir / _MANIFEST_FILENAME
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No manifest at {manifest_path}. Run `boepie context init` first."
        )
    try:
        manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(
            f"The manifest at {manifest_path} is not readable JSON ({error}). "
            f"Run `boepie context apply` to rewrite it."
        ) from error
    if not isinstance(manifest_data, dict):
        raise ValueError(
            f"The manifest at {manifest_path} is not a JSON object. "
            f"Run `boepie context apply` to rewrite it."
        )
    try:
        return BundleManifest(**manifest_data)
    except TypeError as error:
        recorded_version = manifest_data.get("bundle_version", "unknown")
        raise ValueError(
            f"The bundle at {bundle_dir} was written by an older boepie "
            f"(bundle_version {recorded_version}, this one writes "
            f"{_BUNDLE_VERSION}). Run `boepie context apply` to rewrite it."
        ) from error


# ---------------------------------------------------------------------------
# apply-log.md -- newest-first update history
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
# content to copy -- apply-log.md is append-only bundle history (see
# _prepend_log_entry), never a document to converge.
_NON_CONTENT_SOURCE_NAMES = frozenset({_LOG_FILENAME})

# Files that live directly under a bundle but are never source-managed
# content -- manifest.json, apply-log.md and .gitignore are bundle-lifecycle
# bookkeeping that no content source ever provides a counterpart for.
_NON_CONTENT_BUNDLE_NAMES = frozenset({_MANIFEST_FILENAME, _LOG_FILENAME, _GITIGNORE_FILENAME})


def _is_derived_state(relative_path: Path) -> bool:
    """Whether a bundle-relative path is inside the derived `.index/` tree.

    Convergence must skip it outright: those files are binary, so merely
    asking whether they are `managed_by: boepie` (which reads them as UTF-8
    text) would raise.
    """
    return relative_path.parts[0] == _DERIVED_DIRNAME


def _is_source_local(document_path: Path) -> bool:
    frontmatter, _ = read_frontmatter(document_path.read_text(encoding="utf-8"))
    return frontmatter.get("managed_by") == "user"


def _is_boepie_managed(document_path: Path, relative_path: Path) -> bool:
    """`index.md` carries no frontmatter (OKF reserved) but is always
    boepie's to manage; every other document must say `managed_by: boepie`
    explicitly before convergence is allowed to delete it."""
    if relative_path.name == _INDEX_FILENAME:
        return True
    frontmatter, _ = read_frontmatter(document_path.read_text(encoding="utf-8"))
    return frontmatter.get("managed_by") == "boepie"


def list_source_local_files(bundle_dir: Path) -> list[Path]:
    """Every `managed_by: user` file currently in the bundle, bundle-root-relative.

    Public: the CLI layer calls this directly to build `context reset`'s
    confirmation prompt before deciding whether to call `reset_bundle` at
    all, in addition to its use as `reset_bundle`'s own pre-flight listing
    for the discarded-files log entry.
    """
    local_paths: list[Path] = []
    for bundle_path in sorted(bundle_dir.rglob("*")):
        if bundle_path.is_dir():
            continue
        relative_path = bundle_path.relative_to(bundle_dir)
        if relative_path.name in _NON_CONTENT_BUNDLE_NAMES or _is_derived_state(relative_path):
            continue
        if _is_source_local(bundle_path):
            local_paths.append(relative_path)
    return local_paths


def _copy_managed_files(
    bundle_dir: Path, source_dir: Path, *, force_paths: frozenset[Path] = frozenset()
) -> list[Path]:
    """Rewrite every boepie-managed bundle file from its counterpart in `source_dir`.

    `index.md` carries no frontmatter and is always rewritten. A relative
    path in `force_paths` bypasses the `managed_by: user` guard, so it is
    rewritten (and its frontmatter flips back to `managed_by: boepie`, since the
    source copy already carries that) even though it would otherwise be
    skipped.
    """
    rewritten_paths: list[Path] = []
    for source_path in sorted(source_dir.rglob("*")):
        if source_path.is_dir():
            continue
        relative_path = source_path.relative_to(source_dir)
        if relative_path.name in _NON_CONTENT_SOURCE_NAMES:
            continue

        target_path = bundle_dir / relative_path
        if (
            relative_path.name != _INDEX_FILENAME
            and target_path.exists()
            and _is_source_local(target_path)
            and relative_path not in force_paths
        ):
            continue

        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_bytes(source_path.read_bytes())
        rewritten_paths.append(relative_path)
    return rewritten_paths


def _delete_orphaned_managed_files(bundle_dir: Path, source_dir: Path) -> list[Path]:
    """Delete `managed_by: boepie` bundle files whose source counterpart is gone.

    The bundle mirrors the resolved content source; it is not an archive.
    Files without an explicit `managed_by: boepie` marker -- local or otherwise
    -- are left alone even when orphaned (`bundle_status` flags a local
    orphan as informational, never as stale).
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
            f"Bundle already exists at {bundle_dir}. Use `boepie context apply` instead."
        )

    source_dir = context_content_dir()
    bundle_dir.mkdir(parents=True)
    _copy_managed_files(bundle_dir, source_dir)
    ensure_gitignore(bundle_dir)

    manifest = _write_manifest(bundle_dir, source_dir)
    _prepend_log_entry(
        bundle_dir / _LOG_FILENAME,
        f"bundle initialized (boepie {manifest.boepie_version}, "
        f"cult-cargo {manifest.cultcargo_version}, content {manifest.content_sha256[:12]})",
    )
    return manifest


def _normalize_force_path(raw_path: str | Path) -> Path:
    """Bundle-root-relative `Path` for a `--force` target, stripping an
    optional leading `.boepie/` -- the prefix a `search_context` hit's
    `source:` line shows, so a target copied straight from there still works.
    """
    relative_path = Path(raw_path)
    if relative_path.parts and relative_path.parts[0] == _BUNDLE_DIRNAME:
        relative_path = Path(*relative_path.parts[1:])
    return relative_path


def _resolve_force_paths(
    force_paths: Iterable[str | Path], bundle_dir: Path, source_dir: Path
) -> frozenset[Path]:
    """Normalize and validate every `--force` target before any file is
    written, so a bad target fails the whole `apply_bundle` call atomically
    rather than partially reverting some files and erroring on a later one.
    """
    resolved_bundle_dir = bundle_dir.resolve()
    normalized_paths: list[Path] = []
    for raw_path in force_paths:
        relative_path = _normalize_force_path(raw_path)

        resolved_target = (bundle_dir / relative_path).resolve()
        if resolved_target != resolved_bundle_dir and not resolved_target.is_relative_to(
            resolved_bundle_dir
        ):
            raise ValueError(f"--force target '{relative_path}' escapes the bundle directory")

        target_path = bundle_dir / relative_path
        if not target_path.exists():
            raise ValueError(f"no such bundle file to revert: {relative_path}")
        if not _is_source_local(target_path):
            raise ValueError(
                f"'{relative_path}' is not managed_by: user; --force is only for "
                "reverting local files back to boepie-managed"
            )
        if not (source_dir / relative_path).exists():
            raise ValueError(
                f"cannot revert '{relative_path}': no counterpart in the resolved "
                "content source (it may be your own content, never boepie's, so "
                "there's nothing to revert to)"
            )
        normalized_paths.append(relative_path)
    return frozenset(normalized_paths)


def apply_bundle(
    target_dir: Path, source_dir: Path, *, force_paths: Iterable[str | Path] = ()
) -> BundleManifest:
    """Converge `.boepie/` with `source_dir`: rewrite `managed_by: boepie` (and
    `index.md`) files, delete boepie-managed files whose source counterpart
    is gone, and never touch `managed_by: user` files -- except any named in
    `force_paths`, which are reverted back to boepie-managed even though they
    are currently `managed_by: user`.

    Callers choose `source_dir` explicitly -- normally
    `boepie.assets.context_content_dir()` -- so convergence against an
    arbitrary directory (e.g. in tests) is also possible. `force_paths` entries are bundle-root-relative (an optional
    leading `.boepie/` is stripped); every target is validated up front, so a
    bad one raises `ValueError` before anything is written.
    """
    bundle_dir = target_dir / _BUNDLE_DIRNAME
    if not bundle_dir.exists():
        raise FileNotFoundError(
            f"No bundle at {bundle_dir}. Run `boepie context init` first."
        )

    resolved_force_paths = _resolve_force_paths(force_paths, bundle_dir, source_dir)

    rewritten_paths = _copy_managed_files(bundle_dir, source_dir, force_paths=resolved_force_paths)
    deleted_paths = _delete_orphaned_managed_files(bundle_dir, source_dir)
    ensure_gitignore(bundle_dir)

    manifest = _write_manifest(bundle_dir, source_dir)

    log_message = (
        f"bundle applied (boepie {manifest.boepie_version}, "
        f"cult-cargo {manifest.cultcargo_version}, content {manifest.content_sha256[:12]}); "
        f"rewrote {len(rewritten_paths)} file(s)"
    )
    if deleted_paths:
        deleted_list = ", ".join(str(path) for path in deleted_paths)
        log_message += f"; deleted {len(deleted_paths)} orphaned file(s): {deleted_list}"
    forced_reverts = sorted(resolved_force_paths & set(rewritten_paths), key=str)
    if forced_reverts:
        forced_list = ", ".join(str(path) for path in forced_reverts)
        log_message += (
            f"; force-reverted {len(forced_reverts)} managed_by: user file(s) to "
            f"boepie-managed: {forced_list}"
        )
    _prepend_log_entry(bundle_dir / _LOG_FILENAME, log_message)
    return manifest


def reset_bundle(target_dir: Path) -> BundleManifest:
    """Tear `.boepie/` down and rebuild it from scratch, unconditionally.

    Unlike `apply_bundle`'s `force_paths` (a scalpel that only reverts a
    named `managed_by: user` file when boepie still has a counterpart to revert
    it to), this discards every `managed_by: user` file outright, including
    ones with no upstream counterpart at all -- a note the user authored that
    never existed in boepie's content source can only be deleted, not
    reverted.

    This function does not confirm with the user -- that is the CLI layer's
    job (list `list_source_local_files(bundle_dir)`, prompt, only call this
    once confirmed). The rebuilt bundle's `apply-log.md` opens with a record
    of the reset itself, naming every discarded file -- the old log, along
    with everything else, is gone once the directory is removed, so this is
    the one place the reset's effect stays legible afterward.
    """
    bundle_dir = target_dir / _BUNDLE_DIRNAME
    if not bundle_dir.exists():
        raise FileNotFoundError(f"No bundle at {bundle_dir}. Run `boepie context init` first.")

    local_paths = list_source_local_files(bundle_dir)

    shutil.rmtree(bundle_dir)
    manifest = init_bundle(target_dir)

    if local_paths:
        discarded_list = ", ".join(str(path) for path in local_paths)
        _prepend_log_entry(
            bundle_dir / _LOG_FILENAME,
            f"bundle reset from scratch; discarded {len(local_paths)} source: "
            f"local file(s): {discarded_list}",
        )
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
    installed_content_sha256: str


def bundle_status(target_dir: Path) -> BundleStatus:
    """Three-way comparison: bundle manifest vs installed versions vs the
    content the installed boepie ships.

    Every term is local - the content is in the venv, not on a release - so
    this is the whole truth rather than an offline approximation of it: an
    installed cult-cargo/boepie version different from what the bundle was
    generated against, or a content tree whose digest has moved since the
    last `apply`.
    """
    bundle_dir = target_dir / _BUNDLE_DIRNAME
    manifest = _read_manifest(bundle_dir)

    installed_cultcargo_version = _cultcargo_version()
    installed_boepie_version = _installed_boepie_version
    installed_content_sha256 = asset_checksum(context_content_dir())

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
    if manifest.content_sha256 != installed_content_sha256:
        mismatches.append(
            f"content_sha256 {manifest.content_sha256[:12]} != installed "
            f"{installed_content_sha256[:12]}: bundle behind the content this "
            "boepie ships: run `boepie context apply`"
        )

    if mismatches:
        return BundleStatus(
            state="stale",
            detail="; ".join(mismatches),
            manifest=manifest,
            installed_cultcargo_version=installed_cultcargo_version,
            installed_boepie_version=installed_boepie_version,
            installed_content_sha256=installed_content_sha256,
        )

    return BundleStatus(
        state="current",
        detail="bundle matches the installed versions and the content they ship",
        manifest=manifest,
        installed_cultcargo_version=installed_cultcargo_version,
        installed_boepie_version=installed_boepie_version,
        installed_content_sha256=installed_content_sha256,
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
