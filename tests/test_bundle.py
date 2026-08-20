"""Tests for the `.boepie/` OKF context bundle lifecycle.

Covers init -> status(current), a stale manifest, `managed_by: user` files
surviving `apply_bundle` byte-for-byte, `managed_by: boepie` files actually
getting rewritten or deleted when orphaned, `--force`-style reverts of a
named `managed_by: user` file via `force_paths`, a full `reset_bundle` teardown
and rebuild, the AGENTS.md pointer being idempotent, the frontmatter helpers
round-tripping, and the content channel (`resolve_content_source`,
`fetch_content`) that backs convergence.
"""

from __future__ import annotations

import io
import json
import shutil
import tarfile
from importlib.metadata import PackageNotFoundError, version as installed_version
from pathlib import Path

import pytest

from boepie import __version__ as boepie_version
from boepie.context import bundle
from boepie.context.frontmatter import read_frontmatter, write_frontmatter


def _installed_cultcargo_version() -> str:
    try:
        return installed_version("cult-cargo")
    except PackageNotFoundError:
        return "unknown"


@pytest.fixture(autouse=True)
def _isolated_content_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test gets an empty content cache by default, so
    `resolve_content_source()` falls back to the packaged seeds unless a test
    explicitly populates a cache. Keeps tests independent of whatever a
    developer's machine happens to have cached at BOEPIE_CONTENT_DIR."""
    monkeypatch.setattr(bundle, "CONTENT_DIR", tmp_path / "content-cache-unset")


def _write_content_manifest(content_dir: Path, content_version: str) -> None:
    content_dir.mkdir(parents=True, exist_ok=True)
    manifest_data = {"content_version": content_version, "generated_at": "2026-01-01T00:00:00Z"}
    (content_dir / "content-manifest.json").write_text(
        json.dumps(manifest_data, indent=2), encoding="utf-8"
    )


def _build_content_tarball(content_version: str) -> bytes:
    """A tarball shaped like scripts/package_content.py's output: the seed
    files unwrapped at the tar root, plus a content-manifest.json."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for entry in sorted(bundle._seed_content_dir().iterdir()):
            tar.add(entry, arcname=entry.name)
        manifest_bytes = json.dumps(
            {"content_version": content_version, "generated_at": "2026-01-01T00:00:00Z"}
        ).encode("utf-8")
        manifest_info = tarfile.TarInfo(name="content-manifest.json")
        manifest_info.size = len(manifest_bytes)
        tar.addfile(manifest_info, io.BytesIO(manifest_bytes))
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# frontmatter round trip
# ---------------------------------------------------------------------------


def test_frontmatter_round_trips_source_and_okf_fields() -> None:
    original_frontmatter = {
        "type": "Concept",
        "title": "Recipe substitution",
        "description": "How `{recipe.step.param}` substitution resolves.",
        "tags": ["recipes", "substitution"],
        "managed_by": "user",
    }
    document = write_frontmatter(original_frontmatter, "# Recipe substitution\n\nBody text.\n")

    parsed_frontmatter, body = read_frontmatter(document)
    assert parsed_frontmatter == original_frontmatter
    assert body == "# Recipe substitution\n\nBody text.\n"


def test_read_frontmatter_on_reserved_file_with_no_block_returns_empty_mapping() -> None:
    text = "# Stimela knowledge base\n\nNo frontmatter here.\n"
    parsed_frontmatter, body = read_frontmatter(text)
    assert parsed_frontmatter == {}
    assert body == text


# ---------------------------------------------------------------------------
# init_bundle()
# ---------------------------------------------------------------------------


def test_init_bundle_creates_seed_layout_and_manifest(tmp_path: Path) -> None:
    manifest = bundle.init_bundle(tmp_path)

    bundle_dir = tmp_path / ".boepie"
    assert (bundle_dir / "index.md").exists()
    assert (bundle_dir / "apply-log.md").exists()
    assert (bundle_dir / "manifest.json").exists()
    assert (bundle_dir / "concepts" / "skeleton.md").exists()
    assert (bundle_dir / "playbooks" / "skeleton.md").exists()
    assert (bundle_dir / "cabs" / "skeleton.md").exists()
    assert (bundle_dir / "literature" / "skeleton.md").exists()

    assert manifest.boepie_version == boepie_version
    assert manifest.cultcargo_version == _installed_cultcargo_version()
    assert manifest.content_version == bundle._BUNDLE_VERSION

    manifest_on_disk = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest_on_disk == manifest.to_dict()

    log_text = (bundle_dir / "apply-log.md").read_text(encoding="utf-8")
    assert log_text.startswith("# Update log")
    assert "bundle initialized" in log_text


def test_init_bundle_raises_if_already_present(tmp_path: Path) -> None:
    bundle.init_bundle(tmp_path)
    with pytest.raises(FileExistsError):
        bundle.init_bundle(tmp_path)


def test_init_bundle_copies_from_a_populated_cache_when_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_dir = tmp_path / "content-cache"
    shutil.copytree(bundle._seed_content_dir(), cache_dir)
    _write_content_manifest(cache_dir, "2.5.0")
    monkeypatch.setattr(bundle, "CONTENT_DIR", cache_dir)

    manifest = bundle.init_bundle(tmp_path)

    assert manifest.content_version == "2.5.0"


def test_seed_skeleton_files_have_okf_frontmatter(tmp_path: Path) -> None:
    bundle.init_bundle(tmp_path)
    bundle_dir = tmp_path / ".boepie"

    expected_types = {
        "concepts": "Concept",
        "playbooks": "Playbook",
        "cabs": "Cab Note",
        "literature": "Paper",
    }
    for directory_name, expected_type in expected_types.items():
        text = (bundle_dir / directory_name / "skeleton.md").read_text(encoding="utf-8")
        frontmatter, body = read_frontmatter(text)
        assert frontmatter["type"] == expected_type
        assert frontmatter["managed_by"] == "boepie"
        assert "title" in frontmatter and "description" in frontmatter and "tags" in frontmatter
        assert "# Citations" in body


# ---------------------------------------------------------------------------
# bundle_status()
# ---------------------------------------------------------------------------


def test_status_is_current_immediately_after_init(tmp_path: Path) -> None:
    bundle.init_bundle(tmp_path)
    status = bundle.bundle_status(tmp_path)
    assert status.state == "current"


def test_status_is_stale_when_manifest_records_a_different_cultcargo_version(
    tmp_path: Path,
) -> None:
    bundle.init_bundle(tmp_path)
    manifest_path = tmp_path / ".boepie" / "manifest.json"
    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_data["cultcargo_version"] = "0.0.0-fake"
    manifest_path.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")

    status = bundle.bundle_status(tmp_path)
    assert status.state == "stale"
    assert "cultcargo_version" in status.detail


def test_status_flags_bundle_behind_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle.init_bundle(tmp_path)  # resolves to seeds: content_version == _BUNDLE_VERSION

    cache_dir = tmp_path / "content-cache"
    shutil.copytree(bundle._seed_content_dir(), cache_dir)
    _write_content_manifest(cache_dir, "9.9.9")
    monkeypatch.setattr(bundle, "CONTENT_DIR", cache_dir)

    status = bundle.bundle_status(tmp_path)
    assert status.state == "stale"
    assert "bundle behind cache" in status.detail
    assert "boepie context apply" in status.detail
    assert status.resolved_content_version == "9.9.9"


def test_status_raises_when_bundle_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        bundle.bundle_status(tmp_path)


# ---------------------------------------------------------------------------
# apply_bundle()
# ---------------------------------------------------------------------------


def test_apply_preserves_source_local_file_byte_for_byte(tmp_path: Path) -> None:
    bundle.init_bundle(tmp_path)
    concept_path = tmp_path / ".boepie" / "concepts" / "skeleton.md"

    frontmatter, body = read_frontmatter(concept_path.read_text(encoding="utf-8"))
    frontmatter["managed_by"] = "user"
    local_document = write_frontmatter(frontmatter, "# Hand-written notes\n\nDo not touch.\n")
    concept_path.write_bytes(local_document.encode("utf-8"))
    local_bytes_before_apply = concept_path.read_bytes()

    bundle.apply_bundle(tmp_path, bundle.resolve_content_source())

    assert concept_path.read_bytes() == local_bytes_before_apply


def test_apply_rewrites_boepie_managed_file(tmp_path: Path) -> None:
    bundle.init_bundle(tmp_path)
    cab_path = tmp_path / ".boepie" / "cabs" / "skeleton.md"
    seed_bytes = (bundle._seed_content_dir() / "cabs" / "skeleton.md").read_bytes()

    cab_path.write_text("this local edit should be discarded on apply\n", encoding="utf-8")
    assert cab_path.read_bytes() != seed_bytes

    bundle.apply_bundle(tmp_path, bundle.resolve_content_source())

    assert cab_path.read_bytes() == seed_bytes


def test_apply_deletes_orphaned_boepie_managed_file_and_keeps_local_orphan(
    tmp_path: Path,
) -> None:
    bundle.init_bundle(tmp_path)
    bundle_dir = tmp_path / ".boepie"

    # A user-authored file with no counterpart in any content source: an
    # orphan from the start, must survive convergence untouched.
    local_orphan_path = bundle_dir / "concepts" / "my-notes.md"
    local_document = write_frontmatter(
        {
            "type": "Concept",
            "title": "My notes",
            "description": "Scratch notes not part of any content source.",
            "tags": [],
            "managed_by": "user",
        },
        "# My notes\n\nKeep this.\n",
    )
    local_orphan_path.write_text(local_document, encoding="utf-8")
    local_bytes_before_apply = local_orphan_path.read_bytes()

    # A reduced source simulating an upstream content update that dropped
    # cabs/skeleton.md: the bundle's existing boepie-managed copy becomes an
    # orphan and should be deleted.
    reduced_source_dir = tmp_path / "reduced-source"
    shutil.copytree(bundle._seed_content_dir(), reduced_source_dir)
    (reduced_source_dir / "cabs" / "skeleton.md").unlink()

    bundle.apply_bundle(tmp_path, reduced_source_dir)

    assert not (bundle_dir / "cabs" / "skeleton.md").exists()
    assert local_orphan_path.exists()
    assert local_orphan_path.read_bytes() == local_bytes_before_apply

    log_text = (bundle_dir / "apply-log.md").read_text(encoding="utf-8")
    assert "deleted 1 orphaned file" in log_text
    assert "cabs/skeleton.md" in log_text


def test_apply_prepends_a_new_log_entry_without_losing_the_old_one(tmp_path: Path) -> None:
    bundle.init_bundle(tmp_path)
    log_path = tmp_path / ".boepie" / "apply-log.md"
    entries_after_init = [line for line in log_path.read_text(encoding="utf-8").splitlines() if line.startswith("- ")]
    assert len(entries_after_init) == 1

    bundle.apply_bundle(tmp_path, bundle.resolve_content_source())

    entries_after_apply = [line for line in log_path.read_text(encoding="utf-8").splitlines() if line.startswith("- ")]
    assert len(entries_after_apply) == 2
    assert "bundle applied" in entries_after_apply[0]
    assert "bundle initialized" in entries_after_apply[1]


def test_apply_raises_when_bundle_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        bundle.apply_bundle(tmp_path, bundle._seed_content_dir())


def test_apply_with_no_force_paths_is_unchanged_from_before(tmp_path: Path) -> None:
    """Regression guard: `force_paths` defaults to `()`, so a call that never
    mentions it must behave exactly as it did before the parameter existed."""
    bundle.init_bundle(tmp_path)
    cab_path = tmp_path / ".boepie" / "cabs" / "skeleton.md"
    seed_bytes = (bundle._seed_content_dir() / "cabs" / "skeleton.md").read_bytes()
    cab_path.write_text("this local edit should be discarded on apply\n", encoding="utf-8")

    bundle.apply_bundle(tmp_path, bundle.resolve_content_source())

    assert cab_path.read_bytes() == seed_bytes
    log_text = (tmp_path / ".boepie" / "apply-log.md").read_text(encoding="utf-8")
    assert "force-reverted" not in log_text


# ---------------------------------------------------------------------------
# apply_bundle(force_paths=...) -- reverting a named managed_by: user file
# ---------------------------------------------------------------------------


def _mark_source_local(document_path: Path, body: str = "# Hand-written notes\n\nDo not touch.\n") -> None:
    """Rewrite `document_path` in place as a `managed_by: user` file, keeping
    its other OKF frontmatter fields intact."""
    frontmatter, _ = read_frontmatter(document_path.read_text(encoding="utf-8"))
    frontmatter["managed_by"] = "user"
    document_path.write_text(write_frontmatter(frontmatter, body), encoding="utf-8")


def _write_local_orphan(bundle_dir: Path) -> Path:
    """A user-authored file with no counterpart in any content source."""
    orphan_path = bundle_dir / "concepts" / "my-notes.md"
    local_document = write_frontmatter(
        {
            "type": "Concept",
            "title": "My notes",
            "description": "Scratch notes not part of any content source.",
            "tags": [],
            "managed_by": "user",
        },
        "# My notes\n\nKeep this.\n",
    )
    orphan_path.write_text(local_document, encoding="utf-8")
    return orphan_path


def test_apply_force_reverts_a_source_local_file_to_boepie_managed(tmp_path: Path) -> None:
    bundle.init_bundle(tmp_path)
    concept_path = tmp_path / ".boepie" / "concepts" / "skeleton.md"
    seed_bytes = (bundle._seed_content_dir() / "concepts" / "skeleton.md").read_bytes()
    _mark_source_local(concept_path)
    assert concept_path.read_bytes() != seed_bytes

    bundle.apply_bundle(
        tmp_path, bundle.resolve_content_source(), force_paths=["concepts/skeleton.md"]
    )

    assert concept_path.read_bytes() == seed_bytes
    frontmatter, _ = read_frontmatter(concept_path.read_text(encoding="utf-8"))
    assert frontmatter["managed_by"] == "boepie"


def test_apply_force_accepts_a_dot_boepie_prefixed_path(tmp_path: Path) -> None:
    bundle.init_bundle(tmp_path)
    concept_path = tmp_path / ".boepie" / "concepts" / "skeleton.md"
    seed_bytes = (bundle._seed_content_dir() / "concepts" / "skeleton.md").read_bytes()
    _mark_source_local(concept_path)

    bundle.apply_bundle(
        tmp_path, bundle.resolve_content_source(), force_paths=[".boepie/concepts/skeleton.md"]
    )

    assert concept_path.read_bytes() == seed_bytes


def test_apply_force_on_a_boepie_managed_target_raises(tmp_path: Path) -> None:
    bundle.init_bundle(tmp_path)

    with pytest.raises(ValueError, match="not managed_by: user"):
        bundle.apply_bundle(
            tmp_path, bundle.resolve_content_source(), force_paths=["concepts/skeleton.md"]
        )


def test_apply_force_on_a_missing_bundle_file_raises(tmp_path: Path) -> None:
    bundle.init_bundle(tmp_path)

    with pytest.raises(ValueError, match="no such bundle file to revert"):
        bundle.apply_bundle(
            tmp_path, bundle.resolve_content_source(), force_paths=["concepts/nonexistent.md"]
        )


def test_apply_force_on_a_file_with_no_source_counterpart_raises(tmp_path: Path) -> None:
    bundle.init_bundle(tmp_path)
    bundle_dir = tmp_path / ".boepie"
    _write_local_orphan(bundle_dir)

    with pytest.raises(ValueError, match="no counterpart in the resolved content source"):
        bundle.apply_bundle(
            tmp_path, bundle.resolve_content_source(), force_paths=["concepts/my-notes.md"]
        )


def test_apply_force_path_traversal_raises_before_any_write(tmp_path: Path) -> None:
    bundle.init_bundle(tmp_path)

    with pytest.raises(ValueError, match="escapes the bundle directory"):
        bundle.apply_bundle(
            tmp_path, bundle.resolve_content_source(), force_paths=["../../etc/passwd"]
        )


def test_apply_force_log_entry_names_forced_reverts_distinctly(tmp_path: Path) -> None:
    bundle.init_bundle(tmp_path)
    concept_path = tmp_path / ".boepie" / "concepts" / "skeleton.md"
    _mark_source_local(concept_path)

    bundle.apply_bundle(
        tmp_path, bundle.resolve_content_source(), force_paths=["concepts/skeleton.md"]
    )

    log_text = (tmp_path / ".boepie" / "apply-log.md").read_text(encoding="utf-8")
    assert "force-reverted 1 managed_by: user file(s)" in log_text
    assert "concepts/skeleton.md" in log_text


# ---------------------------------------------------------------------------
# list_source_local_files()
# ---------------------------------------------------------------------------


def test_list_source_local_files_finds_only_local_files(tmp_path: Path) -> None:
    bundle.init_bundle(tmp_path)
    bundle_dir = tmp_path / ".boepie"
    concept_path = bundle_dir / "concepts" / "skeleton.md"
    _mark_source_local(concept_path)

    local_paths = bundle.list_source_local_files(bundle_dir)

    assert local_paths == [Path("concepts/skeleton.md")]


def test_list_source_local_files_is_empty_for_a_fresh_bundle(tmp_path: Path) -> None:
    bundle.init_bundle(tmp_path)
    bundle_dir = tmp_path / ".boepie"

    assert bundle.list_source_local_files(bundle_dir) == []


# ---------------------------------------------------------------------------
# reset_bundle()
#
# reset_bundle() itself never prompts -- confirmation is the CLI layer's job
# (see tests/test_cli_context.py for the prompt/--yes/decline coverage).
# These tests only cover the unconditional teardown-and-rebuild behavior.
# ---------------------------------------------------------------------------


def test_reset_bundle_rebuilds_a_bundle_with_no_local_files(tmp_path: Path) -> None:
    bundle.init_bundle(tmp_path)

    manifest = bundle.reset_bundle(tmp_path)

    assert manifest is not None
    bundle_dir = tmp_path / ".boepie"
    assert (bundle_dir / "concepts" / "skeleton.md").exists()
    assert (bundle_dir / "manifest.json").exists()


def test_reset_bundle_discards_local_files_and_rebuilds(tmp_path: Path) -> None:
    bundle.init_bundle(tmp_path)
    bundle_dir = tmp_path / ".boepie"
    local_path = _write_local_orphan(bundle_dir)

    manifest = bundle.reset_bundle(tmp_path)

    assert manifest is not None
    assert not local_path.exists()
    assert (bundle_dir / "concepts" / "skeleton.md").exists()

    log_text = (bundle_dir / "apply-log.md").read_text(encoding="utf-8")
    assert "bundle reset from scratch" in log_text
    assert "concepts/my-notes.md" in log_text
    # apply-log.md is newest-first and the reset entry is written after
    # init_bundle's own "bundle initialized" entry, so it must lead.
    entries = [line for line in log_text.splitlines() if line.startswith("- ")]
    assert "bundle reset from scratch" in entries[0]


def test_reset_bundle_raises_when_bundle_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        bundle.reset_bundle(tmp_path)


# ---------------------------------------------------------------------------
# resolve_content_source() / fetch_content()
# ---------------------------------------------------------------------------


def test_resolve_content_source_falls_back_to_seeds_when_cache_absent() -> None:
    assert bundle.resolve_content_source() == bundle._seed_content_dir()


def test_resolve_content_source_prefers_populated_cache_over_seeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_dir = tmp_path / "content-cache"
    _write_content_manifest(cache_dir, "9.9.9")
    monkeypatch.setattr(bundle, "CONTENT_DIR", cache_dir)

    assert bundle.resolve_content_source() == cache_dir


def test_fetch_content_extracts_verified_tarball_into_content_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_dir = tmp_path / "content-cache"
    monkeypatch.setattr(bundle, "CONTENT_DIR", cache_dir)
    monkeypatch.setattr(bundle, "fetch_asset_checksum", lambda tag, asset_name: "remote-digest")

    tarball_bytes = _build_content_tarball("2.0.0")
    captured_args: list[tuple[str, str, str | None]] = []

    def _fake_download(tag: str, asset_name: str, expected_digest: str | None = None) -> bytes:
        captured_args.append((tag, asset_name, expected_digest))
        return tarball_bytes

    monkeypatch.setattr(bundle, "download_verified_asset", _fake_download)

    result = bundle.fetch_content(tag="v2")

    assert result.content_dir == cache_dir
    assert result.changed is True
    assert captured_args == [("v2", "knowledge-content.tar.gz", "remote-digest")]
    assert (cache_dir / "index.md").exists()
    manifest_data = json.loads((cache_dir / "content-manifest.json").read_text(encoding="utf-8"))
    assert manifest_data["content_version"] == "2.0.0"


def test_fetch_content_replaces_a_previous_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_dir = tmp_path / "content-cache"
    cache_dir.mkdir()
    (cache_dir / "stale-marker.txt").write_text("old", encoding="utf-8")
    monkeypatch.setattr(bundle, "CONTENT_DIR", cache_dir)
    monkeypatch.setattr(bundle, "fetch_asset_checksum", lambda tag, asset_name: "remote-digest")

    tarball_bytes = _build_content_tarball("3.0.0")
    monkeypatch.setattr(
        bundle, "download_verified_asset",
        lambda tag, asset_name, expected_digest=None: tarball_bytes,
    )

    bundle.fetch_content()

    assert not (cache_dir / "stale-marker.txt").exists()
    assert (cache_dir / "content-manifest.json").exists()


def test_fetch_content_skips_download_when_cached_digest_matches_remote(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cache already at the remote digest should be reported unchanged
    without ever calling `download_verified_asset` - the whole point of
    checking the small `.sha256` sidecar first."""
    cache_dir = tmp_path / "content-cache"
    _write_content_manifest(cache_dir, "1.0.0")
    (cache_dir.parent / f".{cache_dir.name}.sha256").write_text("abc123", encoding="utf-8")
    monkeypatch.setattr(bundle, "CONTENT_DIR", cache_dir)
    monkeypatch.setattr(bundle, "fetch_asset_checksum", lambda tag, asset_name: "abc123")
    monkeypatch.setattr(
        bundle, "download_verified_asset",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not download")),
    )

    result = bundle.fetch_content(tag="latest")

    assert result.content_dir == cache_dir
    assert result.changed is False
    manifest_data = json.loads((cache_dir / "content-manifest.json").read_text(encoding="utf-8"))
    assert manifest_data["content_version"] == "1.0.0"


def test_fetch_content_redownloads_when_cached_digest_differs_from_remote(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stale cached digest must not short-circuit the download."""
    cache_dir = tmp_path / "content-cache"
    _write_content_manifest(cache_dir, "1.0.0")
    (cache_dir.parent / f".{cache_dir.name}.sha256").write_text("old-digest", encoding="utf-8")
    monkeypatch.setattr(bundle, "CONTENT_DIR", cache_dir)
    monkeypatch.setattr(bundle, "fetch_asset_checksum", lambda tag, asset_name: "new-digest")

    tarball_bytes = _build_content_tarball("2.0.0")
    monkeypatch.setattr(
        bundle, "download_verified_asset",
        lambda tag, asset_name, expected_digest=None: tarball_bytes,
    )

    result = bundle.fetch_content(tag="latest")

    assert result.changed is True
    manifest_data = json.loads((cache_dir / "content-manifest.json").read_text(encoding="utf-8"))
    assert manifest_data["content_version"] == "2.0.0"
    digest_path = cache_dir.parent / f".{cache_dir.name}.sha256"
    assert digest_path.read_text(encoding="utf-8") == "new-digest"


# ---------------------------------------------------------------------------
# find_bundle() / index_root_for() / ensure_gitignore()
# ---------------------------------------------------------------------------


def test_find_bundle_walks_up_from_a_nested_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bundle.init_bundle(tmp_path)
    nested_dir = tmp_path / "recipes" / "imaging"
    nested_dir.mkdir(parents=True)

    assert bundle.find_bundle(nested_dir) == tmp_path / ".boepie"

    monkeypatch.chdir(nested_dir)
    assert bundle.find_bundle() == tmp_path / ".boepie"


def test_find_bundle_returns_none_when_nothing_above(tmp_path: Path) -> None:
    # tmp_path has no bundle and no ancestor of it does either.
    assert bundle.find_bundle(tmp_path) is None


def test_find_bundle_ignores_a_manifestless_boepie_directory(tmp_path: Path) -> None:
    """A stray `.boepie/` must not shadow the real bundle above it."""
    bundle.init_bundle(tmp_path)
    nested_dir = tmp_path / "sub"
    (nested_dir / ".boepie").mkdir(parents=True)

    assert bundle.find_bundle(nested_dir) == tmp_path / ".boepie"


def test_find_bundle_env_override_is_checked_before_walking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    walked_to = tmp_path / "walked"
    overridden = tmp_path / "overridden"
    walked_to.mkdir()
    overridden.mkdir()
    bundle.init_bundle(walked_to)
    bundle.init_bundle(overridden)

    monkeypatch.setenv("BOEPIE_BUNDLE_DIR", str(overridden / ".boepie"))
    assert bundle.find_bundle(walked_to) == overridden / ".boepie"


def test_find_bundle_env_override_pointing_at_a_non_bundle_finds_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A mistyped override must not silently fall back to another project's
    bundle - that is exactly the wrong-answer failure being designed out."""
    bundle.init_bundle(tmp_path)
    monkeypatch.setenv("BOEPIE_BUNDLE_DIR", str(tmp_path / "nowhere"))

    assert bundle.find_bundle(tmp_path) is None


def test_index_root_for_is_inside_the_bundle(tmp_path: Path) -> None:
    bundle_dir = tmp_path / ".boepie"
    assert bundle.index_root_for(bundle_dir) == bundle_dir / ".index"


def test_init_writes_a_gitignore_for_the_derived_index(tmp_path: Path) -> None:
    bundle.init_bundle(tmp_path)
    gitignore_path = tmp_path / ".boepie" / ".gitignore"
    assert gitignore_path.read_text(encoding="utf-8").splitlines() == [".index/"]


def test_ensure_gitignore_is_idempotent_and_preserves_other_lines(tmp_path: Path) -> None:
    bundle.init_bundle(tmp_path)
    bundle_dir = tmp_path / ".boepie"
    gitignore_path = bundle_dir / ".gitignore"
    gitignore_path.write_text("scratch/\n", encoding="utf-8")

    assert bundle.ensure_gitignore(bundle_dir) is True
    assert bundle.ensure_gitignore(bundle_dir) is False

    lines = gitignore_path.read_text(encoding="utf-8").splitlines()
    assert lines == ["scratch/", ".index/"]


def test_apply_ignores_derived_index_and_gitignore(tmp_path: Path) -> None:
    """Convergence must not treat derived state as an orphaned content file:
    the index holds binary files that reading as text would blow up on."""
    bundle.init_bundle(tmp_path)
    bundle_dir = tmp_path / ".boepie"
    binary_index_file = bundle.index_root_for(bundle_dir) / "context" / "bm25" / "data.npy"
    binary_index_file.parent.mkdir(parents=True)
    binary_index_file.write_bytes(b"\x93NUMPY\x01\x00\xff\xfe")

    bundle.apply_bundle(tmp_path, bundle.resolve_content_source())

    assert binary_index_file.exists()
    assert (bundle_dir / ".gitignore").exists()


# ---------------------------------------------------------------------------
# append_agents_pointer()
# ---------------------------------------------------------------------------


def test_append_agents_pointer_creates_file_when_absent(tmp_path: Path) -> None:
    agents_md = tmp_path / "AGENTS.md"
    wrote_something = bundle.append_agents_pointer(agents_md)

    assert wrote_something is True
    assert agents_md.exists()
    text = agents_md.read_text(encoding="utf-8")
    assert text.count("Stimela knowledge base in `.boepie/`") == 1


def test_append_agents_pointer_is_idempotent(tmp_path: Path) -> None:
    agents_md = tmp_path / "AGENTS.md"
    agents_md.write_text("# Project agents\n\nSome existing instructions.\n", encoding="utf-8")

    first_call_wrote_something = bundle.append_agents_pointer(agents_md)
    second_call_wrote_something = bundle.append_agents_pointer(agents_md)

    assert first_call_wrote_something is True
    assert second_call_wrote_something is False

    text = agents_md.read_text(encoding="utf-8")
    assert text.count("Stimela knowledge base in `.boepie/`") == 1
    assert "Some existing instructions." in text


# ---------------------------------------------------------------------------
# Content cache: a pre-schema-change cache must not win over the seeds
# ---------------------------------------------------------------------------


def _write_cache(cache_dir: Path, content_version: str) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "content-manifest.json").write_text(
        json.dumps({"content_version": content_version, "files": []}), encoding="utf-8"
    )


def test_a_cache_at_the_current_content_version_is_used(tmp_path, monkeypatch):
    cache_dir = tmp_path / "content"
    _write_cache(cache_dir, bundle._BUNDLE_VERSION)
    monkeypatch.setattr(bundle, "CONTENT_DIR", cache_dir)

    assert bundle.resolve_content_source() == cache_dir


def test_a_cache_predating_a_schema_change_falls_back_to_the_packaged_seeds(
    tmp_path, monkeypatch
):
    """A cache built before a required-frontmatter-field change carries the
    old field names, which `apply_bundle` would read as the user's own files
    and then refuse to regenerate - a bundle silently frozen at an old
    revision. Falling back to the seeds is the safe answer."""
    cache_dir = tmp_path / "content"
    _write_cache(cache_dir, "0.1.0")
    monkeypatch.setattr(bundle, "CONTENT_DIR", cache_dir)

    resolved = bundle.resolve_content_source()

    assert resolved != cache_dir
    assert (resolved / "index.md").is_file()


def test_a_cache_newer_than_this_boepie_is_still_used(tmp_path, monkeypatch):
    """Only older caches are rejected: a newer one means the user fetched
    content from a release ahead of their installed boepie, which is their
    call to make."""
    cache_dir = tmp_path / "content"
    _write_cache(cache_dir, "99.0.0")
    monkeypatch.setattr(bundle, "CONTENT_DIR", cache_dir)

    assert bundle.resolve_content_source() == cache_dir


def test_an_unparseable_content_version_is_treated_as_old(tmp_path, monkeypatch):
    cache_dir = tmp_path / "content"
    _write_cache(cache_dir, "not-a-version")
    monkeypatch.setattr(bundle, "CONTENT_DIR", cache_dir)

    assert bundle.resolve_content_source() != cache_dir
