"""Tests for the `.boepie/` OKF knowledge bundle lifecycle.

Covers init -> status(current), a stale manifest, `managed: human` files
surviving `apply_bundle` byte-for-byte, `managed: boepie` files actually
getting rewritten or deleted when orphaned, the AGENTS.md pointer being
idempotent, the frontmatter helpers round-tripping, and the content channel
(`resolve_content_source`, `fetch_content`) that backs convergence.
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
from boepie.knowledge import bundle
from boepie.knowledge.frontmatter import read_frontmatter, write_frontmatter


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


def test_frontmatter_round_trips_managed_and_okf_fields() -> None:
    original_frontmatter = {
        "type": "Concept",
        "title": "Recipe substitution",
        "description": "How `{recipe.step.param}` substitution resolves.",
        "tags": ["recipes", "substitution"],
        "managed": "human",
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
    assert (bundle_dir / "log.md").exists()
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

    log_text = (bundle_dir / "log.md").read_text(encoding="utf-8")
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
        assert frontmatter["managed"] == "boepie"
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
    assert "boepie knowledge apply" in status.detail
    assert status.resolved_content_version == "9.9.9"


def test_status_raises_when_bundle_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        bundle.bundle_status(tmp_path)


# ---------------------------------------------------------------------------
# apply_bundle()
# ---------------------------------------------------------------------------


def test_apply_preserves_human_managed_file_byte_for_byte(tmp_path: Path) -> None:
    bundle.init_bundle(tmp_path)
    concept_path = tmp_path / ".boepie" / "concepts" / "skeleton.md"

    frontmatter, body = read_frontmatter(concept_path.read_text(encoding="utf-8"))
    frontmatter["managed"] = "human"
    human_document = write_frontmatter(frontmatter, "# Hand-written notes\n\nDo not touch.\n")
    concept_path.write_bytes(human_document.encode("utf-8"))
    human_bytes_before_apply = concept_path.read_bytes()

    bundle.apply_bundle(tmp_path, bundle.resolve_content_source())

    assert concept_path.read_bytes() == human_bytes_before_apply


def test_apply_rewrites_boepie_managed_file(tmp_path: Path) -> None:
    bundle.init_bundle(tmp_path)
    cab_path = tmp_path / ".boepie" / "cabs" / "skeleton.md"
    seed_bytes = (bundle._seed_content_dir() / "cabs" / "skeleton.md").read_bytes()

    cab_path.write_text("this local edit should be discarded on apply\n", encoding="utf-8")
    assert cab_path.read_bytes() != seed_bytes

    bundle.apply_bundle(tmp_path, bundle.resolve_content_source())

    assert cab_path.read_bytes() == seed_bytes


def test_apply_deletes_orphaned_boepie_managed_file_and_keeps_human_orphan(
    tmp_path: Path,
) -> None:
    bundle.init_bundle(tmp_path)
    bundle_dir = tmp_path / ".boepie"

    # A human-authored file with no counterpart in any content source: an
    # orphan from the start, must survive convergence untouched.
    human_orphan_path = bundle_dir / "concepts" / "my-notes.md"
    human_document = write_frontmatter(
        {
            "type": "Concept",
            "title": "My notes",
            "description": "Scratch notes not part of any content source.",
            "tags": [],
            "managed": "human",
        },
        "# My notes\n\nKeep this.\n",
    )
    human_orphan_path.write_text(human_document, encoding="utf-8")
    human_bytes_before_apply = human_orphan_path.read_bytes()

    # A reduced source simulating an upstream content update that dropped
    # cabs/skeleton.md: the bundle's existing boepie-managed copy becomes an
    # orphan and should be deleted.
    reduced_source_dir = tmp_path / "reduced-source"
    shutil.copytree(bundle._seed_content_dir(), reduced_source_dir)
    (reduced_source_dir / "cabs" / "skeleton.md").unlink()

    bundle.apply_bundle(tmp_path, reduced_source_dir)

    assert not (bundle_dir / "cabs" / "skeleton.md").exists()
    assert human_orphan_path.exists()
    assert human_orphan_path.read_bytes() == human_bytes_before_apply

    log_text = (bundle_dir / "log.md").read_text(encoding="utf-8")
    assert "deleted 1 orphaned file" in log_text
    assert "cabs/skeleton.md" in log_text


def test_apply_prepends_a_new_log_entry_without_losing_the_old_one(tmp_path: Path) -> None:
    bundle.init_bundle(tmp_path)
    log_path = tmp_path / ".boepie" / "log.md"
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
    binary_index_file = bundle.index_root_for(bundle_dir) / "knowledge" / "bm25" / "data.npy"
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
