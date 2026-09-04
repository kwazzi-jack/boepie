"""`boepie.assets`: the packaged files an installation ships, and their digests.

The point of the module is that there is exactly one place each asset can
come from - the venv boepie is imported from - so these tests are mostly
about the absence of alternatives: no path is configurable, no lookup falls
back, and a missing asset is an error naming what was looked for rather than
an empty result that reads as "nothing to do".
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from boepie import assets
from boepie.docs.manifest import load_default_manifest as load_docs_manifest
from boepie.literature.manifest import load_default_manifest as load_literature_manifest


# ---------------------------------------------------------------------------
# resolution
# ---------------------------------------------------------------------------


def test_every_asset_resolves_inside_the_imported_package() -> None:
    """Each asset is found through the import system, so it comes from
    whichever boepie is installed - not from a checkout that happens to sit
    beside the caller's working directory."""
    package_root = Path(assets.__file__).resolve().parent

    for asset_path in (
        assets.context_content_dir(),
        assets.literature_manifest_path(),
        assets.docs_manifest_path(),
    ):
        assert asset_path.is_absolute()
        assert asset_path.is_relative_to(package_root)


def test_the_context_content_directory_holds_the_bundle_seeds() -> None:
    content_dir = assets.context_content_dir()

    assert (content_dir / "index.md").is_file()
    assert (content_dir / "concepts" / "skeleton.md").is_file()


def test_the_two_manifests_are_the_ones_their_loaders_read() -> None:
    """The manifest modules resolve through `assets` rather than doing their
    own `__file__` arithmetic, so there is one answer to where a manifest
    lives."""
    literature_entries = json.loads(
        assets.literature_manifest_path().read_text(encoding="utf-8")
    )
    docs_entries = json.loads(assets.docs_manifest_path().read_text(encoding="utf-8"))

    assert len(load_literature_manifest()) == len(literature_entries)
    assert len(load_docs_manifest()) == len(docs_entries)


def test_a_missing_asset_is_an_error_naming_where_it_was_looked_for() -> None:
    """A broken install is not a case to recover from. It has to say what is
    missing and where, because the alternative - an empty manifest, an empty
    content directory - reads downstream as "nothing to fetch"."""
    with pytest.raises(assets.MissingAssetError) as error:
        assets._packaged_path("boepie.context", "no-such-asset.json")

    message = str(error.value)
    assert "no-such-asset.json" in message
    assert "boepie/context" in message.replace("\\", "/")


# ---------------------------------------------------------------------------
# checksums
# ---------------------------------------------------------------------------


def test_a_file_digest_is_the_digest_of_its_bytes(tmp_path: Path) -> None:
    import hashlib

    file_path = tmp_path / "manifest.json"
    file_path.write_bytes(b'{"a": 1}')

    assert assets.asset_checksum(file_path) == hashlib.sha256(b'{"a": 1}').hexdigest()


def test_a_directory_digest_covers_its_whole_tree(tmp_path: Path) -> None:
    """Editing a nested file has to move the digest: this is what
    `bundle_status` reads to say the content moved under a bundle."""
    tree_dir = tmp_path / "content"
    (tree_dir / "concepts").mkdir(parents=True)
    (tree_dir / "index.md").write_text("index", encoding="utf-8")
    (tree_dir / "concepts" / "one.md").write_text("one", encoding="utf-8")

    before = assets.asset_checksum(tree_dir)
    (tree_dir / "concepts" / "one.md").write_text("one edited", encoding="utf-8")

    assert assets.asset_checksum(tree_dir) != before


def test_a_directory_digest_moves_when_a_file_is_added_or_removed(tmp_path: Path) -> None:
    """Names are digested alongside bytes, so a removal is visible - which it
    has to be, since `apply` deletes bundle files whose source counterpart is
    gone."""
    tree_dir = tmp_path / "content"
    tree_dir.mkdir()
    (tree_dir / "one.md").write_text("same", encoding="utf-8")

    with_one_file = assets.asset_checksum(tree_dir)
    (tree_dir / "two.md").write_text("same", encoding="utf-8")
    with_two_files = assets.asset_checksum(tree_dir)
    (tree_dir / "two.md").unlink()

    assert with_two_files != with_one_file
    assert assets.asset_checksum(tree_dir) == with_one_file


def test_a_directory_digest_is_stable_across_calls(tmp_path: Path) -> None:
    """Walked in sorted order, so it does not depend on the order a
    filesystem happens to hand entries back."""
    tree_dir = tmp_path / "content"
    (tree_dir / "b").mkdir(parents=True)
    (tree_dir / "a.md").write_text("a", encoding="utf-8")
    (tree_dir / "b" / "c.md").write_text("c", encoding="utf-8")

    assert assets.asset_checksum(tree_dir) == assets.asset_checksum(tree_dir)
