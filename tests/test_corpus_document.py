"""Tests for boepie.corpus.document: reading/writing one leaf document,
bare or asset-wrapped, on top of context.frontmatter's codec.
"""

from __future__ import annotations

import pytest

from boepie.context.frontmatter import write_frontmatter
from boepie.corpus.document import read_document, write_leaf_document


# ---------------------------------------------------------------------------
# write_leaf_document: bare file
# ---------------------------------------------------------------------------


def test_write_leaf_document_writes_a_bare_file_without_assets(tmp_path):
    md_path = tmp_path / "A Test Note.md"

    document = write_leaf_document(
        md_path, document_id="abc1234567",
        frontmatter_fields={"slug": "a-test-note", "title": "A Test Note", "managed_by": "user"},
        body="# A Test Note\n\nBody.\n",
    )

    assert document.md_path == md_path
    assert document.wrapper_dir is None
    assert md_path.exists()
    assert not (tmp_path / "A Test Note").exists()


def test_write_leaf_document_round_trips_frontmatter_and_body(tmp_path):
    md_path = tmp_path / "Doc.md"
    write_leaf_document(
        md_path, document_id="abc1234567",
        frontmatter_fields={"citekey": "smirnov2011", "managed_by": "boepie"},
        body="# Doc\n\nBody text.\n",
    )

    document = read_document(md_path)

    assert document.id == "abc1234567"
    assert document.frontmatter["citekey"] == "smirnov2011"
    assert document.frontmatter["managed_by"] == "boepie"
    assert document.body == "# Doc\n\nBody text.\n"


def test_write_leaf_document_creates_parent_directories(tmp_path):
    md_path = tmp_path / "calibration" / "subtopic" / "Deep Note.md"

    write_leaf_document(
        md_path, document_id="abc1234567",
        frontmatter_fields={"slug": "deep-note", "managed_by": "user"},
        body="Body.\n",
    )

    assert md_path.exists()


def test_write_leaf_document_rejects_id_in_frontmatter_fields(tmp_path):
    with pytest.raises(ValueError, match="must not set 'id'"):
        write_leaf_document(
            tmp_path / "Doc.md", document_id="abc1234567",
            frontmatter_fields={"id": "should-not-be-here"},
            body="Body.\n",
        )


def test_write_leaf_document_overwrites_an_existing_document_in_place(tmp_path):
    md_path = tmp_path / "Doc.md"
    write_leaf_document(
        md_path, document_id="abc1234567",
        frontmatter_fields={"title": "First"}, body="First body.\n",
    )

    write_leaf_document(
        md_path, document_id="abc1234567",
        frontmatter_fields={"title": "Second"}, body="Second body.\n",
    )

    document = read_document(md_path)
    assert document.frontmatter["title"] == "Second"
    assert document.body == "Second body.\n"


# ---------------------------------------------------------------------------
# write_leaf_document: asset-wrapped
# ---------------------------------------------------------------------------


def test_write_leaf_document_with_assets_creates_a_wrapper_directory(tmp_path):
    md_path = tmp_path / "Paper With Figures.md"

    document = write_leaf_document(
        md_path, document_id="abc1234567",
        frontmatter_fields={"citekey": "x2020", "managed_by": "boepie"},
        body="# Paper\n\nSee fig1.\n",
        assets={"fig1.png": b"\x89PNG\r\n"},
    )

    wrapper_dir = tmp_path / "Paper With Figures"
    assert document.md_path == wrapper_dir / "content.md"
    assert document.wrapper_dir == wrapper_dir
    assert (wrapper_dir / "fig1.png").read_bytes() == b"\x89PNG\r\n"
    assert not md_path.exists()


def test_write_leaf_document_allows_overwriting_the_same_id(tmp_path):
    md_path = tmp_path / "Paper.md"
    write_leaf_document(
        md_path, document_id="abc1234567",
        frontmatter_fields={"title": "First"}, body="First body.\n",
        assets={"fig1.png": b"data"},
    )

    document = write_leaf_document(
        md_path, document_id="abc1234567",
        frontmatter_fields={"title": "Refetched"}, body="Second body.\n",
        assets={"fig1.png": b"data"},
    )

    assert read_document(document.md_path).frontmatter["title"] == "Refetched"


def test_write_leaf_document_refuses_to_overwrite_a_different_id(tmp_path):
    md_path = tmp_path / "Paper.md"
    write_leaf_document(
        md_path, document_id="abc1234567",
        frontmatter_fields={"title": "Existing"}, body="Body.\n",
        assets={"fig1.png": b"data"},
    )

    with pytest.raises(ValueError, match="refusing to overwrite"):
        write_leaf_document(
            md_path, document_id="different99",
            frontmatter_fields={"title": "Colliding"}, body="Other body.\n",
            assets={"fig1.png": b"data"},
        )


def test_write_leaf_document_creates_parent_dirs_for_nested_asset_paths(tmp_path):
    md_path = tmp_path / "Paper.md"

    write_leaf_document(
        md_path, document_id="abc1234567",
        frontmatter_fields={}, body="Body.\n",
        assets={"images/fig1.png": b"data"},
    )

    assert (tmp_path / "Paper" / "images" / "fig1.png").read_bytes() == b"data"


# ---------------------------------------------------------------------------
# read_document
# ---------------------------------------------------------------------------


def test_read_document_raises_on_a_pre_migration_file_with_no_id(tmp_path):
    md_path = tmp_path / "Old Format.md"
    md_path.write_text("# Old Format\n\nNo frontmatter at all.\n", encoding="utf-8")

    with pytest.raises(ValueError, match="pre-migration"):
        read_document(md_path)


def test_read_document_raises_on_frontmatter_missing_an_id(tmp_path):
    md_path = tmp_path / "No Id.md"
    md_path.write_text(write_frontmatter({"title": "No Id"}, "Body.\n"), encoding="utf-8")

    with pytest.raises(ValueError):
        read_document(md_path)


def test_read_document_detects_a_wrapped_document_by_directory_shape(tmp_path):
    wrapper_dir = tmp_path / "Wrapped"
    wrapper_dir.mkdir()
    md_path = wrapper_dir / "content.md"
    md_path.write_text(write_frontmatter({"id": "abc1234567"}, "Body.\n"), encoding="utf-8")

    document = read_document(md_path)

    assert document.wrapper_dir == wrapper_dir


def test_read_document_treats_a_bare_file_named_content_as_unwrapped(tmp_path):
    # content.md is only a wrapper leaf when it sits inside a directory a
    # caller reached via classify_child's "leaf-wrapped" shape; a bare file
    # that merely happens to be named content.md (e.g. a document titled
    # "content") is not automatically treated as wrapped by read_document
    # itself - wrapper_dir detection is purely path-shape-based.
    md_path = tmp_path / "content.md"
    md_path.write_text(write_frontmatter({"id": "abc1234567"}, "Body.\n"), encoding="utf-8")

    document = read_document(md_path)

    assert document.wrapper_dir == tmp_path
