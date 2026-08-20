"""Tests for boepie.corpus.layout: the recursive group-walking rule,
full-title filenames, uniqueness disambiguation, and the natural-key/id
index reconcile.py diffs against.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from boepie.context.frontmatter import write_frontmatter
from boepie.corpus.layout import (
    WRAPPED_DOCUMENT_FILENAME,
    IndexedDocument,
    classify_child,
    collection_index,
    full_title_filename,
    iter_documents,
    natural_key_of,
    title_needs_dot_stripped,
    unique_document_name,
)


def _write_leaf(path, *, frontmatter=None, body="Body.\n"):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = write_frontmatter(frontmatter, body) if frontmatter is not None else body
    path.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# classify_child
# ---------------------------------------------------------------------------


def test_classify_child_bare_md_file_is_leaf_bare(tmp_path):
    md_path = tmp_path / "Some Title.md"
    md_path.write_text("Body.\n", encoding="utf-8")

    assert classify_child(md_path) == "leaf-bare"


def test_classify_child_directory_with_content_md_is_leaf_wrapped(tmp_path):
    wrapper_dir = tmp_path / "Some Title"
    wrapper_dir.mkdir()
    (wrapper_dir / "content.md").write_text("Body.\n", encoding="utf-8")

    assert classify_child(wrapper_dir) == "leaf-wrapped"


def test_classify_child_other_directory_is_group(tmp_path):
    group_dir = tmp_path / "calibration"
    group_dir.mkdir()

    assert classify_child(group_dir) == "group"


def test_classify_child_a_same_named_md_file_no_longer_triggers_wrapping(tmp_path):
    # Regression: classify_child used to match a directory against a
    # same-named .md file inside it, so a document titled the same as a
    # group (e.g. "My Note" landing in a group also named "My Note") would
    # turn that whole group into a single leaf, hiding every sibling
    # already there. The fixed WRAPPED_DOCUMENT_FILENAME convention means a
    # same-named .md file alone (with no content.md) is just a bare
    # document that happens to be a group's child - the group is still a
    # group.
    group_dir = tmp_path / "My Note"
    group_dir.mkdir()
    (group_dir / "My Note.md").write_text("Body.\n", encoding="utf-8")

    assert classify_child(group_dir) == "group"


def test_classify_child_raises_on_a_non_markdown_file(tmp_path):
    stray_path = tmp_path / "notes.json"
    stray_path.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError):
        classify_child(stray_path)


# ---------------------------------------------------------------------------
# iter_documents: the recursive group-walking rule
# ---------------------------------------------------------------------------


def test_iter_documents_finds_a_bare_leaf_at_the_root(tmp_path):
    _write_leaf(tmp_path / "Notes on Substitution.md")

    locations = list(iter_documents(tmp_path, collection="notes"))

    assert len(locations) == 1
    assert locations[0].md_path == tmp_path / "Notes on Substitution.md"
    assert locations[0].wrapper_dir is None


def test_iter_documents_finds_a_wrapped_leaf_and_does_not_recurse_into_it(tmp_path):
    wrapper_dir = tmp_path / "Paper With Figures"
    wrapper_dir.mkdir()
    _write_leaf(wrapper_dir / "content.md")
    (wrapper_dir / "fig1.png").write_bytes(b"\x89PNG")

    locations = list(iter_documents(tmp_path, collection="literature"))

    assert len(locations) == 1
    assert locations[0].md_path == wrapper_dir / "content.md"
    assert locations[0].wrapper_dir == wrapper_dir


def test_iter_documents_recurses_into_user_created_groups(tmp_path):
    _write_leaf(tmp_path / "calibration" / "subtopic" / "Deep Note.md")
    _write_leaf(tmp_path / "Shallow Note.md")

    locations = {location.md_path for location in iter_documents(tmp_path, collection="notes")}

    assert locations == {
        tmp_path / "calibration" / "subtopic" / "Deep Note.md",
        tmp_path / "Shallow Note.md",
    }


def test_iter_documents_skips_dotfiles_and_non_markdown_bookkeeping(tmp_path):
    _write_leaf(tmp_path / "A Paper.md")
    (tmp_path / "user-papers.json").write_text("[]", encoding="utf-8")
    (tmp_path / ".gitignore").write_text("ignored\n", encoding="utf-8")

    locations = list(iter_documents(tmp_path, collection="literature"))

    assert [location.md_path for location in locations] == [tmp_path / "A Paper.md"]


def test_iter_documents_yields_nothing_for_a_missing_collection_root(tmp_path):
    assert list(iter_documents(tmp_path / "does-not-exist", collection="notes")) == []


# ---------------------------------------------------------------------------
# full_title_filename / unique_document_name
# ---------------------------------------------------------------------------


def test_full_title_filename_strips_illegal_characters():
    assert full_title_filename('A: Title/With\\Illegal*Chars?') == "A TitleWithIllegalChars.md"


def test_full_title_filename_collapses_internal_whitespace():
    assert full_title_filename("Title   with    gaps") == "Title with gaps.md"


def test_full_title_filename_falls_back_to_untitled_for_an_empty_title():
    assert full_title_filename("***") == "untitled.md"


def test_unique_document_name_returns_base_when_free():
    assert unique_document_name("Some Title.md", set()) == "Some Title.md"


def test_unique_document_name_disambiguates_on_collision():
    existing = {"Some Title.md", "Some Title (2).md"}
    assert unique_document_name("Some Title.md", existing) == "Some Title (3).md"


def test_unique_document_name_never_returns_the_wrapped_document_filename():
    # A bare document titled "content" must never claim WRAPPED_DOCUMENT_FILENAME
    # even if it is not present in existing_names, since claiming it would
    # make classify_child mistake the parent directory for a wrapped
    # document the next time something is added alongside it.
    assert unique_document_name(WRAPPED_DOCUMENT_FILENAME, set()) == "content (2).md"


def test_full_title_filename_strips_a_leading_dot():
    assert full_title_filename(".bashrc") == "bashrc.md"


def test_full_title_filename_strips_multiple_leading_dots():
    assert full_title_filename("..hidden") == "hidden.md"


def test_title_needs_dot_stripped_true_for_a_dotfile_title():
    assert title_needs_dot_stripped(".bashrc") is True


def test_title_needs_dot_stripped_false_for_an_ordinary_title():
    assert title_needs_dot_stripped("Notes on Substitution") is False


# ---------------------------------------------------------------------------
# natural_key_of / collection_index
# ---------------------------------------------------------------------------


def test_natural_key_of_joins_key_fields():
    frontmatter = {"project": "stimela", "page": "guide", "title": "Guide"}
    assert natural_key_of(frontmatter, key_fields=("project", "page")) == "stimela/guide"


def test_natural_key_of_raises_on_a_missing_field():
    with pytest.raises(KeyError, match="citekey"):
        natural_key_of({"title": "No citekey here"}, key_fields=("citekey",))


def test_collection_index_skips_pre_migration_documents_without_an_id(tmp_path):
    _write_leaf(tmp_path / "Old Format.md", body="No frontmatter at all.\n")
    _write_leaf(
        tmp_path / "Migrated.md",
        frontmatter={"id": "abc1234567", "citekey": "smirnov2011", "title": "Migrated"},
    )

    indexed = collection_index(tmp_path, collection="literature", key_fields=("citekey",))

    assert [document.natural_key for document in indexed] == ["smirnov2011"]
    assert indexed[0].id == "abc1234567"


def test_collection_index_raises_on_a_migrated_document_missing_a_key_field(tmp_path):
    _write_leaf(
        tmp_path / "Broken.md",
        frontmatter={"id": "abc1234567", "title": "Broken"},
    )

    with pytest.raises(KeyError):
        collection_index(tmp_path, collection="literature", key_fields=("citekey",))


def test_collection_index_finds_documents_across_nested_groups(tmp_path):
    _write_leaf(
        tmp_path / "calibration" / "Deep Note.md",
        frontmatter={"id": "abc1234567", "slug": "deep-note", "title": "Deep Note"},
    )

    indexed = collection_index(tmp_path, collection="notes", key_fields=("slug",))

    assert len(indexed) == 1
    assert indexed[0].wrapper_dir is None
    assert indexed[0].md_path == tmp_path / "calibration" / "Deep Note.md"


# ---------------------------------------------------------------------------
# IndexedDocument.reserved_filename
# ---------------------------------------------------------------------------


def test_reserved_filename_of_a_bare_document_is_its_own_name():
    document = IndexedDocument(
        id="abc1234567", natural_key="deep-note",
        md_path=Path("/corpus/Deep Note.md"), wrapper_dir=None, frontmatter={},
    )
    assert document.reserved_filename == "Deep Note.md"


def test_reserved_filename_of_a_wrapped_document_is_its_wrapper_directory_name():
    # md_path.name is always the fixed content.md for a wrapped document -
    # the name that actually needs to stay unique is the wrapper directory's.
    document = IndexedDocument(
        id="abc1234567", natural_key="perkins2025",
        md_path=Path("/corpus/Africanus I/content.md"),
        wrapper_dir=Path("/corpus/Africanus I"), frontmatter={},
    )
    assert document.reserved_filename == "Africanus I.md"
