"""Tests for the `list_corpus` MCP tool.

The tool answers "what is in here at all", which `search_*` cannot: a corpus
document is addressed by an opaque surrogate id, so an agent that has not
just run a search has no way to name one. These cover the two shapes a
collection actually takes - grouped, and flat - because the flat one is what
literature and notes look like in practice and it used to report nothing
useful at all.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from boepie.tools import corpus as corpus_tool
from boepie.tools.corpus import ListCorpusInput, list_corpus

from tests.conftest import write_corpus_document


@pytest.fixture
def literature_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    collection_dir = tmp_path / "literature"
    collection_dir.mkdir()
    monkeypatch.setitem(corpus_tool._COLLECTION_DIRS, "literature", collection_dir)
    return collection_dir


@pytest.fixture
def docs_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    collection_dir = tmp_path / "docs"
    collection_dir.mkdir()
    monkeypatch.setitem(corpus_tool._COLLECTION_DIRS, "docs", collection_dir)
    return collection_dir


def _write_paper(collection_dir: Path, document_id: str, title: str, **kwargs) -> None:
    write_corpus_document(
        collection_dir,
        document_id=document_id,
        title=title,
        body="Body text.",
        bib={"citekey": document_id, "year": 2024},
        **kwargs,
    )


def _write_page(collection_dir: Path, document_id: str, title: str, project: str) -> None:
    write_corpus_document(
        collection_dir,
        document_id=document_id,
        title=title,
        body="Body text.",
        group=project,
        docs={"project": project, "page": title},
    )


# ---------------------------------------------------------------------------
# Flat collections - no group structure at all
# ---------------------------------------------------------------------------


async def test_a_flat_collection_names_its_documents(literature_dir: Path):
    """With no groups there is nothing to report at detail='groups'.

    The listing used to answer a one-document collection with a count line
    and a "(top level) 1 document(s)" line that restated it, naming nothing
    and giving a caller no id to read with.
    """
    _write_paper(literature_dir, "aaaaaaaaaa", "Radio interferometry basics")

    output = await list_corpus(ListCorpusInput(collection="literature"))

    assert "literature  1 document(s)" in output
    assert "  Radio interferometry basics  document_id=aaaaaaaaaa" in output


async def test_a_flat_collection_does_not_print_a_top_level_group_header(
    literature_dir: Path,
):
    _write_paper(literature_dir, "aaaaaaaaaa", "Radio interferometry basics")

    output = await list_corpus(ListCorpusInput(collection="literature"))

    assert "(top level)" not in output


async def test_a_flat_collection_sorts_documents_by_title(literature_dir: Path):
    _write_paper(literature_dir, "aaaaaaaaaa", "Zeta paper")
    _write_paper(literature_dir, "bbbbbbbbbb", "Alpha paper")

    output = await list_corpus(ListCorpusInput(collection="literature"))

    assert output.index("Alpha paper") < output.index("Zeta paper")


async def test_a_long_flat_collection_points_at_search_not_at_groups(
    literature_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    """There is no group to narrow to, so suggesting one would be useless."""
    monkeypatch.setattr(corpus_tool, "_MAX_DOCUMENTS_PER_GROUP", 2)
    for index in range(4):
        _write_paper(literature_dir, f"{index}aaaaaaaaa", f"Paper {index}")

    output = await list_corpus(ListCorpusInput(collection="literature"))

    assert "... 2 more (search_literature to find one)" in output


# ---------------------------------------------------------------------------
# Grouped collections
# ---------------------------------------------------------------------------


async def test_a_grouped_collection_reports_counts_per_group(docs_dir: Path):
    _write_page(docs_dir, "aaaaaaaaaa", "Recipes", project="stimela")
    _write_page(docs_dir, "bbbbbbbbbb", "Options", project="stimela")
    _write_page(docs_dir, "cccccccccc", "Gains", project="quartical")

    output = await list_corpus(ListCorpusInput(collection="docs"))

    assert output.splitlines() == [
        "docs  3 document(s)",
        "  quartical  1",
        "  stimela  2",
    ]


async def test_a_grouped_collection_withholds_titles_until_asked(docs_dir: Path):
    """detail='groups' is the cheap first look; titles are the opt-in."""
    _write_page(docs_dir, "aaaaaaaaaa", "Recipes", project="stimela")
    _write_page(docs_dir, "cccccccccc", "Gains", project="quartical")

    output = await list_corpus(ListCorpusInput(collection="docs"))

    assert "Recipes" not in output
    assert "document_id=" not in output


async def test_a_grouped_collection_names_documents_on_request(docs_dir: Path):
    _write_page(docs_dir, "aaaaaaaaaa", "Recipes", project="stimela")
    _write_page(docs_dir, "cccccccccc", "Gains", project="quartical")

    output = await list_corpus(
        ListCorpusInput(collection="docs", detail="documents")
    )

    assert "  stimela  1" in output
    assert "    Recipes  document_id=aaaaaaaaaa" in output


async def test_a_group_filter_restricts_the_listing(docs_dir: Path):
    _write_page(docs_dir, "aaaaaaaaaa", "Recipes", project="stimela")
    _write_page(docs_dir, "cccccccccc", "Gains", project="quartical")

    output = await list_corpus(
        ListCorpusInput(collection="docs", group="stimela", detail="documents")
    )

    assert "Recipes" in output
    assert "Gains" not in output


async def test_an_unknown_group_says_so(docs_dir: Path):
    _write_page(docs_dir, "aaaaaaaaaa", "Recipes", project="stimela")

    output = await list_corpus(ListCorpusInput(collection="docs", group="nope"))

    assert "No group 'nope' in 'docs'." == output


async def test_filtering_to_one_group_still_reports_it_as_a_group(docs_dir: Path):
    """A named group is not the degenerate case a flat collection is.

    "stimela holds 2 documents" is a real answer, so `detail` still governs
    whether the titles come with it. Only the unnamed top-level group
    collapses, because there the header carries no information the count
    line did not already give.
    """
    _write_page(docs_dir, "aaaaaaaaaa", "Recipes", project="stimela")
    _write_page(docs_dir, "bbbbbbbbbb", "Options", project="stimela")

    output = await list_corpus(ListCorpusInput(collection="docs", group="stimela"))

    assert "  stimela  2" in output
    assert "document_id=" not in output


# ---------------------------------------------------------------------------
# Empty collections
# ---------------------------------------------------------------------------


async def test_an_empty_collection_says_how_to_fill_it(literature_dir: Path):
    output = await list_corpus(ListCorpusInput(collection="literature"))

    assert "empty" in output
    assert "boepie corpus add literature" in output


# ---------------------------------------------------------------------------
# Tree structure
# ---------------------------------------------------------------------------


async def test_an_intermediate_group_appears_even_with_no_documents_of_its_own(
    docs_dir: Path,
):
    """A tree that omits a parent misrepresents where its children live.

    With pages only under `quartical/gains` and `quartical/solver` and none
    loose in `quartical/`, that directory is not a key in the grouping at
    all - so the first cut of this rendering indented both children under
    whichever unrelated group happened to sort above them.
    """
    _write_page(docs_dir, "aaaaaaaaaa", "G term", project="quartical/gains")
    _write_page(docs_dir, "bbbbbbbbbb", "Convergence", project="quartical/solver")

    output = await list_corpus(ListCorpusInput(collection="docs"))

    assert output.splitlines() == [
        "docs  2 document(s)",
        "  quartical  2",
        "    gains  1",
        "    solver  1",
    ]


async def test_a_parent_counts_its_whole_subtree(docs_dir: Path):
    """A bare zero on an intermediate directory would say nothing useful."""
    _write_page(docs_dir, "aaaaaaaaaa", "Overview", project="stimela")
    _write_page(docs_dir, "bbbbbbbbbb", "CLI", project="stimela/reference")
    _write_page(docs_dir, "cccccccccc", "Options", project="stimela/reference")

    output = await list_corpus(ListCorpusInput(collection="docs"))

    assert "  stimela  3" in output
    assert "    reference  2" in output


async def test_a_child_is_never_separated_from_its_parent(docs_dir: Path):
    """Sorting on the raw string would interleave the tree.

    "stimela-extras" sorts between "stimela" and "stimela/reference" as a
    string, which would push a child below an unrelated sibling; sorting on
    split path segments keeps it under its parent.
    """
    _write_page(docs_dir, "aaaaaaaaaa", "Overview", project="stimela")
    _write_page(docs_dir, "bbbbbbbbbb", "CLI", project="stimela/reference")
    _write_page(docs_dir, "cccccccccc", "Extra", project="stimela-extras")

    lines = (await list_corpus(ListCorpusInput(collection="docs"))).splitlines()

    assert lines == [
        "docs  3 document(s)",
        "  stimela  2",
        "    reference  1",
        "  stimela-extras  1",
    ]


async def test_documents_loose_at_the_root_sit_beside_the_groups(docs_dir: Path):
    _write_page(docs_dir, "bbbbbbbbbb", "CLI", project="stimela")
    write_corpus_document(
        docs_dir, document_id="aaaaaaaaaa", title="Loose page", body="x",
        docs={"project": "root", "page": "Loose page"},
    )

    output = await list_corpus(ListCorpusInput(collection="docs"))

    assert "  (top level)  1" in output
    assert "  stimela  1" in output
