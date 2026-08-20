"""End-to-end tests for the search_docs and read_docs tools.

Builds a tiny docs corpus (two projects, several pages each) with DocsLoader
and points boepie's default index lookup at it, so the tools are exercised
directly without an embedding backend. Mode is BM25-only (lexical) since
embedding=None.

Assertions target the F2/F4 output structure and payload bounds from
design/interface-spec.md, not the exact prose, which is expected to churn.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from boepie.rag import DocsLoader, engine
from boepie.tools._retrieval import SHORT_SNIPPET_CHARS
from boepie.tools.docs import (
    ReadDocsInput,
    ReadRequest,
    SearchDocsInput,
    read_docs,
    search_docs,
)
from tests.conftest import write_corpus_document

_LONG_TAIL = (
    "Beyond that, every cab declares its inputs and outputs in a schema that "
    "stimela validates before the container is ever launched, which is why a "
    "typo in a parameter name fails fast rather than halfway through a run."
)


class _FakeDocs:
    """Tiny docs corpus fixture builder: two projects, 2-3 pages each.

    Written in the real corpus layout (full-title filenames, surrogate ids,
    a namespaced `docs` frontmatter block) via `write_corpus_document`, so
    what DocsLoader walks here is what it walks in a real installation.
    """

    def __init__(self, tmp_corpus: Path) -> None:
        self.tmp_corpus = tmp_corpus

        pages = [
            (
                "doc0000001", "stimela", "recipe_syntax", "Stimela Recipe Syntax",
                "A stimela recipe defines a pipeline of cabs. "
                "Each cab is a containerised tool wrapped with parameter "
                "validation and input/output handling. " + _LONG_TAIL,
            ),
            (
                "doc0000002", "stimela", "cabs_reference", "Cabs Reference",
                "Cabs are containerised applications. "
                "They run inside Docker or Singularity containers and handle "
                "data I/O, parameter validation, and result reporting.",
            ),
            (
                "doc0000003", "quartical", "solvers", "Quartical Solvers",
                "Quartical supports multiple calibration solvers including "
                "Gauss-Newton, Levenberg-Marquardt, and direct matrix inversion. "
                "Each solver has trade-offs in convergence speed and accuracy.",
            ),
            (
                "doc0000004", "quartical", "gains", "Gain Calibration",
                "Gain calibration computes per-antenna complex gains to correct "
                "for instrumental effects and atmospheric distortions.",
            ),
        ]
        versions = {"stimela": "1.3.0", "quartical": "0.6.0"}
        base_urls = {
            "stimela": "https://stimela.readthedocs.io/",
            "quartical": "https://quartical.readthedocs.io/",
        }

        for document_id, project, page, title, body in pages:
            write_corpus_document(
                tmp_corpus,
                document_id=document_id,
                title=title,
                body=f"# {title}\n\n{body}",
                group=project,
                origin=f"{base_urls[project]}{page}.html",
                via="sphinx",
                source_format="html",
                docs={
                    "project": project,
                    "page": page,
                    "base_url": base_urls[project],
                    "version": versions[project],
                },
            )


def _hit_lines(payload: str) -> list[str]:
    return [line for line in payload.splitlines() if re.match(r"^\[\d+\] ", line)]


@pytest.fixture
async def docs_index(tmp_path, monkeypatch):
    """Build a stub docs index and make it the default one boepie loads."""
    # Named after the real corpus root so source-path relativisation has its
    # anchor to cut on, exactly as it would in a real installation.
    corpus_dir = tmp_path / "deep" / "nested" / "docs-corpus"
    _FakeDocs(corpus_dir)

    index_root = tmp_path / "index_root"
    index_root.mkdir(parents=True)

    await engine.build(DocsLoader(corpus_dir), index_root=index_root, embedding=None)

    # search_docs() calls boepie.rag.search() with no index_root override,
    # so it resolves the index root from engine.INDEX_DIR. Patch it so the
    # tool finds this stub index.
    monkeypatch.setattr(engine, "INDEX_DIR", index_root)

    yield corpus_dir
    await engine.clear_cache()


# ---------------------------------------------------------------------------
# F2: ranked hits
# ---------------------------------------------------------------------------


async def test_search_docs_emits_f2_count_line_and_numbered_blocks(docs_index):
    result = await search_docs(SearchDocsInput(question="recipe syntax", mode="bm25"))
    first = result.splitlines()[0]
    assert re.match(r'^\d+ hits for "recipe syntax" in docs$', first)
    hits = _hit_lines(result)
    assert hits
    assert int(first.split()[0]) == len(hits)


async def test_search_docs_hit_line_identifies_project_and_page(docs_index):
    result = await search_docs(SearchDocsInput(question="recipe syntax", mode="bm25"))
    hit = _hit_lines(result)[0]
    assert hit.startswith("[1] stimela: Stimela Recipe Syntax")


async def test_search_docs_hit_line_carries_no_numeric_score(docs_index):
    # MCP output drops the score entirely; rank order is the relevance signal.
    result = await search_docs(SearchDocsInput(question="recipe syntax", mode="bm25"))
    hit = _hit_lines(result)[0]
    assert "score=" not in hit
    assert not re.search(r"\bbm25=|\bcos=|\brrf=", hit)


async def test_search_docs_surfaces_copy_pasteable_read_handles(docs_index):
    result = await search_docs(SearchDocsInput(question="recipe syntax", mode="bm25"))
    assert "    read: document_id=doc0000001 chunk_index=" in result


async def test_search_docs_source_path_is_relative(docs_index):
    result = await search_docs(SearchDocsInput(question="recipe syntax", mode="bm25"))
    assert "source: stimela/Stimela Recipe Syntax.md (chars " in result
    assert str(docs_index) not in result  # no absolute path


async def test_search_docs_emits_no_banned_markdown(docs_index):
    result = await search_docs(SearchDocsInput(question="recipe syntax", mode="bm25"))
    generated = [line for line in result.splitlines() if not line.startswith("    ")]
    assert not [line for line in generated if line.startswith(("#", "_"))]
    assert "---" not in result


async def test_search_docs_filters_by_project(docs_index):
    result_stimela = await search_docs(
        SearchDocsInput(question="gains solvers", mode="bm25", project="stimela")
    )
    result_quartical = await search_docs(
        SearchDocsInput(question="gains solvers", mode="bm25", project="quartical")
    )
    assert _hit_lines(result_stimela) and _hit_lines(result_quartical)
    for line in _hit_lines(result_stimela):
        assert line.startswith("[1] stimela: ") or "stimela: " in line
    for line in _hit_lines(result_quartical):
        assert "quartical: " in line


# ---------------------------------------------------------------------------
# Section 4: the snippet policy
# ---------------------------------------------------------------------------


async def test_search_docs_snippet_short_is_the_default_and_is_bounded(docs_index):
    default = await search_docs(SearchDocsInput(question="recipe syntax", mode="bm25"))
    explicit = await search_docs(
        SearchDocsInput(question="recipe syntax", mode="bm25", snippet="short")
    )
    assert default == explicit
    assert "halfway through a run" not in default  # past the cutoff
    body_lines = [
        line
        for line in default.splitlines()
        if line.startswith("    ") and not line.lstrip().startswith(("read:", "source:"))
    ]
    assert body_lines
    for line in body_lines:
        assert len(line.strip()) <= SHORT_SNIPPET_CHARS + len(" ...")


async def test_search_docs_snippet_none_drops_bodies_and_shrinks_payload(docs_index):
    none_payload = await search_docs(
        SearchDocsInput(question="recipe syntax", mode="bm25", snippet="none")
    )
    short_payload = await search_docs(
        SearchDocsInput(question="recipe syntax", mode="bm25", snippet="short")
    )
    assert "A stimela recipe defines" not in none_payload
    assert len(none_payload) < len(short_payload)
    assert "read: document_id=" in none_payload


async def test_search_docs_snippet_full_returns_whole_chunk(docs_index):
    full_payload = await search_docs(
        SearchDocsInput(question="recipe syntax", mode="bm25", snippet="full")
    )
    short_payload = await search_docs(
        SearchDocsInput(question="recipe syntax", mode="bm25", snippet="short")
    )
    assert "halfway through a run" in full_payload
    assert len(full_payload) > len(short_payload)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


async def test_search_docs_no_index_reports_one_line_error_naming_a_command(
    tmp_path, monkeypatch
):
    # Point INDEX_DIR at an empty dir so this is deterministic.
    monkeypatch.setattr(engine, "INDEX_DIR", tmp_path)
    await engine.clear_cache()
    result = await search_docs(SearchDocsInput(question="anything"))
    assert result.startswith("Error:")
    assert len(result.splitlines()) == 1
    assert "boepie index build --collection docs" in result


async def test_search_docs_top_k_upper_bound_is_twenty(docs_index):
    with pytest.raises(ValueError):
        SearchDocsInput(question="anything", top_k=21)
    assert SearchDocsInput(question="anything", top_k=20).top_k == 20


# ---------------------------------------------------------------------------
# F4: spans
# ---------------------------------------------------------------------------


async def test_read_docs_emits_f4_provenance_header(docs_index):
    result = await read_docs(
        ReadDocsInput(requests=[ReadRequest(document_id="doc0000001")])
    )
    lines = result.splitlines()
    assert re.match(
        r"^document_id=doc0000001 chunks=\d+-\d+ chars=\d+-\d+$", lines[0]
    )
    assert lines[1] == "source: stimela/Stimela Recipe Syntax.md"
    assert lines[2] == "sections: Stimela Recipe Syntax"
    assert lines[3] == ""
    assert "A stimela recipe defines a pipeline of cabs." in result
    assert "---" not in result


async def test_read_docs_batches_without_a_horizontal_rule(docs_index):
    result = await read_docs(
        ReadDocsInput(
            requests=[
                ReadRequest(document_id="doc0000001"),
                ReadRequest(document_id="doc0000003"),
            ]
        )
    )
    assert result.count("document_id=") == 2
    assert "---" not in result


async def test_read_docs_reports_unknown_document_on_one_line(docs_index):
    result = await read_docs(
        ReadDocsInput(requests=[ReadRequest(document_id="nosuchid99")])
    )
    assert result.startswith("Error:")
    assert len(result.splitlines()) == 1
    assert "search_docs" in result
