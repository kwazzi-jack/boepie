"""Tests for the `boepie search` and `boepie read` CLI commands.

These commands are thin frontends over the same retrieval helpers the
search_*/read_* MCP tools use, so the assertions here target parity of
behaviour (routing, error messages) and the CLI-only additions: the `--json`
machine-readable mode and cross-collection dispatch (literature/docs/context
for search; literature/docs for read).

Indices are built lexical-only (BM25, embedding=None) so the tests stay fully
offline - no embedding backend is downloaded or contacted. Docs/literature
searches therefore run with `--mode bm25`; the resolved (fastembed) embedding
is ignored by a lexical-only index, exactly as the engine documents.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from boepie import cli
from boepie.context import index_root_for
from boepie.rag import engine
from boepie.rag.loaders import ContextLoader, DocsLoader
from tests.conftest import write_corpus_document


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


# ---------------------------------------------------------------------------
# Fixtures: tiny lexical-only indices on disk
# ---------------------------------------------------------------------------

_DOC_BODY = (
    "# Recipes\n\n"
    "A stimela recipe chains cabs such as wsclean for wide-field imaging with "
    "w-stacking and multiscale deconvolution. Each step wires one cab's output "
    "into the next cab's input, keeping a long imaging pipeline honest when an "
    "early parameter changes partway through a run.\n"
)

_CONTEXT_FRONTMATTER = "---\ntype: Concept\ntitle: {title}\nmanaged_by: boepie\n---\n\n"


def _write_docs_index(index_dir: Path, corpus_dir: Path) -> None:
    write_corpus_document(
        corpus_dir,
        document_id="doc0000001",
        title="Recipes",
        body=_DOC_BODY,
        group="stimela",
        origin="https://stimela.readthedocs.io/recipes.html",
        via="sphinx",
        source_format="html",
        docs={"project": "stimela", "page": "recipes"},
    )
    asyncio.run(engine.build(DocsLoader(corpus_dir), index_root=index_dir, embedding=None))


def _write_context_bundle(bundle_dir: Path) -> None:
    page = bundle_dir / "playbooks" / "imaging.md"
    page.parent.mkdir(parents=True, exist_ok=True)
    page.write_text(
        _CONTEXT_FRONTMATTER.format(title="Imaging strategy")
        + "# Imaging strategy\n\nA stimela imaging pipeline chains wsclean multiscale "
        "clean with successive masking rounds to recover faint extended emission.\n",
        encoding="utf-8",
    )
    (bundle_dir / "manifest.json").write_text(
        json.dumps({"bundle_version": "0.1.0"}), encoding="utf-8"
    )
    asyncio.run(engine.build(ContextLoader(bundle_dir), index_root=index_root_for(bundle_dir), embedding=None))


@pytest.fixture
def docs_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A lexical-only docs index under a patched INDEX_DIR."""
    index_dir = tmp_path / "indices"
    corpus_dir = tmp_path / "docs-corpus"
    _write_docs_index(index_dir, corpus_dir)
    monkeypatch.setattr(cli, "INDEX_DIR", index_dir)
    yield index_dir
    asyncio.run(engine.clear_cache())


@pytest.fixture
def empty_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A patched INDEX_DIR with no indices built."""
    index_dir = tmp_path / "indices"
    index_dir.mkdir()
    monkeypatch.setattr(cli, "INDEX_DIR", index_dir)
    yield index_dir
    asyncio.run(engine.clear_cache())


@pytest.fixture
def context_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A context bundle with its own index, run from inside the project."""
    project_dir = tmp_path / "project"
    bundle_dir = project_dir / ".boepie"
    _write_context_bundle(bundle_dir)
    monkeypatch.chdir(project_dir)
    yield bundle_dir
    asyncio.run(engine.clear_cache())


# ---------------------------------------------------------------------------
# search: text output
# ---------------------------------------------------------------------------


def test_search_docs_text_lists_hits(runner: CliRunner, docs_index: Path) -> None:
    result = runner.invoke(
        cli.cli, ["search", "wsclean imaging", "--collection", "docs", "--mode", "bm25"]
    )
    assert result.exit_code == 0, result.output
    assert 'hits for "wsclean imaging" in docs' in result.output
    assert "[1] stimela: Recipes" in result.output
    # F2 read handle line is present for docs (it has a read_* tool).
    assert "read: document_id=doc0000001" in result.output


def test_search_docs_text_shows_raw_per_leg_score_detail(runner: CliRunner, docs_index: Path) -> None:
    # Unlike the MCP tools, CLI text search shows the raw scores that back
    # the ranking - a bm25-only mode query shows just the bm25 leg plus rrf.
    result = runner.invoke(
        cli.cli, ["search", "wsclean imaging", "--collection", "docs", "--mode", "bm25"]
    )
    assert result.exit_code == 0, result.output
    hit_line = next(line for line in result.output.splitlines() if line.startswith("[1]"))
    assert "bm25=" in hit_line
    assert "rrf=" in hit_line
    assert "cos=" not in hit_line  # no dense leg ran in mode='bm25'


def test_search_context_text_uses_bundle_paths(runner: CliRunner, context_project: Path) -> None:
    result = runner.invoke(cli.cli, ["search", "wsclean imaging", "--collection", "context"])
    assert result.exit_code == 0, result.output
    assert "in context" in result.output
    assert "source: .boepie/playbooks/imaging.md" in result.output
    # Context has no read tool: no read handle line.
    assert "read:" not in result.output


# ---------------------------------------------------------------------------
# search: --json output
# ---------------------------------------------------------------------------


def test_search_docs_json_is_machine_readable(runner: CliRunner, docs_index: Path) -> None:
    result = runner.invoke(
        cli.cli,
        ["search", "wsclean imaging", "--collection", "docs", "--mode", "bm25", "--json"],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["collections"] == ["docs"]
    assert payload["question"] == "wsclean imaging"
    assert payload["hits"]
    top = payload["hits"][0]
    assert top["document_id"] == "doc0000001"
    assert top["rank"] == 1
    assert top["read_handle"] is True
    assert isinstance(top["rrf_score"], float)
    assert isinstance(top["bm25_score"], float)  # bm25 mode: lexical leg ran
    assert top["dense_score"] is None  # no dense leg in mode='bm25'
    assert "chars" not in top["source"]  # source is a clean relative path


def test_search_context_json_marks_no_read_handle(runner: CliRunner, context_project: Path) -> None:
    result = runner.invoke(
        cli.cli, ["search", "imaging", "--collection", "context", "--json"]
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["collections"] == ["context"]
    assert payload["hits"][0]["read_handle"] is False
    assert payload["hits"][0]["source"].startswith(".boepie/")


def test_search_context_json_carries_bm25_score_and_null_dense_score(
    runner: CliRunner, context_project: Path
) -> None:
    # context is BM25-only: the raw bm25 leg is populated, the dense leg
    # never ran so dense_score stays null.
    result = runner.invoke(
        cli.cli, ["search", "imaging", "--collection", "context", "--json"]
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    top = payload["hits"][0]
    assert isinstance(top["bm25_score"], float)
    assert top["dense_score"] is None
    assert isinstance(top["rrf_score"], float)


def test_search_json_snippet_option_does_not_break_json(runner: CliRunner, docs_index: Path) -> None:
    # --snippet only affects the text renderer; JSON always carries full text.
    result = runner.invoke(
        cli.cli,
        ["search", "recipe", "--collection", "docs", "--mode", "bm25", "--json", "--snippet", "none"],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["hits"][0]["text"]


# ---------------------------------------------------------------------------
# search: errors and routing
# ---------------------------------------------------------------------------


def test_search_missing_index_names_the_fix(runner: CliRunner, empty_index: Path) -> None:
    result = runner.invoke(
        cli.cli, ["search", "anything", "--collection", "docs", "--mode", "bm25"]
    )
    assert result.exit_code != 0
    assert "no 'docs' index" in result.output
    assert "index build --collection docs" in result.output


def test_search_context_without_bundle_says_init(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    empty = tmp_path / "not-a-project"
    empty.mkdir()
    monkeypatch.chdir(empty)
    result = runner.invoke(cli.cli, ["search", "anything", "--collection", "context"])
    assert result.exit_code != 0
    assert "boepie context init" in result.output


def test_search_rejects_unknown_collection(runner: CliRunner) -> None:
    result = runner.invoke(cli.cli, ["search", "x", "--collection", "nope"])
    assert result.exit_code != 0
    assert "nope" in result.output


# ---------------------------------------------------------------------------
# read
# ---------------------------------------------------------------------------


def test_read_docs_text_returns_span(runner: CliRunner, docs_index: Path) -> None:
    result = runner.invoke(
        cli.cli, ["read", "doc0000001", "--collection", "docs"]
    )
    assert result.exit_code == 0, result.output
    assert "document_id=doc0000001" in result.output
    assert "source: stimela/Recipes.md" in result.output
    assert "wsclean" in result.output


def test_read_docs_json_carries_text_and_provenance(runner: CliRunner, docs_index: Path) -> None:
    result = runner.invoke(
        cli.cli, ["read", "doc0000001", "--collection", "docs", "--json"]
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["document_id"] == "doc0000001"
    assert payload["source"] == "stimela/Recipes.md"
    assert "wsclean" in payload["text"]


def test_read_bad_document_id_points_to_search(runner: CliRunner, docs_index: Path) -> None:
    result = runner.invoke(
        cli.cli, ["read", "nosuchid99", "--collection", "docs"]
    )
    assert result.exit_code != 0
    assert "boepie search --collection docs" in result.output


def test_read_rejects_context_collection(runner: CliRunner) -> None:
    # context has no read counterpart; the choice must not accept it.
    result = runner.invoke(cli.cli, ["read", "x", "--collection", "context"])
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# command surface
# ---------------------------------------------------------------------------


def test_search_and_read_replace_ask(runner: CliRunner) -> None:
    top = runner.invoke(cli.cli, ["--help"])
    assert "search" in top.output
    assert "read" in top.output
    assert "ask" not in top.output


def test_search_help_advertises_json(runner: CliRunner) -> None:
    result = runner.invoke(cli.cli, ["search", "--help"])
    assert result.exit_code == 0
    assert "--json" in result.output
    assert "--embedding-binding" in result.output
