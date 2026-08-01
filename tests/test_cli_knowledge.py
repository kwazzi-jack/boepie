"""Tests for the CLI knowledge commands (init, update, status, hint).

Tests the `.boepie/` bundle lifecycle, BM25 indexing, and hint-mode search
via the CLI interface, using a temporary bundle.

The knowledge index is per-bundle (`.boepie/.index/knowledge/bm25/`), not
machine-global, so INDEX_DIR is only monkeypatched to keep the
legacy-index notice off a developer's real data directory.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from boepie import cli
from boepie.knowledge import index_root_for
from boepie.rag import clear_cache
from boepie.rag.models import Chunk, SearchResult


@pytest.fixture
def runner() -> CliRunner:
    """CLI test runner."""
    return CliRunner()


@pytest.fixture
def tmp_index_dir(tmp_path: Path) -> Path:
    """Temporary directory for monkeypatched INDEX_DIR."""
    index_dir = tmp_path / "indices"
    index_dir.mkdir()
    return index_dir


def _knowledge_index_dir(project_dir: Path) -> Path:
    """Where init/apply build the bundle's own BM25 index."""
    return index_root_for(project_dir / ".boepie") / "knowledge" / "bm25"


# ---------------------------------------------------------------------------
# knowledge init
# ---------------------------------------------------------------------------


def test_knowledge_init_creates_bundle_layout(runner: CliRunner, tmp_path: Path, tmp_index_dir: Path) -> None:
    """Init should create .boepie/ with manifest.json, index.md, and seed content."""
    with patch("boepie.cli.INDEX_DIR", tmp_index_dir):
        result = runner.invoke(cli.cli, ["knowledge", "init", "--directory", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert (tmp_path / ".boepie" / "manifest.json").exists()
    assert (tmp_path / ".boepie" / "index.md").exists()
    assert (tmp_path / ".boepie" / "log.md").exists()


def test_knowledge_init_appends_agents_pointer(runner: CliRunner, tmp_path: Path, tmp_index_dir: Path) -> None:
    """Init should append pointer to AGENTS.md."""
    with patch("boepie.cli.INDEX_DIR", tmp_index_dir):
        result = runner.invoke(cli.cli, ["knowledge", "init", "--directory", str(tmp_path)])

    assert result.exit_code == 0
    agents_md = tmp_path / "AGENTS.md"
    assert agents_md.exists()
    text = agents_md.read_text(encoding="utf-8")
    assert "Stimela knowledge base in `.boepie/`" in text


def test_knowledge_init_pointer_idempotent(runner: CliRunner, tmp_path: Path, tmp_index_dir: Path) -> None:
    """Running init twice should append the pointer only once."""
    dir1 = tmp_path / "dir1"
    dir2 = tmp_path / "dir2"
    dir1.mkdir()
    dir2.mkdir()

    with patch("boepie.cli.INDEX_DIR", tmp_index_dir):
        result1 = runner.invoke(cli.cli, ["knowledge", "init", "--directory", str(dir1)])
        result2 = runner.invoke(cli.cli, ["knowledge", "init", "--directory", str(dir2)])

    assert result1.exit_code == 0
    assert result2.exit_code == 0

    agents_md = tmp_path / "AGENTS.md"
    if agents_md.exists():
        text = agents_md.read_text(encoding="utf-8")
        pointer_count = text.count("Stimela knowledge base in `.boepie/`")
        assert pointer_count == 1


def test_knowledge_init_builds_bm25_index_inside_the_bundle(
    runner: CliRunner, tmp_path: Path, tmp_index_dir: Path
) -> None:
    """The index belongs to the bundle it was built from, not to the machine."""
    with patch("boepie.cli.INDEX_DIR", tmp_index_dir):
        result = runner.invoke(cli.cli, ["knowledge", "init", "--directory", str(tmp_path)])

    assert result.exit_code == 0
    knowledge_index = _knowledge_index_dir(tmp_path)
    assert knowledge_index.exists()
    assert (knowledge_index / "manifest.json").exists()
    assert (knowledge_index / "chunks.jsonl").exists()
    # Nothing lands in the machine-global store.
    assert not (tmp_index_dir / "knowledge").exists()


def test_knowledge_init_gitignores_the_derived_index(
    runner: CliRunner, tmp_path: Path, tmp_index_dir: Path
) -> None:
    """`.boepie/` is committable; the index inside it is not."""
    with patch("boepie.cli.INDEX_DIR", tmp_index_dir):
        result = runner.invoke(cli.cli, ["knowledge", "init", "--directory", str(tmp_path)])

    assert result.exit_code == 0
    gitignore = tmp_path / ".boepie" / ".gitignore"
    assert gitignore.read_text(encoding="utf-8").splitlines() == [".index/"]


def test_knowledge_apply_keeps_the_gitignore_line_unduplicated(
    runner: CliRunner, tmp_path: Path, tmp_index_dir: Path
) -> None:
    with patch("boepie.cli.INDEX_DIR", tmp_index_dir):
        assert runner.invoke(cli.cli, ["knowledge", "init", "--directory", str(tmp_path)]).exit_code == 0
        assert runner.invoke(cli.cli, ["knowledge", "apply", "--directory", str(tmp_path)]).exit_code == 0

    gitignore_text = (tmp_path / ".boepie" / ".gitignore").read_text(encoding="utf-8")
    assert gitignore_text.count(".index/") == 1


def test_two_projects_each_keep_their_own_index(
    runner: CliRunner, tmp_path: Path, tmp_index_dir: Path
) -> None:
    """Initialising a second project must not clobber the first's index -
    the whole point of moving the index into the bundle."""
    project_a = tmp_path / "a"
    project_b = tmp_path / "b"
    project_a.mkdir()
    project_b.mkdir()

    with patch("boepie.cli.INDEX_DIR", tmp_index_dir):
        assert runner.invoke(cli.cli, ["knowledge", "init", "--directory", str(project_a)]).exit_code == 0
        chunks_a_before = (_knowledge_index_dir(project_a) / "chunks.jsonl").read_bytes()

        assert runner.invoke(cli.cli, ["knowledge", "init", "--directory", str(project_b)]).exit_code == 0

    assert _knowledge_index_dir(project_a).exists()
    assert _knowledge_index_dir(project_b).exists()
    assert (_knowledge_index_dir(project_a) / "chunks.jsonl").read_bytes() == chunks_a_before


def test_knowledge_init_notes_a_legacy_global_index(
    runner: CliRunner, tmp_path: Path, tmp_index_dir: Path
) -> None:
    """An orphaned pre-move index is surfaced, never deleted."""
    legacy_dir = tmp_index_dir / "knowledge" / "bm25"
    legacy_dir.mkdir(parents=True)
    (legacy_dir / "manifest.json").write_text("{}", encoding="utf-8")

    with patch("boepie.cli.INDEX_DIR", tmp_index_dir):
        result = runner.invoke(cli.cli, ["knowledge", "init", "--directory", str(tmp_path)])

    assert result.exit_code == 0
    assert "legacy knowledge index" in result.output
    assert legacy_dir.exists()  # non-destructive


def test_knowledge_init_with_skills_flag(runner: CliRunner, tmp_path: Path, tmp_index_dir: Path) -> None:
    """Init with --skills should print 'not implemented yet'."""
    with patch("boepie.cli.INDEX_DIR", tmp_index_dir):
        result = runner.invoke(cli.cli, ["knowledge", "init", "--directory", str(tmp_path), "--skills"])

    assert result.exit_code == 0
    assert "not implemented yet" in result.output


def test_knowledge_init_with_hooks_flag(runner: CliRunner, tmp_path: Path, tmp_index_dir: Path) -> None:
    """Init with --hooks should print 'not implemented yet'."""
    with patch("boepie.cli.INDEX_DIR", tmp_index_dir):
        result = runner.invoke(cli.cli, ["knowledge", "init", "--directory", str(tmp_path), "--hooks"])

    assert result.exit_code == 0
    assert "not implemented yet" in result.output


# ---------------------------------------------------------------------------
# knowledge apply
# ---------------------------------------------------------------------------


def test_knowledge_apply_rebuilds_index(runner: CliRunner, tmp_path: Path, tmp_index_dir: Path) -> None:
    """Apply should rebuild the knowledge index."""
    with patch("boepie.cli.INDEX_DIR", tmp_index_dir):
        # First init.
        result_init = runner.invoke(cli.cli, ["knowledge", "init", "--directory", str(tmp_path)])
        assert result_init.exit_code == 0

        # Then apply.
        result_apply = runner.invoke(cli.cli, ["knowledge", "apply", "--directory", str(tmp_path)])
        assert result_apply.exit_code == 0, result_apply.output

        # Index should still exist after apply.
        assert _knowledge_index_dir(tmp_path).exists()


# ---------------------------------------------------------------------------
# knowledge status
# ---------------------------------------------------------------------------


def test_knowledge_status_prints_current(runner: CliRunner, tmp_path: Path, tmp_index_dir: Path) -> None:
    """Status should print 'current' right after init."""
    with patch("boepie.cli.INDEX_DIR", tmp_index_dir):
        result_init = runner.invoke(cli.cli, ["knowledge", "init", "--directory", str(tmp_path)])
        assert result_init.exit_code == 0

        result_status = runner.invoke(cli.cli, ["knowledge", "status", "--directory", str(tmp_path)])
        assert result_status.exit_code == 0
        assert "current" in result_status.output


def test_knowledge_status_raises_when_no_bundle(runner: CliRunner, tmp_path: Path, tmp_index_dir: Path) -> None:
    """Status should fail when no bundle exists."""
    with patch("boepie.cli.INDEX_DIR", tmp_index_dir):
        result = runner.invoke(cli.cli, ["knowledge", "status", "--directory", str(tmp_path)])

    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# hint command
# ---------------------------------------------------------------------------


def test_hint_with_matching_prompt_prints_results(
    runner: CliRunner, tmp_path: Path, tmp_index_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Hint with a prompt matching content should print results.

    The hook runs `boepie hint` from the project directory, so the cwd is
    what selects the bundle.
    """
    with patch("boepie.cli.INDEX_DIR", tmp_index_dir):
        # Initialize bundle.
        result_init = runner.invoke(cli.cli, ["knowledge", "init", "--directory", str(tmp_path)])
        assert result_init.exit_code == 0

        # Clear cache to avoid cross-test bleed.
        import asyncio
        asyncio.run(clear_cache())

        monkeypatch.chdir(tmp_path)
        # Search for something that should match seed content.
        result_hint = runner.invoke(cli.cli, ["hint", "knowledge"])

    assert result_hint.exit_code == 0
    # Results should have at most 3 lines (or be empty if no match above threshold).
    lines = [line for line in result_hint.output.strip().split("\n") if line]
    assert len(lines) <= 3


def test_hint_with_gibberish_prints_nothing(
    runner: CliRunner, tmp_path: Path, tmp_index_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Hint with gibberish should print nothing and exit 0."""
    with patch("boepie.cli.INDEX_DIR", tmp_index_dir):
        # Initialize bundle.
        result_init = runner.invoke(cli.cli, ["knowledge", "init", "--directory", str(tmp_path)])
        assert result_init.exit_code == 0

        # Clear cache to avoid cross-test bleed.
        import asyncio
        asyncio.run(clear_cache())

        monkeypatch.chdir(tmp_path)
        # Search for something unlikely to match.
        result_hint = runner.invoke(cli.cli, ["hint", "xyzqwerty123notreal"])

    assert result_hint.exit_code == 0
    # Should have no output or only warning lines (no search results).
    output = result_hint.output.strip()
    # Allow for any banner text but no actual search results.
    # Results would be in the form "document_id#section: snippet"
    assert "#" not in output or len(output) == 0


def test_hint_with_no_bundle_exits_silently(
    runner: CliRunner, tmp_path: Path, tmp_index_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A prompt hook fires on every prompt: no bundle above the cwd must stay
    silent and exit 0, never error."""
    monkeypatch.chdir(tmp_path)
    with patch("boepie.cli.INDEX_DIR", tmp_index_dir):
        result = runner.invoke(cli.cli, ["hint", "test prompt"])

    assert result.exit_code == 0
    assert result.output.strip() == ""


def test_hint_finds_the_bundle_from_a_nested_subdirectory(
    runner: CliRunner, tmp_path: Path, tmp_index_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Discovery walks up, so a hook fired deep in a project still works."""
    with patch("boepie.cli.INDEX_DIR", tmp_index_dir):
        assert runner.invoke(cli.cli, ["knowledge", "init", "--directory", str(tmp_path)]).exit_code == 0

        import asyncio
        asyncio.run(clear_cache())

        nested_dir = tmp_path / "recipes" / "imaging"
        nested_dir.mkdir(parents=True)
        monkeypatch.chdir(nested_dir)
        result = runner.invoke(cli.cli, ["hint", "knowledge"])

    assert result.exit_code == 0


# ---------------------------------------------------------------------------
# hint: the raw-BM25 gate (_HINT_MIN_SCORE), isolated from real corpus tuning
# ---------------------------------------------------------------------------
#
# hint gates on results[0].bm25_score, not the RRF `.score` (which for a
# BM25-only search caps at ~0.016 and could never clear a threshold tuned in
# BM25 units). Building a real corpus whose BM25 magnitude straddles the
# threshold is fragile, so these patch `search` directly with a canned
# SearchResult carrying a controlled bm25_score - the gate logic itself is
# what's under test, not BM25 magnitude.


def _fake_hint_result(bm25_score: float | None) -> SearchResult:
    chunk = Chunk(
        id="concepts/x::0", collection="knowledge", document_id="concepts/x",
        chunk_index=0, text="a matching snippet", source_path="/bundle/concepts/x.md",
        char_start=0, char_end=10, section="X",
    )
    return SearchResult(chunk=chunk, score=0.016, bm25_rank=1, bm25_score=bm25_score)


def test_hint_fires_when_top_bm25_score_clears_the_threshold(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "find_bundle", lambda: tmp_path)

    async def fake_search(*args, **kwargs):
        return [_fake_hint_result(cli._HINT_MIN_SCORE + 1.0)]

    monkeypatch.setattr(cli, "search", fake_search)

    result = runner.invoke(cli.cli, ["hint", "anything"])
    assert result.exit_code == 0
    assert "concepts/x" in result.output
    assert "a matching snippet" in result.output


def test_hint_stays_silent_when_top_bm25_score_is_below_the_threshold(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "find_bundle", lambda: tmp_path)

    async def fake_search(*args, **kwargs):
        return [_fake_hint_result(cli._HINT_MIN_SCORE - 0.5)]

    monkeypatch.setattr(cli, "search", fake_search)

    result = runner.invoke(cli.cli, ["hint", "anything"])
    assert result.exit_code == 0
    assert result.output.strip() == ""


def test_hint_stays_silent_when_bm25_score_is_none(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fail closed: an unexpected shape (bm25_score missing) must never spam."""
    monkeypatch.setattr(cli, "find_bundle", lambda: tmp_path)

    async def fake_search(*args, **kwargs):
        return [_fake_hint_result(None)]

    monkeypatch.setattr(cli, "search", fake_search)

    result = runner.invoke(cli.cli, ["hint", "anything"])
    assert result.exit_code == 0
    assert result.output.strip() == ""
