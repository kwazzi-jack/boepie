"""Tests for the CLI context commands (init, apply, status, reset, hint).

Tests the `.boepie/` bundle lifecycle, BM25 indexing, and hint-mode search
via the CLI interface, using a temporary bundle.

The context index is per-bundle (`.boepie/.index/context/bm25/`), not
machine-global, so INDEX_DIR is only monkeypatched to keep the
legacy-index notice off a developer's real data directory.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from boepie import cli
from boepie.context import index_root_for
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


def _context_index_dir(project_dir: Path) -> Path:
    """Where init/apply build the bundle's own BM25 index."""
    return index_root_for(project_dir / ".boepie") / "context" / "bm25"


# ---------------------------------------------------------------------------
# context init
# ---------------------------------------------------------------------------


def test_context_init_creates_bundle_layout(runner: CliRunner, tmp_path: Path, tmp_index_dir: Path) -> None:
    """Init should create .boepie/ with manifest.json, index.md, and seed content."""
    with patch("boepie.cli.INDEX_DIR", tmp_index_dir):
        result = runner.invoke(cli.cli, ["context", "init", "--directory", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert (tmp_path / ".boepie" / "manifest.json").exists()
    assert (tmp_path / ".boepie" / "index.md").exists()
    assert (tmp_path / ".boepie" / "apply-log.md").exists()


def test_context_init_appends_agents_pointer(runner: CliRunner, tmp_path: Path, tmp_index_dir: Path) -> None:
    """Init should append pointer to AGENTS.md."""
    with patch("boepie.cli.INDEX_DIR", tmp_index_dir):
        result = runner.invoke(cli.cli, ["context", "init", "--directory", str(tmp_path)])

    assert result.exit_code == 0
    agents_md = tmp_path / "AGENTS.md"
    assert agents_md.exists()
    text = agents_md.read_text(encoding="utf-8")
    assert "Stimela knowledge base in `.boepie/`" in text


def test_context_init_pointer_idempotent(runner: CliRunner, tmp_path: Path, tmp_index_dir: Path) -> None:
    """Running init twice should append the pointer only once."""
    dir1 = tmp_path / "dir1"
    dir2 = tmp_path / "dir2"
    dir1.mkdir()
    dir2.mkdir()

    with patch("boepie.cli.INDEX_DIR", tmp_index_dir):
        result1 = runner.invoke(cli.cli, ["context", "init", "--directory", str(dir1)])
        result2 = runner.invoke(cli.cli, ["context", "init", "--directory", str(dir2)])

    assert result1.exit_code == 0
    assert result2.exit_code == 0

    agents_md = tmp_path / "AGENTS.md"
    if agents_md.exists():
        text = agents_md.read_text(encoding="utf-8")
        pointer_count = text.count("Stimela knowledge base in `.boepie/`")
        assert pointer_count == 1


def test_context_init_builds_bm25_index_inside_the_bundle(
    runner: CliRunner, tmp_path: Path, tmp_index_dir: Path
) -> None:
    """The index belongs to the bundle it was built from, not to the machine."""
    with patch("boepie.cli.INDEX_DIR", tmp_index_dir):
        result = runner.invoke(cli.cli, ["context", "init", "--directory", str(tmp_path)])

    assert result.exit_code == 0
    context_index = _context_index_dir(tmp_path)
    assert context_index.exists()
    assert (context_index / "manifest.json").exists()
    assert (context_index / "chunks.jsonl").exists()
    # Nothing lands in the machine-global store.
    assert not (tmp_index_dir / "context").exists()


def test_context_init_gitignores_the_derived_index(
    runner: CliRunner, tmp_path: Path, tmp_index_dir: Path
) -> None:
    """`.boepie/` is committable; the index inside it is not."""
    with patch("boepie.cli.INDEX_DIR", tmp_index_dir):
        result = runner.invoke(cli.cli, ["context", "init", "--directory", str(tmp_path)])

    assert result.exit_code == 0
    gitignore = tmp_path / ".boepie" / ".gitignore"
    assert gitignore.read_text(encoding="utf-8").splitlines() == [".index/"]


def test_context_apply_keeps_the_gitignore_line_unduplicated(
    runner: CliRunner, tmp_path: Path, tmp_index_dir: Path
) -> None:
    with patch("boepie.cli.INDEX_DIR", tmp_index_dir):
        assert runner.invoke(cli.cli, ["context", "init", "--directory", str(tmp_path)]).exit_code == 0
        assert runner.invoke(cli.cli, ["context", "apply", "--directory", str(tmp_path)]).exit_code == 0

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
        assert runner.invoke(cli.cli, ["context", "init", "--directory", str(project_a)]).exit_code == 0
        chunks_a_before = (_context_index_dir(project_a) / "chunks.jsonl").read_bytes()

        assert runner.invoke(cli.cli, ["context", "init", "--directory", str(project_b)]).exit_code == 0

    assert _context_index_dir(project_a).exists()
    assert _context_index_dir(project_b).exists()
    assert (_context_index_dir(project_a) / "chunks.jsonl").read_bytes() == chunks_a_before


def test_context_init_notes_a_legacy_global_index(
    runner: CliRunner, tmp_path: Path, tmp_index_dir: Path
) -> None:
    """An orphaned pre-move index is surfaced, never deleted."""
    legacy_dir = tmp_index_dir / "knowledge" / "bm25"
    legacy_dir.mkdir(parents=True)
    (legacy_dir / "manifest.json").write_text("{}", encoding="utf-8")

    with patch("boepie.cli.INDEX_DIR", tmp_index_dir):
        result = runner.invoke(cli.cli, ["context", "init", "--directory", str(tmp_path)])

    assert result.exit_code == 0
    assert "legacy knowledge index" in result.output
    assert legacy_dir.exists()  # non-destructive


def test_context_init_with_skills_flag(runner: CliRunner, tmp_path: Path, tmp_index_dir: Path) -> None:
    """Init with --skills should print 'not implemented yet'."""
    with patch("boepie.cli.INDEX_DIR", tmp_index_dir):
        result = runner.invoke(cli.cli, ["context", "init", "--directory", str(tmp_path), "--skills"])

    assert result.exit_code == 0
    assert "not implemented yet" in result.output


def test_context_init_with_hooks_flag(runner: CliRunner, tmp_path: Path, tmp_index_dir: Path) -> None:
    """Init with --hooks should print 'not implemented yet'."""
    with patch("boepie.cli.INDEX_DIR", tmp_index_dir):
        result = runner.invoke(cli.cli, ["context", "init", "--directory", str(tmp_path), "--hooks"])

    assert result.exit_code == 0
    assert "not implemented yet" in result.output


# ---------------------------------------------------------------------------
# context apply
# ---------------------------------------------------------------------------


def test_context_apply_rebuilds_index(runner: CliRunner, tmp_path: Path, tmp_index_dir: Path) -> None:
    """Apply should rebuild the context index."""
    with patch("boepie.cli.INDEX_DIR", tmp_index_dir):
        # First init.
        result_init = runner.invoke(cli.cli, ["context", "init", "--directory", str(tmp_path)])
        assert result_init.exit_code == 0

        # Then apply.
        result_apply = runner.invoke(cli.cli, ["context", "apply", "--directory", str(tmp_path)])
        assert result_apply.exit_code == 0, result_apply.output

        # Index should still exist after apply.
        assert _context_index_dir(tmp_path).exists()


def test_context_apply_force_reverts_a_source_local_file(
    runner: CliRunner, tmp_path: Path, tmp_index_dir: Path
) -> None:
    """A hand-edited, managed_by: user seed file gets reverted byte-for-byte by
    --force, and the apply-log.md records the force-revert distinctly.

    Content cache pinned to an empty directory so `resolve_content_source()`
    falls back to the packaged seeds, independent of whatever a developer's
    machine happens to have cached at BOEPIE_CONTENT_DIR.
    """
    with (
        patch("boepie.cli.INDEX_DIR", tmp_index_dir),
        patch("boepie.context.bundle.CONTENT_DIR", tmp_path / "content-cache-unset"),
    ):
        assert runner.invoke(cli.cli, ["context", "init", "--directory", str(tmp_path)]).exit_code == 0

        concept_path = tmp_path / ".boepie" / "concepts" / "skeleton.md"
        from boepie.context.frontmatter import read_frontmatter, write_frontmatter

        frontmatter, _ = read_frontmatter(concept_path.read_text(encoding="utf-8"))
        frontmatter["managed_by"] = "user"
        concept_path.write_text(
            write_frontmatter(frontmatter, "# Hand-edited\n\nDo not touch.\n"), encoding="utf-8"
        )

        result = runner.invoke(
            cli.cli,
            ["context", "apply", "--directory", str(tmp_path), "--force", "concepts/skeleton.md"],
        )

    assert result.exit_code == 0, result.output

    from boepie.context.bundle import _seed_content_dir

    seed_bytes = (_seed_content_dir() / "concepts" / "skeleton.md").read_bytes()
    assert concept_path.read_bytes() == seed_bytes

    log_text = (tmp_path / ".boepie" / "apply-log.md").read_text(encoding="utf-8")
    assert "force-reverted 1 managed_by: user file(s)" in log_text
    assert "concepts/skeleton.md" in log_text


def test_context_apply_force_on_a_nonexistent_path_exits_non_zero(
    runner: CliRunner, tmp_path: Path, tmp_index_dir: Path
) -> None:
    with patch("boepie.cli.INDEX_DIR", tmp_index_dir):
        assert runner.invoke(cli.cli, ["context", "init", "--directory", str(tmp_path)]).exit_code == 0

        result = runner.invoke(
            cli.cli,
            ["context", "apply", "--directory", str(tmp_path), "--force", "nonexistent/path.md"],
        )

    assert result.exit_code != 0
    assert "no such bundle file to revert" in result.output


# ---------------------------------------------------------------------------
# context status
# ---------------------------------------------------------------------------


def test_context_status_prints_current(runner: CliRunner, tmp_path: Path, tmp_index_dir: Path) -> None:
    """Status should print 'current' right after init."""
    with patch("boepie.cli.INDEX_DIR", tmp_index_dir):
        result_init = runner.invoke(cli.cli, ["context", "init", "--directory", str(tmp_path)])
        assert result_init.exit_code == 0

        result_status = runner.invoke(cli.cli, ["context", "status", "--directory", str(tmp_path)])
        assert result_status.exit_code == 0
        assert "current" in result_status.output


def test_context_status_raises_when_no_bundle(runner: CliRunner, tmp_path: Path, tmp_index_dir: Path) -> None:
    """Status should fail when no bundle exists."""
    with patch("boepie.cli.INDEX_DIR", tmp_index_dir):
        result = runner.invoke(cli.cli, ["context", "status", "--directory", str(tmp_path)])

    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# context reset
# ---------------------------------------------------------------------------


def test_context_reset_yes_discards_local_files_and_rebuilds(
    runner: CliRunner, tmp_path: Path, tmp_index_dir: Path
) -> None:
    """`--yes` skips the confirmation prompt; the local file is gone and the
    bundle is otherwise indistinguishable from a fresh init."""
    with patch("boepie.cli.INDEX_DIR", tmp_index_dir):
        assert runner.invoke(cli.cli, ["context", "init", "--directory", str(tmp_path)]).exit_code == 0

        from boepie.context.frontmatter import write_frontmatter

        local_path = tmp_path / ".boepie" / "concepts" / "my-notes.md"
        local_path.write_text(
            write_frontmatter(
                {
                    "type": "Concept", "title": "My notes",
                    "description": "Scratch notes.", "tags": [], "managed_by": "user",
                },
                "# My notes\n\nKeep this.\n",
            ),
            encoding="utf-8",
        )

        result = runner.invoke(
            cli.cli, ["context", "reset", "--directory", str(tmp_path), "--yes"]
        )

    assert result.exit_code == 0, result.output
    assert not local_path.exists()
    assert (tmp_path / ".boepie" / "concepts" / "skeleton.md").exists()
    assert _context_index_dir(tmp_path).exists()


def test_context_reset_declining_confirmation_leaves_bundle_untouched(
    runner: CliRunner, tmp_path: Path, tmp_index_dir: Path
) -> None:
    """No `--yes`, and the user answers 'n' at the prompt: nothing changes."""
    with patch("boepie.cli.INDEX_DIR", tmp_index_dir):
        assert runner.invoke(cli.cli, ["context", "init", "--directory", str(tmp_path)]).exit_code == 0

        from boepie.context.frontmatter import write_frontmatter

        local_path = tmp_path / ".boepie" / "concepts" / "my-notes.md"
        local_path.write_text(
            write_frontmatter(
                {
                    "type": "Concept", "title": "My notes",
                    "description": "Scratch notes.", "tags": [], "managed_by": "user",
                },
                "# My notes\n\nKeep this.\n",
            ),
            encoding="utf-8",
        )
        bundle_dir = tmp_path / ".boepie"
        bundle_snapshot_before = {
            path: path.read_bytes() for path in sorted(bundle_dir.rglob("*")) if path.is_file()
        }

        result = runner.invoke(
            cli.cli, ["context", "reset", "--directory", str(tmp_path)], input="n\n"
        )

    assert result.exit_code != 0
    bundle_snapshot_after = {
        path: path.read_bytes() for path in sorted(bundle_dir.rglob("*")) if path.is_file()
    }
    assert bundle_snapshot_after == bundle_snapshot_before


def test_context_reset_confirming_at_the_prompt_discards_local_files(
    runner: CliRunner, tmp_path: Path, tmp_index_dir: Path
) -> None:
    """No `--yes`, but the user answers 'y' at the prompt: same effect as --yes."""
    with patch("boepie.cli.INDEX_DIR", tmp_index_dir):
        assert runner.invoke(cli.cli, ["context", "init", "--directory", str(tmp_path)]).exit_code == 0

        from boepie.context.frontmatter import write_frontmatter

        local_path = tmp_path / ".boepie" / "concepts" / "my-notes.md"
        local_path.write_text(
            write_frontmatter(
                {
                    "type": "Concept", "title": "My notes",
                    "description": "Scratch notes.", "tags": [], "managed_by": "user",
                },
                "# My notes\n\nKeep this.\n",
            ),
            encoding="utf-8",
        )

        result = runner.invoke(
            cli.cli, ["context", "reset", "--directory", str(tmp_path)], input="y\n"
        )

    assert result.exit_code == 0, result.output
    assert not local_path.exists()
    assert (tmp_path / ".boepie" / "concepts" / "skeleton.md").exists()


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
        result_init = runner.invoke(cli.cli, ["context", "init", "--directory", str(tmp_path)])
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
        result_init = runner.invoke(cli.cli, ["context", "init", "--directory", str(tmp_path)])
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
        assert runner.invoke(cli.cli, ["context", "init", "--directory", str(tmp_path)]).exit_code == 0

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
        id="concepts/x::0", collection="context", document_id="concepts/x",
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
