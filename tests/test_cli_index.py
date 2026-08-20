"""Tests for the CLI index commands (build, fetch, status, list).

Tests the index group operations via the CLI interface, using a temporary
index root and monkeypatched configuration.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from boepie import cli


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


# ---------------------------------------------------------------------------
# index status
# ---------------------------------------------------------------------------


def test_index_status_when_empty(runner: CliRunner, tmp_index_dir: Path) -> None:
    """Status should report no indices when none are present."""
    with patch("boepie.cli.INDEX_DIR", tmp_index_dir):
        result = runner.invoke(cli.cli, ["index", "status"])

    assert result.exit_code == 0
    assert "No indices built or fetched yet" in result.output


def test_index_status_with_collections(runner: CliRunner, tmp_index_dir: Path) -> None:
    """Status should list collections and their active indices."""
    with patch("boepie.cli.INDEX_DIR", tmp_index_dir):
        # Create a fake index structure
        lit_index_dir = tmp_index_dir / "literature" / "test-index-1"
        lit_index_dir.mkdir(parents=True)

        # Create a manifest for the index
        manifest_data = {
            "embedding_kind": "ollama",
            "embedding_model": "nomic-embed-text",
        }
        (lit_index_dir / "manifest.json").write_text(
            json.dumps(manifest_data), encoding="utf-8"
        )

        # Set this as the active index
        latest_path = tmp_index_dir / "literature" / "latest.json"
        latest_path.write_text(
            json.dumps({"index_id": "test-index-1"}), encoding="utf-8"
        )

        result = runner.invoke(cli.cli, ["index", "status"])

    assert result.exit_code == 0
    assert "literature" in result.output
    assert "test-index-1" in result.output
    assert "nomic-embed-text" in result.output


def test_index_status_no_index_dir(runner: CliRunner, tmp_path: Path) -> None:
    """Status should handle missing index directory gracefully."""
    nonexistent_dir = tmp_path / "nonexistent"

    with patch("boepie.cli.INDEX_DIR", nonexistent_dir):
        result = runner.invoke(cli.cli, ["index", "status"])

    assert result.exit_code == 0
    assert "No index directory yet" in result.output


# ---------------------------------------------------------------------------
# index list
# ---------------------------------------------------------------------------


def test_index_list_when_empty(runner: CliRunner, tmp_index_dir: Path) -> None:
    """List should report no indices when none are present."""
    with patch("boepie.cli.INDEX_DIR", tmp_index_dir):
        result = runner.invoke(cli.cli, ["index", "list"])

    assert result.exit_code == 0
    assert "No indices found" in result.output


def test_index_list_with_indices(runner: CliRunner, tmp_index_dir: Path) -> None:
    """List should enumerate all indices by collection."""
    with patch("boepie.cli.INDEX_DIR", tmp_index_dir):
        # Create multiple indices
        (tmp_index_dir / "literature" / "index-1").mkdir(parents=True)
        (tmp_index_dir / "literature" / "index-2").mkdir(parents=True)
        (tmp_index_dir / "docs" / "docs-index-1").mkdir(parents=True)

        result = runner.invoke(cli.cli, ["index", "list"])

    assert result.exit_code == 0
    # Output should contain both collections
    assert "literature:" in result.output or "literature" in result.output
    assert "docs:" in result.output or "docs" in result.output
    # And their indices
    assert "index-1" in result.output
    assert "index-2" in result.output
    assert "docs-index-1" in result.output


def test_index_list_no_index_dir(runner: CliRunner, tmp_path: Path) -> None:
    """List should handle missing index directory gracefully."""
    nonexistent_dir = tmp_path / "nonexistent"

    with patch("boepie.cli.INDEX_DIR", nonexistent_dir):
        result = runner.invoke(cli.cli, ["index", "list"])

    assert result.exit_code == 0
    assert "No index directory yet" in result.output


# ---------------------------------------------------------------------------
# CLI help and structure
# ---------------------------------------------------------------------------


def test_index_group_help(runner: CliRunner) -> None:
    """The index group should exist and have help."""
    result = runner.invoke(cli.cli, ["index", "--help"])

    assert result.exit_code == 0
    assert "build" in result.output
    assert "fetch" in result.output
    assert "status" in result.output
    assert "list" in result.output


def test_context_group_has_fetch(runner: CliRunner) -> None:
    """The context group should have a fetch command."""
    result = runner.invoke(cli.cli, ["context", "--help"])

    assert result.exit_code == 0
    assert "fetch" in result.output


def test_context_group_has_apply_not_update(runner: CliRunner) -> None:
    """The context group should have apply, not update."""
    result = runner.invoke(cli.cli, ["context", "--help"])

    assert result.exit_code == 0
    assert "apply" in result.output
    assert "update" not in result.output


def test_top_level_no_fetch_index(runner: CliRunner) -> None:
    """Top-level commands should not have fetch-index."""
    result = runner.invoke(cli.cli, ["--help"])

    assert "fetch-index" not in result.output


def test_top_level_no_index_command(runner: CliRunner) -> None:
    """Top-level commands should not have the old index command."""
    # The old 'index' command is now a group, so this checks it's not there as a standalone.
    result = runner.invoke(cli.cli, ["--help"])

    # Should have 'index' and 'context' groups (plus the shrunk 'knowledge' group).
    assert "index" in result.output
    assert "context" in result.output
