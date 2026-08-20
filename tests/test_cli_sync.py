"""Tests for `boepie sync` (composite of `context fetch` -> `index fetch`
(docs) + `corpus fetch --collection literature` -> `context apply`/`init`).

`sync` adds no fetch logic of its own: it calls the same underlying seams
the `context fetch`, `index fetch` and `corpus fetch --collection
literature` commands call (`boepie.cli.fetch_content`,
`boepie.cli.download_verified_asset`, `boepie.cli.sync_literature`), so
those are what gets monkeypatched here. The final convergence step
(`apply`/`init`) runs for real against a tmp INDEX_DIR, mirroring
tests/test_cli_context.py.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import httpx
import pytest
from click.testing import CliRunner

from boepie import cli
from boepie.context import ContentFetchResult


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


@pytest.fixture(autouse=True)
def _isolate_literature(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test in this module exercises sync's step ordering, not the real
    arXiv fetch or the developer's own machine state: point LITERATURE_DIR at
    an empty tmp directory (so the post-fetch index build is skipped - see
    `sync`'s `if LITERATURE_DIR.exists()` guard) and stub the one call that
    would otherwise hit arxiv.org/ar5iv."""
    monkeypatch.setattr(cli, "LITERATURE_DIR", tmp_path / "literature-corpus-unused")
    monkeypatch.setattr(cli, "sync_literature", MagicMock(return_value=[]))


def _failing_fetch_content(tag: str = "latest") -> Path:
    raise ValueError("simulated content-fetch failure")


def _failing_download_verified_asset(tag: str, asset_name: str) -> bytes:
    raise ValueError(f"simulated download failure for {asset_name}")


def _failing_sync_literature(collection_dir, manifest, **kwargs) -> list:
    # `_sync_network_step` only treats (SystemExit, httpx.HTTPError) as a
    # warn-and-continue network failure - the same shape a real network
    # outage would raise, before `fetch_paper`'s own per-source handling of
    # httpx.HTTPError ever gets a chance to swallow it.
    raise httpx.HTTPError("simulated arxiv-fetch failure")


# ---------------------------------------------------------------------------
# step order: context fetch -> index fetch (per collection) -> apply/init
# ---------------------------------------------------------------------------


def test_sync_runs_steps_in_order_and_still_initializes_on_full_network_failure(
    runner: CliRunner, tmp_path: Path, tmp_index_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With every network leg failing, sync still converges the bundle by
    calling `init` (no prior `.boepie/`), and the warnings appear in the
    documented order: context fetch, then docs index fetch, then literature
    fetch, then the final init. Uses --verbose since the per-step
    "Initialized" message is otherwise suppressed by the default one-line
    summary."""
    monkeypatch.setattr(cli, "fetch_content", _failing_fetch_content)
    monkeypatch.setattr(cli, "download_verified_asset", _failing_download_verified_asset)
    monkeypatch.setattr(cli, "sync_literature", _failing_sync_literature)
    monkeypatch.setattr(cli, "INDEX_DIR", tmp_index_dir)

    result = runner.invoke(cli.cli, ["sync", "--verbose", "--directory", str(tmp_path)])

    assert result.exit_code == 0, result.output
    output = result.output

    context_fetch_pos = output.index("context fetch --tag latest failed")
    docs_fetch_pos = output.index("index fetch --collection docs --tag latest failed")
    literature_fetch_pos = output.index("corpus fetch --collection literature failed")
    initialized_pos = output.index("Initialized")

    assert context_fetch_pos < docs_fetch_pos < literature_fetch_pos < initialized_pos
    assert (tmp_path / ".boepie" / "manifest.json").exists()


# ---------------------------------------------------------------------------
# --only restriction
# ---------------------------------------------------------------------------


def test_sync_only_context_never_calls_index_fetch(
    runner: CliRunner, tmp_path: Path, tmp_index_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--only context should skip the index-fetch half entirely."""
    fake_fetch_content = MagicMock(
        return_value=ContentFetchResult(content_dir=tmp_path / "content-cache", changed=True)
    )
    fake_download = MagicMock(side_effect=AssertionError("download_verified_asset should not be called"))
    monkeypatch.setattr(cli, "fetch_content", fake_fetch_content)
    monkeypatch.setattr(cli, "download_verified_asset", fake_download)
    monkeypatch.setattr(cli, "INDEX_DIR", tmp_index_dir)

    result = runner.invoke(cli.cli, ["sync", "--only", "context", "--directory", str(tmp_path)])

    assert result.exit_code == 0, result.output
    fake_fetch_content.assert_called_once()
    fake_download.assert_not_called()
    assert (tmp_path / ".boepie" / "manifest.json").exists()


def test_sync_only_indices_never_touches_the_bundle(
    runner: CliRunner, tmp_path: Path, tmp_index_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--only indices should skip both `context fetch` and `apply`/`init`."""
    fake_fetch_content = MagicMock(side_effect=AssertionError("fetch_content should not be called"))
    monkeypatch.setattr(cli, "fetch_content", fake_fetch_content)
    monkeypatch.setattr(cli, "download_verified_asset", _failing_download_verified_asset)
    monkeypatch.setattr(cli, "sync_literature", _failing_sync_literature)
    monkeypatch.setattr(cli, "INDEX_DIR", tmp_index_dir)

    result = runner.invoke(cli.cli, ["sync", "--only", "indices", "--directory", str(tmp_path)])

    assert result.exit_code == 0, result.output
    fake_fetch_content.assert_not_called()
    assert "index fetch --collection docs --tag latest failed" in result.output
    assert "corpus fetch --collection literature failed" in result.output
    assert not (tmp_path / ".boepie").exists()


# ---------------------------------------------------------------------------
# apply vs init depending on whether a bundle already exists
# ---------------------------------------------------------------------------


def test_sync_calls_apply_not_init_when_bundle_already_exists(
    runner: CliRunner, tmp_path: Path, tmp_index_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pre-existing `.boepie/` must converge via `apply`, never re-`init`
    (which would raise FileExistsError). Uses --verbose to see which of the
    two ran."""
    monkeypatch.setattr(cli, "INDEX_DIR", tmp_index_dir)
    init_result = runner.invoke(cli.cli, ["context", "init", "--directory", str(tmp_path)])
    assert init_result.exit_code == 0, init_result.output

    monkeypatch.setattr(cli, "fetch_content", _failing_fetch_content)
    monkeypatch.setattr(cli, "download_verified_asset", _failing_download_verified_asset)

    result = runner.invoke(cli.cli, ["sync", "--verbose", "--directory", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert "Applied" in result.output
    assert "Initialized" not in result.output


def test_sync_failing_content_fetch_still_completes_apply_from_seeds(
    runner: CliRunner, tmp_path: Path, tmp_index_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failing `context fetch` warns but does not block `apply`, which
    still converges the bundle from whatever content is already resolvable
    (the packaged seeds, absent a populated cache). Uses --verbose to see
    the "Applied" message alongside the warning."""
    monkeypatch.setattr(cli, "INDEX_DIR", tmp_index_dir)
    init_result = runner.invoke(cli.cli, ["context", "init", "--directory", str(tmp_path)])
    assert init_result.exit_code == 0, init_result.output

    monkeypatch.setattr(cli, "fetch_content", _failing_fetch_content)
    monkeypatch.setattr(cli, "download_verified_asset", _failing_download_verified_asset)

    result = runner.invoke(cli.cli, ["sync", "--verbose", "--directory", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert "Warning" in result.output
    assert "context fetch --tag latest failed" in result.output
    assert "Applied" in result.output
    assert (tmp_path / ".boepie" / "manifest.json").exists()


# ---------------------------------------------------------------------------
# default verbosity: one-line summary, per-step detail behind --verbose
# ---------------------------------------------------------------------------


def test_sync_default_run_prints_one_line_summary_not_step_detail(
    runner: CliRunner, tmp_path: Path, tmp_index_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without --verbose, sync's own step messages ("Applied", "Initialized",
    "Fetched", "Indexed") are suppressed in favour of a single summary line;
    warnings from a failing step still surface."""
    monkeypatch.setattr(cli, "fetch_content", _failing_fetch_content)
    monkeypatch.setattr(cli, "download_verified_asset", _failing_download_verified_asset)
    monkeypatch.setattr(cli, "INDEX_DIR", tmp_index_dir)

    result = runner.invoke(cli.cli, ["sync", "--directory", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert "Synced" in result.output
    assert "Initialized" not in result.output
    assert "Fetched" not in result.output
    assert "Indexed" not in result.output
    assert "context fetch --tag latest failed" in result.output


# ---------------------------------------------------------------------------
# --tag passthrough
# ---------------------------------------------------------------------------


def test_sync_tag_passes_through_to_both_fetch_kinds(
    runner: CliRunner, tmp_path: Path, tmp_index_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--tag must reach both `fetch_content` and `download_verified_asset`.

    `corpus fetch --collection literature` has no --tag of its own (arXiv
    fetch is not a release asset), so only the context and docs-index legs
    are release-tagged; the docs leg is the sole remaining
    `download_verified_asset` call.
    """
    observed_tags: list[str] = []

    def _recording_fetch_content(tag: str = "latest") -> Path:
        observed_tags.append(f"context:{tag}")
        raise ValueError("stop before touching the real cache")

    def _recording_download(tag: str, asset_name: str) -> bytes:
        observed_tags.append(f"index:{tag}")
        raise ValueError("stop before touching the real network")

    monkeypatch.setattr(cli, "fetch_content", _recording_fetch_content)
    monkeypatch.setattr(cli, "download_verified_asset", _recording_download)
    monkeypatch.setattr(cli, "INDEX_DIR", tmp_index_dir)

    result = runner.invoke(cli.cli, ["sync", "--tag", "v1.2.3", "--directory", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert "context:v1.2.3" in observed_tags
    assert observed_tags.count("index:v1.2.3") == 1


# ---------------------------------------------------------------------------
# help / structure
# ---------------------------------------------------------------------------


def test_sync_help_lists_options(runner: CliRunner) -> None:
    """`sync` should exist at the top level with --only, --tag, --directory,
    --verbose."""
    result = runner.invoke(cli.cli, ["sync", "--help"])

    assert result.exit_code == 0
    assert "--only" in result.output
    assert "--tag" in result.output
    assert "--directory" in result.output
    assert "--verbose" in result.output


def test_top_level_help_lists_sync(runner: CliRunner) -> None:
    """sync should show up as a top-level command."""
    result = runner.invoke(cli.cli, ["--help"])

    assert result.exit_code == 0
    assert "sync" in result.output
