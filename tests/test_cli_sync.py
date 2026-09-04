"""Tests for `boepie sync` (composite of `corpus fetch --collection
literature,docs` -> `index build` for both -> `context apply`/`init`).

`sync` adds no fetch logic of its own: it calls the same underlying seams
the `corpus fetch` command calls (`boepie.cli.sync_literature`,
`boepie.cli.sync_docs`), so those are what gets monkeypatched here. The final
convergence step (`apply`/`init`) runs for real against a tmp INDEX_DIR,
mirroring tests/test_cli_context.py.

Nothing here downloads anything. boepie publishes no prebuilt index, so
`sync`'s middle leg fetches corpora and then builds over them - and with the
corpora stubbed empty, each build is skipped rather than run. The bundle's
own content is not fetched at all: it ships in the venv, which is why there
is no context leg in the order below and no --tag on the command.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import httpx
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


@pytest.fixture(autouse=True)
def _isolate_corpora(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test in this module exercises sync's step ordering, not the real
    arXiv fetch, the real docs crawl, or the developer's own machine state.

    Both corpus directories point at empty tmp directories, so each index
    build finds nothing and is skipped, and the two calls that would
    otherwise reach arxiv.org and readthedocs.io are stubbed."""
    monkeypatch.setattr(cli, "LITERATURE_DIR", tmp_path / "literature-unused")
    monkeypatch.setattr(cli, "DOCS_DIR", tmp_path / "docs-unused")
    monkeypatch.setattr(cli, "sync_literature", MagicMock(return_value=[]))
    monkeypatch.setattr(cli, "sync_docs", MagicMock(return_value=[]))


def _failing_sync_literature(collection_dir, manifest, **kwargs) -> list:
    # `_sync_network_step` only treats (SystemExit, httpx.HTTPError) as a
    # warn-and-continue network failure - the same shape a real network
    # outage would raise, before `fetch_paper`'s own per-source handling of
    # httpx.HTTPError ever gets a chance to swallow it.
    raise httpx.HTTPError("simulated arxiv-fetch failure")


# ---------------------------------------------------------------------------
# step order: corpus fetch -> index build -> apply/init
# ---------------------------------------------------------------------------


def test_sync_runs_steps_in_order_and_still_initializes_on_network_failure(
    runner: CliRunner, tmp_path: Path, tmp_index_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the one network leg failing, sync still converges the bundle by
    calling `init` (no prior `.boepie/`), and the warning precedes it. Uses
    --verbose since the per-step "Initialized" message is otherwise
    suppressed by the default one-line summary."""
    monkeypatch.setattr(cli, "sync_literature", _failing_sync_literature)
    monkeypatch.setattr(cli, "INDEX_DIR", tmp_index_dir)

    result = runner.invoke(cli.cli, ["sync", "--verbose", "--directory", str(tmp_path)])

    assert result.exit_code == 0, result.output
    output = result.output

    corpus_fetch_pos = output.index("corpus fetch --collection literature,docs failed")
    initialized_pos = output.index("Initialized")

    assert corpus_fetch_pos < initialized_pos
    assert (tmp_path / ".boepie" / "manifest.json").exists()


# ---------------------------------------------------------------------------
# --only restriction
# ---------------------------------------------------------------------------


def test_sync_only_context_never_touches_a_corpus(
    runner: CliRunner, tmp_path: Path, tmp_index_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--only context should skip the corpus half entirely."""
    fake_sync_literature = MagicMock(
        side_effect=AssertionError("sync_literature should not be called")
    )
    monkeypatch.setattr(cli, "sync_literature", fake_sync_literature)
    monkeypatch.setattr(cli, "INDEX_DIR", tmp_index_dir)

    result = runner.invoke(cli.cli, ["sync", "--only", "context", "--directory", str(tmp_path)])

    assert result.exit_code == 0, result.output
    fake_sync_literature.assert_not_called()
    assert (tmp_path / ".boepie" / "manifest.json").exists()


def test_sync_only_indices_never_touches_the_bundle(
    runner: CliRunner, tmp_path: Path, tmp_index_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--only indices should skip `apply`/`init`."""
    monkeypatch.setattr(cli, "sync_literature", _failing_sync_literature)
    monkeypatch.setattr(cli, "INDEX_DIR", tmp_index_dir)

    result = runner.invoke(cli.cli, ["sync", "--only", "indices", "--directory", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert "corpus fetch --collection literature,docs failed" in result.output
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

    result = runner.invoke(cli.cli, ["sync", "--verbose", "--directory", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert "Applied" in result.output
    assert "Initialized" not in result.output


def test_sync_converges_the_bundle_with_no_network_at_all(
    runner: CliRunner, tmp_path: Path, tmp_index_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The bundle half of sync reaches nothing outside the venv.

    Its content used to come from a GitHub release asset cached per machine,
    so a machine that had never fetched one converged from the packaged
    seeds as a *fallback*. There is no fallback now because there is only
    one source, and the way to pin that is to make any HTTP client an error:
    `--only context` must still succeed.
    """
    monkeypatch.setattr(cli, "INDEX_DIR", tmp_index_dir)
    monkeypatch.setattr(
        httpx, "Client", MagicMock(side_effect=AssertionError("sync must not reach the network"))
    )

    result = runner.invoke(
        cli.cli, ["sync", "--only", "context", "--verbose", "--directory", str(tmp_path)]
    )

    assert result.exit_code == 0, result.output
    assert "Initialized" in result.output
    assert (tmp_path / ".boepie" / "manifest.json").exists()


# ---------------------------------------------------------------------------
# default verbosity: one-line summary, per-step detail behind --verbose
# ---------------------------------------------------------------------------


def test_sync_default_run_prints_one_line_summary_not_step_detail(
    runner: CliRunner, tmp_path: Path, tmp_index_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without --verbose, sync's own step messages ("Applied", "Initialized",
    "Indexed") are suppressed in favour of a single summary line; warnings
    from a failing step still surface."""
    monkeypatch.setattr(cli, "sync_literature", _failing_sync_literature)
    monkeypatch.setattr(cli, "INDEX_DIR", tmp_index_dir)

    result = runner.invoke(cli.cli, ["sync", "--directory", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert "Synced" in result.output
    assert "Initialized" not in result.output
    assert "Indexed" not in result.output
    assert "corpus fetch --collection literature,docs failed" in result.output


# ---------------------------------------------------------------------------
# no release tag to pass through any more
# ---------------------------------------------------------------------------


def test_sync_has_no_tag_option(
    runner: CliRunner, tmp_path: Path, tmp_index_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--tag had exactly one consumer, `context fetch`, and that command is
    gone: the bundle's content ships in the venv, and neither arXiv nor a
    docs site is a release asset. So the option is a usage error, not an
    ignored argument."""
    monkeypatch.setattr(cli, "INDEX_DIR", tmp_index_dir)

    result = runner.invoke(cli.cli, ["sync", "--tag", "v1.2.3", "--directory", str(tmp_path)])

    assert result.exit_code != 0
    assert "--tag" in result.output


# ---------------------------------------------------------------------------
# help / structure
# ---------------------------------------------------------------------------


def test_sync_help_lists_options(runner: CliRunner) -> None:
    """`sync` should exist at the top level with --only, --directory,
    --verbose."""
    result = runner.invoke(cli.cli, ["sync", "--help"])

    assert result.exit_code == 0
    assert "--only" in result.output
    assert "--directory" in result.output
    assert "--verbose" in result.output


def test_top_level_help_lists_sync(runner: CliRunner) -> None:
    """sync should show up as a top-level command."""
    result = runner.invoke(cli.cli, ["--help"])

    assert result.exit_code == 0
    assert "sync" in result.output
