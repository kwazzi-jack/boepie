"""Tests for `boepie sync` (composite of `corpus sync --collection
literature,docs` -> `context sync`, then a report on what that did to
each index).

`sync` adds no fetch logic of its own: it calls the same underlying seams
the `corpus sync` command calls (`boepie.cli.sync_literature`,
`boepie.cli.sync_docs`), so those are what gets monkeypatched here. The final
convergence step (`context sync`) runs for real against a tmp INDEX_DIR,
mirroring tests/test_cli_context.py.

`sync` scaffolds nothing now, so every ordering test starts from the
`initialised` fixture. That is the split working: before it, `sync` created
the bundle on the way past and no test could tell a scaffolded workspace from
a bare one.

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
    otherwise reach arxiv.org and readthedocs.io are stubbed.

    They are *created*, because `sync` no longer scaffolds: an absent corpus
    is now a refusal naming `boepie init`, so a directory that does not exist
    is a different fixture (see the refusal tests) rather than a stand-in for
    an empty one."""
    for attribute, name in (
        ("LITERATURE_DIR", "literature-unused"),
        ("DOCS_DIR", "docs-unused"),
        ("NOTES_DIR", "notes-unused"),
    ):
        corpus_dir = tmp_path / name
        corpus_dir.mkdir(exist_ok=True)
        monkeypatch.setattr(cli, attribute, corpus_dir)
    monkeypatch.setattr(cli, "sync_literature", MagicMock(return_value=[]))
    monkeypatch.setattr(cli, "sync_docs", MagicMock(return_value=[]))


@pytest.fixture
def initialised(runner: CliRunner, tmp_path: Path, tmp_index_dir: Path,
                monkeypatch: pytest.MonkeyPatch) -> Path:
    """A workspace `boepie init` has already scaffolded.

    `sync` refuses an uninitialised one now, so every ordering test needs
    this - which is itself the point of the split: before it, `sync` created
    the bundle on the way past and no test could tell a scaffolded workspace
    from a bare one.
    """
    monkeypatch.setattr(cli, "INDEX_DIR", tmp_index_dir)
    result = runner.invoke(cli.cli, ["context", "init", "--directory", str(tmp_path)])
    assert result.exit_code == 0, result.output
    return tmp_path


def _failing_sync_literature(collection_dir, manifest, **kwargs) -> list:
    # `_sync_network_step` only treats (SystemExit, httpx.HTTPError) as a
    # warn-and-continue network failure - the same shape a real network
    # outage would raise, before `fetch_paper`'s own per-source handling of
    # httpx.HTTPError ever gets a chance to swallow it.
    raise httpx.HTTPError("simulated arxiv-fetch failure")


# ---------------------------------------------------------------------------
# step order: corpus sync -> context sync -> index drift report
# ---------------------------------------------------------------------------


def test_sync_still_converges_the_bundle_when_the_network_leg_fails(
    runner: CliRunner, initialised: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unreachable arXiv must not cost the local convergence that follows
    it: the warning comes first and `context sync` still runs."""
    monkeypatch.setattr(cli, "sync_literature", _failing_sync_literature)

    result = runner.invoke(cli.cli, ["sync", "--directory", str(initialised)])

    assert result.exit_code == 0, result.output
    output = result.output
    assert output.index("corpus sync --collection literature,docs failed") < output.index(
        "Applied"
    )


def test_sync_refuses_a_workspace_nothing_has_initialised(
    runner: CliRunner, tmp_path: Path, tmp_index_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """It used to create the bundle itself, which made a bare directory look
    like a working one - the command reported success against state it had
    just invented, and nobody had said "set this up here"."""
    monkeypatch.setattr(cli, "INDEX_DIR", tmp_index_dir)

    result = runner.invoke(cli.cli, ["sync", "--directory", str(tmp_path)])

    assert result.exit_code != 0
    assert "no context bundle" in result.output
    assert "boepie init" in result.output
    assert not (tmp_path / ".boepie").exists()


def test_sync_refuses_when_the_corpus_has_no_directory(
    runner: CliRunner, initialised: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The bundle is only half of being initialised. A corpus directory used
    to spring into existence the first time a document was written into it,
    so a mistyped BOEPIE_LITERATURE_DIR silently made a second corpus at the
    wrong path instead of an error."""
    monkeypatch.setattr(cli, "LITERATURE_DIR", tmp_path / "never-created")

    result = runner.invoke(cli.cli, ["sync", "--directory", str(initialised)])

    assert result.exit_code != 0
    assert "no corpus on this machine" in result.output
    assert "boepie init" in result.output


# ---------------------------------------------------------------------------
# --only restriction
# ---------------------------------------------------------------------------


def test_sync_has_no_only_option(runner: CliRunner, initialised: Path) -> None:
    """`--only context` was a second spelling of `boepie context sync`, and a
    second place for the two to disagree. One noun at a time is what the
    per-noun commands already are."""
    result = runner.invoke(
        cli.cli, ["sync", "--only", "context", "--directory", str(initialised)]
    )

    assert result.exit_code != 0
    assert "--only" in result.output


def test_one_noun_at_a_time_is_that_noun_s_own_sync(
    runner: CliRunner, initialised: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """What replaced `--only context`."""
    fake_sync_literature = MagicMock(
        side_effect=AssertionError("sync_literature should not be called")
    )
    monkeypatch.setattr(cli, "sync_literature", fake_sync_literature)

    result = runner.invoke(
        cli.cli, ["context", "sync", "--directory", str(initialised)]
    )

    assert result.exit_code == 0, result.output
    fake_sync_literature.assert_not_called()
    assert "Applied" in result.output


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
    runner: CliRunner, initialised: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The bundle half of sync reaches nothing outside the venv.

    Its content used to come from a GitHub release asset cached per machine,
    so a machine that had never fetched one converged from the packaged
    seeds as a *fallback*. There is no fallback now because there is only
    one source, and the way to pin that is to make any HTTP client an error:
    `--only context` must still succeed.
    """
    monkeypatch.setattr(
        httpx, "Client", MagicMock(side_effect=AssertionError("sync must not reach the network"))
    )

    result = runner.invoke(
        cli.cli, ["context", "sync", "--directory", str(initialised)]
    )

    assert result.exit_code == 0, result.output
    assert "Applied" in result.output


# ---------------------------------------------------------------------------
# default verbosity: one-line summary, per-step detail behind --verbose
# ---------------------------------------------------------------------------


def test_sync_reports_each_phase_like_setup_does(
    runner: CliRunner, initialised: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """It used to run every step inside `console.capture()` and print one
    `Synced` line, so its slowest leg - a corpus sync that can take minutes
    - showed nothing at all. `setup` printed each phase; two composites of
    the same shape had opposite defaults, and the silent one was the one that
    needed the output.

    The capture also disabled the progress bars as a side effect, because a
    captured console is not a terminal - which nobody decided.
    """
    monkeypatch.setattr(cli, "sync_literature", _failing_sync_literature)

    result = runner.invoke(cli.cli, ["sync", "--directory", str(initialised)])

    assert result.exit_code == 0, result.output
    assert "Applied" in result.output
    assert "corpus sync --collection literature,docs failed" in result.output
    # No summary line standing in for the report it used to hide.
    assert "Synced" not in result.output


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


# ---------------------------------------------------------------------------
# cancellation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("command", "arguments"),
    [
        ("sync", ["sync"]),
        ("setup", ["setup", "--agents", "claude"]),
    ],
)
def test_ctrl_c_during_the_fetch_stops_the_whole_run(
    command: str,
    arguments: list[str],
    runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
    initialised: Path,
) -> None:
    """A composite carries on past a *failed* fetch on purpose - an
    unreachable arXiv should still leave whatever was fetched before indexed.
    Cancellation used to be signalled as `SystemExit(130)`, which that same
    wrapper caught as exactly such a failure: Ctrl-C was reported as `corpus
    fetch failed: 130`, every later phase ran anyway, and both commands
    exited 0.
    """

    def interrupted(*args: object, **keywords: object) -> None:
        raise KeyboardInterrupt

    later_phases: list[str] = []
    monkeypatch.setattr(cli, "_corpus_fetch_literature", interrupted)
    monkeypatch.setattr(cli, "_corpus_fetch_docs", interrupted)
    # `sync` reports index drift rather than building, so the phase to catch
    # after a Ctrl-C is that report; `setup` still goes on to build.
    monkeypatch.setattr(
        cli, "_report_index_drift", lambda *_: later_phases.append("report")
    )
    monkeypatch.setattr(
        cli, "_build_corpus_index", lambda name, **_: later_phases.append(name)
    )
    # A real result, not None: `init` registers agents before `sync` runs, so
    # this stub stands in for a registration that succeeded.
    monkeypatch.setattr(
        cli,
        "apply_target",
        lambda name, *a, **k: cli.TargetResult(name, "current", None, ""),
    )

    result = runner.invoke(cli.cli, [*arguments, "--directory", str(initialised)])

    assert result.exit_code == 130, result.output
    assert later_phases == []
    assert "Cancelled:" in result.output
    # Not reported as a failure: nothing went wrong.
    assert "failed" not in result.output
