"""Indexing, now that it is a verb on the noun that owns the index.

There is no `boepie index` group any more: the corpus collections index to
the machine-global `INDEX_DIR/<collection>/`, the context bundle indexes to
`<bundle>/.index/` inside the project, and the two share no storage, no scope
and no retrieval stack. `index status` proved the grouping wrong by
enumerating INDEX_DIR alone - the noun that claimed to own every index could
not see one of them.

So the state of an index is reported by `corpus status` / `context status`
beside the thing it was built from, and building it is `corpus index` /
`context index`.
"""

from __future__ import annotations

from pathlib import Path

import click

import pytest
from click.testing import CliRunner

from boepie import cli
from tests.conftest import write_corpus_document


@pytest.fixture
def runner() -> CliRunner:
    """CLI test runner."""
    return CliRunner()


# ---------------------------------------------------------------------------
# the noun is gone
# ---------------------------------------------------------------------------


def test_index_is_a_composite_not_a_group(runner: CliRunner) -> None:
    """`boepie index` came back as a *command* - `corpus index` then `context
    index`, the way `boepie sync` composes the two converging verbs. What it
    is not is a group: there is no `index status`, because an index's state
    belongs to the thing it was built from."""
    assert not isinstance(cli.cli.commands["index"], click.Group)

    result = runner.invoke(cli.cli, ["index", "status"])

    assert result.exit_code != 0


def test_indexing_is_a_verb_on_both_owners(runner: CliRunner) -> None:
    assert "index" in cli.cli.commands["corpus"].commands
    assert "index" in cli.cli.commands["context"].commands


# ---------------------------------------------------------------------------
# index state is reported beside the corpus it was built from
# ---------------------------------------------------------------------------


def test_corpus_status_reports_an_index_that_was_never_built(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`index status` used to answer this, and could only answer it for the
    machine-global collections."""
    for name in ("literature", "docs", "notes"):
        directory = tmp_path / name
        directory.mkdir()
        monkeypatch.setattr(cli, f"{name.upper()}_DIR", directory)
    monkeypatch.setattr(cli, "INDEX_DIR", tmp_path / "indices")

    write_corpus_document(
        tmp_path / "notes", document_id="note000001", title="A Note",
        body="# A Note\n\nBody.\n",
    )

    result = runner.invoke(cli.cli, ["corpus", "status", "--collection", "notes"])

    assert result.exit_code == 0, result.output
    assert "not built yet" in result.output
    assert "boepie corpus index --collection notes" in result.output


def test_corpus_index_check_only_builds_nothing_and_exits_non_zero(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The read half of the same command, as `register --check-only` is."""
    index_dir = tmp_path / "indices"
    monkeypatch.setattr(cli, "INDEX_DIR", index_dir)
    for name in ("literature", "docs", "notes"):
        directory = tmp_path / name
        directory.mkdir()
        monkeypatch.setattr(cli, f"{name.upper()}_DIR", directory)

    result = runner.invoke(
        cli.cli, ["corpus", "index", "--check-only", "--collection", "notes"]
    )

    assert result.exit_code != 0
    assert "not built yet" in result.output
    assert not index_dir.exists()


def test_corpus_index_refuses_a_collection_and_a_shorthand_at_once(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same rule `corpus add` follows: two ways of naming a collection
    can disagree, and silence would build the wrong index."""
    for name in ("literature", "docs", "notes"):
        directory = tmp_path / name
        directory.mkdir()
        monkeypatch.setattr(cli, f"{name.upper()}_DIR", directory)
    monkeypatch.setattr(cli, "INDEX_DIR", tmp_path / "indices")

    result = runner.invoke(
        cli.cli, ["corpus", "index", "--collection", "docs", "-l"]
    )

    assert result.exit_code != 0
    assert "both name a collection" in result.output


def test_context_group_has_no_fetch(runner: CliRunner) -> None:
    """There is no `context fetch` either, for the same reason as `index
    fetch`: the bundle's content ships in the venv boepie is installed in,
    so `init`/`apply` copy it from there and nothing is downloaded."""
    result = runner.invoke(cli.cli, ["context", "--help"])

    assert result.exit_code == 0
    assert "fetch" not in result.output


def test_context_group_converges_with_sync(runner: CliRunner) -> None:
    """`apply` was context's own word for "make this match its source", which
    `corpus` spelled `fetch` and `index` spelled `build`. One word now.
    `update` was rejected long before that, for apt-style ambiguity."""
    result = runner.invoke(cli.cli, ["context", "--help"])

    assert result.exit_code == 0
    assert "sync" in result.output
    assert "apply" not in result.output
    assert "update" not in result.output


def test_top_level_no_fetch_index(runner: CliRunner) -> None:
    """Top-level commands should not have fetch-index."""
    result = runner.invoke(cli.cli, ["--help"])

    assert "fetch-index" not in result.output


