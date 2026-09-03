"""`boepie setup`: one command from a fresh install to a working workspace.

Composite, like `sync`, and adds only one step of its own - registering the
MCP server. What is tested here is the orchestration: that the machine-global
corpora are fetched only when this machine has none, that the bundle is
created or converged, and that nothing outside the workspace is touched by a
default run.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

from boepie import cli
from boepie.context import ContentFetchResult
from tests.conftest import write_corpus_document


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture(autouse=True)
def _isolate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No network, no developer corpus, no real index root."""
    monkeypatch.setattr(cli, "LITERATURE_DIR", tmp_path / "literature-corpus")
    monkeypatch.setattr(cli, "DOCS_DIR", tmp_path / "docs-corpus")
    monkeypatch.setattr(cli, "NOTES_DIR", tmp_path / "notes")
    monkeypatch.setattr(cli, "INDEX_DIR", tmp_path / "indices")
    monkeypatch.setattr(cli, "sync_literature", MagicMock(return_value=[]))
    monkeypatch.setattr(cli, "sync_docs", MagicMock(return_value=[]))
    monkeypatch.setattr(
        cli,
        "fetch_content",
        MagicMock(
            return_value=ContentFetchResult(
                content_dir=tmp_path / "content-cache", changed=False
            )
        ),
    )


def _installed_except(*absent: str):
    """PATH lookup reporting every agent installed but the named ones."""
    return lambda name: None if name in absent else f"/usr/bin/{name}"


@pytest.fixture(autouse=True)
def _agents_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every agent present, and none of them actually spawned.

    Whether `claude` or `codex` happens to be installed on the machine
    running the suite must not change what these tests exercise, and no test
    may shell out to a real agent and edit that user's configuration - so
    spawning is an error rather than merely unstubbed.
    """

    def unexpected(argv, **_):
        raise AssertionError(f"a test spawned {argv}")

    # Only the agents boepie configures by writing a file. Every CLI-driven
    # one is absent by default, so a test cannot spawn a real agent by
    # forgetting to say otherwise - the guard above turns that into an error
    # rather than a config edit on the machine running the suite.
    monkeypatch.setattr(
        "boepie.mcp_config.shutil.which",
        _installed_except("claude", "codex", "gemini"),
    )
    monkeypatch.setattr("boepie.mcp_config.subprocess.run", unexpected)


def _seed_corpora(tmp_path: Path) -> None:
    """One boepie-managed document in each machine-global collection."""
    write_corpus_document(
        tmp_path / "literature-corpus", document_id="litseed001",
        title="A Paper", body="# A Paper\n\nCalibration.\n",
        bib={"citekey": "smirnov2011"},
    )
    write_corpus_document(
        tmp_path / "docs-corpus", document_id="docsseed001", title="Guide",
        body="# Guide\n\nUsage.\n", group="quartical",
        docs={"project": "quartical", "page": "guide"},
    )


@pytest.fixture
def built_indices(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record which collections setup decided to index, without embedding.

    A real build would download an embedding model and take seconds per run;
    what these tests are about is the decision, which is `_index_plan`.
    """
    built: list[str] = []
    monkeypatch.setattr(
        cli, "_build_synced_index", lambda collection, **_: built.append(collection)
    )
    return built


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    directory = tmp_path / "workspace"
    directory.mkdir()
    return directory


def _run(runner: CliRunner, workspace: Path, *extra: str):
    return runner.invoke(
        cli.cli, ["setup", "--directory", str(workspace), *extra]
    )


def test_setup_writes_both_project_configs_and_creates_the_bundle(
    runner: CliRunner, workspace: Path
) -> None:
    result = _run(runner, workspace)

    assert result.exit_code == 0, result.output
    assert (workspace / ".boepie" / "manifest.json").is_file()
    assert (workspace / ".mcp.json").is_file()
    assert (workspace / ".vscode" / "mcp.json").is_file()


def test_the_registered_command_is_the_running_installations_own(
    runner: CliRunner, workspace: Path
) -> None:
    """The whole point of the step: the config has to name the venv boepie
    and stimela share, not whichever boepie a PATH lookup would find."""
    _run(runner, workspace)

    config = json.loads((workspace / ".mcp.json").read_text(encoding="utf-8"))
    entry = config["mcpServers"]["boepie"]
    assert entry["command"] == cli.server_command()[0]
    assert entry["args"] == cli.server_command()[1:]
    assert Path(entry["command"]).is_absolute()


def test_setup_is_repeatable(
    runner: CliRunner, workspace: Path
) -> None:
    """Every step converges rather than duplicating: the second run applies
    the bundle instead of initializing it, and leaves the configs alone."""
    first = _run(runner, workspace)
    second = _run(runner, workspace)

    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    assert "Initialized" in first.output
    assert "Applied" in second.output
    assert "already correct" in second.output


def test_an_empty_machine_fetches_the_global_corpora(
    runner: CliRunner, workspace: Path
) -> None:
    _run(runner, workspace)

    assert cli.sync_literature.called
    assert cli.sync_docs.called


def test_a_populated_machine_is_still_reconciled(
    runner: CliRunner, workspace: Path, tmp_path: Path
) -> None:
    """An existing corpus is merged rather than left alone: `reconcile` only
    touches `managed_by: boepie` documents and skips what is already
    converted, so a second run costs what is new and nothing else."""
    _seed_corpora(tmp_path)

    result = _run(runner, workspace)

    assert result.exit_code == 0, result.output
    assert cli.sync_literature.called
    assert cli.sync_docs.called


def test_no_corpus_leaves_an_empty_machine_alone(
    runner: CliRunner, workspace: Path
) -> None:
    result = _run(runner, workspace, "--no-corpus")

    assert result.exit_code == 0, result.output
    assert not cli.sync_literature.called
    assert (workspace / ".mcp.json").is_file()


def test_setup_does_not_advise_the_step_it_is_about_to_take(
    runner: CliRunner, workspace: Path, tmp_path: Path, built_indices: list[str]
) -> None:
    """`corpus fetch` closes by telling you to run `index build` - which is
    the next phase here, once per collection."""
    _seed_corpora(tmp_path)

    result = _run(runner, workspace)

    assert "index build" not in result.output


def test_a_later_command_still_gets_its_next_step(
    runner: CliRunner, workspace: Path, tmp_path: Path
) -> None:
    """The suppression is scoped to setup's corpus phase, not switched off
    for the process."""
    _run(runner, workspace)
    source = tmp_path / "note.md"
    source.write_text("# A Note\n\nBody.\n", encoding="utf-8")

    result = runner.invoke(cli.cli, ["corpus", "add", "-n", str(source)])

    assert result.exit_code == 0, result.output
    assert "index build" in result.output


# ---------------------------------------------------------------------------
# index: build what is missing, rebuild what has moved, keep what is in step
# ---------------------------------------------------------------------------


def _record_index(tmp_path: Path, collection: str, documents: dict[str, str]) -> None:
    """Put a manifest on disk claiming `documents` were what got indexed."""
    from boepie.rag.loaders import CorpusRevision

    corpus = tmp_path / f"{collection}-corpus"
    index_dir = tmp_path / "indices" / collection / "an-index"
    index_dir.mkdir(parents=True)
    (index_dir / "manifest.json").write_text(
        json.dumps(
            {
                "built_from": CorpusRevision(
                    path=str(corpus.resolve()), documents=documents
                ).__dict__
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "indices" / collection / "latest.json").write_text(
        json.dumps({"index_id": "an-index"}), encoding="utf-8"
    )


def _current_documents(tmp_path: Path, collection: str) -> dict[str, str]:
    loader = cli._loader_for(collection)
    revision = loader.corpus_revision()
    assert revision is not None
    return dict(revision.documents)


def test_an_empty_collection_is_not_indexed(
    runner: CliRunner, workspace: Path, built_indices: list[str]
) -> None:
    """A corpus with nothing in it is a normal state on a machine that has
    just been set up offline, not a failure to report."""
    result = _run(runner, workspace)

    assert result.exit_code == 0, result.output
    assert built_indices == []
    assert "no documents yet" in result.output


def test_a_collection_with_no_index_is_built(
    runner: CliRunner, workspace: Path, tmp_path: Path, built_indices: list[str]
) -> None:
    _seed_corpora(tmp_path)

    result = _run(runner, workspace)

    assert built_indices == ["literature", "docs"]
    assert "no index yet" in result.output


def test_an_index_in_step_with_its_corpus_is_kept(
    runner: CliRunner, workspace: Path, tmp_path: Path, built_indices: list[str]
) -> None:
    """Rebuilding an index that already matches would make `setup` cost
    minutes every time it is run to check anything else."""
    _seed_corpora(tmp_path)
    for collection in ("literature", "docs"):
        _record_index(tmp_path, collection, _current_documents(tmp_path, collection))

    result = _run(runner, workspace)

    assert built_indices == []
    assert "in step with its corpus" in result.output


def test_a_document_added_since_the_build_forces_a_rebuild(
    runner: CliRunner, workspace: Path, tmp_path: Path, built_indices: list[str]
) -> None:
    """Wider than what serving refuses: an index missing a document is
    incomplete rather than wrong, and closing that gap is setup's job."""
    _seed_corpora(tmp_path)
    _record_index(tmp_path, "literature", {})
    _record_index(tmp_path, "docs", _current_documents(tmp_path, "docs"))

    result = _run(runner, workspace)

    assert built_indices == ["literature"]
    assert "1 new document(s)" in result.output


def test_a_changed_document_forces_a_rebuild(
    runner: CliRunner, workspace: Path, tmp_path: Path, built_indices: list[str]
) -> None:
    _seed_corpora(tmp_path)
    documents = _current_documents(tmp_path, "literature")
    _record_index(
        tmp_path, "literature", {key: "0" * 16 for key in documents}
    )

    result = _run(runner, workspace)

    assert "literature" in built_indices
    assert "1 changed" in result.output


def test_a_document_gone_since_the_build_forces_a_rebuild(
    runner: CliRunner, workspace: Path, tmp_path: Path, built_indices: list[str]
) -> None:
    _seed_corpora(tmp_path)
    documents = _current_documents(tmp_path, "literature")
    _record_index(tmp_path, "literature", {**documents, "vanished01": "0" * 16})

    result = _run(runner, workspace)

    assert "literature" in built_indices
    assert "1 gone" in result.output


def test_an_index_predating_the_freshness_check_is_rebuilt(
    runner: CliRunner, workspace: Path, tmp_path: Path, built_indices: list[str]
) -> None:
    """Three ways of not knowing are all unverifiable, never fresh - and
    setup is where an unverifiable one gets replaced by a known one."""
    _seed_corpora(tmp_path)
    index_dir = tmp_path / "indices" / "literature" / "an-index"
    index_dir.mkdir(parents=True)
    (index_dir / "manifest.json").write_text(
        json.dumps({"built_from": None}), encoding="utf-8"
    )
    (tmp_path / "indices" / "literature" / "latest.json").write_text(
        json.dumps({"index_id": "an-index"}), encoding="utf-8"
    )

    result = _run(runner, workspace)

    assert "literature" in built_indices
    assert "before the freshness check" in result.output


def test_an_unreadable_manifest_is_a_reason_to_build_not_to_abort(
    runner: CliRunner, workspace: Path, tmp_path: Path, built_indices: list[str]
) -> None:
    _seed_corpora(tmp_path)
    index_dir = tmp_path / "indices" / "literature" / "an-index"
    index_dir.mkdir(parents=True)
    (index_dir / "manifest.json").write_text("{ not json", encoding="utf-8")
    (tmp_path / "indices" / "literature" / "latest.json").write_text(
        json.dumps({"index_id": "an-index"}), encoding="utf-8"
    )

    result = _run(runner, workspace)

    assert result.exit_code == 0, result.output
    assert "literature" in built_indices


def test_a_default_run_registers_no_user_level_config(
    runner: CliRunner, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """codex and gemini edit configuration outside this workspace through
    another program, which is more than a default should do unasked."""
    spawned: list[list[str]] = []
    monkeypatch.setattr(
        cli, "apply_target",
        lambda name, directory, command, force: spawned.append(name)
        or cli.TargetResult(name, "skipped"),
    )

    _run(runner, workspace)

    assert spawned == list(cli.DEFAULT_TARGETS)
    assert "codex" not in spawned and "gemini" not in spawned


def test_two_agents_reading_one_file_are_reported_once_each(
    runner: CliRunner, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Claude Code and Copilot CLI both read `.mcp.json`. Writing it twice
    would be a lie about what happened; saying nothing would hide that
    Copilot is covered."""
    import subprocess

    def fake_run(argv, **_):
        # `mcp get` says absent, `mcp add` succeeds - see test_mcp_config.
        return subprocess.CompletedProcess(argv, 1 if argv[2] == "get" else 0, "", "")

    monkeypatch.setattr("boepie.mcp_config.shutil.which", _installed_except())
    monkeypatch.setattr("boepie.mcp_config.subprocess.run", fake_run)

    result = _run(runner, workspace)

    assert "same file as claude" in result.output


def test_an_agent_that_is_not_installed_is_reported_not_configured(
    runner: CliRunner, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing is written for a tool the user does not have - the report is a
    claim about what will now work, not a list of files touched."""
    monkeypatch.setattr("boepie.mcp_config.shutil.which", lambda name: None)

    result = _run(runner, workspace)

    assert result.exit_code == 0, result.output
    assert "not installed" in result.output
    assert not (workspace / ".mcp.json").exists()
    assert not (workspace / ".vscode").exists()
    # The bundle is not an agent config, so it is still set up.
    assert (workspace / ".boepie" / "manifest.json").is_file()


def test_naming_an_absent_agent_explicitly_still_reports_it_absent(
    runner: CliRunner, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Being told it is not there is the useful answer; writing config for it
    anyway is what asking would otherwise have to mean."""
    monkeypatch.setattr("boepie.mcp_config.shutil.which", _installed_except("gemini"))

    result = _run(runner, workspace, "--agents", "gemini")

    assert result.exit_code == 0, result.output
    assert "gemini" in result.output and "not installed" in result.output


def test_uncovered_agents_get_a_definition_to_paste(
    runner: CliRunner, workspace: Path
) -> None:
    """Covering every agent is not a race boepie can win, so the ones it did
    not register are named along with the block that registers them by hand."""
    result = _run(runner, workspace)

    assert "not registered:" in result.output
    assert "codex" in result.output and "gemini" in result.output
    # Each is named with what it is, so the reader can tell whether they
    # wanted it, and the block below registers anything boepie has never
    # heard of.
    assert "codex mcp add" in result.output
    assert '"boepie"' in result.output


def test_naming_every_agent_leaves_nothing_to_paste(
    runner: CliRunner, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli.shutil, "which", lambda name: None)

    result = _run(runner, workspace, "--agents", "all")

    assert result.exit_code == 0, result.output
    assert "not registered:" not in result.output
