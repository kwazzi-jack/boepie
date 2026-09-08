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
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from boepie import cli
from tests.conftest import write_corpus_document


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture(autouse=True)
def _isolate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No network, no developer corpus, no real index root."""
    monkeypatch.setattr(cli, "LITERATURE_DIR", tmp_path / "literature")
    monkeypatch.setattr(cli, "DOCS_DIR", tmp_path / "docs")
    monkeypatch.setattr(cli, "NOTES_DIR", tmp_path / "notes")
    monkeypatch.setattr(cli, "INDEX_DIR", tmp_path / "indices")
    monkeypatch.setattr(cli, "sync_literature", MagicMock(return_value=[]))
    monkeypatch.setattr(cli, "sync_docs", MagicMock(return_value=[]))


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
        tmp_path / "literature", document_id="litseed001",
        title="A Paper", body="# A Paper\n\nCalibration.\n",
        bib={"citekey": "smirnov2011"},
    )
    write_corpus_document(
        tmp_path / "docs", document_id="docsseed001", title="Guide",
        body="# Guide\n\nUsage.\n", group="quartical",
        docs={"project": "quartical", "page": "guide"},
    )


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
    assert "already current" in second.output


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


def test_init_scaffolds_without_fetching_or_registering_anything(
    runner: CliRunner, workspace: Path
) -> None:
    """What `setup --no-corpus` used to be for. Splitting the scaffold from
    the fetch made the flag redundant: `init` *is* the run that touches no
    network, so there is nothing left to opt out of.

    It touches no agent either - registration is the genuinely optional part
    of setting a workspace up, so it is `boepie register`'s own command.
    """
    result = runner.invoke(cli.cli, ["init", "--directory", str(workspace)])

    assert result.exit_code == 0, result.output
    assert not cli.sync_literature.called
    assert (workspace / ".boepie" / "manifest.json").is_file()
    assert not (workspace / ".mcp.json").exists()


def test_setup_does_not_advise_the_step_it_is_about_to_take(
    runner: CliRunner, workspace: Path, tmp_path: Path
) -> None:
    """`corpus sync` closes by naming `corpus index`, and `sync` closes by
    naming `boepie index` - both of which are phases `setup` runs itself."""
    _seed_corpora(tmp_path)

    result = _run(runner, workspace)

    assert "hint: run `boepie corpus index" not in result.output
    assert "hint: run `boepie index`" not in result.output


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
    assert "corpus index" in result.output


# ---------------------------------------------------------------------------
# index: build what is missing, rebuild what has moved, keep what is in step
# ---------------------------------------------------------------------------


def _record_index(tmp_path: Path, collection: str, documents: dict[str, str]) -> None:
    """Put a manifest on disk claiming `documents` were what got indexed."""
    from boepie.rag.loaders import CorpusRevision

    corpus = tmp_path / collection
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
    runner: CliRunner, workspace: Path
) -> None:
    """A corpus with nothing in it is a normal state on a machine that has
    just been set up offline, not a failure to report."""
    result = _run(runner, workspace)

    assert result.exit_code == 0, result.output
    assert "Skipped literature index - no documents" in result.output


def _plan(tmp_path: Path, collection: str) -> tuple[str, str]:
    """What `sync` would report about this collection's index.

    Asserted at the decision rather than at a build seam, because there is no
    longer a build to intercept: `sync` converges content and *reports* what
    that did to each index, naming `boepie index`. The old fixtures stubbed
    `_build_synced_index`, a function that existed only so sync could rebuild
    silently - and both it and `_setup_index` went with that behaviour.
    """
    with patch("boepie.cli.INDEX_DIR", tmp_path / "indices"):
        return cli._index_plan(collection)


def test_a_collection_with_no_index_reads_as_needing_one(
    tmp_path: Path,
) -> None:
    _seed_corpora(tmp_path)

    action, reason = _plan(tmp_path, "literature")

    assert action == "build"
    assert "no index yet" in reason


def test_an_empty_collection_is_not_an_index_that_is_missing(
    tmp_path: Path,
) -> None:
    """Nothing to index is a different answer from an index that should
    exist, which is why `sync` says nothing about an empty collection."""
    (tmp_path / "literature").mkdir(exist_ok=True)

    assert _plan(tmp_path, "literature")[0] == "empty"


def test_an_index_in_step_with_its_corpus_is_kept(tmp_path: Path) -> None:
    _seed_corpora(tmp_path)
    _record_index(tmp_path, "literature", _current_documents(tmp_path, "literature"))

    action, reason = _plan(tmp_path, "literature")

    assert action == "keep"
    assert "current" in reason


def test_a_document_added_since_the_build_needs_a_rebuild(tmp_path: Path) -> None:
    """`index_freshness` calls this `in step` on purpose - incomplete is not
    wrong, and refusing to serve it would break the ordinary `corpus add` ->
    `corpus index` gap. The question `sync` asks is wider: is this index
    *complete*."""
    _seed_corpora(tmp_path)
    _record_index(tmp_path, "literature", {})
    write_corpus_document(
        tmp_path / "literature", document_id="litseed002", title="Another",
        body="# Another\n\nText.\n", bib={"citekey": "another2020"},
    )

    action, reason = _plan(tmp_path, "literature")

    assert action == "rebuild"
    assert "added" in reason


def test_a_changed_document_needs_a_rebuild(tmp_path: Path) -> None:
    _seed_corpora(tmp_path)
    documents = _current_documents(tmp_path, "literature")
    _record_index(
        tmp_path, "literature", dict.fromkeys(documents, "0" * 16)
    )

    assert "1 changed" in _plan(tmp_path, "literature")[1]


def test_a_document_gone_since_the_build_needs_a_rebuild(tmp_path: Path) -> None:
    _seed_corpora(tmp_path)
    documents = _current_documents(tmp_path, "literature")
    _record_index(tmp_path, "literature", {**documents, "vanished01": "0" * 16})

    assert "1 gone" in _plan(tmp_path, "literature")[1]


def test_an_index_predating_the_freshness_check_needs_a_rebuild(
    tmp_path: Path,
) -> None:
    """Three ways of not knowing are all unverifiable, never fresh."""
    _seed_corpora(tmp_path)
    index_dir = tmp_path / "indices" / "literature" / "an-index"
    index_dir.mkdir(parents=True)
    (index_dir / "manifest.json").write_text(
        json.dumps({"built_from": None}), encoding="utf-8"
    )
    (tmp_path / "indices" / "literature" / "latest.json").write_text(
        json.dumps({"index_id": "an-index"}), encoding="utf-8"
    )

    assert "before the freshness check" in _plan(tmp_path, "literature")[1]


def test_an_unreadable_manifest_is_a_reason_to_build_not_to_abort(
    tmp_path: Path,
) -> None:
    _seed_corpora(tmp_path)
    index_dir = tmp_path / "indices" / "literature" / "an-index"
    index_dir.mkdir(parents=True)
    (index_dir / "manifest.json").write_text("{ not json", encoding="utf-8")
    (tmp_path / "indices" / "literature" / "latest.json").write_text(
        json.dumps({"index_id": "an-index"}), encoding="utf-8"
    )

    assert _plan(tmp_path, "literature")[0] == "build"


def test_sync_reports_index_drift_and_names_the_command_that_fixes_it(
    runner: CliRunner, workspace: Path, tmp_path: Path
) -> None:
    """The behaviour that replaced the silent rebuild. Converging content and
    rebuilding the index over it are two different costs - a fetch is
    seconds, an embed is minutes - so `sync` says what it did to each index
    rather than spending that time unasked."""
    _seed_corpora(tmp_path)
    runner.invoke(cli.cli, ["init", "--directory", str(workspace)])

    result = runner.invoke(cli.cli, ["sync", "--directory", str(workspace)])

    assert result.exit_code == 0, result.output
    assert "literature index - not built yet" in result.output
    assert "boepie index" in result.output


def test_sync_says_nothing_to_do_when_every_index_is_current(
    runner: CliRunner, workspace: Path, tmp_path: Path
) -> None:
    _seed_corpora(tmp_path)
    runner.invoke(cli.cli, ["init", "--directory", str(workspace)])
    for collection in ("literature", "docs"):
        _record_index(tmp_path, collection, _current_documents(tmp_path, collection))

    result = runner.invoke(cli.cli, ["sync", "--directory", str(workspace)])

    assert result.exit_code == 0, result.output
    assert "boepie index" not in result.output


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

    # One line names every agent that can now launch boepie, and the detail
    # lines name each file written - so one `.mcp.json` under two agent names
    # shows the sharing without needing a sentence to explain it away.
    assert "Registered boepie with claude, copilot, vscode" in result.output
    assert result.output.count(".mcp.json") == 1


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


def test_agents_that_were_not_asked_for_are_not_mentioned(
    runner: CliRunner, workspace: Path
) -> None:
    """A run used to close by naming every agent boepie knows and did not
    touch, each with a sentence about what it is, followed by a JSON block to
    paste. That is help text in the middle of a report: it made the tail of a
    successful run longer than the run, and none of it was about what
    happened. `--agents` and `--help` carry it instead."""
    result = _run(runner, workspace)

    assert result.exit_code == 0, result.output
    assert "not registered" not in result.output
    assert "codex" not in result.output
    assert "gemini" not in result.output
    assert '"boepie"' not in result.output


def test_a_definition_to_paste_appears_only_when_registration_fails(
    runner: CliRunner, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The paste-in block is still the right answer to "boepie could not do
    this for me" - it is just no longer the answer to a question nobody
    asked. It appears when, and only when, a registration actually failed."""
    monkeypatch.setattr(
        cli,
        "apply_target",
        lambda name, directory, command, force: cli.TargetResult(
            name, "failed", None, "permission denied"
        ),
    )

    result = _run(runner, workspace)

    assert "could not register" in result.output
    assert "permission denied" in result.output
    assert '"boepie"' in result.output


# ---------------------------------------------------------------------------
# register: its own command, and able to check itself
# ---------------------------------------------------------------------------


def test_register_writes_the_workspace_configs(
    runner: CliRunner, workspace: Path
) -> None:
    result = runner.invoke(cli.cli, ["register", "--directory", str(workspace)])

    assert result.exit_code == 0, result.output
    assert (workspace / ".mcp.json").is_file()
    assert (workspace / ".vscode" / "mcp.json").is_file()


def test_register_check_reports_an_unregistered_workspace_and_exits_non_zero(
    runner: CliRunner, workspace: Path
) -> None:
    """Usable as a precondition in a script, the way `black --check` is."""
    result = runner.invoke(
        cli.cli, ["register", "--check-only", "--directory", str(workspace)]
    )

    assert result.exit_code != 0
    assert "not registered" in result.output
    assert "boepie register" in result.output
    # It changed nothing on its way to saying so.
    assert not (workspace / ".mcp.json").exists()


def test_register_check_passes_once_registered(
    runner: CliRunner, workspace: Path
) -> None:
    runner.invoke(cli.cli, ["register", "--directory", str(workspace)])

    result = runner.invoke(
        cli.cli, ["register", "--check-only", "--directory", str(workspace)]
    )

    assert result.exit_code == 0, result.output
    assert "points at this boepie" in result.output


def test_register_check_catches_a_registration_pointing_elsewhere(
    runner: CliRunner, workspace: Path
) -> None:
    """The failure this command exists for. boepie's pipeline tools drive
    stimela's config chain in-process, so a registration naming a boepie in
    another venv - one installed as an isolated tool, or left behind by a
    rebuilt environment - starts cleanly and then sees zero cabs. The agent
    lists no boepie tools and explains nothing.
    """
    runner.invoke(cli.cli, ["register", "--directory", str(workspace)])
    config_path = workspace / ".mcp.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["mcpServers"]["boepie"]["command"] = "/elsewhere/bin/boepie"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    result = runner.invoke(
        cli.cli,
        ["register", "--check-only", "--agents", "copilot", "--directory", str(workspace)],
    )

    assert result.exit_code != 0
    assert "points somewhere else" in result.output
    # Names the venv it actually points at, not only that it is wrong.
    assert "/elsewhere/bin/boepie" in result.output
    # And names --force, because plain `register` leaves an existing entry.
    assert "boepie register --force" in result.output


def test_register_check_does_not_claim_a_cli_only_agent_is_correct(
    runner: CliRunner, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """codex and gemini can be asked *whether* they know the server but not
    *what* they would launch, so `opaque` is its own answer. Calling it
    `current` would be the same unfounded claim `--check` exists to catch."""
    monkeypatch.setattr(
        "boepie.mcp_config.shutil.which", lambda name: f"/usr/bin/{name}"
    )
    monkeypatch.setattr(
        "boepie.mcp_config.subprocess.run",
        lambda *a, **k: __import__("subprocess").CompletedProcess(a[0], 0, "", ""),
    )

    result = runner.invoke(
        cli.cli,
        ["register", "--check-only", "--agents", "codex", "--directory", str(workspace)],
    )

    assert result.exit_code == 0, result.output
    assert "does not say with which boepie" in result.output
    assert "points at this boepie" not in result.output
