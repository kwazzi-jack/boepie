"""Registering the boepie MCP server with the agents that launch it.

The failure this module exists to prevent is a silent one: an agent that
starts, lists no boepie tools, and says nothing about why. Two ways to get
there - a command that launches a boepie which cannot see stimela, and a
config file written in a shape the agent does not read - so these tests pin
the command's provenance and the exact shape of each file that is written.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from boepie import mcp_config


def _venv(tmp_path: Path, *, with_script: bool) -> Path:
    """A directory shaped like a venv, optionally holding a console script."""
    prefix = tmp_path / "venv"
    scripts = prefix / ("Scripts" if os.name == "nt" else "bin")
    scripts.mkdir(parents=True)
    if with_script:
        suffix = ".exe" if os.name == "nt" else ""
        (scripts / f"boepie{suffix}").write_text("#!/usr/bin/env python\n")
    return prefix


# ---------------------------------------------------------------------------
# Which boepie the config launches
# ---------------------------------------------------------------------------


def test_the_command_is_this_venvs_own_console_script(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An absolute path, not a bare name: the server has to run in the venv
    stimela is installed in, and a PATH lookup could find any other boepie."""
    prefix = _venv(tmp_path, with_script=True)
    monkeypatch.setattr(mcp_config.sys, "prefix", str(prefix))

    command = mcp_config.server_command()

    assert Path(command[0]).is_absolute()
    assert Path(command[0]).parent.parent == prefix
    assert command[1:] == ["serve"]


def test_without_a_console_script_the_interpreter_runs_the_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An uninstalled source checkout still has an interpreter, and `-m`
    reaches the same entry point through it - which is the part that matters."""
    monkeypatch.setattr(mcp_config.sys, "prefix", str(_venv(tmp_path, with_script=False)))

    assert mcp_config.server_command() == [sys.executable, "-m", "boepie", "serve"]


def test_no_launcher_is_named_in_the_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`uv run` was what the old dev script wrote, and it required uv on the
    PATH of whatever process spawns the server. Nothing here may need one."""
    monkeypatch.setattr(mcp_config.sys, "prefix", str(_venv(tmp_path, with_script=True)))

    assert Path(mcp_config.server_command()[0]).name.startswith("boepie")


# ---------------------------------------------------------------------------
# The shape of each file
# ---------------------------------------------------------------------------


_COMMAND = ["/venv/bin/boepie", "serve"]
_ENTRY = {"type": "stdio", "command": "/venv/bin/boepie", "args": ["serve"]}


@pytest.fixture
def agents_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every agent found on PATH, and no agent CLI actually run.

    Applied rather than left to the machine: whether `claude` or `code`
    happens to be installed where the suite runs must not decide which code
    path a test takes. Spawning is made an error rather than merely
    unstubbed, because a test that quietly shells out to a real agent
    changes that user's configuration - use the `agent_cli` fixture where a
    CLI is meant to be called.
    """

    def unexpected(argv, **_):
        raise AssertionError(f"a test spawned {argv}")

    monkeypatch.setattr(mcp_config.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(mcp_config.subprocess, "run", unexpected)


@pytest.fixture
def nothing_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mcp_config.shutil, "which", lambda name: None)


def _written(tmp_path: Path, name: str) -> dict:
    result = mcp_config.apply_target(name, tmp_path, _COMMAND, force=False)
    assert result.path is not None
    return json.loads(result.path.read_text(encoding="utf-8"))


def test_the_workspace_file_holds_the_project_scoped_definition(tmp_path: Path, agents_installed: None) -> None:
    config = _written(tmp_path, "copilot")

    assert (tmp_path / ".mcp.json").is_file()
    assert config == {"mcpServers": {"boepie": _ENTRY}}


def test_vscode_gets_the_servers_shape_with_an_explicit_transport(tmp_path: Path, agents_installed: None) -> None:
    """VS Code reads `servers`, not `mcpServers`, and requires `type`."""
    config = _written(tmp_path, "vscode")

    assert (tmp_path / ".vscode" / "mcp.json").is_file()
    assert config["servers"]["boepie"]["type"] == "stdio"
    assert config["servers"]["boepie"]["args"] == ["serve"]


def test_a_missing_directory_is_created(tmp_path: Path, agents_installed: None) -> None:
    mcp_config.apply_target("vscode", tmp_path, _COMMAND, force=False)

    assert (tmp_path / ".vscode").is_dir()


# ---------------------------------------------------------------------------
# Merging into a config that is already there
# ---------------------------------------------------------------------------


def test_other_servers_and_other_keys_survive(tmp_path: Path, agents_installed: None) -> None:
    """These files are the user's, not boepie's. Only one key is ours."""
    path = tmp_path / ".mcp.json"
    path.write_text(
        json.dumps(
            {
                "mcpServers": {"other": {"command": "other", "args": []}},
                "somethingElse": {"keep": "me"},
            }
        ),
        encoding="utf-8",
    )

    config = _written(tmp_path, "copilot")

    assert config["mcpServers"]["other"] == {"command": "other", "args": []}
    assert config["somethingElse"] == {"keep": "me"}
    assert config["mcpServers"]["boepie"]["command"] == "/venv/bin/boepie"


def test_an_identical_entry_reports_current_rather_than_rewriting(tmp_path: Path, agents_installed: None) -> None:
    mcp_config.apply_target("copilot", tmp_path, _COMMAND, force=False)
    before = (tmp_path / ".mcp.json").stat().st_mtime_ns

    result = mcp_config.apply_target("copilot", tmp_path, _COMMAND, force=False)

    assert result.status == "current"
    assert (tmp_path / ".mcp.json").stat().st_mtime_ns == before


def test_a_different_boepie_entry_is_left_alone_without_force(tmp_path: Path, agents_installed: None) -> None:
    """A user who hand-edited the command had a reason; setup says so instead
    of overwriting it."""
    mcp_config.apply_target("copilot", tmp_path, ["/elsewhere/boepie", "serve"], force=False)

    result = mcp_config.apply_target("copilot", tmp_path, _COMMAND, force=False)

    assert result.status == "skipped"
    assert "--force" in result.detail
    config = json.loads((tmp_path / ".mcp.json").read_text(encoding="utf-8"))
    assert config["mcpServers"]["boepie"]["command"] == "/elsewhere/boepie"


def test_force_replaces_it(tmp_path: Path, agents_installed: None) -> None:
    mcp_config.apply_target("copilot", tmp_path, ["/elsewhere/boepie", "serve"], force=False)

    result = mcp_config.apply_target("copilot", tmp_path, _COMMAND, force=True)

    assert result.status == "written"
    config = json.loads((tmp_path / ".mcp.json").read_text(encoding="utf-8"))
    assert config["mcpServers"]["boepie"]["command"] == "/venv/bin/boepie"


def test_unparseable_json_names_the_file_rather_than_clobbering_it(tmp_path: Path, agents_installed: None) -> None:
    path = tmp_path / ".mcp.json"
    path.write_text("{ not json", encoding="utf-8")

    with pytest.raises(mcp_config.McpConfigError) as raised:
        mcp_config.apply_target("copilot", tmp_path, _COMMAND, force=False)

    assert ".mcp.json" in str(raised.value)
    assert path.read_text(encoding="utf-8") == "{ not json"


def test_a_server_container_of_the_wrong_type_is_refused(tmp_path: Path, agents_installed: None) -> None:
    (tmp_path / ".mcp.json").write_text(
        json.dumps({"mcpServers": ["not", "an", "object"]}), encoding="utf-8"
    )

    with pytest.raises(mcp_config.McpConfigError):
        mcp_config.apply_target("copilot", tmp_path, _COMMAND, force=False)


# ---------------------------------------------------------------------------
# Targets driven through their own CLI
# ---------------------------------------------------------------------------


def test_an_absent_cli_is_a_skip_naming_the_binary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(mcp_config.shutil, "which", lambda name: None)

    result = mcp_config.apply_target("codex", tmp_path, _COMMAND, force=False)

    assert result.status == "skipped"
    assert "codex" in result.detail
    assert not list(tmp_path.iterdir())


@pytest.fixture
def agent_cli(monkeypatch: pytest.MonkeyPatch):
    """An installed agent CLI whose `mcp get` says the server is not there.

    Separating the probe from the add is the point: `mcp add` refuses a name
    the agent already has, so what `get` answers decides whether `add` is
    even attempted.
    """

    def install(*, registered: bool = False, add_returncode: int = 0, stderr: str = ""):
        seen: list[list[str]] = []

        def fake_run(argv, **kwargs):
            seen.append(list(argv))
            if argv[2] == "get":
                return subprocess.CompletedProcess(
                    argv, 0 if registered else 1, "", ""
                )
            return subprocess.CompletedProcess(argv, add_returncode, "", stderr)

        monkeypatch.setattr(
            mcp_config.shutil, "which", lambda name: f"/usr/bin/{name}"
        )
        monkeypatch.setattr(mcp_config.subprocess, "run", fake_run)
        return seen

    return install


def test_the_cli_is_handed_the_command_after_a_separator(
    tmp_path: Path, agent_cli
) -> None:
    """`--` is what keeps `serve` from being read as a flag of codex's own."""
    seen = agent_cli()

    result = mcp_config.apply_target("codex", tmp_path, _COMMAND, force=False)

    assert result.status == "written"
    assert seen[-1] == ["codex", "mcp", "add", "boepie", "--", *_COMMAND]


def test_a_cli_only_target_is_probed_before_it_is_added(
    tmp_path: Path, agent_cli
) -> None:
    """`mcp add` refuses a name the agent already has, and `setup` promises
    to be repeatable - so an existing registration is recognised rather than
    discovered by failing."""
    seen = agent_cli(registered=True)

    result = mcp_config.apply_target("codex", tmp_path, _COMMAND, force=False)

    assert result.status == "current"
    assert seen == [["codex", "mcp", "get", "boepie"]]


def test_a_file_backed_target_is_settled_by_its_file_not_a_probe(
    tmp_path: Path, agent_cli
) -> None:
    """A probe would also see a user-level registration and call the
    workspace done; the file is exact about scope."""
    seen = agent_cli(registered=True)
    (tmp_path / ".mcp.json").write_text(
        json.dumps({"mcpServers": {}}), encoding="utf-8"
    )

    result = mcp_config.apply_target("claude", tmp_path, _COMMAND, force=False)

    assert result.status == "written"
    assert [argv[2] for argv in seen] == ["add"]


def test_force_replaces_through_the_file_because_add_cannot(
    tmp_path: Path, agent_cli
) -> None:
    """`mcp add` refuses an existing name, so the CLI has no way to express
    a replacement."""
    seen = agent_cli()
    (tmp_path / ".mcp.json").write_text(
        json.dumps({"mcpServers": {"boepie": {"command": "/elsewhere", "args": []}}}),
        encoding="utf-8",
    )

    result = mcp_config.apply_target("claude", tmp_path, _COMMAND, force=True)

    assert result.status == "written"
    assert seen == []
    config = json.loads((tmp_path / ".mcp.json").read_text(encoding="utf-8"))
    assert config["mcpServers"]["boepie"]["command"] == "/venv/bin/boepie"


def test_the_cli_is_preferred_over_writing_the_file(
    tmp_path: Path, agent_cli
) -> None:
    """The tool that owns the schema is the one that should write it, and
    will keep owning it when the schema changes."""
    seen = agent_cli()

    result = mcp_config.apply_target("claude", tmp_path, _COMMAND, force=False)

    assert result.status == "written"
    assert seen[-1][:3] == ["claude", "mcp", "add"]
    assert not (tmp_path / ".mcp.json").exists()


def test_a_project_scope_cli_runs_in_the_workspace_not_the_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`claude mcp add --scope project` resolves `.mcp.json` against the cwd
    of the process running it, so `setup --directory elsewhere` would
    otherwise configure whichever directory the user was standing in."""
    seen: dict[str, object] = {}

    def fake_run(argv, **kwargs):
        seen.update(kwargs)
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(mcp_config.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(mcp_config.subprocess, "run", fake_run)

    mcp_config.apply_target("claude", tmp_path, _COMMAND, force=False)

    assert seen["cwd"] == tmp_path


def test_copilot_alone_still_gets_the_workspace_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`.mcp.json` is read by Copilot CLI too, and its own `mcp add` writes
    user config - so requiring Claude Code to be installed in order to
    configure Copilot's workspace would be absurd."""
    monkeypatch.setattr(
        mcp_config.shutil,
        "which",
        lambda name: "/usr/bin/copilot" if name == "copilot" else None,
    )

    absent = mcp_config.apply_target("claude", tmp_path, _COMMAND, force=False)
    present = mcp_config.apply_target("copilot", tmp_path, _COMMAND, force=False)

    assert absent.status == "skipped"
    assert "not installed" in absent.detail
    assert present.status == "written"
    assert (tmp_path / ".mcp.json").is_file()


# ---------------------------------------------------------------------------
# Agents that are not there
# ---------------------------------------------------------------------------


def test_an_agent_that_is_not_installed_is_reported_not_configured(
    tmp_path: Path, nothing_installed: None
) -> None:
    """A config for a tool the user does not have is clutter they never
    asked for, and it turns the report into a list of files touched rather
    than a claim about what will now work."""
    result = mcp_config.apply_target("vscode", tmp_path, _COMMAND, force=False)

    assert result.status == "skipped"
    assert "not installed" in result.detail
    assert not list(tmp_path.iterdir())


def test_the_skip_names_what_was_looked_for(
    tmp_path: Path, nothing_installed: None
) -> None:
    """An agent can ship under more than one name, and which ones boepie
    tried is the difference between "install it" and "it is installed, just
    not as that"."""
    result = mcp_config.apply_target("vscode", tmp_path, _COMMAND, force=False)

    assert "code" in result.detail and "code-insiders" in result.detail


def test_any_one_of_an_agents_executables_is_enough(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        mcp_config.shutil,
        "which",
        lambda name: "/usr/bin/code-insiders" if name == "code-insiders" else None,
    )

    result = mcp_config.apply_target("vscode", tmp_path, _COMMAND, force=False)

    assert result.status == "written"


def test_every_target_declares_something_to_look_for() -> None:
    """A target with no executables would be configured unconditionally,
    which is the behaviour this check exists to remove."""
    for target in mcp_config.TARGETS:
        assert target.executables


def test_a_failing_cli_is_not_worked_around_by_writing_the_file(
    tmp_path: Path, agent_cli
) -> None:
    """A refusal from the tool that owns the format is not boepie's to
    overrule; only its absence falls through."""

    agent_cli(add_returncode=1, stderr="refused\n")

    result = mcp_config.apply_target("claude", tmp_path, _COMMAND, force=False)

    assert result.status == "failed"
    assert not (tmp_path / ".mcp.json").exists()


def test_copilot_reads_the_same_workspace_file_as_claude(
    tmp_path: Path, agents_installed: None
) -> None:
    """Copilot CLI lists `.mcp.json` among its own sources, so one file
    covers both and its own `mcp add` (user scope) is not what we want."""
    claude = mcp_config.target_named("claude")
    copilot = mcp_config.target_named("copilot")

    assert copilot.relative_path == claude.relative_path
    assert copilot.register is None
    assert copilot.scope == "workspace"


def test_an_entry_a_cli_wrote_is_recognised_as_the_same_server(
    tmp_path: Path, agents_installed: None
) -> None:
    """`claude mcp add` writes an empty `env` boepie does not. Rewriting the
    file to strip it would report a change on every run for no difference in
    what gets launched."""
    (tmp_path / ".mcp.json").write_text(
        json.dumps({"mcpServers": {"boepie": {**_ENTRY, "env": {}}}}), encoding="utf-8"
    )

    result = mcp_config.apply_target("copilot", tmp_path, _COMMAND, force=False)

    assert result.status == "current"


def test_a_failing_cli_reports_its_own_last_line(tmp_path: Path, agent_cli) -> None:
    """boepie cannot know why another tool refused; quoting it beats guessing.
    This is the whole reason these targets are delegated rather than written."""

    agent_cli(add_returncode=1, stderr="error: unknown flag\n")

    result = mcp_config.apply_target("gemini", tmp_path, _COMMAND, force=False)

    assert result.status == "failed"
    assert result.detail == "error: unknown flag"


def test_a_cli_that_never_returns_is_not_waited_on_forever(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_run(argv, **_):
        raise subprocess.TimeoutExpired(argv, 30)

    monkeypatch.setattr(mcp_config.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(mcp_config.subprocess, "run", fake_run)

    result = mcp_config.apply_target("codex", tmp_path, _COMMAND, force=False)

    assert result.status == "failed"


# ---------------------------------------------------------------------------
# What is offered for everything else
# ---------------------------------------------------------------------------


def test_every_default_target_stays_inside_the_workspace() -> None:
    """A default may write inside the directory being set up; changing a
    user-level config for every project on the machine is a bigger promise."""
    for name in mcp_config.DEFAULT_TARGETS:
        assert mcp_config.target_named(name).scope == "workspace"
    assert set(mcp_config.DEFAULT_TARGETS) < set(mcp_config.TARGET_NAMES)


def test_the_manual_definition_is_the_common_shape() -> None:
    pasted = json.loads(mcp_config.manual_definition(_COMMAND))

    assert pasted == {"boepie": _ENTRY}
