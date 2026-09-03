# src/boepie/mcp_config.py
"""Registers the boepie MCP server with the agents that will launch it.

**The command has to name a venv, not a package.** boepie's pipeline tools
drive stimela's own configuration chain in-process, so the server must run in
the environment stimela is installed in - which is why boepie is installed
*alongside* stimela in one venv rather than as an isolated tool. A tool
install (`uv tool install`, `pipx`) puts boepie somewhere stimela is not, and
`uvx boepie` does the same thing on every launch; both produce a server that
starts cleanly and then cannot see a single cab.

So what gets written is an **absolute path to that venv's own console
script**. Its shebang points at the venv's interpreter, so nothing has to be
activated first, and no launcher - `uv` included - has to be installed for
the agent to start the server. `uv run --directory <repo>` was what the
dev-only `scripts/setup_dev_mcp.py` wrote until this replaced it, and that
required uv on the PATH of whatever process spawns the server.

**An agent's own CLI first, a file only where there is none.** A config
written in a shape the agent does not read fails *silently*: the agent
starts, lists no boepie tools, and explains nothing. The tool that owns the
schema is the one that should write it, and it will keep owning it when the
schema changes. So each target names the CLI that registers it, and the file
is the fallback for when that CLI is not installed - which matters, because
`.mcp.json` is read by **both Claude Code and GitHub Copilot CLI**, and
requiring Claude Code to be installed in order to configure Copilot would be
absurd.

Two things the CLIs will not do, which is why the file writer stays:
- **VS Code has no workspace-scope command.** `code --add-mcp` adds to the
  *user* config; `.vscode/mcp.json` can only be reached by writing it.
- `copilot mcp add` and `codex mcp add` write user-level config with no
  scope flag of their own. Workspace scope for Copilot comes from
  `.mcp.json`, which is written anyway.

The fallback writes exactly what `claude mcp add --scope project` writes -
`type: "stdio"` included - so the two routes cannot produce different
entries. Verified by running it.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SERVER_NAME = "boepie"

# How long an agent's own CLI gets to register the server before boepie stops
# waiting. These are local config edits; anything slower has gone wrong.
_CLI_TIMEOUT_SECONDS = 30


class McpConfigError(Exception):
    """A target could not be written, with the reason a user can act on."""


def server_command() -> list[str]:
    """The argv an agent should run to start this boepie's MCP server.

    Derived from the running interpreter rather than from `shutil.which`: the
    answer must be *this* installation, in the venv that also holds stimela,
    not whichever boepie happens to be first on some other process's PATH.
    """
    scripts = "Scripts" if os.name == "nt" else "bin"
    suffix = ".exe" if os.name == "nt" else ""
    console_script = Path(sys.prefix) / scripts / f"{SERVER_NAME}{suffix}"
    if console_script.is_file():
        return [str(console_script), "serve"]
    # No console script - an uninstalled source checkout, or an interpreter
    # invoked directly. `-m` reaches the same entry point through the same
    # interpreter, which is the part that matters.
    return [sys.executable, "-m", SERVER_NAME, "serve"]


def _stdio_entry(command: list[str]) -> dict[str, Any]:
    """One server definition, in the shape `claude mcp add` produces.

    `type` is redundant for every agent that defaults to stdio, and carried
    anyway: VS Code requires it, and matching what the CLI writes is what
    keeps the two registration routes from disagreeing.
    """
    return {"type": "stdio", "command": command[0], "args": command[1:]}


@dataclass(frozen=True)
class AgentTarget:
    """One agent boepie knows how to register itself with.

    A target may have a CLI, a file, or both. With both, the CLI wins and the
    file is what happens when it is not installed.
    """

    name: str
    # The executables whose presence proves this agent is installed. Nothing
    # is written for an agent that is not here: a config for a tool the user
    # does not have is clutter they never asked for, and it makes `setup`'s
    # report a claim about what will work rather than a list of files
    # touched. Several names because an agent can ship under more than one
    # (VS Code stable and Insiders).
    executables: tuple[str, ...] = ()
    # The argv that registers the server through the agent's own tool.
    register: Callable[[list[str]], list[str]] | None = None
    # The argv that asks whether it is registered already. `mcp add` is not
    # idempotent - every one of these CLIs refuses a name it already has -
    # and `setup` promises to be repeatable, so an existing registration has
    # to be recognised before `add` is attempted rather than after it fails.
    probe: list[str] | None = None
    # Where the definition goes otherwise, relative to the workspace.
    relative_path: str | None = None
    # The key holding the map of servers, e.g. `mcpServers` or `servers`.
    container: str = "mcpServers"
    # `workspace` targets stay inside the directory being set up; `user`
    # targets change configuration for every project on the machine, which
    # is why they are never in the default selection.
    scope: str = "workspace"
    note: str = ""


def _claude_argv(command: list[str]) -> list[str]:
    return ["claude", "mcp", "add", SERVER_NAME, "--scope", "project", *command]


def _copilot_argv(command: list[str]) -> list[str]:
    return ["copilot", "mcp", "add", SERVER_NAME, "--", *command]


def _codex_argv(command: list[str]) -> list[str]:
    return ["codex", "mcp", "add", SERVER_NAME, "--", *command]


def _gemini_argv(command: list[str]) -> list[str]:
    # Unverified: gemini was not installed on the machine this was written on,
    # so this argv follows `claude mcp add`'s shape, which it is modelled on.
    # Being CLI-driven is what makes that acceptable - a wrong argv comes back
    # as the CLI's own error message, where a wrongly-shaped config file would
    # come back as an agent that lists no boepie tools and says nothing. It
    # has no file fallback for the same reason: boepie has never seen the
    # shape it would be writing.
    return ["gemini", "mcp", "add", SERVER_NAME, *command]


# Order is the order they are reported in.
TARGETS: tuple[AgentTarget, ...] = (
    AgentTarget(
        name="claude",
        executables=("claude",),
        register=_claude_argv,
        probe=["claude", "mcp", "get", SERVER_NAME],
        relative_path=".mcp.json",
        note="Claude Code, project scope - committable, approved per project.",
    ),
    AgentTarget(
        name="copilot",
        executables=("copilot",),
        relative_path=".mcp.json",
        note="GitHub Copilot CLI, workspace scope - it reads the same "
        ".mcp.json. Its own `copilot mcp add` writes user config instead.",
    ),
    AgentTarget(
        name="vscode",
        executables=("code", "code-insiders"),
        relative_path=".vscode/mcp.json",
        container="servers",
        note="VS Code, workspace scope. `code --add-mcp` only reaches the "
        "user config, so this one is written.",
    ),
    AgentTarget(
        name="codex",
        executables=("codex",),
        register=_codex_argv,
        probe=["codex", "mcp", "get", SERVER_NAME],
        scope="user",
        note="Codex, through its own `codex mcp add` - user scope, not this "
        "project.",
    ),
    AgentTarget(
        name="gemini",
        executables=("gemini",),
        register=_gemini_argv,
        probe=["gemini", "mcp", "get", SERVER_NAME],
        scope="user",
        note="Gemini CLI, through its own `gemini mcp add` - user scope.",
    ),
)

TARGET_NAMES: tuple[str, ...] = tuple(target.name for target in TARGETS)
# Everything that stays inside the workspace being set up. A user-scope
# target changes configuration for every project on the machine, which is a
# bigger promise than a default should make.
DEFAULT_TARGETS: tuple[str, ...] = tuple(
    target.name for target in TARGETS if target.scope == "workspace"
)


@dataclass(frozen=True)
class TargetResult:
    """What became of one target."""

    name: str
    status: str  # written | current | skipped | failed
    path: Path | None = None
    detail: str = ""


def target_named(name: str) -> AgentTarget:
    for target in TARGETS:
        if target.name == name:
            return target
    raise KeyError(name)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise McpConfigError(
            f"{path} exists but could not be read as JSON ({error}). Fix or "
            f"move it, then run setup again."
        ) from error
    if not isinstance(loaded, dict):
        raise McpConfigError(f"{path} is not a JSON object, so it cannot be merged.")
    return loaded


def _write_json(path: Path, data: dict[str, Any]) -> None:
    """Replace `path` atomically, so an interrupted write cannot truncate a
    config the user already had."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(data, stream, indent=2)
            stream.write("\n")
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def write_file_target(
    target: AgentTarget, directory: Path, command: list[str], *, force: bool
) -> TargetResult:
    """Merge boepie's entry into one config file, leaving the rest of it alone."""
    if target.relative_path is None:
        raise ValueError(f"{target.name} has no file to write")

    path = directory / target.relative_path
    config = _read_json(path)
    servers = config.get(target.container)
    if servers is not None and not isinstance(servers, dict):
        raise McpConfigError(
            f"{path} has a '{target.container}' key that is not an object."
        )
    servers = dict(servers or {})

    entry = _stdio_entry(command)
    existing = servers.get(SERVER_NAME)
    if existing == entry or _same_server(existing, entry):
        return TargetResult(target.name, "current", path)
    if existing is not None and not force:
        return TargetResult(
            target.name,
            "skipped",
            path,
            "a different boepie entry is already there; --force replaces it",
        )

    servers[SERVER_NAME] = entry
    config[target.container] = servers
    _write_json(path, config)
    return TargetResult(target.name, "written", path)


def _same_server(existing: Any, entry: dict[str, Any]) -> bool:
    """Whether an entry already launches the same server.

    Compared on command and args only. An entry written by an agent's own
    CLI carries fields boepie does not write (`claude mcp add` adds an empty
    `env`), and rewriting the file to strip them would report a change on
    every run for no difference in what gets launched.
    """
    if not isinstance(existing, dict):
        return False
    return (
        existing.get("command") == entry["command"]
        and list(existing.get("args") or []) == entry["args"]
    )


def register_via_cli(
    target: AgentTarget, directory: Path, command: list[str]
) -> TargetResult:
    """Hand the registration to the agent's own CLI.

    Preferred over writing a file because the tool that owns the schema is
    the one that should write it - and will keep owning it when the schema
    changes. Reports `skipped` when the CLI is not installed, which is the
    caller's cue to fall back to a file.

    **Run in `directory`, not in boepie's own working directory.** A
    project-scope registration resolves against the cwd of the process doing
    it - `claude mcp add --scope project` writes `./.mcp.json` - so without
    this, `boepie setup --directory elsewhere` would quietly configure
    whichever directory the user happened to be standing in.
    """
    if target.register is None:
        raise ValueError(f"{target.name} has no CLI registration")

    argv = target.register(command)
    # The agent was found, but not necessarily under the name that registers
    # it - `installed_executable` accepts several spellings and a CLI may be
    # a different program from the one that proves the agent is there.
    if shutil.which(argv[0]) is None:
        return TargetResult(
            target.name, "skipped", None, f"'{argv[0]}' is not on PATH"
        )
    try:
        completed = subprocess.run(
            argv,
            cwd=directory,
            capture_output=True,
            text=True,
            timeout=_CLI_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return TargetResult(target.name, "failed", None, str(error))
    if completed.returncode != 0:
        reason = (completed.stderr or completed.stdout or "").strip().splitlines()
        return TargetResult(
            target.name,
            "failed",
            None,
            reason[-1] if reason else f"exit status {completed.returncode}",
        )
    return TargetResult(target.name, "written", None, " ".join(argv))


def installed_executable(target: AgentTarget) -> str | None:
    """The first of this agent's executables found on PATH, or None.

    The presence of the program is what boepie treats as "the user has this
    agent". It is not a perfect proxy - a VS Code installed without its
    `code` shell command would be missed - but the alternative is writing
    configuration for tools nobody has, and a false negative says so out
    loud where a false positive leaves a file behind in silence.
    """
    for executable in target.executables:
        if shutil.which(executable) is not None:
            return executable
    return None


def _not_installed(target: AgentTarget) -> TargetResult:
    looked_for = ", ".join(target.executables) or "nothing"
    return TargetResult(
        target.name, "skipped", None, f"not installed (looked for {looked_for})"
    )


def _already_registered(target: AgentTarget, directory: Path) -> bool:
    """Whether the agent's own CLI already knows this server."""
    if target.probe is None or shutil.which(target.probe[0]) is None:
        return False
    try:
        completed = subprocess.run(
            target.probe,
            cwd=directory,
            capture_output=True,
            text=True,
            timeout=_CLI_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0


def apply_target(
    name: str, directory: Path, command: list[str], *, force: bool
) -> TargetResult:
    """Register the server with one agent, by whichever route it offers.

    **Nothing is done for an agent that is not installed.** That check comes
    first, and a missing one is reported rather than configured: a config
    file for a tool the user does not have is clutter they never asked for,
    and it turns the report into a list of files touched instead of a claim
    about what will now work. Naming an agent explicitly does not override
    it - being told it is absent is the useful answer there too.

    After that the order is what makes `setup` repeatable and precise at
    once:

    1. An entry already in the target's own file settles it - that answer is
       exact about scope, where an `mcp get` probe would also see a
       user-level registration and call the workspace done.
    2. Otherwise the agent's own CLI creates it, because the tool that owns
       the schema should write it.
    3. The file is what happens when the agent has no CLI that writes this
       scope. `.mcp.json` is read by Copilot CLI, whose own `mcp add` writes
       user config, and `.vscode/mcp.json` has no command at all.

    *Replacing* an entry always goes through the file writer: `mcp add`
    refuses a name it already has, so the CLI cannot express `--force`. A CLI
    failure is reported rather than worked around - the tool that owns the
    format refused, and writing behind its back would be boepie overruling
    it.
    """
    target = target_named(name)
    if installed_executable(target) is None:
        return _not_installed(target)

    if target.relative_path is not None:
        settled = _settled_by_file(target, directory, command, force=force)
        if settled is not None:
            return settled
    elif _already_registered(target, directory):
        return TargetResult(target.name, "current", None, "already registered")

    if target.register is not None:
        result = register_via_cli(target, directory, command)
        if result.status != "skipped" or target.relative_path is None:
            return result
    if target.relative_path is None:
        return TargetResult(target.name, "skipped", None, "no way to register it")
    return write_file_target(target, directory, command, force=force)


def _settled_by_file(
    target: AgentTarget, directory: Path, command: list[str], *, force: bool
) -> TargetResult | None:
    """The answer the existing file already gives, or None to go on.

    None means "nothing of ours is in there", which is the only case where
    an `mcp add` can succeed.
    """
    path = directory / str(target.relative_path)
    config = _read_json(path)
    servers = config.get(target.container)
    existing = servers.get(SERVER_NAME) if isinstance(servers, dict) else None
    if existing is None:
        return None
    if _same_server(existing, _stdio_entry(command)):
        return TargetResult(target.name, "current", path)
    if not force:
        return TargetResult(
            target.name,
            "skipped",
            path,
            "a different boepie entry is already there; --force replaces it",
        )
    return write_file_target(target, directory, command, force=True)


def manual_definition(command: list[str]) -> str:
    """The definition to paste into an agent boepie does not cover.

    Deliberately the plain `mcpServers` shape, which is what most agents
    took from Claude Desktop's original file - and the command is the part
    that actually matters anyway.
    """
    return json.dumps({SERVER_NAME: _stdio_entry(command)}, indent=2)
