"""Generate local MCP server config for Claude Code and VS Code.

Creates config files that point both tools at the boepie MCP server
for development testing. Only writes files if they don't already exist
(or if --force is passed).

Usage:
    uv run scripts/setup_dev_mcp.py
    uv run scripts/setup_dev_mcp.py --force   # overwrite existing
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HOME_DIR = Path.home()

# Claude Code: $HOME/.claude.json
CLAUDE_CONFIG_PATH = HOME_DIR / ".claude.json"

# VS Code: .vscode/mcp.json
VSCODE_CONFIG_PATH = PROJECT_ROOT / ".vscode" / "mcp.json"

# The MCP server definition used by both tools
MCP_SERVER_DEF: dict[str, Any] = {
    "command": "uv",
    "args": [
        "--directory", str(PROJECT_ROOT),
        "run",
        "boepie", "serve",
    ],
}


def claude_code_config() -> dict[str, Any]:
    """Build Claude Code settings with the boepie MCP server."""
    return {
        "mcpServers": {
            "boepie": MCP_SERVER_DEF,
        },
    }


def vscode_config() -> dict[str, Any]:
    """Build VS Code MCP config with the boepie MCP server."""
    return {
        "servers": {
            "boepie": {
                "type": "stdio",
                **MCP_SERVER_DEF,
            },
        },
    }


def write_config(path: Path, data: dict[str, Any], force: bool = False) -> bool:
    """Write a JSON config file if it doesn't exist (or force is set).

    If the file already exists and contains an mcpServers/servers key,
    merges the boepie entry rather than overwriting the whole file.

    Returns True if the file was written/updated.
    """
    if path.exists() and not force:
        try:
            existing: dict[str, Any] = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            print(f"  SKIP {path} (exists, could not parse; use --force)")
            return False

        updated = False
        if "mcpServers" in data:
            existing.setdefault("mcpServers", {})
            if "boepie" not in existing["mcpServers"]:
                existing["mcpServers"]["boepie"] = data["mcpServers"]["boepie"]
                updated = True
        elif "servers" in data:
            existing.setdefault("servers", {})
            if "boepie" not in existing["servers"]:
                existing["servers"]["boepie"] = data["servers"]["boepie"]
                updated = True

        if updated:
            path.write_text(json.dumps(existing, indent=2) + "\n")
            print(f"  MERGED boepie server into {path}")
            return True
        else:
            print(f"  SKIP {path} (boepie entry already present)")
            return False

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")
    print(f"  WROTE {path}")
    return True


def main() -> None:
    force = "--force" in sys.argv

    print("Setting up local MCP server configs for development...\n")

    write_config(CLAUDE_CONFIG_PATH, claude_code_config(), force=force)
    write_config(VSCODE_CONFIG_PATH, vscode_config(), force=force)

    print(f"\nServer command: uv --directory {PROJECT_ROOT} run boepie serve")
    print("Transport: stdio (default)")


if __name__ == "__main__":
    main()
