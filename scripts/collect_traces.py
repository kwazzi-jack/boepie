"""Collect and rename conversation trace JSONL files.

Gathers session traces from Claude Code and VS Code Copilot, copies them
into a ``traces/`` directory with human-readable names.

Each conversation gets its own folder::

    traces/{YYYYMMDD}_{source}_{msg_count}msg_{title_slug}/trace.jsonl

Sources:
- Claude Code: ``~/.claude/projects/<project-hash>/*.jsonl``
- VS Code Copilot: ``~/.config/Code/User/workspaceStorage/<hash>/chatSessions/*.jsonl``

Usage::

    uv run python scripts/collect_traces.py
    uv run python scripts/collect_traces.py --out traces_export
    uv run python scripts/collect_traces.py --summarize   # requires Ollama
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
PROJECT_PATH = str(PROJECT_DIR)

# Claude Code stores sessions per-project under this base.
CLAUDE_CODE_BASE = Path.home() / ".claude" / "projects"

# VS Code stores workspace data here.
VSCODE_STORAGE_BASE = (
    Path.home() / ".config" / "Code" / "User" / "workspaceStorage"
)

MAX_SLUG_WORDS = 8


# ---------------------------------------------------------------------------
# Slug helpers
# ---------------------------------------------------------------------------

def _slugify(text: str, max_words: int = MAX_SLUG_WORDS) -> str:
    """Turn arbitrary text into a filesystem-safe slug.

    Keeps only alphanumeric words, lowercased, joined by hyphens,
    capped at *max_words* words (not character length) to avoid
    weird mid-word truncation.
    """
    # Strip markdown fences, backticks, special chars
    text = re.sub(r"```[\s\S]*?```", "", text)
    text = re.sub(r"`[^`]*`", "", text)
    text = re.sub(r"[^\w\s-]", " ", text)
    words = text.lower().split()
    words = [w for w in words if w]
    if not words:
        return "untitled"
    return "-".join(words[:max_words])


# ---------------------------------------------------------------------------
# Claude Code
# ---------------------------------------------------------------------------

def _find_claude_code_project_dir() -> Path | None:
    """Find the Claude Code project directory matching this repo."""
    if not CLAUDE_CODE_BASE.is_dir():
        return None
    # Claude Code hashes the project path into the directory name.
    # Convention: the cwd appears as a slug like -home-brian-PhD-boepie
    slug = PROJECT_PATH.replace("/", "-")
    for d in CLAUDE_CODE_BASE.iterdir():
        if d.is_dir() and slug in d.name:
            return d
    return None


def _parse_claude_code(path: Path) -> dict[str, str | int]:
    """Extract metadata from a Claude Code JSONL session file."""
    first_user_text = ""
    first_timestamp = ""
    user_msg_count = 0

    with path.open() as f:
        for line in f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("type") == "user":
                user_msg_count += 1
                if not first_user_text:
                    first_timestamp = obj.get("timestamp", "")
                    msg = obj.get("message", {})
                    if isinstance(msg, dict):
                        content = msg.get("content", "")
                        if isinstance(content, list):
                            for part in content:
                                if (
                                    isinstance(part, dict)
                                    and part.get("type") == "text"
                                ):
                                    first_user_text = part["text"]
                                    break
                        elif isinstance(content, str):
                            first_user_text = content
                    elif isinstance(msg, str):
                        first_user_text = msg

    # Parse timestamp (ISO 8601)
    date_str = ""
    if first_timestamp:
        try:
            dt = datetime.fromisoformat(
                first_timestamp.replace("Z", "+00:00")
            )
            date_str = dt.strftime("%Y%m%d")
        except ValueError:
            pass

    title = _slugify(first_user_text)
    return {
        "date": date_str or "unknown",
        "source": "claude-code",
        "msg_count": user_msg_count,
        "title": title,
    }


def collect_claude_code() -> list[tuple[Path, dict[str, str | int]]]:
    """Return (source_path, metadata) for each Claude Code session."""
    project_dir = _find_claude_code_project_dir()
    if project_dir is None:
        return []
    results: list[tuple[Path, dict[str, str | int]]] = []
    for jsonl in sorted(project_dir.glob("*.jsonl")):
        meta = _parse_claude_code(jsonl)
        if meta["msg_count"] > 0:
            results.append((jsonl, meta))
    return results


# ---------------------------------------------------------------------------
# VS Code Copilot
# ---------------------------------------------------------------------------

def _find_vscode_workspace_dirs() -> list[Path]:
    """Find VS Code workspace storage directories for this project."""
    if not VSCODE_STORAGE_BASE.is_dir():
        return []
    matches: list[Path] = []
    for d in VSCODE_STORAGE_BASE.iterdir():
        ws_json = d / "workspace.json"
        if not ws_json.is_file():
            continue
        try:
            data = json.loads(ws_json.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        folder = data.get("folder", "")
        # Decode URI: file:///home/... or vscode-remote://ssh-remote%2B.../home/...
        parsed = urllib.parse.urlparse(folder)
        decoded_path = urllib.parse.unquote(parsed.path)
        if decoded_path.rstrip("/") == PROJECT_PATH.rstrip("/"):
            matches.append(d)
        # Also check remote paths where the project name matches
        elif decoded_path.rstrip("/").endswith("/boepie"):
            matches.append(d)
    return matches


def _parse_vscode_copilot(path: Path) -> dict[str, str | int]:
    """Extract metadata from a VS Code Copilot JSONL session file."""
    title = "untitled"
    date_str = "unknown"
    msg_count = 0

    with path.open() as f:
        first_line = f.readline()
        try:
            obj = json.loads(first_line)
        except json.JSONDecodeError:
            return {
                "date": date_str,
                "source": "vscode-copilot",
                "msg_count": 0,
                "title": title,
            }

        v = obj.get("v", {})
        if isinstance(v, dict):
            custom_title = v.get("customTitle", "")
            if custom_title:
                title = _slugify(custom_title)

            creation_date = v.get("creationDate")
            if creation_date:
                try:
                    dt = datetime.fromtimestamp(
                        creation_date / 1000, tz=timezone.utc
                    )
                    date_str = dt.strftime("%Y%m%d")
                except (ValueError, OSError):
                    pass

            requests = v.get("requests", [])
            msg_count = len(requests)

            # Fallback: use first user message if no customTitle
            if title == "untitled" and requests:
                first_msg = requests[0].get("message", {})
                if isinstance(first_msg, dict):
                    text = first_msg.get("text", "")
                elif isinstance(first_msg, str):
                    text = first_msg
                else:
                    text = ""
                if text:
                    title = _slugify(text)

        # Count kind=2 lines (new request batches) and extract
        # first user message for title fallback.
        first_request_text = ""
        for line in f:
            try:
                update = json.loads(line)
            except json.JSONDecodeError:
                continue
            if update.get("kind") == 2:
                v_list = update.get("v", [])
                if isinstance(v_list, list):
                    msg_count += len(v_list)
                    if not first_request_text and v_list:
                        req_msg = v_list[0].get("message", {})
                        if isinstance(req_msg, dict):
                            first_request_text = req_msg.get("text", "")
                        elif isinstance(req_msg, str):
                            first_request_text = req_msg

        # Use first request text if we still have no title
        if title == "untitled" and first_request_text:
            title = _slugify(first_request_text)

    return {
        "date": date_str,
        "source": "vscode-copilot",
        "msg_count": msg_count,
        "title": title,
    }


def collect_vscode_copilot() -> list[tuple[Path, dict[str, str | int]]]:
    """Return (source_path, metadata) for each VS Code Copilot session."""
    ws_dirs = _find_vscode_workspace_dirs()
    results: list[tuple[Path, dict[str, str | int]]] = []
    for ws_dir in ws_dirs:
        chat_dir = ws_dir / "chatSessions"
        if not chat_dir.is_dir():
            continue
        for jsonl in sorted(chat_dir.glob("*.jsonl")):
            meta = _parse_vscode_copilot(jsonl)
            if meta["msg_count"] > 0:
                results.append((jsonl, meta))
    return results


# ---------------------------------------------------------------------------
# Ollama summarization (optional)
# ---------------------------------------------------------------------------

def _ollama_available(base_url: str = "http://localhost:11434") -> bool:
    """Check if Ollama API is reachable."""
    import urllib.request

    try:
        req = urllib.request.Request(f"{base_url}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=3):
            return True
    except Exception:
        return False


def _summarize_title(
    text: str,
    base_url: str = "http://localhost:11434",
    model: str = "llama3.1:8b",
) -> str:
    """Ask Ollama to generate a short title from the first user message."""
    import urllib.request

    payload = json.dumps({
        "model": model,
        "prompt": (
            "Generate a short title (max 6 words) for this conversation. "
            "Reply with ONLY the title, nothing else.\n\n"
            f"First message: {text[:500]}"
        ),
        "stream": False,
    }).encode()

    req = urllib.request.Request(
        f"{base_url}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read())
            return _slugify(result.get("response", ""))
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _build_dirname(meta: dict[str, str | int]) -> str:
    """Build the output directory name from metadata."""
    return (
        f"{meta['date']}_{meta['source']}"
        f"_{meta['msg_count']}msg_{meta['title']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect conversation trace JSONL files."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_DIR / "traces",
        help="Output directory (default: traces/)",
    )
    parser.add_argument(
        "--summarize",
        action="store_true",
        help="Use Ollama to generate better titles (requires running Ollama)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llama3.1:8b",
        help="Ollama model to use for --summarize (default: llama3.1:8b)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be copied without copying.",
    )
    args = parser.parse_args()

    # Collect all sessions
    sessions: list[tuple[Path, dict[str, str | int]]] = []
    sessions.extend(collect_claude_code())
    sessions.extend(collect_vscode_copilot())

    if not sessions:
        print("No conversation traces found.")
        return

    # Optional Ollama summarization for Claude Code sessions
    if args.summarize:
        if _ollama_available():
            print("Ollama detected - generating titles for Claude Code sessions...")
            for src_path, meta in sessions:
                if meta["source"] == "claude-code":
                    # Read the first user message for summarization
                    with src_path.open() as f:
                        for line in f:
                            obj = json.loads(line)
                            if obj.get("type") == "user":
                                msg = obj.get("message", {})
                                text = ""
                                if isinstance(msg, dict):
                                    content = msg.get("content", "")
                                    if isinstance(content, str):
                                        text = content
                                    elif isinstance(content, list):
                                        for part in content:
                                            if (
                                                isinstance(part, dict)
                                                and part.get("type") == "text"
                                            ):
                                                text = part["text"]
                                                break
                                elif isinstance(msg, str):
                                    text = msg
                                if text:
                                    summary = _summarize_title(
                                        text, model=args.model
                                    )
                                    if summary:
                                        meta["title"] = summary
                                break
        else:
            print("Warning: --summarize requested but Ollama is not reachable.")

    # Copy files into per-conversation folders
    out_dir: Path = args.out
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    for src_path, meta in sessions:
        dirname = _build_dirname(meta)
        conv_dir = out_dir / dirname
        dest = conv_dir / "trace.jsonl"

        if args.dry_run:
            print(f"  [dry-run] {src_path.name} -> {dirname}/trace.jsonl")
        else:
            conv_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, dest)
            print(f"  {src_path.name} -> {dirname}/trace.jsonl")

    print(f"\nCollected {len(sessions)} sessions into {out_dir}/")


if __name__ == "__main__":
    main()
