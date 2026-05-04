"""Convert collected JSONL conversation traces to readable Markdown.

Reads JSONL files from the ``traces/`` directory (or a custom path) and
writes a ``.md`` file alongside each one. The markdown shows the
user/assistant conversation with tool calls summarised.

Usage::

    uv run python scripts/traces_to_markdown.py
    uv run python scripts/traces_to_markdown.py --dir traces_export
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Claude Code -> Markdown
# ---------------------------------------------------------------------------

def _extract_text_from_content(content: list[dict[str, object]] | str) -> str:
    """Pull plain text from a Claude Code message content field."""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        ptype = part.get("type", "")
        if ptype == "text":
            text = part.get("text", "")
            if isinstance(text, str):
                parts.append(text)
        elif ptype == "tool_use":
            name = part.get("name", "unknown_tool")
            parts.append(f"*[Called tool: `{name}`]*")
        elif ptype == "tool_result":
            parts.append("*[Tool result received]*")
    return "\n\n".join(parts)


def _format_timestamp(ts: str) -> str:
    """Format an ISO timestamp to a readable string."""
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    except (ValueError, AttributeError):
        return ts or ""


def _role_header(role: str, timestamp: str = "") -> str:
    """Build a role header line: small bold role + optional timestamp."""
    header = f"#### {role}"
    if timestamp:
        header += f"  \n*{timestamp}*"
    return header


def convert_claude_code(jsonl_path: Path) -> str:
    """Convert a Claude Code JSONL session to markdown."""
    blocks: list[str] = []
    session_id = ""
    msg_count = 0

    with jsonl_path.open() as f:
        for raw_line in f:
            try:
                obj = json.loads(raw_line)
            except json.JSONDecodeError:
                continue

            msg_type = obj.get("type", "")

            if msg_type == "permission-mode" and not session_id:
                session_id = obj.get("sessionId", "")
                continue

            if msg_type == "user":
                msg = obj.get("message", {})
                text = ""
                if isinstance(msg, dict):
                    text = _extract_text_from_content(msg.get("content", ""))
                elif isinstance(msg, str):
                    text = msg

                # Skip tool-result-only messages (no real user text)
                if not text or text == "*[Tool result received]*":
                    continue

                msg_count += 1
                ts = _format_timestamp(obj.get("timestamp", ""))
                blocks.append(
                    _role_header("User", ts) + "\n\n" + text
                )

            elif msg_type == "assistant":
                msg = obj.get("message", {})
                text = ""
                if isinstance(msg, dict):
                    text = _extract_text_from_content(msg.get("content", ""))
                elif isinstance(msg, str):
                    text = msg

                # Skip tool-call-only messages (no prose)
                if not text or text == "*[Tool result received]*":
                    continue

                ts = _format_timestamp(obj.get("timestamp", ""))
                blocks.append(
                    _role_header("Assistant", ts) + "\n\n" + text
                )

            elif msg_type == "summary":
                summary = obj.get("summary", "")
                if isinstance(summary, str) and summary:
                    blocks.append(
                        _role_header("Summary (context compacted)")
                        + "\n\n" + summary
                    )

    # Build header
    title = jsonl_path.parent.name
    header_lines = [f"# {title}", ""]
    if session_id:
        header_lines.append(f"**Session:** `{session_id}`  ")
    header_lines.append("**Source:** Claude Code  ")
    header_lines.append(f"**Messages:** {msg_count}  ")
    header_lines.append("")
    header_lines.append("---")
    header_lines.append("")

    body = ("\n\n---\n\n").join(blocks)
    return "\n".join(header_lines) + body + "\n"


# ---------------------------------------------------------------------------
# VS Code Copilot -> Markdown
# ---------------------------------------------------------------------------

def _extract_vscode_response_text(response: list[object]) -> str:
    """Extract text from VS Code Copilot response items."""
    parts: list[str] = []
    for item in response:
        if not isinstance(item, dict):
            continue
        # Text content items have a 'value' key with markdown text
        if "value" in item and isinstance(item["value"], str):
            parts.append(item["value"])
        # Inline references (file mentions, etc.)
        elif item.get("kind") == "inlineReference":
            ref = item.get("inlineReference", {})
            if isinstance(ref, dict):
                uri = ref.get("uri", ref.get("path", ""))
                if isinstance(uri, dict):
                    uri = uri.get("path", uri.get("fsPath", ""))
                if uri:
                    parts.append(f"`{uri}`")
    return "".join(parts)


def convert_vscode_copilot(jsonl_path: Path) -> str:
    """Convert a VS Code Copilot JSONL session to markdown."""
    title = jsonl_path.stem
    custom_title = ""
    creation_date = ""
    session_id = ""

    # Deduplicate requests by requestId, keeping the version with text.
    requests_by_id: dict[str, dict[str, object]] = {}
    request_order: list[str] = []

    with jsonl_path.open() as f:
        for raw_line in f:
            try:
                obj = json.loads(raw_line)
            except json.JSONDecodeError:
                continue

            kind = obj.get("kind")

            if kind == 0:
                v = obj.get("v", {})
                if isinstance(v, dict):
                    custom_title = v.get("customTitle", "")
                    session_id = v.get("sessionId", "")
                    cd = v.get("creationDate")
                    if cd:
                        try:
                            dt = datetime.fromtimestamp(
                                cd / 1000, tz=timezone.utc
                            )
                            creation_date = dt.strftime(
                                "%Y-%m-%d %H:%M:%S UTC"
                            )
                        except (ValueError, OSError):
                            pass
                    reqs = v.get("requests", [])
                    if isinstance(reqs, list):
                        for req in reqs:
                            if isinstance(req, dict):
                                _merge_request(
                                    req, requests_by_id, request_order
                                )

            elif kind == 2:
                v = obj.get("v", [])
                if isinstance(v, list):
                    for req in v:
                        if isinstance(req, dict):
                            _merge_request(
                                req, requests_by_id, request_order
                            )

    # Convert to markdown blocks
    blocks: list[str] = []
    for rid in request_order:
        req = requests_by_id[rid]

        # User message
        msg = req.get("message", {})
        user_text = ""
        if isinstance(msg, dict):
            user_text = msg.get("text", "")
        elif isinstance(msg, str):
            user_text = msg

        # Skip requests with no user text (agent continuations)
        if not user_text:
            continue

        ts = req.get("timestamp")
        ts_str = ""
        if ts:
            try:
                dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
                ts_str = dt.strftime("%Y-%m-%d %H:%M:%S UTC")
            except (ValueError, OSError):
                pass

        block_parts = [_role_header("User", ts_str) + "\n\n" + user_text]

        # Assistant response
        response = req.get("response", [])
        if isinstance(response, list) and response:
            resp_text = _extract_vscode_response_text(response)
            if resp_text:
                block_parts.append(
                    _role_header("Assistant") + "\n\n" + resp_text
                )

        blocks.append("\n\n---\n\n".join(block_parts))

    # Build header
    title = jsonl_path.parent.name
    display_title = custom_title or title
    msg_count = len(blocks)
    header_lines = [f"# {display_title}", ""]
    if session_id:
        header_lines.append(f"**Session:** `{session_id}`  ")
    header_lines.append("**Source:** VS Code Copilot  ")
    if creation_date:
        header_lines.append(f"**Created:** {creation_date}  ")
    header_lines.append(f"**Messages:** {msg_count}  ")
    header_lines.append("")
    header_lines.append("---")
    header_lines.append("")

    body = ("\n\n---\n\n").join(blocks)
    return "\n".join(header_lines) + body + "\n"


def _merge_request(
    req: dict[str, object],
    by_id: dict[str, dict[str, object]],
    order: list[str],
) -> None:
    """Merge a request into the dedup map, preferring versions with text."""
    rid = req.get("requestId", "")
    if not rid:
        return

    msg = req.get("message", {})
    has_text = bool(
        msg.get("text", "") if isinstance(msg, dict) else msg
    )

    if rid not in by_id:
        by_id[rid] = req
        order.append(rid)
    elif has_text:
        # Prefer the version with actual message text and merge response
        existing = by_id[rid]
        existing_msg = existing.get("message", {})
        existing_has_text = bool(
            existing_msg.get("text", "")
            if isinstance(existing_msg, dict)
            else existing_msg
        )
        if not existing_has_text:
            by_id[rid] = req
        # If existing has no response but new one does, merge it in
        if not existing.get("response") and req.get("response"):
            by_id[rid]["response"] = req["response"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _detect_source(jsonl_path: Path) -> str:
    """Detect whether a JSONL file is Claude Code or VS Code Copilot."""
    with jsonl_path.open() as f:
        first_line = f.readline()
    try:
        obj = json.loads(first_line)
    except json.JSONDecodeError:
        return "unknown"

    # VS Code Copilot files start with kind=0
    if "kind" in obj:
        return "vscode-copilot"
    # Claude Code files start with type=permission-mode or type=system
    if obj.get("type") in ("permission-mode", "system"):
        return "claude-code"
    return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert JSONL conversation traces to Markdown."
    )
    parser.add_argument(
        "--dir",
        type=Path,
        default=PROJECT_DIR / "traces",
        help="Directory containing JSONL traces (default: traces/)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .md files.",
    )
    args = parser.parse_args()

    traces_dir: Path = args.dir
    if not traces_dir.is_dir():
        print(f"Error: directory not found: {traces_dir}")
        return

    jsonl_files = sorted(traces_dir.glob("*/trace.jsonl"))
    if not jsonl_files:
        print(f"No trace.jsonl files found in {traces_dir}/*/")
        return

    converted = 0
    skipped = 0
    for jsonl_path in jsonl_files:
        md_path = jsonl_path.with_name("trace.md")
        if md_path.exists() and not args.overwrite:
            skipped += 1
            continue

        source = _detect_source(jsonl_path)
        if source == "claude-code":
            markdown = convert_claude_code(jsonl_path)
        elif source == "vscode-copilot":
            markdown = convert_vscode_copilot(jsonl_path)
        else:
            print(f"  Skipping (unknown format): {jsonl_path.name}")
            continue

        md_path.write_text(markdown)
        converted += 1
        print(f"  {jsonl_path.parent.name}/trace.md")

    print(
        f"\nConverted {converted} files"
        + (f", skipped {skipped} existing" if skipped else "")
    )


if __name__ == "__main__":
    main()
