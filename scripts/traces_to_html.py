"""Convert collected JSONL conversation traces to self-contained HTML.

Produces a single ``.html`` file per trace with all CSS inline - no
external dependencies. Includes rich metadata (model, tokens, timings,
tool calls) in collapsible panels.

Usage::

    uv run python scripts/traces_to_html.py
    uv run python scripts/traces_to_html.py --pdf
    uv run python scripts/traces_to_html.py --overwrite
"""

from __future__ import annotations

import argparse
import html
import json
from datetime import datetime, timezone
from pathlib import Path

import markdown
import weasyprint

PROJECT_DIR = Path(__file__).resolve().parent.parent

_md = markdown.Markdown(extensions=["fenced_code", "tables"])


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

CSS = """\
:root {
  --bg: #f8f9fa;
  --surface: #ffffff;
  --user-bg: #e8f0fe;
  --user-border: #4a90d9;
  --assistant-bg: #f0f0f0;
  --assistant-border: #6b7280;
  --summary-bg: #fef9e7;
  --summary-border: #d4a017;
  --text: #1a1a1a;
  --text-muted: #6b7280;
  --mono: 'SF Mono', 'Cascadia Code', 'Fira Code', Consolas, monospace;
  --sans: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  --radius: 8px;
}
*, *::before, *::after { box-sizing: border-box; }
body {
  font-family: var(--sans);
  background: var(--bg);
  color: var(--text);
  margin: 0;
  padding: 2rem 1rem;
  line-height: 1.6;
}
.container { max-width: 860px; margin: 0 auto; }

/* Session header */
.session-header {
  background: var(--surface);
  border: 1px solid #ddd;
  border-radius: var(--radius);
  padding: 1.5rem;
  margin-bottom: 2rem;
}
.session-header h1 {
  margin: 0 0 0.75rem;
  font-size: 1.4rem;
  line-height: 1.3;
}
.session-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem 1.5rem;
  font-size: 0.85rem;
  color: var(--text-muted);
}
.session-meta dt { font-weight: 600; display: inline; }
.session-meta dd { display: inline; margin: 0; margin-right: 0.5rem; }

/* Messages */
.message {
  background: var(--surface);
  border-radius: var(--radius);
  padding: 1rem 1.25rem;
  margin-bottom: 1rem;
  border-left: 4px solid #ddd;
  position: relative;
}
.message--user { background: var(--user-bg); border-left-color: var(--user-border); }
.message--assistant { background: var(--assistant-bg); border-left-color: var(--assistant-border); }
.message--summary { background: var(--summary-bg); border-left-color: var(--summary-border); }

.message__header {
  display: flex;
  align-items: baseline;
  gap: 0.75rem;
  margin-bottom: 0.5rem;
  flex-wrap: wrap;
}
.message__role {
  font-weight: 700;
  font-size: 0.8rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}
.message--user .message__role { color: var(--user-border); }
.message--assistant .message__role { color: var(--assistant-border); }
.message--summary .message__role { color: var(--summary-border); }
.message__time {
  font-size: 0.75rem;
  color: var(--text-muted);
}

/* Badges */
.badges { display: flex; flex-wrap: wrap; gap: 0.35rem; margin-top: 0.25rem; }
.badge {
  display: inline-block;
  font-size: 0.7rem;
  font-family: var(--mono);
  padding: 0.15rem 0.5rem;
  border-radius: 999px;
  background: #e2e8f0;
  color: #334155;
  white-space: nowrap;
}
.badge--model { background: #dbeafe; color: #1e40af; }
.badge--input { background: #dcfce7; color: #166534; }
.badge--output { background: #fef3c7; color: #92400e; }
.badge--cache { background: #f3e8ff; color: #6b21a8; }
.badge--agent { background: #ffe4e6; color: #9f1239; }
.badge--timing { background: #e0f2fe; color: #075985; }

/* Content */
.message__content {
  font-size: 0.92rem;
  overflow-wrap: break-word;
}
.message__content p { margin: 0.5em 0; }
.message__content p:first-child { margin-top: 0; }
.message__content p:last-child { margin-bottom: 0; }
.message__content pre {
  background: #1e293b;
  color: #e2e8f0;
  padding: 0.75rem 1rem;
  border-radius: 6px;
  overflow-x: auto;
  font-size: 0.82rem;
  line-height: 1.5;
}
.message__content code {
  font-family: var(--mono);
  font-size: 0.85em;
}
.message__content :not(pre) > code {
  background: rgba(0,0,0,0.06);
  padding: 0.15em 0.35em;
  border-radius: 3px;
}
.message__content table {
  border-collapse: collapse;
  margin: 0.5em 0;
  font-size: 0.85rem;
}
.message__content th, .message__content td {
  border: 1px solid #ddd;
  padding: 0.35rem 0.6rem;
  text-align: left;
}
.message__content th { background: #f1f5f9; }
.message__content blockquote {
  border-left: 3px solid #cbd5e1;
  margin: 0.5em 0;
  padding: 0.25rem 0.75rem;
  color: var(--text-muted);
}

/* Tool calls */
.tool-calls { margin-top: 0.75rem; }
.tool-call {
  border: 1px solid #e2e8f0;
  border-radius: 6px;
  margin-bottom: 0.4rem;
  background: #f8fafc;
}
.tool-call summary {
  cursor: pointer;
  padding: 0.4rem 0.75rem;
  font-size: 0.8rem;
  font-family: var(--mono);
  color: #475569;
  user-select: none;
}
.tool-call summary:hover { background: #f1f5f9; }
.tool-call__input {
  margin: 0;
  padding: 0.5rem 0.75rem;
  font-size: 0.75rem;
  background: #1e293b;
  color: #e2e8f0;
  overflow-x: auto;
  max-height: 300px;
  overflow-y: auto;
}
.tool-call__result { border-top: 1px solid #334155; }
.tool-call__result-label {
  font-size: 0.7rem;
  font-weight: 600;
  color: #94a3b8;
  padding: 0.3rem 0.75rem 0;
  background: #0f172a;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}
.tool-call__output {
  margin: 0;
  padding: 0.5rem 0.75rem;
  font-size: 0.75rem;
  background: #0f172a;
  color: #a5f3fc;
  border-radius: 0 0 6px 6px;
  overflow-x: auto;
  max-height: 300px;
  overflow-y: auto;
  white-space: pre-wrap;
  word-wrap: break-word;
}

/* Metadata panel */
.metadata-panel {
  margin-top: 0.75rem;
  border: 1px solid #e2e8f0;
  border-radius: 6px;
  font-size: 0.78rem;
}
.metadata-panel summary {
  cursor: pointer;
  padding: 0.35rem 0.75rem;
  color: var(--text-muted);
  user-select: none;
}
.metadata-panel summary:hover { background: #f8fafc; }
.metadata-panel table {
  width: 100%;
  border-collapse: collapse;
  font-family: var(--mono);
  font-size: 0.75rem;
}
.metadata-panel td {
  padding: 0.25rem 0.75rem;
  border-top: 1px solid #f1f5f9;
  vertical-align: top;
}
.metadata-panel td:first-child {
  color: var(--text-muted);
  white-space: nowrap;
  width: 1%;
}

/* Separator */
.separator {
  border: none;
  border-top: 1px solid #e5e7eb;
  margin: 1.25rem 0;
}

/* Print */
@media print {
  body { padding: 0; background: white; }
  .message { break-inside: avoid; }
  details[open] summary ~ * { display: block; }
  details { open: true; }
  .tool-call, .metadata-panel { border-color: #ccc; }
}
"""

# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>{css}</style>
</head>
<body>
  <div class="container">
    {header}
    <div class="messages">
{messages}
    </div>
  </div>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _render_md(text: str) -> str:
    """Convert markdown text to HTML."""
    _md.reset()
    return _md.convert(text)


def _format_timestamp(ts: str) -> str:
    """Format an ISO timestamp to a readable string."""
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    except (ValueError, AttributeError):
        return ts or ""


def _format_ms_timestamp(ms: int | float) -> str:
    """Format a millisecond epoch timestamp to a readable string."""
    try:
        dt = datetime.fromtimestamp(ms / 1000, tz=timezone.utc)
        return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    except (ValueError, OSError):
        return ""


def _badge(label: str, value: str, cls: str = "") -> str:
    """Build a badge HTML span."""
    css = f"badge badge--{cls}" if cls else "badge"
    return f'<span class="{css}">{html.escape(label)}: {html.escape(value)}</span>'


def _format_tokens(usage: dict[str, object]) -> list[str]:
    """Build badge list from a Claude Code usage dict."""
    badges: list[str] = []
    inp = usage.get("input_tokens")
    out = usage.get("output_tokens")
    cache_read = usage.get("cache_read_input_tokens")
    cache_create = usage.get("cache_creation_input_tokens")
    if inp is not None:
        badges.append(_badge("in", f"{inp:,}", "input"))
    if out is not None:
        badges.append(_badge("out", f"{out:,}", "output"))
    if cache_read:
        badges.append(_badge("cache read", f"{cache_read:,}", "cache"))
    if cache_create:
        badges.append(_badge("cache write", f"{cache_create:,}", "cache"))
    return badges


def _extract_tool_result_text(content: str | list[object]) -> str:
    """Extract displayable text from a tool_result content field."""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for item in content:
        if isinstance(item, dict) and item.get("type") == "text":
            text = item.get("text", "")
            if isinstance(text, str) and text:
                parts.append(text)
    return "\n".join(parts)


def _build_tool_calls_html(
    content: list[dict[str, object]],
    tool_results: dict[str, str] | None = None,
) -> str:
    """Extract tool_use items from Claude Code content and render HTML.

    *tool_results* maps ``tool_use_id`` to the result text so that each
    tool call can show its output in the same collapsible block.
    """
    tool_results = tool_results or {}
    tools: list[dict[str, object]] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get("type") == "tool_use":
            tools.append(part)
    if not tools:
        return ""
    parts = ['<div class="tool-calls">']
    for tool in tools:
        name = html.escape(str(tool.get("name", "unknown")))
        tid = str(tool.get("id", ""))
        inp = tool.get("input", {})
        try:
            inp_str = json.dumps(inp, indent=2, ensure_ascii=False)
        except (TypeError, ValueError):
            inp_str = str(inp)

        result_html = ""
        result_text = tool_results.get(tid, "")
        if result_text:
            # Try to unwrap JSON for readability
            try:
                parsed = json.loads(result_text)
                if isinstance(parsed, dict) and len(parsed) == 1:
                    # Single-key result (e.g. {"result": "..."}) - show
                    # the value directly for readability
                    val = next(iter(parsed.values()))
                    if isinstance(val, str):
                        result_text = val
                    else:
                        result_text = json.dumps(
                            val, indent=2, ensure_ascii=False
                        )
                else:
                    result_text = json.dumps(
                        parsed, indent=2, ensure_ascii=False
                    )
            except (json.JSONDecodeError, TypeError):
                pass
            # Truncate very large results in the display
            display = result_text if len(result_text) <= 5000 else (
                result_text[:5000] + f"\n\n... ({len(result_text):,} chars total)"
            )
            result_html = (
                f'<div class="tool-call__result">'
                f'<div class="tool-call__result-label">Result</div>'
                f'<pre class="tool-call__output">{html.escape(display)}</pre>'
                f"</div>"
            )

        parts.append(
            f'<details class="tool-call">'
            f"<summary>Tool: <code>{name}</code></summary>"
            f'<pre class="tool-call__input">{html.escape(inp_str)}</pre>'
            f"{result_html}"
            f"</details>"
        )
    parts.append("</div>")
    return "\n".join(parts)


def _build_metadata_html(meta: dict[str, str]) -> str:
    """Build a collapsible metadata panel."""
    if not meta:
        return ""
    rows = "".join(
        f"<tr><td>{html.escape(k)}</td><td>{html.escape(v)}</td></tr>"
        for k, v in meta.items()
        if v
    )
    if not rows:
        return ""
    return (
        '<details class="metadata-panel">'
        "<summary>Metadata</summary>"
        f"<table>{rows}</table>"
        "</details>"
    )


def _extract_text_parts(content: list[dict[str, object]] | str) -> str:
    """Pull only text parts from Claude Code message content."""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get("type") == "text":
            text = part.get("text", "")
            if isinstance(text, str) and text:
                parts.append(text)
    return "\n\n".join(parts)


def _build_message_html(
    role: str,
    text_html: str,
    timestamp: str = "",
    badges: list[str] | None = None,
    tool_calls_html: str = "",
    metadata: dict[str, str] | None = None,
) -> str:
    """Assemble a complete message div."""
    role_lower = role.lower().replace(" ", "-")
    cls = f"message message--{role_lower}"

    badge_html = ""
    if badges:
        badge_html = f'<div class="badges">{"".join(badges)}</div>'

    ts_html = ""
    if timestamp:
        ts_html = f'<span class="message__time">{html.escape(timestamp)}</span>'

    meta_html = _build_metadata_html(metadata or {})

    return (
        f'<div class="{cls}">'
        f'<div class="message__header">'
        f'<span class="message__role">{html.escape(role)}</span>'
        f"{ts_html}"
        f"{badge_html}"
        f"</div>"
        f'<div class="message__content">{text_html}</div>'
        f"{tool_calls_html}"
        f"{meta_html}"
        f"</div>"
    )


# ---------------------------------------------------------------------------
# Claude Code converter
# ---------------------------------------------------------------------------

def convert_claude_code(
    jsonl_path: Path, *, include_tool_results: bool = True,
) -> str:
    """Convert a Claude Code JSONL session to HTML."""
    # First pass: collect all tool results keyed by tool_use_id.
    tool_results: dict[str, str] = {}
    if include_tool_results:
        with jsonl_path.open() as f:
            for raw_line in f:
                try:
                    obj = json.loads(raw_line)
                except json.JSONDecodeError:
                    continue
                if obj.get("type") != "user":
                    continue
                msg = obj.get("message", {})
                if not isinstance(msg, dict):
                    continue
                content = msg.get("content", [])
                if not isinstance(content, list):
                    continue
                for part in content:
                    if (
                        isinstance(part, dict)
                        and part.get("type") == "tool_result"
                    ):
                        tid = part.get("tool_use_id", "")
                        if tid:
                            tool_results[tid] = _extract_tool_result_text(
                                part.get("content", "")
                            )

    # Second pass: build messages.
    messages: list[str] = []
    session_id = ""
    cli_version = ""
    git_branch = ""
    user_count = 0

    with jsonl_path.open() as f:
        for raw_line in f:
            try:
                obj = json.loads(raw_line)
            except json.JSONDecodeError:
                continue

            msg_type = obj.get("type", "")

            # Capture session metadata from any line that has it
            if not session_id and obj.get("sessionId"):
                session_id = obj["sessionId"]
            if not cli_version and obj.get("version"):
                cli_version = obj["version"]
            if not git_branch and obj.get("gitBranch"):
                git_branch = obj["gitBranch"]

            if msg_type == "user":
                msg = obj.get("message", {})
                content = msg.get("content", "") if isinstance(msg, dict) else msg
                text = _extract_text_parts(content)

                # Skip tool-result-only messages
                if not text:
                    continue

                user_count += 1
                ts = _format_timestamp(obj.get("timestamp", ""))
                text_html = _render_md(text)
                messages.append(_build_message_html("User", text_html, ts))

            elif msg_type == "assistant":
                msg = obj.get("message", {})
                if not isinstance(msg, dict):
                    continue

                content = msg.get("content", "")
                text = _extract_text_parts(content)

                # Build tool calls HTML from content (with results)
                tool_html = ""
                if isinstance(content, list):
                    tool_html = _build_tool_calls_html(
                        content, tool_results if include_tool_results else None
                    )

                # Skip if no text and no tool calls
                if not text and not tool_html:
                    continue

                ts = _format_timestamp(obj.get("timestamp", ""))
                text_html = _render_md(text) if text else ""

                # Badges
                badges: list[str] = []
                model = msg.get("model", "")
                if model:
                    badges.append(_badge("model", str(model), "model"))
                usage = msg.get("usage", {})
                if isinstance(usage, dict):
                    badges.extend(_format_tokens(usage))

                # Extra metadata
                meta: dict[str, str] = {}
                stop = msg.get("stop_reason")
                if stop:
                    meta["stop_reason"] = str(stop)
                msg_id = msg.get("id", "")
                if msg_id:
                    meta["message_id"] = str(msg_id)

                messages.append(
                    _build_message_html(
                        "Assistant",
                        text_html,
                        ts,
                        badges=badges,
                        tool_calls_html=tool_html,
                        metadata=meta,
                    )
                )

            elif msg_type == "summary":
                summary = obj.get("summary", "")
                if isinstance(summary, str) and summary:
                    ts = _format_timestamp(obj.get("timestamp", ""))
                    messages.append(
                        _build_message_html(
                            "Summary",
                            _render_md(summary),
                            ts,
                        )
                    )

    # Session header
    title = jsonl_path.parent.name
    meta_items = [
        ("Source", "Claude Code"),
        ("Session", session_id),
        ("CLI Version", cli_version),
        ("Git Branch", git_branch),
        ("User Messages", str(user_count)),
    ]
    header = _build_session_header(title, meta_items)

    sep = '<hr class="separator">\n'
    return HTML_TEMPLATE.format(
        title=html.escape(title),
        css=CSS,
        header=header,
        messages=sep.join(messages),
    )


# ---------------------------------------------------------------------------
# VS Code Copilot converter
# ---------------------------------------------------------------------------

def _extract_vscode_response_text(response: list[object]) -> str:
    """Extract text from VS Code Copilot response items."""
    parts: list[str] = []
    for item in response:
        if not isinstance(item, dict):
            continue
        if "value" in item and isinstance(item["value"], str):
            parts.append(item["value"])
        elif item.get("kind") == "inlineReference":
            ref = item.get("inlineReference", {})
            if isinstance(ref, dict):
                uri = ref.get("uri", ref.get("path", ""))
                if isinstance(uri, dict):
                    uri = uri.get("path", uri.get("fsPath", ""))
                if uri:
                    parts.append(f"`{uri}`")
    return "".join(parts)


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
        existing = by_id[rid]
        existing_msg = existing.get("message", {})
        existing_has_text = bool(
            existing_msg.get("text", "")
            if isinstance(existing_msg, dict)
            else existing_msg
        )
        if not existing_has_text:
            by_id[rid] = req
        if not existing.get("response") and req.get("response"):
            by_id[rid]["response"] = req["response"]


def convert_vscode_copilot(jsonl_path: Path) -> str:
    """Convert a VS Code Copilot JSONL session to HTML."""
    title = jsonl_path.parent.name
    custom_title = ""
    creation_date = ""
    session_id = ""

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
                        creation_date = _format_ms_timestamp(cd)
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

    # Convert to HTML message blocks
    blocks: list[str] = []
    for rid in request_order:
        req = requests_by_id[rid]

        msg = req.get("message", {})
        user_text = ""
        if isinstance(msg, dict):
            user_text = msg.get("text", "")
        elif isinstance(msg, str):
            user_text = msg

        if not user_text:
            continue

        ts = req.get("timestamp")
        ts_str = _format_ms_timestamp(ts) if ts else ""

        # User message
        blocks.append(
            _build_message_html("User", _render_md(user_text), ts_str)
        )

        # Assistant response
        response = req.get("response", [])
        if isinstance(response, list) and response:
            resp_text = _extract_vscode_response_text(response)
            if resp_text:
                # Badges
                badges: list[str] = []
                model_id = req.get("modelId", "")
                if model_id:
                    badges.append(_badge("model", str(model_id), "model"))

                agent = req.get("agent", {})
                if isinstance(agent, dict):
                    agent_id = agent.get("id", "")
                    if agent_id:
                        badges.append(
                            _badge("agent", str(agent_id), "agent")
                        )

                # Timings
                result = req.get("result", {})
                if isinstance(result, dict):
                    timings = result.get("timings", {})
                    if isinstance(timings, dict):
                        first = timings.get("firstProgress")
                        total = timings.get("totalElapsed")
                        if first is not None:
                            badges.append(
                                _badge(
                                    "first token",
                                    f"{first / 1000:.1f}s",
                                    "timing",
                                )
                            )
                        if total is not None:
                            badges.append(
                                _badge(
                                    "total",
                                    f"{total / 1000:.1f}s",
                                    "timing",
                                )
                            )

                # Content references as metadata
                meta: dict[str, str] = {}
                refs = req.get("contentReferences", [])
                if isinstance(refs, list) and refs:
                    ref_strs: list[str] = []
                    for ref in refs:
                        if isinstance(ref, dict):
                            r = ref.get("reference", {})
                            if isinstance(r, dict):
                                uri = r.get("uri", r.get("path", ""))
                                if isinstance(uri, dict):
                                    uri = uri.get("path", "")
                                if uri:
                                    ref_strs.append(str(uri))
                    if ref_strs:
                        meta["references"] = ", ".join(ref_strs)

                blocks.append(
                    _build_message_html(
                        "Assistant",
                        _render_md(resp_text),
                        badges=badges,
                        metadata=meta,
                    )
                )

    # Session header
    display_title = custom_title or title
    msg_count = len(
        [rid for rid in request_order if _has_user_text(requests_by_id[rid])]
    )
    meta_items = [
        ("Source", "VS Code Copilot"),
        ("Session", session_id),
        ("Created", creation_date),
        ("User Messages", str(msg_count)),
    ]
    header = _build_session_header(display_title, meta_items)

    sep = '<hr class="separator">\n'
    return HTML_TEMPLATE.format(
        title=html.escape(display_title),
        css=CSS,
        header=header,
        messages=sep.join(blocks),
    )


def _has_user_text(req: dict[str, object]) -> bool:
    """Check if a VS Code request has user text."""
    msg = req.get("message", {})
    if isinstance(msg, dict):
        return bool(msg.get("text", ""))
    return bool(msg)


# ---------------------------------------------------------------------------
# Shared
# ---------------------------------------------------------------------------

def _build_session_header(
    title: str, meta_items: list[tuple[str, str]]
) -> str:
    """Build the session header HTML block."""
    dl = ""
    for label, value in meta_items:
        if value:
            dl += (
                f"<dt>{html.escape(label)}:</dt>"
                f"<dd><code>{html.escape(value)}</code></dd> "
            )
    return (
        '<div class="session-header">'
        f"<h1>{html.escape(title)}</h1>"
        f'<dl class="session-meta">{dl}</dl>'
        "</div>"
    )


def _detect_source(jsonl_path: Path) -> str:
    """Detect whether a JSONL file is Claude Code or VS Code Copilot."""
    with jsonl_path.open() as f:
        first_line = f.readline()
    try:
        obj = json.loads(first_line)
    except json.JSONDecodeError:
        return "unknown"

    if "kind" in obj:
        return "vscode-copilot"
    if obj.get("type") in ("permission-mode", "system"):
        return "claude-code"
    return "unknown"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert JSONL conversation traces to HTML (and optionally PDF)."
    )
    parser.add_argument(
        "--dir",
        type=Path,
        default=PROJECT_DIR / "traces",
        help="Directory containing trace folders (default: traces/)",
    )
    parser.add_argument(
        "--pdf",
        action="store_true",
        help="Also generate PDF files from the HTML.",
    )
    parser.add_argument(
        "--no-tool-results",
        action="store_true",
        help="Omit tool result outputs (reduces file size for large sessions).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .html/.pdf files.",
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

    html_count = 0
    pdf_count = 0
    skipped = 0
    for jsonl_path in jsonl_files:
        conv_dir = jsonl_path.parent
        html_path = conv_dir / "trace.html"
        pdf_path = conv_dir / "trace.pdf"

        # Generate HTML
        if html_path.exists() and not args.overwrite:
            skipped += 1
            html_content = html_path.read_text()
        else:
            source = _detect_source(jsonl_path)
            if source == "claude-code":
                html_content = convert_claude_code(
                    jsonl_path,
                    include_tool_results=not args.no_tool_results,
                )
            elif source == "vscode-copilot":
                html_content = convert_vscode_copilot(jsonl_path)
            else:
                print(f"  Skipping (unknown format): {conv_dir.name}")
                continue

            html_path.write_text(html_content)
            html_count += 1
            print(f"  {conv_dir.name}/trace.html")

        # Generate PDF
        if args.pdf:
            if pdf_path.exists() and not args.overwrite:
                pass
            else:
                wp_html = weasyprint.HTML(string=html_content)
                wp_html.write_pdf(pdf_path)
                pdf_count += 1
                print(f"  {conv_dir.name}/trace.pdf")

    parts = []
    if html_count:
        parts.append(f"{html_count} HTML")
    if pdf_count:
        parts.append(f"{pdf_count} PDF")
    if skipped:
        parts.append(f"{skipped} skipped")
    print(f"\n{', '.join(parts) or 'Nothing to do'}")


if __name__ == "__main__":
    main()
