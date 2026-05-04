"""Interactive REPL for exercising boepie's MCP tools.

Runs boepie in-process and exposes its tools through a slash-command
shell. Useful for manual development and regression checks.

Slash commands:
    /list             Show all tools on the connected server
    /show <tool>      Show docstring and parameters for a tool
    /call <tool>      Prompt for parameters and invoke a tool
    /help             Show this help
    /quit  /exit      Leave the shell
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Iterable

from fastmcp import Client
from prompt_toolkit import PromptSession
from prompt_toolkit.application.current import get_app
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import HTML
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from boepie.server import mcp as boepie_mcp

console = Console()

SLASH_COMMANDS: dict[str, str] = {
    "/list": "list all tools on the server",
    "/show": "show docstring and parameters for a tool",
    "/call": "prompt for parameters and invoke a tool",
    "/help": "show this help",
    "/quit": "exit the shell",
    "/exit": "exit the shell",
}


class SlashCompleter(Completer):
    """Completer that hints slash commands and tool names contextually."""

    def __init__(self, tool_names: Iterable[str]) -> None:
        self.tool_names = sorted(tool_names)

    def get_completions(self, document: Document, complete_event):  # type: ignore[override]
        text = document.text_before_cursor
        if not text.startswith("/"):
            return
        parts = text.split(maxsplit=1)
        head = parts[0]
        if len(parts) == 1 and not text.endswith(" "):
            for command, description in SLASH_COMMANDS.items():
                if command.startswith(head):
                    yield Completion(
                        command,
                        start_position=-len(head),
                        display_meta=description,
                    )
            return
        if head in ("/show", "/call"):
            fragment = parts[1] if len(parts) > 1 else ""
            for tool_name in self.tool_names:
                if tool_name.startswith(fragment):
                    yield Completion(tool_name, start_position=-len(fragment))


def _tool_signature(tool: Any) -> str:
    """One-line signature like ``name(req: str, opt?: int)``."""
    schema = tool.inputSchema or {}
    properties = schema.get("properties") or {}
    required_params = set(schema.get("required") or [])
    parts: list[str] = []
    for param_name, param_schema in properties.items():
        marker = "" if param_name in required_params else "?"
        parts.append(f"{param_name}{marker}: {_format_param_type(param_schema)}")
    return f"{tool.name}({', '.join(parts)})"


def _tool_summary(tool: Any) -> str:
    return (tool.description or "").strip().split("\n", 1)[0]


def _make_bottom_toolbar(tool_map: dict[str, Any]):
    """Return a toolbar callable that mirrors Claude Code's inline hints."""

    def toolbar():
        try:
            text = get_app().current_buffer.document.text_before_cursor
        except Exception:
            return ""
        if not text.startswith("/"):
            return HTML("<b>/</b> for commands")
        parts = text.split(maxsplit=1)
        head = parts[0]
        if head in SLASH_COMMANDS and head not in ("/show", "/call"):
            return HTML(f"<b>{head}</b> {SLASH_COMMANDS[head]}")
        if head in ("/show", "/call"):
            fragment = parts[1].strip() if len(parts) > 1 else ""
            tool = tool_map.get(fragment)
            if tool is None:
                return HTML(f"<b>{head}</b> &lt;tool&gt;  (tab to complete)")
            signature = _tool_signature(tool)
            summary = _tool_summary(tool)
            hint = f"{signature}"
            if summary:
                hint += f"  :: {summary}"
            return HTML(f"<b>{head}</b> {hint}")
        if head in SLASH_COMMANDS:
            return HTML(f"<b>{head}</b> {SLASH_COMMANDS[head]}")
        return HTML(f"unknown: {head}  (/help)")

    return toolbar


def _param_toolbar(
    tool: Any,
    param_name: str,
    param_schema: dict[str, Any],
    is_required: bool,
):
    """Toolbar for per-parameter prompts showing type, default, description."""
    parts = [f"<b>{tool.name}.{param_name}</b>", _format_param_type(param_schema)]
    parts.append("required" if is_required else "optional")
    if "default" in param_schema and param_schema["default"] not in (None, ""):
        parts.append(f"default={param_schema['default']}")
    description = (param_schema.get("description") or "").strip().split("\n", 1)[0]
    hint = "  ".join(parts)
    if description:
        hint += f"  :: {description}"
    return lambda: HTML(hint)


def _humanize_bytes(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes} B"
    if num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.1f} KiB"
    return f"{num_bytes / (1024 * 1024):.2f} MiB"


def _humanize_duration(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.0f} ms"
    if seconds < 60:
        return f"{seconds:.1f} s"
    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60
    return f"{minutes}m {remaining_seconds:.0f}s"


def _enum_hint(param_schema: dict[str, Any]) -> str:
    """Return a short hint string if the schema (or its items) has an enum."""
    enum = param_schema.get("enum")
    if not enum:
        items_enum = (param_schema.get("items") or {}).get("enum")
        if items_enum:
            return f"  choices: {', '.join(str(v) for v in items_enum)}"
        return ""
    return f"  choices: {', '.join(str(v) for v in enum)}"


def _format_param_type(param_schema: dict[str, Any]) -> str:
    """Produce a short type label from a JSON Schema property dict."""
    json_type = param_schema.get("type", "?")
    if json_type == "array":
        items = param_schema.get("items") or {}
        return f"array[{items.get('type', '?')}]"
    return str(json_type)


def show_help() -> None:
    lines = [f"{command:<14} {description}" for command, description in SLASH_COMMANDS.items()]
    console.print(Panel("\n".join(lines), title="slash commands", border_style="cyan"))


def show_tool_list(tool_map: dict[str, Any]) -> None:
    table = Table(title="boepie MCP tools", show_lines=False)
    table.add_column("name", style="bold cyan")
    table.add_column("summary")
    for tool_name in sorted(tool_map.keys()):
        tool = tool_map[tool_name]
        first_line = (tool.description or "").strip().split("\n", 1)[0]
        table.add_row(tool_name, first_line)
    console.print(table)


def show_tool_details(tool_map: dict[str, Any], name: str) -> None:
    if not name:
        console.print("[red]usage: show <tool>[/red]")
        return
    tool = tool_map.get(name)
    if tool is None:
        console.print(f"[red]unknown tool: {name}[/red]")
        return

    console.print(
        Panel(
            tool.description or "(no description)",
            title=name,
            border_style="cyan",
        )
    )

    schema = tool.inputSchema or {}
    properties = schema.get("properties") or {}
    required_params = set(schema.get("required") or [])

    if not properties:
        console.print("[dim](no parameters)[/dim]")
        return

    table = Table(title="parameters")
    table.add_column("name", style="bold")
    table.add_column("type")
    table.add_column("required")
    table.add_column("default")
    for param_name, param_schema in properties.items():
        table.add_row(
            param_name,
            _format_param_type(param_schema),
            "yes" if param_name in required_params else "no",
            str(param_schema.get("default", "")),
        )
    console.print(table)


async def _prompt_array_values(
    session: PromptSession[str],
    tool: Any,
    param_name: str,
    items_schema: dict[str, Any],
    toolbar,
) -> list[Any]:
    """Collect array items until a blank line (scalars) or empty object (objects)."""
    items_type = _resolve_schema_type(items_schema)
    values: list[Any] = []

    if items_type == "object":
        console.print(f"[dim]{param_name}: enter one object per iteration, empty object to finish[/dim]")
        while True:
            item = await _prompt_object_value(session, tool, f"{param_name}[{len(values)}]", items_schema, False)
            if not item:
                return values
            values.append(item)
    else:
        hint = _enum_hint(items_schema)
        console.print(f"[dim]{param_name}: enter one value per line, blank line to finish{hint}[/dim]")
        while True:
            entry = await session.prompt_async(
                f"  {param_name}[{len(values)}]: ",
                bottom_toolbar=toolbar,
            )
            if not entry:
                return values
            values.append(entry)


def _resolve_schema_type(param_schema: dict[str, Any]) -> str:
    """Return a simple type string, resolving anyOf/nullable unions to the non-null branch."""
    if "type" in param_schema:
        return str(param_schema["type"])
    for branch in param_schema.get("anyOf", []):
        if branch.get("type") != "null":
            return str(branch.get("type", "string"))
    return "string"


async def _prompt_dict_value(
    session: PromptSession[str],
    tool: Any,
    param_name: str,
    value_schema: dict[str, Any],
) -> dict[str, Any]:
    """Prompt for arbitrary string keys whose values follow value_schema.

    Used for dict[str, SomeModel] parameters that appear in JSON schema as
    an object with ``additionalProperties`` and no fixed ``properties``.
    An empty key ends the loop.
    """
    console.print(f"[dim]{param_name}: enter a key (e.g. cab name), blank key to finish[/dim]")
    result: dict[str, Any] = {}
    while True:
        key = (await session.prompt_async(f"  {param_name} key: ")).strip()
        if not key:
            return result
        value = await _prompt_object_value(session, tool, f"{param_name}[{key!r}]", value_schema, True)
        if value is not None:
            result[key] = value


async def _prompt_object_value(
    session: PromptSession[str],
    tool: Any,
    param_name: str,
    param_schema: dict[str, Any],
    is_required: bool,
) -> dict[str, Any] | None:
    """Expand an object parameter into per-field prompts.

    If the schema has ``additionalProperties`` and no fixed ``properties``,
    delegates to ``_prompt_dict_value`` for free key-value entry.

    Returns None if the user leaves any required field blank, which signals
    end-of-list when collecting arrays of objects.
    """
    add_props: dict[str, Any] | None = param_schema.get("additionalProperties")
    sub_properties: dict[str, Any] = param_schema.get("properties") or {}

    if add_props and isinstance(add_props, dict) and not sub_properties:
        return await _prompt_dict_value(session, tool, param_name, add_props) or None

    if not sub_properties:
        return None

    required_sub = set(param_schema.get("required") or [])
    console.print(f"[dim]{param_name} (object) - enter sub-fields:[/dim]")
    result: dict[str, Any] = {}
    for sub_name, sub_schema in sub_properties.items():
        sub_required = sub_name in required_sub
        hint = _enum_hint(sub_schema)
        if hint:
            console.print(f"[dim]{hint}[/dim]")
        value = await _prompt_param_value(
            session, tool, f"{param_name}.{sub_name}", sub_schema, sub_required
        )
        if sub_required and not value:
            return None
        if value is not None:
            result[sub_name] = value
    return result or None


async def _prompt_param_value(
    session: PromptSession[str],
    tool: Any,
    param_name: str,
    param_schema: dict[str, Any],
    is_required: bool,
) -> Any:
    param_type = _resolve_schema_type(param_schema)
    toolbar = _param_toolbar(tool, param_name, param_schema, is_required)

    if param_type == "array":
        items_schema = param_schema.get("items") or {}
        values = await _prompt_array_values(session, tool, param_name, items_schema, toolbar)
        if not values and not is_required:
            return None
        return values

    if param_type == "object":
        return await _prompt_object_value(session, tool, param_name, param_schema, is_required)

    requirement_tag = " [required]" if is_required else ""
    default_value = param_schema.get("default")
    default_hint = f", default={default_value}" if default_value not in (None, "") else ""
    label = f"{param_name} ({param_type}{default_hint}){requirement_tag}: "
    raw_value = await session.prompt_async(label, bottom_toolbar=toolbar)

    if raw_value == "" and not is_required:
        return None

    if param_type == "boolean":
        return raw_value.lower() in ("1", "true", "yes", "y")
    if param_type == "integer":
        return int(raw_value)
    if param_type == "number":
        return float(raw_value)
    return raw_value


def _render_tool_output(
    tool_name: str,
    result: Any,
    elapsed_seconds: float,
    request_bytes: int,
) -> None:
    text_parts: list[str] = []
    content_list = getattr(result, "content", None)
    if content_list:
        for content_item in content_list:
            text_parts.append(getattr(content_item, "text", str(content_item)))
    else:
        text_parts.append(str(result))

    payload = "\n".join(text_parts).rstrip()
    response_bytes = len(payload.encode("utf-8"))
    line_count = payload.count("\n") + 1 if payload else 0
    subtitle = (
        f"{_humanize_duration(elapsed_seconds)}  "
        f"req {_humanize_bytes(request_bytes)}  "
        f"resp {_humanize_bytes(response_bytes)} / {line_count} lines"
    )

    lexer = _guess_lexer(payload)
    body: Any = Syntax(payload, lexer, theme="ansi_dark", word_wrap=True) if lexer else payload
    console.print(
        Panel(
            body,
            title=f"{tool_name} output",
            subtitle=subtitle,
            border_style="green",
        )
    )


def _guess_lexer(payload: str) -> str | None:
    stripped = payload.lstrip()
    if stripped.startswith("{") or stripped.startswith("["):
        try:
            json.loads(stripped)
        except ValueError:
            return None
        return "json"
    first_line = stripped.split("\n", 1)[0]
    if "," in first_line and "\n" in stripped:
        return None
    return None


async def call_tool_interactively(
    client: Client,
    session: PromptSession[str],
    tool_map: dict[str, Any],
    name: str,
) -> None:
    if not name:
        console.print("[red]usage: call <tool>[/red]")
        return
    tool = tool_map.get(name)
    if tool is None:
        console.print(f"[red]unknown tool: {name}[/red]")
        return

    schema = tool.inputSchema or {}
    properties = schema.get("properties") or {}
    required_params = set(schema.get("required") or [])

    arguments: dict[str, Any] = {}
    for param_name, param_schema in properties.items():
        is_required = param_name in required_params
        try:
            value = await _prompt_param_value(
                session, tool, param_name, param_schema, is_required
            )
        except (EOFError, KeyboardInterrupt):
            console.print("[yellow]cancelled[/yellow]")
            return
        except ValueError as error:
            console.print(f"[red]invalid value: {error}[/red]")
            return
        if value is not None:
            arguments[param_name] = value

    serialized_args = json.dumps(arguments, default=str)
    request_bytes = len(serialized_args.encode("utf-8"))
    console.print(f"[dim]calling {name}({serialized_args})[/dim]")
    start = time.perf_counter()
    try:
        result = await client.call_tool(name, arguments)
    except Exception as error:
        elapsed = time.perf_counter() - start
        console.print(
            f"[red]call failed after {elapsed * 1000:.1f} ms: {error}[/red]"
        )
        return
    elapsed = time.perf_counter() - start

    _render_tool_output(name, result, elapsed, request_bytes)


async def repl() -> None:
    console.print(
        Panel(
            "in-process boepie server (no network)",
            title="boepie test shell",
            border_style="cyan",
        )
    )
    async with Client(boepie_mcp) as client:
        tool_objects = await client.list_tools()
        tool_map: dict[str, Any] = {tool.name: tool for tool in tool_objects}

        console.print(f"[green]connected, {len(tool_map)} tools available[/green]")
        console.print("[dim]type / to see slash commands[/dim]")
        show_tool_list(tool_map)

        completer = SlashCompleter(tool_map.keys())
        session: PromptSession[str] = PromptSession(
            completer=completer,
            bottom_toolbar=_make_bottom_toolbar(tool_map),
            complete_while_typing=True,
        )

        while True:
            try:
                line = await session.prompt_async("boepie> ")
            except (EOFError, KeyboardInterrupt):
                console.print("[cyan]totsiens[/cyan]")
                return

            line = line.strip()
            if not line:
                continue
            if not line.startswith("/"):
                console.print("[red]commands must start with /[/red] (try /help)")
                continue

            parts = line.split(maxsplit=1)
            command = parts[0].lower()
            argument = parts[1] if len(parts) > 1 else ""

            if command in ("/quit", "/exit"):
                console.print("[cyan]bye[/cyan]")
                return
            if command == "/help":
                show_help()
                continue
            if command == "/list":
                show_tool_list(tool_map)
                continue
            if command == "/show":
                show_tool_details(tool_map, argument)
                continue
            if command == "/call":
                await call_tool_interactively(client, session, tool_map, argument)
                continue

            console.print(f"[red]unknown command: {command}[/red] (try /help)")


def main() -> None:
    try:
        asyncio.run(repl())
    except Exception as error:
        console.print(f"[red]fatal: {error}[/red]")


if __name__ == "__main__":
    main()
