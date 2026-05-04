"""MCP tools for validating and running stimela pipelines.

These tools let the AI validate generated YAML recipes via
``stimela run --dry-run`` and execute them when ready. Both tools pass
``-B`` / ``--boring`` to stimela so captured output is free of ANSI/Rich
markup, and they return log file paths instead of forwarding full log
content through MCP. Call sites should use the Read tool on the reported
paths whenever they need more detail than the inline tail provides.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from boepie.runner import find_recipe_logs, stimela_run


_SUCCESS_TAIL_LINES = 20
_ERROR_TAIL_LINES = 40


def _clean_lines(text: str) -> list[str]:
    """Split ``text`` into lines and strip the trailing whitespace padding
    stimela's formatter adds to each line (it pads every line to the console
    width even in boring mode, which is pure token waste over MCP)."""
    return [line.rstrip() for line in text.splitlines()]


def _clean(text: str) -> str:
    """Return ``text`` with per-line trailing whitespace stripped."""
    return "\n".join(_clean_lines(text))


def _tail(text: str, max_lines: int) -> str:
    """Return the last ``max_lines`` lines of ``text`` with an omission marker."""
    lines = _clean_lines(text)
    if len(lines) <= max_lines:
        return "\n".join(lines)
    omitted = len(lines) - max_lines
    kept = "\n".join(lines[-max_lines:])
    return f"... [{omitted} earlier lines omitted, read the log files for full context]\n{kept}"


def _format_log_paths(log_paths: list[Path]) -> str:
    """Format a list of log file paths for inclusion in an MCP response."""
    if not log_paths:
        return "Log files: (none found in recipe directory)"
    lines = ["Log files (use the Read tool to fetch full content):"]
    for path in log_paths:
        lines.append(f"  {path}")
    return "\n".join(lines)


def validate_recipe(
    recipe_path: str = "",
    yaml_content: str = "",
    recipe_name: str = "",
) -> str:
    """Validate a stimela recipe via ``stimela run --dry-run``.

    Provide exactly one of ``recipe_path`` or ``yaml_content``.

    Prefer ``recipe_path`` as soon as the recipe grows past a handful of
    lines: write the YAML to a file once and pass the path, so the full
    content does not have to travel through the MCP message stream on
    every validation call. Reserve ``yaml_content`` for tiny inline
    snippets where writing to disk first is not worth the round-trip.

    On success the response is ``VALID`` followed by a tail of the dry-run
    output. On failure it is ``INVALID`` followed by an error tail. When a
    real ``recipe_path`` was supplied, the response also lists the log
    files stimela wrote alongside the recipe. Use the Read tool on those
    paths when you need the full per-step log content; do not ask for it
    to be inlined into the response.

    Parameters
    ----------
    recipe_path:
        Path to an existing recipe YAML file to validate in place.
        Preferred for anything non-trivial.
    yaml_content:
        Inline YAML content. Written to a temporary file before validation.
        Use only for very small recipes.
    recipe_name:
        Optional recipe name if the file defines multiple recipes.
    """
    if bool(recipe_path) == bool(yaml_content):
        return "Error: provide exactly one of 'recipe_path' or 'yaml_content'."

    if recipe_path:
        resolved_path = Path(recipe_path)
        if not resolved_path.is_file():
            return f"Error: recipe file not found: {recipe_path}"

        yaml_path = resolved_path.resolve()
        log_dir = yaml_path.parent

        result = stimela_run(
            recipe_yml=str(yaml_path),
            recipe_name=recipe_name,
            dry_run=True,
            cwd=str(log_dir),
        )
        log_paths = find_recipe_logs(log_dir)

        if result.ok:
            return (
                f"VALID\n\n"
                f"Recipe: {yaml_path}\n\n"
                f"{_format_log_paths(log_paths)}\n\n"
                f"Output tail:\n{_tail(result.stdout, _SUCCESS_TAIL_LINES)}"
            )
        error_body = result.stderr or result.stdout
        return (
            f"INVALID (exit code {result.returncode})\n\n"
            f"Recipe: {yaml_path}\n\n"
            f"Error tail:\n{_tail(error_body, _ERROR_TAIL_LINES)}\n\n"
            f"{_format_log_paths(log_paths)}"
        )

    # yaml_content mode: use a temp dir so the log files stimela emits
    # land beside the temporary recipe instead of polluting the caller's
    # working directory. The directory is cleaned up on exit, so the log
    # content is embedded inline (it is already clean thanks to --boring).
    with tempfile.TemporaryDirectory(prefix="boepie-validate-") as temp_dir:
        temp_recipe = Path(temp_dir) / "recipe.yml"
        temp_recipe.write_text(yaml_content)
        result = stimela_run(
            recipe_yml=str(temp_recipe),
            recipe_name=recipe_name,
            dry_run=True,
            cwd=temp_dir,
        )
        if result.ok:
            return f"VALID\n\n{_clean(result.stdout)}"
        error_body = result.stderr or result.stdout
        return (
            f"INVALID (exit code {result.returncode})\n\n"
            f"{_clean(error_body)}"
        )


def run_recipe(
    recipe_path: str,
    recipe_name: str = "",
    params: dict[str, str] | None = None,
    backend: str = "native",
) -> str:
    """Execute a stimela recipe from a YAML file.

    Runs ``stimela run`` on an existing recipe file with clean/boring
    output mode. Use ``validate_recipe`` first to check for config errors.

    On success the response is ``SUCCESS`` followed by a tail of stdout
    and the list of log files stimela wrote beside the recipe. On failure
    it is ``FAILED`` followed by an error tail and the same log paths.
    Use the Read tool on the reported log paths when you need the full
    per-step execution log: the inline tail is intentionally short to
    keep MCP responses token-efficient.

    No timeout is enforced, since real pipelines can run for hours.

    Parameters
    ----------
    recipe_path:
        Path to the recipe YAML file on disk.
    recipe_name:
        Optional recipe name if the file defines multiple recipes.
    params:
        Optional parameter overrides as key=value pairs.
    backend:
        Backend to use: "native", "singularity", or "kube".
    """
    path = Path(recipe_path)
    if not path.is_file():
        return f"Error: recipe file not found: {recipe_path}"

    resolved = path.resolve()
    log_dir = resolved.parent

    result = stimela_run(
        recipe_yml=str(resolved),
        recipe_name=recipe_name,
        params=params,
        backend=backend,
        cwd=str(log_dir),
        timeout=None,
    )
    log_paths = find_recipe_logs(log_dir)

    if result.ok:
        return (
            f"SUCCESS\n\n"
            f"Recipe: {resolved}\n"
            f"Backend: {backend}\n\n"
            f"{_format_log_paths(log_paths)}\n\n"
            f"Output tail:\n{_tail(result.stdout, _SUCCESS_TAIL_LINES)}"
        )
    error_body = result.stderr or result.stdout
    return (
        f"FAILED (exit code {result.returncode})\n\n"
        f"Recipe: {resolved}\n"
        f"Backend: {backend}\n\n"
        f"Error tail:\n{_tail(error_body, _ERROR_TAIL_LINES)}\n\n"
        f"{_format_log_paths(log_paths)}"
    )
