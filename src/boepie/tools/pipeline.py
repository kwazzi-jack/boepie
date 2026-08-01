"""MCP tools for validating and running stimela pipelines.

These tools let the AI validate generated YAML recipes via
``stimela run --dry-run`` and execute them when ready. Both tools pass
``-B`` / ``--boring`` to stimela so captured output is free of ANSI/Rich
markup, and ``runner.stimela_run`` sets a wide ``COLUMNS`` in the child
environment so stimela's rich-based formatter does not wrap long lines
(file paths above all) mid-token - it wraps to the console width
regardless of TTY or boring mode. ``_clean_lines`` still strips the
trailing whitespace padding rich adds to fill each line out to that width.

Both tools return log file paths instead of forwarding full log content
through MCP. Call sites should use the Read tool on the reported paths
whenever they need more detail than the inline tail provides.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from boepie.runner import find_recipe_logs, stimela_run


_SUCCESS_TAIL_LINES = 20
_ERROR_TAIL_LINES = 40

# What `stimela run --dry-run` was empirically verified to catch (unknown cab
# names, unknown parameter names, missing required parameters - all raised as
# a pre-validation error before any step runs) versus what it does not (a
# dry run exits immediately after pre-validation, before ever touching input
# files or resolving `{substitution}` expressions). A caller reading a bare
# "valid" would otherwise be tempted to execute on the strength of a check
# that never happened.
_SCHEMA_OK_CHECKED = "checked: cab names, parameter names, required parameters"
_SCHEMA_OK_NOT_CHECKED = (
    "not checked: input file existence, {substitution} resolvability, container availability"
)


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
    """Format a list of log file paths for inclusion in an MCP response.

    Kept absolute (not relative to cwd): these are meant to be fed straight
    to the Read tool, and the server's cwd is not guaranteed to match the
    caller's working directory, so a relative path here could resolve to the
    wrong file or nothing at all.
    """
    if not log_paths:
        return "log files: none found in recipe directory"
    lines = ["log files (use the Read tool for full content):"]
    for path in log_paths:
        lines.append(f"  {path}")
    return "\n".join(lines)


class ValidateRecipeInput(BaseModel):
    recipe_path: str = Field(
        default="",
        description=(
            "Path to an existing recipe YAML file to validate in place. Preferred as "
            "soon as the recipe grows past a handful of lines."
        ),
    )
    yaml_content: str = Field(
        default="",
        description="Inline YAML content, written to a temp file before validation. Use only for tiny recipes.",
    )
    recipe_name: str = Field(
        default="", description="Recipe name to select, if the file defines more than one."
    )

    @model_validator(mode="after")
    def _exactly_one_source(self) -> "ValidateRecipeInput":
        if bool(self.recipe_path) == bool(self.yaml_content):
            raise ValueError("Provide exactly one of 'recipe_path' or 'yaml_content'.")
        return self


def validate_recipe(input: ValidateRecipeInput) -> str:
    """Validate a stimela recipe via ``stimela run --dry-run``.

    Checks cab names, parameter names, and required parameters. Does not
    check input file existence, ``{substitution}`` resolvability, or
    container availability - see the ``not checked`` line in the response.
    Use the Read tool on any reported log paths for full per-step detail.
    """
    if input.recipe_path:
        given_path = input.recipe_path
        resolved_path = Path(given_path)
        if not resolved_path.is_file():
            return f"Error: recipe file not found: {given_path}. Check the path and try again."

        yaml_path = resolved_path.resolve()
        log_dir = yaml_path.parent

        result = stimela_run(
            recipe_yml=str(yaml_path),
            recipe_name=input.recipe_name,
            dry_run=True,
            cwd=str(log_dir),
        )
        log_paths = find_recipe_logs(log_dir)

        if result.ok:
            return "\n".join([
                "SCHEMA-OK",
                f"recipe: {given_path}",
                _SCHEMA_OK_CHECKED,
                _SCHEMA_OK_NOT_CHECKED,
                "",
                _format_log_paths(log_paths),
                "",
                f"Output tail:\n{_tail(result.stdout, _SUCCESS_TAIL_LINES)}",
            ])
        error_body = result.stderr or result.stdout
        return "\n".join([
            f"FAILED (exit code {result.returncode})",
            f"recipe: {given_path}",
            "",
            f"Error tail:\n{_tail(error_body, _ERROR_TAIL_LINES)}",
            "",
            _format_log_paths(log_paths),
        ])

    # yaml_content mode: use a temp dir so the log files stimela emits land
    # beside the temporary recipe instead of polluting the caller's working
    # directory. The directory is cleaned up on exit, so there are no log
    # paths to report - the output is already clean thanks to --boring.
    with tempfile.TemporaryDirectory(prefix="boepie-validate-") as temp_dir:
        temp_recipe = Path(temp_dir) / "recipe.yml"
        temp_recipe.write_text(input.yaml_content)
        result = stimela_run(
            recipe_yml=str(temp_recipe),
            recipe_name=input.recipe_name,
            dry_run=True,
            cwd=temp_dir,
        )
        recipe_lines = [f"recipe: {input.recipe_name}"] if input.recipe_name else []
        if result.ok:
            return "\n".join([
                "SCHEMA-OK",
                *recipe_lines,
                _SCHEMA_OK_CHECKED,
                _SCHEMA_OK_NOT_CHECKED,
                "",
                _clean(result.stdout),
            ])
        error_body = result.stderr or result.stdout
        return "\n".join([
            f"FAILED (exit code {result.returncode})",
            *recipe_lines,
            "",
            _clean(error_body),
        ])


class RunRecipeInput(BaseModel):
    recipe_path: str = Field(description="Path to the recipe YAML file on disk.")
    recipe_name: str = Field(
        default="", description="Recipe name to select, if the file defines more than one."
    )
    params: dict[str, str] | None = Field(
        default=None, description="Parameter overrides as key=value pairs."
    )
    backend: Literal["native", "singularity", "kube"] = Field(
        default="native", description="Execution backend."
    )


def run_recipe(input: RunRecipeInput) -> str:
    """Execute a stimela recipe from a YAML file.

    Call ``validate_recipe`` first to catch cab/parameter/config errors
    before spending time on execution. No timeout is enforced, since real
    pipelines can run for hours. Use the Read tool on any reported log
    paths for full per-step detail.
    """
    given_path = input.recipe_path
    path = Path(given_path)
    if not path.is_file():
        return f"Error: recipe file not found: {given_path}. Check the path and try again."

    resolved = path.resolve()
    log_dir = resolved.parent

    result = stimela_run(
        recipe_yml=str(resolved),
        recipe_name=input.recipe_name,
        params=input.params,
        backend=input.backend,
        cwd=str(log_dir),
        timeout=None,
    )
    log_paths = find_recipe_logs(log_dir)

    if result.ok:
        return "\n".join([
            "OK",
            f"recipe: {given_path}",
            f"backend: {input.backend}",
            "",
            _format_log_paths(log_paths),
            "",
            f"Output tail:\n{_tail(result.stdout, _SUCCESS_TAIL_LINES)}",
        ])
    error_body = result.stderr or result.stdout
    return "\n".join([
        f"FAILED (exit code {result.returncode})",
        f"recipe: {given_path}",
        f"backend: {input.backend}",
        "",
        f"Error tail:\n{_tail(error_body, _ERROR_TAIL_LINES)}",
        "",
        _format_log_paths(log_paths),
    ])
