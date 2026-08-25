"""MCP tools for discovering, inspecting, validating and running stimela recipes.

Discovery and inspection (`list_recipes`, `get_recipe_docs`) read the same
resolved config the cab tools do - the libraries named by
`pipeline.sources`, optionally with one `recipe_file` layered on top for
the recipe the user is actually working on. That split is deliberate:
libraries are stable for a session and cost ten seconds to load, so they
are cached; a recipe file changes between calls and is merged into a copy
that is thrown away afterwards.

A finalized `Recipe` exposes the same `inputs`/`outputs` shape as a `Cab`,
including every step parameter the recipe leaves unset promoted to
`{step}.{param}` - which is why both project through `runner.CabSchema`.

Validation and execution (`validate_recipe`, `run_recipe`) instead shell
out to the ``stimela`` binary, so a recipe runs exactly as it would from
the user's own terminal. Both pass
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

import fnmatch
import tempfile
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from boepie.pipeline._table import write_csv
from boepie.pipeline.runner import (
    DEFAULT_DETAIL,
    DetailLevel,
    find_recipe_logs,
    load_recipe_schema,
    stimela_run,
)
from boepie.pipeline.stimela_config import (
    StimelaConfigError,
    describe_sources,
    loaded_config,
)


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


_SOURCE_HELP = (
    "One extra place to read recipes from, on top of the libraries boepie "
    "already found installed. Normally a path to a recipe YAML file you are "
    "working on. Omit it unless you have such a file: installed libraries "
    "are discovered automatically and are already included."
)

_RECIPE_DETAIL_HELP = (
    "How much of the recipe's schema to include, as stimela's own parameter "
    "categories: 'required', 'optional' (the default), 'implicit', "
    "'obscure', 'hidden'. A recipe promotes every unset step parameter to "
    "its own inputs, so the higher levels get large quickly."
)


class ListRecipesInput(BaseModel):
    source: str = Field(default="", description=_SOURCE_HELP)
    pattern: str = Field(
        default="",
        description="fnmatch pattern to narrow the list, e.g. '*selfcal*'. Omit for all.",
    )


def list_recipes(input: ListRecipesInput) -> str:
    """List the stimela recipes available to boepie.

    Every stimela library installed alongside boepie is included
    automatically - you do not need to name one. Use this before
    ``get_recipe_docs`` the way you use ``list_cabs`` before
    ``get_cab_schema``, then pass a name straight to ``get_recipe_docs``.

    Returns a count line followed by a CSV table of ``recipe``,
    ``description``, ``origin``, where ``origin`` names the library or file
    the recipe was written in.
    """
    try:
        config = loaded_config(input.source or None)
    except StimelaConfigError as error:
        return f"Error: {error}"

    names = config.recipe_names_all()
    total = len(names)
    if not total:
        return (
            f"No recipes available. The stimela libraries boepie found "
            f"({describe_sources()}) provide cabs but no recipes; pass "
            f"source=<path to a recipe .yml> to read one from disk."
        )

    if input.pattern:
        names = fnmatch.filter(names, input.pattern)
        if not names:
            return f"No recipes match the pattern '{input.pattern}'."

    origins = config.recipe_origins()
    rows = [
        {
            "recipe": name,
            "description": str(config.config.lib.recipes[name].get("info") or ""),
            "origin": _origin_label(origins.get(name)),
        }
        for name in names
    ]
    header = f"# showing {len(rows)} of {total} recipes\n"
    return header + write_csv(rows, ["recipe", "description", "origin"])


class GetRecipeDocsInput(BaseModel):
    recipe_name: str = Field(
        description="Recipe name as list_recipes reports it, e.g. 'selfcal'."
    )
    source: str = Field(default="", description=_SOURCE_HELP)
    detail: DetailLevel = Field(
        default=DEFAULT_DETAIL, description=_RECIPE_DETAIL_HELP
    )
    raw: bool = Field(
        default=False,
        description=(
            "Return the recipe's YAML source verbatim instead of the "
            "structured summary. Use it to see how a recipe is actually "
            "written - step ordering, substitutions, `_use` references - "
            "which the summary does not show."
        ),
    )


def get_recipe_docs(input: GetRecipeDocsInput) -> str:
    """Get a recipe's inputs, outputs and steps - or its raw YAML.

    The default is a structured summary: the recipe's own parameters, then
    each step in order with the cab or sub-recipe it runs. Pass ``raw=True``
    for the YAML file the recipe was written in, when you need to see the
    wiring between steps rather than the schema.

    Returns a text report, or the file's contents when ``raw`` is set.
    """
    try:
        config = loaded_config(input.source or None)
    except StimelaConfigError as error:
        return f"Error: {error}"

    if input.recipe_name not in config.config.lib.recipes:
        known = ", ".join(config.recipe_names_all()) or "none"
        return (
            f"Error: unknown recipe '{input.recipe_name}'. Available: {known}. "
            f"Call list_recipes, or pass source=<path> if it lives in a file."
        )

    if input.raw:
        return _raw_recipe(config, input.recipe_name)

    try:
        schema = load_recipe_schema(
            input.recipe_name, detail=input.detail, config=config
        )
    except (ValueError, StimelaConfigError) as error:
        return f"Error: {error}"

    return "\n".join([schema.to_compact(), "", _format_steps(config, input.recipe_name)])


def _origin_label(path: Path | None) -> str:
    """Name the library or file a recipe came from, for a listing column.

    An installed library is reported by its package spec (`breifast.recipes`)
    rather than an absolute path into site-packages, since that is both
    shorter and the name a user would recognise. The package is found by
    walking up while directories still hold an `__init__.py`, which works
    wherever the library was installed from - matching on the literal string
    "site-packages" would miss an editable install or a PYTHONPATH entry.
    A file outside any package keeps its path.
    """
    if path is None:
        return "unknown"
    package_parts: list[str] = []
    directory = path.parent
    while (directory / "__init__.py").is_file():
        package_parts.append(directory.name)
        directory = directory.parent
    if not package_parts:
        return str(path)
    return ".".join(reversed(package_parts))


def _raw_recipe(config: Any, recipe_name: str) -> str:
    """The YAML file a recipe was written in, with its path as a header.

    stimela keeps no provenance for recipes, so the file is found by looking
    for the name as a top-level key in the loaded sources. When that fails -
    a recipe reached through `_include`, say - say so rather than
    substituting a re-serialised config: `OmegaConf.to_yaml` of a finalized
    recipe emits every unset schema field as an explicit null, which is
    both much larger than the original and not what the user wrote.
    """
    path = config.recipe_source_file(recipe_name)
    if path is None:
        return (
            f"Error: could not locate the YAML file defining '{recipe_name}'. "
            f"It was probably reached through an `_include` rather than "
            f"declared directly in a source file. Call get_recipe_docs "
            f"without raw=True for the structured summary."
        )
    try:
        body = path.read_text(encoding="utf-8")
    except OSError as error:
        return f"Error: cannot read {path}: {error}"
    return f"# {path}\n{body}"


def _format_steps(config: Any, recipe_name: str) -> str:
    """The recipe's steps in declaration order, each with what it runs.

    Read off the config node rather than the finalized `Recipe`, because
    step order is the one thing a recipe author controls that the schema
    projection loses.
    """
    steps = config.config.lib.recipes[recipe_name].get("steps") or {}
    if not steps:
        return "Steps: none"
    lines = ["Steps:"]
    for position, (name, step) in enumerate(steps.items(), start=1):
        runs = step.get("cab") or step.get("recipe") or "?"
        kind = "cab" if step.get("cab") else "recipe"
        info = f" - {step.get('info')}" if step.get("info") else ""
        skipped = " [skip]" if step.get("skip") else ""
        lines.append(f"  {position}. {name} ({kind}: {runs}){skipped}{info}")
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
