"""Running stimela, and projecting its schemas into boepie's own shapes.

Two halves, and they use stimela differently on purpose:

- **Execution** (`stimela_run`) shells out to the `stimela` binary in the
  same venv, so a recipe runs exactly as it would from the user's terminal.
- **Inspection** (`load_cab_schema`, `load_recipe_schema`) goes in-process
  through `boepie.pipeline.stimela_config`, which drives stimela's own
  config-loading chain. Nothing here parses YAML: the flattening of nested
  parameter groups, `_use` inheritance, and parameter categories are all
  `Cab.finalize`/`Recipe.finalize`'s work.

`CabParam` and `CabSchema` exist because scabha's `Parameter` carries far
more than an MCP response should - path policies, argument-passing
policies, substitution flags, metavars. These are the projection down to
what an agent writing a recipe actually needs.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel
from scabha.basetypes import UNSET
from scabha.cargo import Parameter, ParameterCategory

from boepie.pipeline.stimela_config import LoadedConfig, loaded_config

# stimela executable from the same venv as the running interpreter.
_STIMELA_CMD: list[str] = [str(Path(sys.executable).parent / "stimela")]

# stimela renders its (rich-based) log formatter to this many columns
# regardless of TTY, wrapping long lines - including file paths - mid-token.
# Wide enough that no realistic log line wraps; passed via COLUMNS in the
# child env (see stimela_run). _clean_lines still strips the padding rich
# adds to fill each line out to this width.
_WIDE_COLUMNS = "2000"


@dataclass(frozen=True)
class RunResult:
    """Structured result from a subprocess invocation."""

    command: list[str]
    stdout: str
    stderr: str
    returncode: int

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    @property
    def output(self) -> str:
        """Return stdout if successful, otherwise stderr."""
        return self.stdout if self.ok else self.stderr


def _run(
    args: list[str],
    timeout: float | None = 300,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
) -> RunResult:
    """Run a command and capture output.

    ``args`` is the full command. ``timeout`` is the maximum seconds to wait
    (None for no limit). ``cwd`` is the subprocess working directory (None
    inherits the parent's). ``env`` replaces the child environment entirely
    when given (None inherits the parent's).
    """
    proc = subprocess.run(
        args,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=cwd,
        env=env,
    )
    return RunResult(
        command=args,
        stdout=proc.stdout,
        stderr=proc.stderr,
        returncode=proc.returncode,
    )


def find_recipe_logs(directory: Path) -> list[Path]:
    """Return sorted log-*.txt files in a directory (non-recursive).

    Stimela writes one or more log files with names matching log-*.txt into
    its working directory by default (see opts.log.name). This helper locates
    those files so MCP tools can point callers at them instead of shovelling
    full log content through the response.
    """
    if not directory.is_dir():
        return []
    return sorted(directory.glob("log-*.txt"))


# ---------------------------------------------------------------------------
# Schema projection
# ---------------------------------------------------------------------------

# stimela's own detail levels, lowest first. `stimela doc` shows Required and
# Optional by default and needs -I/-O/-A to go further; boepie takes the same
# default, because Implicit and Obscure parameters are numerous, rarely what
# an agent is looking for, and paid for on every call.
type DetailLevel = Literal["required", "optional", "implicit", "obscure", "hidden"]

_DETAIL_LEVELS: dict[str, ParameterCategory] = {
    "required": ParameterCategory.Required,
    "optional": ParameterCategory.Optional,
    "implicit": ParameterCategory.Implicit,
    "obscure": ParameterCategory.Obscure,
    "hidden": ParameterCategory.Hidden,
}

_CATEGORY_NAMES: dict[ParameterCategory, str] = {
    category: name for name, category in _DETAIL_LEVELS.items()
}

DEFAULT_DETAIL: DetailLevel = "optional"


class CabParam(BaseModel):
    """One cab or recipe parameter, reduced to what a recipe author needs."""

    dtype: str = ""
    info: str = ""
    required: bool = False
    default: Any = None
    choices: list[str] | None = None
    writable: bool = False
    # The value an implicit parameter always takes, as written in the cab
    # definition - often a substitution like `{current.prefix}-sources.txt`.
    # Never settable by the caller, which is the point of surfacing it.
    implicit: str | None = None
    category: str = "optional"


class CabSchema(BaseModel):
    name: str
    info: str = ""
    inputs: dict[str, CabParam] = {}
    outputs: dict[str, CabParam] = {}

    def to_compact(self) -> str:
        """Compact, token-efficient representation for LLM consumption."""
        lines = [f"{self.name}: {self.info}" if self.info else self.name]

        required_inputs = {param_name: param for param_name, param in self.inputs.items() if param.required}
        optional_inputs = {param_name: param for param_name, param in self.inputs.items() if not param.required}

        if required_inputs:
            lines.append("\nRequired inputs:")
            for param_name, param in required_inputs.items():
                lines.append(f"  {_fmt_param(param_name, param)}")

        if optional_inputs:
            lines.append("\nOptional inputs:")
            for param_name, param in optional_inputs.items():
                lines.append(f"  {_fmt_param(param_name, param)}")

        if self.outputs:
            lines.append("\nOutputs:")
            for param_name, param in self.outputs.items():
                lines.append(f"  {_fmt_param(param_name, param)}")

        return "\n".join(lines)


def _fmt_param(param_name: str, param: CabParam) -> str:
    parts = [param.dtype] if param.dtype else []
    if param.writable:
        parts.append("writable")
    if param.default is not None:
        parts.append(f"default={param.default}")
    if param.choices:
        parts.append(f"choices={param.choices}")
    if param.implicit is not None:
        parts.append(f"implicit={param.implicit}")
    meta = f" ({', '.join(parts)})" if parts else ""
    info = f" - {param.info}" if param.info else ""
    return f"{param_name}{meta}{info}"


def _project_param(param: Parameter) -> CabParam:
    """Reduce a finalized scabha `Parameter` to a `CabParam`.

    `default` needs care: scabha marks "no default" with the `UNSET`
    sentinel *class*, not `None`, so a naive read renders it into output as
    the literal text `<class 'scabha.basetypes.UNSET'>`.
    """
    default = param.default
    if isinstance(default, type) and issubclass(default, UNSET):
        default = None
    return CabParam(
        dtype=str(param.dtype or ""),
        info=str(param.info or ""),
        required=bool(param.required),
        default=default,
        choices=[str(choice) for choice in param.choices] if param.choices else None,
        writable=bool(param.writable),
        implicit=str(param.implicit) if param.implicit is not None else None,
        category=_CATEGORY_NAMES.get(param.get_category(), "optional"),
    )


def _project_params(
    params: dict[str, Parameter], detail: DetailLevel
) -> dict[str, CabParam]:
    """Project a finalized parameter mapping, dropping anything above `detail`.

    Ordering is the schema's own, which follows the cab definition - a cab
    author groups related parameters together, and that grouping is worth
    more to a reader than an alphabetical sort would be.
    """
    ceiling = _DETAIL_LEVELS[detail]
    projected: dict[str, CabParam] = {}
    for name, param in params.items():
        if param.get_category() > ceiling:
            continue
        projected[name] = _project_param(param)
    return projected


def list_cab_names(config: LoadedConfig | None = None) -> list[str]:
    """Every cab name in the configured stimela sources, sorted."""
    return (config or loaded_config()).cab_names()


def list_cabs_with_info(config: LoadedConfig | None = None) -> list[dict[str, str]]:
    """Every cab as a ``{cab, description}`` dict, sorted by name.

    Reads the raw config node rather than constructing each `Cab`, which is
    both far cheaper and immune to a single malformed definition - the
    catalogue has to stay listable even when one entry cannot be finalized.
    """
    return [
        {"cab": definition.name, "description": definition.info}
        for definition in (config or loaded_config()).cab_definitions()
    ]


def load_cab_schema(
    cab_name: str,
    detail: DetailLevel = DEFAULT_DETAIL,
    config: LoadedConfig | None = None,
) -> CabSchema:
    """Load, finalize and project one cab's schema.

    Raises `ValueError` when no such cab exists, and `StimelaConfigError`
    when the cab exists but stimela rejects its definition - two different
    problems, and only the first is the caller's fault.
    """
    resolved = config or loaded_config()
    if not resolved.has_cab(cab_name):
        raise ValueError(
            f"Unknown cab '{cab_name}'. Call list_cabs to see what is available."
        )
    cab = resolved.finalized_cab(cab_name)
    return CabSchema(
        name=cab_name,
        info=str(cab.info or ""),
        inputs=_project_params(cab.inputs, detail),
        outputs=_project_params(cab.outputs, detail),
    )


def load_recipe_schema(
    recipe_name: str,
    detail: DetailLevel = DEFAULT_DETAIL,
    config: LoadedConfig | None = None,
) -> CabSchema:
    """Load, finalize and project one recipe's schema.

    A finalized recipe exposes the same `inputs`/`outputs` shape as a cab -
    including each step parameter the recipe leaves unset, promoted to
    `{step}.{param}` - so it projects through the same code.
    """
    resolved = config or loaded_config()
    if recipe_name not in resolved.config.lib.recipes:
        raise ValueError(
            f"Unknown recipe '{recipe_name}'. Call list_recipes to see what is available."
        )
    recipe = resolved.finalized_recipe(recipe_name)
    return CabSchema(
        name=recipe_name,
        info=str(recipe.info or ""),
        inputs=_project_params(recipe.inputs, detail),
        outputs=_project_params(recipe.outputs, detail),
    )


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def stimela_run(
    recipe_yml: str,
    recipe_name: str = "",
    params: dict[str, str] | None = None,
    dry_run: bool = False,
    backend: str = "native",
    cwd: str | None = None,
    timeout: float | None = 300,
) -> RunResult:
    """Run ``stimela -B run <recipe_yml> [recipe_name] [PARAM=VALUE ...]``.

    The ``-B`` / ``--boring`` flag is always passed so captured stdout/stderr
    are free of Rich markup and ANSI escape codes, which keeps token cost low
    when the output is forwarded through MCP. ``COLUMNS`` is set wide in the
    child environment (rich, which stimela's formatter uses, wraps to it
    regardless of TTY or boring mode) so long lines - file paths above all -
    are not split mid-token; the trailing padding rich adds up to that width
    is stripped by the caller.

    Parameters
    ----------
    recipe_yml:
        Path to the recipe YAML file.
    recipe_name:
        Optional recipe name if the file defines multiple recipes.
    params:
        Optional parameter overrides as key=value pairs.
    dry_run:
        If True, pass --dry-run to validate without executing.
    backend:
        Backend to use (native, singularity, kube).
    cwd:
        Working directory for the subprocess. Stimela writes its log files
        relative to this directory, so set it to the recipe's parent folder
        when you want logs colocated with the recipe.
    timeout:
        Maximum seconds to wait. None for no limit. Use None for real
        execution of a recipe since pipelines can take hours.
    """
    cmd = [*_STIMELA_CMD, "-B", "--backend", backend, "run"]

    if dry_run:
        cmd.append("--dry-run")

    cmd.append(recipe_yml)

    if recipe_name:
        cmd.append(recipe_name)

    if params:
        for key, value in params.items():
            cmd.append(f"{key}={value}")

    env = {**os.environ, "COLUMNS": _WIDE_COLUMNS}
    return _run(cmd, timeout=timeout, cwd=cwd, env=env)
