"""Subprocess wrapper for invoking stimela.

All stimela interaction goes through this module. stimela and cult-cargo
are installed into boepie's own environment, so schema inspection can use
direct imports while recipe execution still runs stimela as a subprocess.
"""

from __future__ import annotations

import functools
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cultcargo
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel
import scabha.configuratt as configuratt

# stimela executable from the same venv as the running interpreter.
_STIMELA_CMD: list[str] = [str(Path(sys.executable).parent / "stimela")]

# cultcargo package directory - where cab YAML definitions live.
_CULTCARGO_DIR: Path = Path(cultcargo.__file__).parent

# Glob patterns matching cab YAMLs in the order stimela's MANIFEST loads them.
_CAB_GLOB_PATTERNS: list[str] = ["*.yml", "casa/*.yml"]


class StimelaType(BaseModel):
    """Base model for a stimela/scabha built-in parameter type.

    ``base`` is the immediate parent in the scabha class hierarchy.
    ``python_type`` is the underlying Python primitive it ultimately wraps.
    ``examples`` are sample values that would be valid for this type.
    """

    name: str
    base: str
    python_type: str
    description: str
    examples: list[str]


class URIType(StimelaType):
    name: str = "URI"
    base: str = "str"
    python_type: str = "str"
    description: str = (
        "A Uniform Resource Identifier - a string that supports local paths "
        "and remote schemes like file://, s3://."
    )
    examples: list[str] = [
        "/local/path/to/file",
        "file:///abs/path/to/file",
        "s3://bucket-name/key",
    ]


class FileType(StimelaType):
    name: str = "File"
    base: str = "URI"
    python_type: str = "str"
    description: str = "A file path (local or remote URI) pointing at a single file."
    examples: list[str] = [
        "config.yaml",
        "/data/model.fits",
        "s3://bucket/model.fits",
    ]


class DirectoryType(StimelaType):
    name: str = "Directory"
    base: str = "File"
    python_type: str = "str"
    description: str = "A directory path (local or remote URI)."
    examples: list[str] = [
        "/output/images/",
        "/tmp/workdir",
        "s3://bucket/output/",
    ]


class MSType(StimelaType):
    name: str = "MS"
    base: str = "Directory"
    python_type: str = "str"
    description: str = (
        "Measurement Set - radio astronomy visibility data format stored as a directory "
        "(CASA table). Typically ends in .ms or .MS."
    )
    examples: list[str] = [
        "/data/observation.ms",
        "target_field.MS",
        "s3://bucket/calibrator.ms",
    ]


_STIMELA_TYPES: dict[str, StimelaType] = {
    type_instance.name: type_instance
    for type_instance in (URIType(), FileType(), DirectoryType(), MSType())
}


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
) -> RunResult:
    """Run a command and capture output.

    Parameters
    ----------
    args:
        Full command as a list of strings.
    timeout:
        Maximum seconds to wait. None for no limit.
    cwd:
        Working directory for the subprocess. None inherits the parent's cwd.
    """
    proc = subprocess.run(
        args,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=cwd,
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


@functools.cache
def _load_cultcargo_config() -> DictConfig:
    """Load and merge all cultcargo YAMLs into one resolved config.

    Loads in manifest order so that cross-file ``_use`` references resolve
    correctly. YAMLs that fail to load are skipped with a warning.
    """
    merged: DictConfig = OmegaConf.create({})
    for glob_pattern in _CAB_GLOB_PATTERNS:
        for yml_path in sorted(_CULTCARGO_DIR.glob(glob_pattern)):
            try:
                file_conf, _ = configuratt.load(yml_path, use_cache=False, use_sources=[merged])
                merged = OmegaConf.merge(merged, file_conf)
            except Exception:
                pass
    return merged


def list_cab_names() -> list[str]:
    """Return the names of all available cabs from the merged cultcargo config."""
    cultcargo_config = _load_cultcargo_config()
    return sorted(cultcargo_config.get("cabs", {}).keys())


def list_cabs_with_info() -> list[dict[str, str]]:
    """Return all cabs as a list of ``{cab, description}`` dicts, sorted by name."""
    cultcargo_config = _load_cultcargo_config()
    cabs = cultcargo_config.get("cabs", {})
    return [
        {"cab": name, "description": str(cabs[name].get("info") or "")}
        for name in sorted(cabs.keys())
    ]


class CabParam(BaseModel):
    dtype: str = ""
    info: str = ""
    required: bool = False
    default: Any = None
    choices: list[str] | None = None
    writable: bool = False


class CabSchema(BaseModel):
    name: str
    info: str = ""
    inputs: dict[str, CabParam] = {}
    outputs: dict[str, CabParam] = {}

    def to_json_ref(self) -> str:
        """Minimal JSON reference: names and dtypes only, no descriptions.

        Use this when an LLM already knows what the cab does and only needs
        to confirm parameter names and types.
        """
        required_inputs = {
            param_name: param.dtype
            for param_name, param in self.inputs.items()
            if param.required
        }
        optional_inputs = {
            param_name: param.dtype
            for param_name, param in self.inputs.items()
            if not param.required
        }
        outputs = {param_name: param.dtype for param_name, param in self.outputs.items()}
        payload: dict[str, Any] = {
            "cab": self.name,
            "info": self.info,
            "required_inputs": required_inputs,
            "optional_inputs": optional_inputs,
            "outputs": outputs,
        }
        return json.dumps(payload, indent=2)

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
    meta = f" ({', '.join(parts)})" if parts else ""
    info = f" - {param.info}" if param.info else ""
    return f"{param_name}{meta}{info}"


def _extract_params(params_node: Any) -> dict[str, CabParam]:
    if not params_node:
        return {}
    result: dict[str, CabParam] = {}
    for param_name, param_node in params_node.items():
        param_dict = OmegaConf.to_container(param_node, resolve=True, throw_on_missing=False) or {}
        # Skip namespace containers (dict whose values are themselves dicts)
        if param_dict and all(isinstance(value, dict) for value in param_dict.values()):
            continue
        raw_choices = param_dict.get("choices") or None
        result[param_name] = CabParam(
            dtype=str(param_dict.get("dtype") or ""),
            info=str(param_dict.get("info") or ""),
            required=bool(param_dict.get("required") or False),
            default=param_dict.get("default"),
            choices=[str(choice) for choice in raw_choices] if raw_choices else None,
            writable=bool(param_dict.get("writable") or False),
        )
    return result


def load_cab_schema(cab_name: str) -> CabSchema:
    """Load and resolve a cab's schema from the merged cultcargo config.

    Parameters
    ----------
    cab_name:
        Name of the cab (e.g. "wsclean", "casa.bandpass").
    """
    cultcargo_config = _load_cultcargo_config()
    available_cabs = cultcargo_config.get("cabs", {})
    if cab_name not in available_cabs:
        available_names = ", ".join(sorted(available_cabs.keys()))
        raise ValueError(f"Unknown cab '{cab_name}'. Available: {available_names}")

    cab_node = available_cabs[cab_name]
    return CabSchema(
        name=cab_name,
        info=str(cab_node.get("info") or ""),
        inputs=_extract_params(cab_node.get("inputs")),
        outputs=_extract_params(cab_node.get("outputs")),
    )


def _param_info_dict(param: CabParam) -> dict[str, Any]:
    """Return only the set fields of a CabParam as a dict."""
    info: dict[str, Any] = {"dtype": param.dtype}
    if param.info:
        info["info"] = param.info
    if param.required:
        info["required"] = True
    if param.default is not None:
        info["default"] = param.default
    if param.choices:
        info["choices"] = param.choices
    if param.writable:
        info["writable"] = True
    return info


def _lookup_cab_param(cab_name: str, section: str, param_name: str) -> CabParam:
    if section not in ("inputs", "outputs"):
        raise ValueError(f"Section must be 'inputs' or 'outputs', got '{section}'")
    schema = load_cab_schema(cab_name)
    params = schema.inputs if section == "inputs" else schema.outputs
    if param_name not in params:
        raise ValueError(
            f"Parameter '{param_name}' not found in {cab_name}.{section}"
        )
    return params[param_name]


def _parse_query_path(path: str) -> tuple[str, str, str]:
    """Split a path like 'casa.bandpass.inputs.ms' into (cab, section, param).

    Parses right-to-left so cab names containing dots (e.g. 'casa.bandpass')
    are preserved.
    """
    segments = path.split(".")
    if len(segments) < 3:
        raise ValueError(
            f"Path '{path}' must have at least 3 segments: <cab>.<section>.<param>"
        )
    param_name = segments[-1]
    section = segments[-2]
    cab_name = ".".join(segments[:-2])
    return cab_name, section, param_name


def resolve_query(query: str) -> Any:
    """Resolve a single 'operation:path' query against cabs or stimela types.

    Supported forms:
      - 'type:<cab>.inputs.<param>'       -> dtype string
      - 'type:<cab>.outputs.<param>'      -> dtype string
      - 'info:<cab>.inputs.<param>'       -> dict of set fields
      - 'info:<cab>.outputs.<param>'      -> dict of set fields
      - 'type:stimela.types.<TypeName>'   -> base type string
      - 'info:stimela.types.<TypeName>'   -> full type info dict
    """
    if ":" not in query:
        raise ValueError(f"Query '{query}' must be in 'operation:path' form")
    operation, path = query.split(":", 1)
    operation = operation.strip()
    path = path.strip()

    if operation not in ("type", "info"):
        raise ValueError(f"Unknown operation '{operation}'. Use 'type' or 'info'")

    if path.startswith("stimela.types."):
        type_name = path[len("stimela.types."):]
        if type_name not in _STIMELA_TYPES:
            available = ", ".join(sorted(_STIMELA_TYPES.keys()))
            raise ValueError(
                f"Unknown stimela type '{type_name}'. Available: {available}"
            )
        stimela_type = _STIMELA_TYPES[type_name]
        if operation == "type":
            return stimela_type.base
        return stimela_type.model_dump()

    cab_name, section, param_name = _parse_query_path(path)
    param = _lookup_cab_param(cab_name, section, param_name)
    if operation == "type":
        return param.dtype
    return _param_info_dict(param)


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
    when the output is forwarded through MCP.

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

    return _run(cmd, timeout=timeout, cwd=cwd)
