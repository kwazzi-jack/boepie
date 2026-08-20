"""MCP tools for querying stimela cab definitions and documentation.

These tools let the AI discover what software is available through
cult-cargo and inspect their parameters via ``stimela doc``.
"""

from __future__ import annotations

import fnmatch
from typing import Any, Literal

from pydantic import BaseModel, Field

from boepie.pipeline.runner import CabParam, list_cabs_with_info, load_cab_schema
from boepie.pipeline._table import write_csv


# Canonical field order for get_cab_params output.
_AVAILABLE_FIELDS: list[str] = [
    "dtype", "info", "required", "default", "choices", "writable", "examples",
]


def list_cabs(pattern: str | None = None) -> str:
    """List available cab (containerised application bundle) definitions.

    Use this to discover what tools cult-cargo provides before building a
    recipe. Narrow with an fnmatch ``pattern`` (e.g. ``casa.*``, ``*clean*``);
    omit it to list everything.

    Returns a count line followed by a CSV table of ``cab``, ``description``.
    """
    all_cabs = list_cabs_with_info()
    total = len(all_cabs)
    if not all_cabs:
        return "Error: no cab definitions found. Run 'uv sync' to install project dependencies."

    if pattern:
        all_cabs = [row for row in all_cabs if fnmatch.fnmatch(row["cab"], pattern)]
        if not all_cabs:
            return f"No cabs match the pattern '{pattern}'."

    header = f"# showing {len(all_cabs)} of {total} cabs\n"
    return header + write_csv(all_cabs, ["cab", "description"])


class GetCabDocsInput(BaseModel):
    cab_name: str = Field(
        description="Cab name, e.g. 'wsclean', 'quartical', 'casa.bandpass'. See list_cabs for names."
    )
    params: list[str] | None = Field(
        default=None,
        description="Restrict output to these parameter names. Omit for every parameter.",
    )


def get_cab_docs(input: GetCabDocsInput) -> str:
    """Get full parameter documentation for a cab: descriptions, defaults, choices.

    Use this when learning a new tool. For quick name/type confirmation,
    use ``get_cab_schema`` instead.

    Returns a compact text report of the cab's inputs and outputs, including
    descriptions, defaults, and choices.
    """
    try:
        schema = load_cab_schema(input.cab_name)
    except ValueError as error:
        return f"Error: {error}"

    if input.params is not None:
        params_set = set(input.params)
        schema = schema.model_copy(update={
            "inputs": {name: param for name, param in schema.inputs.items() if name in params_set},
            "outputs": {name: param for name, param in schema.outputs.items() if name in params_set},
        })

    return schema.to_compact()


class GetCabSchemaInput(BaseModel):
    cab_name: str = Field(description="Cab name, e.g. 'wsclean', 'quartical', 'casa.bandpass'.")
    section: Literal["inputs", "outputs", "all"] = Field(
        default="all", description="Which section(s) to include."
    )


def get_cab_schema(input: GetCabSchemaInput) -> str:
    """Get a compact CSV schema for a cab: parameter names, types, and flags.

    Cheapest way to confirm a parameter exists and see its dtype - no
    descriptions, defaults, or choices; use ``get_cab_docs`` for those.

    Returns required inputs (``param,dtype,writable``), then optional inputs
    and outputs (``param,dtype``), each under its own header line.
    """
    try:
        schema = load_cab_schema(input.cab_name)
    except ValueError as error:
        return f"Error: {error}"

    parts: list[str] = []

    if input.section in ("inputs", "all"):
        required = [(name, param) for name, param in schema.inputs.items() if param.required]
        optional = [(name, param) for name, param in schema.inputs.items() if not param.required]

        if required:
            parts.append(f"# {input.cab_name} inputs")
            parts.append(write_csv(
                [{"param": name, "dtype": param.dtype, "writable": str(param.writable).lower()} for name, param in required],
                ["param", "dtype", "writable"],
            ))

        if optional:
            parts.append("# optional")
            parts.append(write_csv(
                [{"param": name, "dtype": param.dtype} for name, param in optional],
                ["param", "dtype"],
            ))

    if input.section in ("outputs", "all") and schema.outputs:
        parts.append(f"# {input.cab_name} outputs")
        parts.append(write_csv(
            [{"param": name, "dtype": param.dtype} for name, param in schema.outputs.items()],
            ["param", "dtype"],
        ))

    return "\n".join(parts)


class CabParamSpec(BaseModel):
    section: Literal["inputs", "outputs"] = Field(
        default="inputs", description="Which side of the cab schema to look up params in."
    )
    params: list[str] = Field(
        description="Parameter names or fnmatch patterns, e.g. ['*', '*freq*', 'niter']."
    )


class GetCabParamsInput(BaseModel):
    fields: list[Literal["dtype", "info", "required", "default", "choices", "writable", "examples"]] = Field(
        default=[], description="Fields to return for every matched parameter. Empty means all fields."
    )
    cabs: dict[str, CabParamSpec] = Field(
        description="Cab name mapped to the section and parameter patterns to query in it."
    )


def _param_field_value(param: CabParam, field: str) -> str:
    """Return a string-serialised value for one field of a CabParam."""
    if field == "dtype":
        return param.dtype or "null"
    if field == "info":
        return param.info or "null"
    if field == "required":
        return str(param.required).lower()
    if field == "default":
        return str(param.default) if param.default is not None else "null"
    if field == "choices":
        return ";".join(param.choices) if param.choices else "null"
    if field == "writable":
        return str(param.writable).lower()
    # examples: not present on cab params
    return "null"


def get_cab_params(input: GetCabParamsInput) -> str:
    """Batch lookup of parameter fields across one or more cabs.

    Most token-efficient way to verify or compare many parameters across one
    or more cabs in a single call. ``cabs`` maps each cab name to a section
    and a list of fnmatch parameter patterns (``["*"]`` for all).

    Returns a count line followed by a CSV table: ``cab``, ``section``,
    ``param``, then the requested fields. A cab that does not exist is
    reported on its own ``# error`` line before the count line instead of
    sinking the batch; a pattern that matches nothing is silently skipped.

    Example:
      fields: ["dtype", "default", "choices"]
      cabs:
        wsclean: {section: inputs, params: ["ms", "niter", "auto-threshold"]}
        casa.bandpass: {section: inputs, params: ["*"]}
    """
    field_columns = [f for f in _AVAILABLE_FIELDS if not input.fields or f in set(input.fields)]
    columns = ["cab", "section", "param"] + field_columns

    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    shown_cabs: set[str] = set()

    for cab_name, spec in input.cabs.items():
        try:
            schema = load_cab_schema(cab_name)
        except ValueError:
            errors.append(f"# error: no cab '{cab_name}'")
            continue

        param_pool = schema.inputs if spec.section == "inputs" else schema.outputs

        # Expand fnmatch patterns, preserving schema order and deduplicating.
        seen: set[str] = set()
        matched: list[str] = []
        for pattern in spec.params:
            for param_name in fnmatch.filter(param_pool.keys(), pattern):
                if param_name not in seen:
                    matched.append(param_name)
                    seen.add(param_name)

        for param_name in matched:
            row: dict[str, Any] = {"cab": cab_name, "section": spec.section, "param": param_name}
            for field in field_columns:
                row[field] = _param_field_value(param_pool[param_name], field)
            rows.append(row)
        if matched:
            shown_cabs.add(cab_name)

    count_line = f"# showing {len(rows)} params across {len(shown_cabs)} cabs"
    return "\n".join([*errors, count_line, write_csv(rows, columns)])
