"""MCP tools for querying stimela cab definitions and documentation.

These tools let the AI discover what software is available through
cult-cargo and inspect their parameters via ``stimela doc``.
"""

from __future__ import annotations

import fnmatch
from typing import Any, Literal

from pydantic import BaseModel

from boepie.runner import CabParam, list_cabs_with_info, load_cab_schema
from boepie.utilities import write_csv


# Canonical field order for query_cab_params output.
_AVAILABLE_FIELDS: list[str] = [
    "dtype", "info", "required", "default", "choices", "writable", "examples",
]


def list_cabs(pattern: str | None = None) -> str:
    """List available cab (containerised application bundle) definitions.

    Returns a CSV table with ``cab`` and ``description`` columns for every
    tool available through cult-cargo. Use this to discover what tools are
    available before building a recipe.

    Parameters
    ----------
    pattern:
        Optional fnmatch-style pattern to narrow results (e.g. ``casa.*``,
        ``*clean*``, ``wsclean``). If omitted, all cabs are returned.
    """
    all_cabs = list_cabs_with_info()
    if not all_cabs:
        return "No cab definitions found. Is cult-cargo installed?"

    if pattern:
        all_cabs = [row for row in all_cabs if fnmatch.fnmatch(row["cab"], pattern)]
        if not all_cabs:
            return f"No cabs match the pattern '{pattern}'."

    total = len(list_cabs_with_info()) if pattern else len(all_cabs)
    header = f"# Showing {len(all_cabs)} of {total} cabs\n"
    return header + write_csv(all_cabs, ["cab", "description"])


class GetCabDocsInput(BaseModel):
    cab_name: str
    params: list[str] | None = None


def get_cab_docs(input: GetCabDocsInput) -> str:
    """Get full parameter documentation for a specific cab.

    Returns a detailed text summary of the cab's inputs and outputs,
    including descriptions, defaults and choices. Use this when you are
    learning a new tool. For quick name/type confirmation, use
    ``get_cab_schema`` instead.

    Supply ``params`` to restrict the output to specific parameters when you
    only have a few uncertainties and don't need the full cab docs.

    Parameters
    ----------
    input.cab_name:
        Name of the cab (e.g. "wsclean", "quartical", "casa.bandpass").
        Use ``list_cabs`` first to see available names.
    input.params:
        Optional list of parameter names to show (e.g. ["niter", "auto-threshold"]).
        If omitted, docs for all parameters are returned.
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
    cab_name: str
    section: Literal["inputs", "outputs", "all"] = "all"


def get_cab_schema(input: GetCabSchemaInput) -> str:
    """Get a compact CSV schema for a cab showing parameter names, types, and flags.

    Required inputs are listed first with ``param,dtype,writable`` columns, followed
    by optional inputs with ``param,dtype`` columns, then outputs with ``param,dtype``.
    No descriptions, defaults, or choices are included - use ``get_cab_docs`` for those.

    Use the ``section`` field to limit output to ``inputs``, ``outputs``, or ``all``
    (default). Useful when you only need to check one side.

    Parameters
    ----------
    input.cab_name:
        Name of the cab (e.g. "wsclean", "quartical", "casa.bandpass").
    input.section:
        Which sections to include - "inputs", "outputs", or "all" (default).
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
    section: Literal["inputs", "outputs"] = "inputs"
    params: list[str]
    """Parameter names or fnmatch patterns (e.g. ["*", "*freq*", "niter"])."""


class QueryCabParamsInput(BaseModel):
    fields: list[Literal["dtype", "info", "required", "default", "choices", "writable", "examples"]] = []
    """Fields to return for every matched parameter. Empty means all fields."""
    cabs: dict[str, CabParamSpec]


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


def query_cab_params(input: QueryCabParamsInput) -> str:
    """Batch lookup of parameter fields across one or more cabs.

    Supply a shared ``fields`` list and a ``cabs`` dict mapping each cab name
    to its section and a list of parameter name patterns. Patterns follow
    fnmatch syntax - use ``["*"]`` for all params, ``["*freq*", "*mem*"]`` for
    params whose names contain "freq" or "mem".

    Available fields: dtype, info, required, default, choices, writable, examples
    If ``fields`` is empty, all available fields are returned.

    Returns a single CSV table with ``cab``, ``section``, ``param`` as identifier
    columns followed by the requested fields. A bad cab name produces an error
    row for that cab; patterns that match nothing are silently skipped.

    Example input:
      fields: ["dtype", "default", "choices"]
      cabs:
        wsclean:         {section: mcp.run("stdio")inputs,  params: ["ms", "niter", "auto-threshold"]}
        msutils.renamecol: {section: outputs, params: ["dds-out"]}
        casa.bandpass:   {section: inputs,  params: ["*"]}
    """
    field_columns = [f for f in _AVAILABLE_FIELDS if not input.fields or f in set(input.fields)]
    columns = ["cab", "section", "param"] + field_columns

    rows: list[dict[str, Any]] = []
    has_error = False

    for cab_name, spec in input.cabs.items():
        try:
            schema = load_cab_schema(cab_name)
        except ValueError as error:
            rows.append({"cab": cab_name, "section": spec.section, "param": "", "error": str(error)})
            has_error = True
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

    if has_error:
        columns.append("error")

    return write_csv(rows, columns)
