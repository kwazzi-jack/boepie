"""MCP tools for inspecting CASA MeasurementSets.

Thin wrappers around :class:`boepie.ms.MSInfo` that let the AI pull
structured metadata for one or more MeasurementSets on disk without
shelling out to casacore directly.
"""

from __future__ import annotations

import fnmatch
import json
from typing import Any

from pydantic import BaseModel, Field

from boepie.ms import MSInfo
from boepie.utilities import write_csv


# Section aliases the AI can ask for by intent, expanding to the
# MSInfo.__display_groups__ keys they group. "physical" covers
# observational, array, frequency, temporal, spatial, polarization, data
# quality, and UV coverage. "data" covers columns, file info,
# software/history, and scan/state bookkeeping. Aliases and individual
# group keys are freely mixable in a single call.
_SECTION_ALIASES: dict[str, list[str]] = {
    "physical": [
        "observation", "array", "frequency", "data_dimensions",
        "temporal", "spatial", "polarization", "data_quality", "uv_coverage",
    ],
    "data": ["data_columns", "file_info", "software", "scan_state"],
}
_DISPLAY_ORDER: list[str] = [group["key"] for group in MSInfo.__display_groups__]
_ALL_SECTION_KEYS: set[str] = set(_DISPLAY_ORDER)
_VALID_SECTIONS_HINT = ", ".join([*_SECTION_ALIASES, *_DISPLAY_ORDER])


def _expand_sections(sections: list[str]) -> list[str]:
    """Expand alias entries, mix with individual keys, dedup, and order per
    ``MSInfo.__display_groups__``."""
    expanded: set[str] = set()
    for key in sections:
        expanded.update(_SECTION_ALIASES.get(key, [key]))
    return [key for key in _DISPLAY_ORDER if key in expanded]


def _queryable_field_names() -> list[str]:
    """Every attribute on MSInfo that a query can project onto.

    Combines regular model fields with pydantic's computed fields so that
    derived values like ``observing_frequency_ghz`` and ``total_visibilities``
    are queryable the same way as raw ones.
    """
    return list(MSInfo.model_fields.keys()) + list(MSInfo.model_computed_fields.keys())


_QUERYABLE_FIELDS: list[str] = _queryable_field_names()


def _collapse(text: str) -> str:
    """Flatten a possibly multi-line exception message onto one line."""
    return " ".join(str(text).split())


def _load(ms_path: str) -> MSInfo | str:
    """Load an MSInfo, or return a one-line failure reason.

    The returned string has no ``Error:`` prefix and no leading capital, so
    callers can embed it either as a standalone ``Error: {reason}`` payload
    or inline in a batch ``# error: {ms_path}: {reason}`` line.
    """
    try:
        return MSInfo.from_path(ms_path)
    except FileNotFoundError as exc:
        return f"{_collapse(str(exc))}. Check the path and try again."
    except ValueError as exc:
        return f"{_collapse(str(exc))}. Check the path and try again."
    except Exception as exc:
        return f"failed to read MS '{ms_path}': {_collapse(str(exc))}. Check the path and try again."


def get_ms_summary(ms_path: str) -> str:
    """Quick-glance report on a MeasurementSet.

    Covers telescope, observation date, antenna and field counts,
    frequency/bandwidth, available data columns, size on disk, and flagged
    fraction. Start here to confirm an MS is what you expect before
    reaching for ``get_ms_info`` or ``get_ms_fields``.

    Returns an F3 report: a title line naming the MS, then key: value lines.
    """
    info = _load(ms_path)
    if isinstance(info, str):
        return f"Error: {info}"
    return info.summary_report(title=ms_path)


class GetMsInfoInput(BaseModel):
    ms_path: str = Field(description="Filesystem path to the MeasurementSet directory.")
    sections: list[str] = Field(
        default=[],
        description=(
            "Section keys or aliases to include; omit or empty for everything. "
            "Aliases: 'physical' (observation, array, frequency, data_dimensions, "
            "temporal, spatial, polarization, data_quality, uv_coverage), 'data' "
            "(data_columns, file_info, software, scan_state). Individual group "
            "keys are also accepted and may be mixed with aliases."
        ),
    )


def get_ms_info(input: GetMsInfoInput) -> str:
    """Inspect selected sections of a MeasurementSet.

    Use for a structured look beyond ``get_ms_summary``: the physical setup
    (``sections=["physical"]``), file/bookkeeping data (``sections=["data"]``),
    any individual group key, or a mix of both.

    Returns an F3 report grouped by section; each section's keys are MSInfo
    field names, the same vocabulary ``get_ms_fields`` uses.
    """
    if input.sections:
        unknown = [key for key in input.sections if key not in _ALL_SECTION_KEYS and key not in _SECTION_ALIASES]
        if unknown:
            return f"Error: unknown section(s) {', '.join(unknown)}. Valid: {_VALID_SECTIONS_HINT}."
        selected = _expand_sections(input.sections)
    else:
        selected = None

    info = _load(input.ms_path)
    if isinstance(info, str):
        return f"Error: {info}"
    return info.to_report(sections=selected, title=input.ms_path)


class GetMsFieldsInput(BaseModel):
    ms_paths: list[str] = Field(description="One or more MeasurementSet paths to query.")
    fields: list[str] = Field(
        description="MSInfo field names or fnmatch patterns, e.g. ['telescope', 'number_of_*', '*frequency*']."
    )


def _serialise_field_value(value: Any) -> str:
    """Flatten one field value to a single CSV cell."""
    if value is None:
        return "null"
    if isinstance(value, (list, tuple)):
        return ";".join(str(item) for item in value) if value else "null"
    if isinstance(value, dict):
        return json.dumps(value, default=str)
    return str(value)


def _expand_field_patterns(patterns: list[str]) -> list[str]:
    """Expand fnmatch patterns against known MSInfo fields, preserving order.

    Duplicates are removed so the same field does not appear twice when a
    specific name and a wildcard both match it.
    """
    seen: set[str] = set()
    matched: list[str] = []
    for pattern in patterns:
        for field_name in fnmatch.filter(_QUERYABLE_FIELDS, pattern):
            if field_name not in seen:
                matched.append(field_name)
                seen.add(field_name)
    return matched


def get_ms_fields(input: GetMsFieldsInput) -> str:
    """Batch lookup of MSInfo field values across one or more MeasurementSets.

    Most token-efficient way to verify or compare many fields across one or
    more MeasurementSets in a single call. ``fields`` accepts exact MSInfo
    field names or fnmatch patterns (``["*"]`` for all), matched against both
    raw fields (``telescope``, ``number_of_antennas``) and computed ones
    (``observing_frequency_ghz``, ``total_visibilities``,
    ``angular_resolution_arcsec``).

    Returns a count line followed by a CSV table: ``ms``, ``field``,
    ``value``. An MS path that fails to load is reported on its own
    ``# error`` line before the count line instead of sinking the batch; a
    pattern that matches nothing is silently skipped.

    Example:
      ms_paths: ["obs1.ms", "obs2.ms"]
      fields: ["telescope", "observing_frequency_ghz", "number_of_*"]
    """
    field_columns = _expand_field_patterns(input.fields)
    columns = ["ms", "field", "value"]

    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    shown_ms: set[str] = set()

    for ms_path in input.ms_paths:
        info = _load(ms_path)
        if isinstance(info, str):
            errors.append(f"# error: {ms_path}: {info}")
            continue

        for field_name in field_columns:
            rows.append({
                "ms": ms_path,
                "field": field_name,
                "value": _serialise_field_value(getattr(info, field_name, None)),
            })
        if field_columns:
            shown_ms.add(ms_path)

    count_line = f"# showing {len(rows)} fields across {len(shown_ms)} ms"
    return "\n".join([*errors, count_line, write_csv(rows, columns)])
