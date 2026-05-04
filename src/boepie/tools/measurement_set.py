"""MCP tools for inspecting CASA MeasurementSets.

Thin wrappers around :class:`boepie.ms.MSInfo` that let the AI pull
structured metadata (physical parameters, data/file parameters, and
arbitrary field queries) for one or more MeasurementSets on disk without
shelling out to casacore directly.
"""

from __future__ import annotations

import fnmatch
import json
from typing import Any

from pydantic import BaseModel

from boepie.ms import MSInfo
from boepie.utilities import write_csv


# Valid section keys from MSInfo.__display_groups__, grouped into two
# buckets the AI can ask for by intent. "physical" covers observational,
# array, frequency, temporal, spatial, polarization, data quality, and UV
# coverage. "data" covers columns, file info, software/history, and
# scan/state bookkeeping.
_PHYSICAL_SECTIONS = [
    "observation", "array", "frequency", "data_dimensions",
    "temporal", "spatial", "polarization", "data_quality", "uv_coverage",
]
_DATA_SECTIONS = [
    "data_columns", "file_info", "software", "scan_state",
]
_ALL_SECTION_KEYS = {group["key"] for group in MSInfo.__display_groups__}


def _queryable_field_names() -> list[str]:
    """Every attribute on MSInfo that a query can project onto.

    Combines regular model fields with pydantic's computed fields so that
    derived values like ``observing_frequency_ghz`` and ``total_visibilities``
    are queryable the same way as raw ones.
    """
    return list(MSInfo.model_fields.keys()) + list(MSInfo.model_computed_fields.keys())


_QUERYABLE_FIELDS: list[str] = _queryable_field_names()


def _load(ms_path: str) -> MSInfo | str:
    """Load an MSInfo or return a human-readable error string."""
    try:
        return MSInfo.from_path(ms_path)
    except FileNotFoundError as exc:
        return f"Error: {exc}"
    except ValueError as exc:
        return f"Error: {exc}"
    except Exception as exc:
        return f"Error: failed to read MS '{ms_path}': {exc}"


def get_ms_physical(ms_path: str) -> str:
    """Inspect the physical parameters of a MeasurementSet.

    Returns a markdown document covering observation, array configuration,
    frequency setup, temporal coverage, pointing, polarization, data
    quality, and UV coverage. Use this for "what was observed".

    Parameters
    ----------
    ms_path:
        Filesystem path to the MeasurementSet directory.
    """
    info = _load(ms_path)
    if isinstance(info, str):
        return info
    return info.to_markdown(sections=_PHYSICAL_SECTIONS)


def get_ms_data(ms_path: str) -> str:
    """Inspect the data and file-level parameters of a MeasurementSet.

    Returns a markdown document covering available visibility columns, the
    main-table column list, on-disk size, creation and last-modified
    timestamps, CASA software version, history, and scan/state bookkeeping.
    Use this for "what data is present and when was it last touched".

    Parameters
    ----------
    ms_path:
        Filesystem path to the MeasurementSet directory.
    """
    info = _load(ms_path)
    if isinstance(info, str):
        return info
    return info.to_markdown(sections=_DATA_SECTIONS)


def get_ms_info(ms_path: str, sections: list[str] | None = None) -> str:
    """Inspect arbitrary sections of a MeasurementSet as markdown.

    Returns a markdown report containing the requested display groups. If
    ``sections`` is omitted or empty, all groups are rendered.

    Parameters
    ----------
    ms_path:
        Filesystem path to the MeasurementSet directory.
    sections:
        Optional list of section keys to include. Valid keys:
        ``observation``, ``array``, ``frequency``, ``data_dimensions``,
        ``temporal``, ``spatial``, ``polarization``, ``data_columns``,
        ``data_quality``, ``uv_coverage``, ``scan_state``, ``file_info``,
        ``software``. Unknown keys produce an error.
    """
    if sections:
        unknown = [key for key in sections if key not in _ALL_SECTION_KEYS]
        if unknown:
            return (
                f"Error: unknown section(s) {unknown}. "
                f"Valid keys: {sorted(_ALL_SECTION_KEYS)}"
            )
    info = _load(ms_path)
    if isinstance(info, str):
        return info
    return info.to_markdown(sections=sections or None)


def get_ms_summary(ms_path: str) -> str:
    """Return a compact JSON summary of a MeasurementSet.

    Token-efficient quick look: telescope, observation date, antenna and
    field counts, frequency/bandwidth, available data columns, size on
    disk, and flagged fraction. Prefer this over ``get_ms_info`` /
    ``get_ms_physical`` / ``get_ms_data`` when you just need to confirm an
    MS is what you expect.

    Parameters
    ----------
    ms_path:
        Filesystem path to the MeasurementSet directory.
    """
    info = _load(ms_path)
    if isinstance(info, str):
        return info
    summary: dict[str, Any] = info.summary()
    return json.dumps(summary, indent=2, default=str)


class QueryMsFieldsInput(BaseModel):
    ms_paths: list[str]
    """One or more MeasurementSet paths to query."""
    fields: list[str]
    """MSInfo field names or fnmatch patterns (e.g. ['telescope', 'number_of_*', '*frequency*'])."""


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


def query_ms_fields(input: QueryMsFieldsInput) -> str:
    """Batch lookup of MSInfo field values across one or more MeasurementSets.

    Supply a list of ``ms_paths`` and a list of ``fields`` (exact names or
    fnmatch patterns over MSInfo field names). Returns a CSV with ``ms``,
    ``field``, ``value`` columns plus an ``error`` column when any path
    fails to load.

    Field names include everything on ``MSInfo``: raw fields like
    ``telescope``, ``number_of_antennas``, ``available_data_columns``, plus
    computed fields like ``observing_frequency_ghz``, ``total_visibilities``,
    ``angular_resolution_arcsec``. Patterns follow fnmatch syntax - use
    ``['*']`` for every field, ``['number_of_*']`` for counts,
    ``['*frequency*']`` for frequency-related fields.

    Use this when comparing the same fields across multiple MSes, or when
    projecting a small set of fields without pulling full markdown
    sections. A bad MS path produces an error row for that MS; patterns
    that match nothing are silently skipped.

    Example input:
      ms_paths: ["obs1.ms", "obs2.ms"]
      fields: ["telescope", "observing_frequency_ghz", "number_of_*"]
    """
    field_columns = _expand_field_patterns(input.fields)
    columns = ["ms", "field", "value"]
    rows: list[dict[str, Any]] = []
    has_error = False

    for ms_path in input.ms_paths:
        info = _load(ms_path)
        if isinstance(info, str):
            rows.append({"ms": ms_path, "field": "", "value": "", "error": info})
            has_error = True
            continue
        for field_name in field_columns:
            rows.append(
                {
                    "ms": ms_path,
                    "field": field_name,
                    "value": _serialise_field_value(getattr(info, field_name, None)),
                }
            )

    if has_error:
        columns.append("error")
    return write_csv(rows, columns)
