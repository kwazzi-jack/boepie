"""Tests for the MeasurementSet MCP tools: get_ms_summary, get_ms_info, get_ms_fields.

``data/meerkat.ms`` is a real (simulated) MeerKAT MeasurementSet checked into
the repo, so most tests load it directly through MSInfo.from_path rather than
synthesising a fixture. A couple of tests monkeypatch MSInfo.from_path where
the point is exercising the error path, not real casacore I/O.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastmcp import Client
from fastmcp.client.transports import FastMCPTransport

from boepie.ms import MSInfo
from boepie.tools import measurement_set
from boepie.tools.measurement_set import (
    GetMsFieldsInput,
    GetMsInfoInput,
    _expand_sections,
    _load,
    get_ms_fields,
    get_ms_info,
    get_ms_summary,
)

_MS_PATH = "data/meerkat.ms"


def _text(result: object) -> str:
    """Extract the first text item from an MCP call_tool result."""
    content = getattr(result, "content", None)
    if content:
        return str(getattr(content[0], "text", content[0]))
    return str(result)


@pytest.fixture(scope="module")
def ms_info() -> MSInfo:
    """Loaded MSInfo for the real MeerKAT fixture, shared across tests."""
    assert Path(_MS_PATH).is_dir(), f"expected MS fixture at {_MS_PATH}"
    return MSInfo.from_path(_MS_PATH)


def _group_names(report: str) -> list[str]:
    """Pull out the group-name lines (unindented, no colon) from an F3 report."""
    lines = report.splitlines()[1:]  # drop the "# <path>" title line
    return [line for line in lines if line and ":" not in line]


# ---------------------------------------------------------------------------
# Section alias expansion (physical / data), dedup, ordering
# ---------------------------------------------------------------------------


def test_physical_alias_expands_to_nine_groups_in_display_order():
    expanded = _expand_sections(["physical"])
    assert expanded == [
        "observation", "array", "frequency", "data_dimensions",
        "temporal", "spatial", "polarization", "data_quality", "uv_coverage",
    ]


def test_data_alias_expands_to_four_groups_in_display_order():
    expanded = _expand_sections(["data"])
    # Display order (from MSInfo.__display_groups__), not alias-declaration order.
    assert expanded == ["data_columns", "scan_state", "file_info", "software"]


def test_aliases_and_individual_keys_are_mixable_and_deduplicated():
    # "scan_state" is already covered by the "data" alias - no duplicate.
    expanded = _expand_sections(["data", "scan_state", "frequency"])
    assert expanded == ["frequency", "data_columns", "scan_state", "file_info", "software"]
    assert expanded.count("scan_state") == 1


def test_physical_and_individual_key_mix_preserves_display_order():
    expanded = _expand_sections(["physical", "scan_state"])
    assert expanded == [
        "observation", "array", "frequency", "data_dimensions",
        "temporal", "spatial", "polarization", "data_quality", "uv_coverage",
        "scan_state",
    ]


def test_get_ms_info_physical_alias_contains_only_physical_groups(ms_info: MSInfo):
    report = get_ms_info(GetMsInfoInput(ms_path=_MS_PATH, sections=["physical"]))
    groups = _group_names(report)
    assert groups == [
        "observation", "array", "frequency", "data_dimensions",
        "temporal", "spatial", "polarization", "data_quality", "uv_coverage",
    ]
    assert "data_columns" not in groups
    assert "scan_state" not in groups


def test_get_ms_info_data_alias_contains_only_data_groups(ms_info: MSInfo):
    report = get_ms_info(GetMsInfoInput(ms_path=_MS_PATH, sections=["data"]))
    groups = _group_names(report)
    # "software" (casa_version, history_summary) renders no rows for this
    # fixture and is omitted entirely, same as get_cab_schema omitting an
    # empty inputs/outputs section rather than printing a bare header.
    assert groups == ["data_columns", "scan_state", "file_info"]
    assert "observation" not in groups


def test_get_ms_info_omitted_sections_returns_everything_with_renderable_data(ms_info: MSInfo):
    report = get_ms_info(GetMsInfoInput(ms_path=_MS_PATH))
    groups = _group_names(report)
    # Every group except "software" has at least one non-null field in this
    # fixture; empty groups are omitted rather than printed with no rows.
    assert set(groups) == {g["key"] for g in MSInfo.__display_groups__} - {"software"}


def test_get_ms_info_individual_key_still_works(ms_info: MSInfo):
    report = get_ms_info(GetMsInfoInput(ms_path=_MS_PATH, sections=["array"]))
    assert _group_names(report) == ["array"]
    assert "number_of_antennas: 64" in report


# ---------------------------------------------------------------------------
# Unknown section error
# ---------------------------------------------------------------------------


def test_unknown_section_is_one_line_error_listing_aliases_and_keys():
    result = get_ms_info(GetMsInfoInput(ms_path=_MS_PATH, sections=["bogus"]))
    assert result.startswith("Error:")
    assert "\n" not in result
    assert "bogus" in result
    assert "physical" in result
    assert "data" in result
    assert "observation" in result  # an individual group key is still listed


def test_unknown_section_mixed_with_valid_alias_reports_only_the_bad_one():
    result = get_ms_info(GetMsInfoInput(ms_path=_MS_PATH, sections=["physical", "bogus"]))
    assert result.startswith("Error: unknown section(s) bogus.")


def test_unknown_section_short_circuits_before_touching_disk():
    # Bad ms_path + bad section: the section error must win without ever
    # trying (and failing) to load the MS.
    result = get_ms_info(GetMsInfoInput(ms_path="/no/such/ms", sections=["bogus"]))
    assert result.startswith("Error: unknown section(s) bogus.")


# ---------------------------------------------------------------------------
# get_ms_summary / get_ms_info: F3 shape
# ---------------------------------------------------------------------------


def test_get_ms_summary_title_line_echoes_given_path_not_absolute(ms_info: MSInfo):
    report = get_ms_summary(_MS_PATH)
    assert report.splitlines()[0] == f"# {_MS_PATH}"


def test_get_ms_summary_has_no_markdown_beyond_leading_hash(ms_info: MSInfo):
    report = get_ms_summary(_MS_PATH)
    body = "\n".join(report.splitlines()[1:])
    assert "##" not in body
    assert "**" not in body
    assert "---" not in body


def test_get_ms_summary_is_key_value_not_json(ms_info: MSInfo):
    report = get_ms_summary(_MS_PATH)
    assert "{" not in report.splitlines()[0]
    assert "telescope: meerkat" in report


def test_get_ms_info_title_line_echoes_given_path(ms_info: MSInfo):
    report = get_ms_info(GetMsInfoInput(ms_path=_MS_PATH, sections=["array"]))
    assert report.splitlines()[0] == f"# {_MS_PATH}"


def test_get_ms_info_has_no_markdown_beyond_leading_hash(ms_info: MSInfo):
    report = get_ms_info(GetMsInfoInput(ms_path=_MS_PATH, sections=["physical"]))
    body = "\n".join(report.splitlines()[1:])
    assert "##" not in body
    assert "**" not in body
    assert "---" not in body


# ---------------------------------------------------------------------------
# get_ms_fields: rename, batch shape, partial failure
# ---------------------------------------------------------------------------


def test_query_ms_fields_no_longer_exists():
    assert not hasattr(measurement_set, "query_ms_fields")
    assert not hasattr(measurement_set, "QueryMsFieldsInput")


def test_get_ms_fields_count_line_matches_get_cab_params_convention(ms_info: MSInfo):
    result = get_ms_fields(GetMsFieldsInput(ms_paths=[_MS_PATH], fields=["telescope"]))
    count_line = next(line for line in result.splitlines() if line.startswith("# showing"))
    assert count_line == "# showing 1 fields across 1 ms"


def test_get_ms_fields_returns_requested_values(ms_info: MSInfo):
    result = get_ms_fields(GetMsFieldsInput(ms_paths=[_MS_PATH], fields=["telescope", "number_of_antennas"]))
    lines = [line for line in result.splitlines() if line and not line.startswith("#")]
    assert lines[0] == "ms,field,value"
    assert f"{_MS_PATH},telescope,meerkat" in lines
    assert f"{_MS_PATH},number_of_antennas,64" in lines


def test_get_ms_fields_wildcard_pattern_matches_multiple_fields(ms_info: MSInfo):
    result = get_ms_fields(GetMsFieldsInput(ms_paths=[_MS_PATH], fields=["number_of_*"]))
    data_rows = [line for line in result.splitlines() if line.startswith(_MS_PATH)]
    assert len(data_rows) > 1
    assert all(",number_of_" in row for row in data_rows)


def test_get_ms_fields_partial_failure_reports_error_line_not_column(ms_info: MSInfo):
    result = get_ms_fields(
        GetMsFieldsInput(ms_paths=[_MS_PATH, "/no/such.ms"], fields=["telescope"])
    )
    error_line = next(line for line in result.splitlines() if line.startswith("# error"))
    assert "/no/such.ms" in error_line
    assert "\n" not in error_line  # single physical line

    header = next(line for line in result.splitlines() if line.startswith("ms,"))
    assert "error" not in header.split(",")  # no phantom error column

    count_line = next(line for line in result.splitlines() if line.startswith("# showing"))
    assert "1 fields across 1 ms" in count_line

    data_rows = [line for line in result.splitlines() if line.startswith(_MS_PATH)]
    assert len(data_rows) == 1
    assert not any("/no/such.ms" in line for line in data_rows)


def test_get_ms_fields_bad_pattern_is_silently_skipped(ms_info: MSInfo):
    result = get_ms_fields(GetMsFieldsInput(ms_paths=[_MS_PATH], fields=["zzz_no_such_field_*"]))
    count_line = next(line for line in result.splitlines() if line.startswith("# showing"))
    assert "0 fields across 0 ms" in count_line


# ---------------------------------------------------------------------------
# Errors: one line, well-formed, names a real fix
# ---------------------------------------------------------------------------


def test_missing_ms_is_one_line_error_with_fix_clause():
    result = get_ms_summary("/no/such/path.ms")
    assert result.startswith("Error:")
    assert "\n" not in result
    assert result.endswith("Check the path and try again.")


def test_load_collapses_multiline_exception_to_one_line(monkeypatch: pytest.MonkeyPatch):
    def _raise_multiline(_ms_path: str) -> MSInfo:
        raise ValueError("first line\nsecond line\n   third line with padding")

    monkeypatch.setattr(MSInfo, "from_path", staticmethod(_raise_multiline))
    result = _load("whatever.ms")
    assert isinstance(result, str)
    assert "\n" not in result
    assert "first line second line third line with padding" in result


def test_load_collapses_arbitrary_exception_to_one_line(monkeypatch: pytest.MonkeyPatch):
    def _raise_weird(_ms_path: str) -> MSInfo:
        raise RuntimeError("casacore blew up:\n  table locked\n  retry later")

    monkeypatch.setattr(MSInfo, "from_path", staticmethod(_raise_weird))
    result = _load("whatever.ms")
    assert isinstance(result, str)
    assert "\n" not in result
    assert result.startswith("failed to read MS 'whatever.ms':")


def test_get_ms_summary_wraps_load_error_with_error_prefix(monkeypatch: pytest.MonkeyPatch):
    def _raise(_ms_path: str) -> MSInfo:
        raise RuntimeError("boom\nwith a second line")

    monkeypatch.setattr(MSInfo, "from_path", staticmethod(_raise))
    result = get_ms_summary("whatever.ms")
    assert result.startswith("Error: failed to read MS 'whatever.ms':")
    assert "\n" not in result


# ---------------------------------------------------------------------------
# Tool registration (rename landed in server.py)
# ---------------------------------------------------------------------------


@pytest.fixture
async def boepie_mcp_client():
    from boepie.server import mcp as _mcp

    async with Client(_mcp) as client:
        yield client


async def test_server_exposes_get_ms_fields_not_query_ms_fields(
    boepie_mcp_client: Client[FastMCPTransport],
):
    tools = await boepie_mcp_client.list_tools()
    names = {tool.name for tool in tools}
    assert "get_ms_fields" in names
    assert "query_ms_fields" not in names
    assert "get_ms_physical" not in names
    assert "get_ms_data" not in names


async def test_get_ms_fields_reachable_through_mcp_client(
    boepie_mcp_client: Client[FastMCPTransport],
):
    result = await boepie_mcp_client.call_tool(
        "get_ms_fields",
        {"input": {"ms_paths": [_MS_PATH], "fields": ["telescope"]}},
    )
    text = _text(result)
    assert f"{_MS_PATH},telescope,meerkat" in text
