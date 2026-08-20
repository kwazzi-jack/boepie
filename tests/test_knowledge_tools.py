"""Tests for the cab MCP tools: list_cabs, get_cab_docs, get_cab_schema, get_cab_params.

All tests run in-process via FastMCP's Client transport - no network, no subprocess.
"""

from __future__ import annotations

from fastmcp import Client
from fastmcp.client.transports import FastMCPTransport

from boepie.pipeline.runner import CabSchema


def _text(result: object) -> str:
    """Extract the first text item from a call_tool result."""
    content = getattr(result, "content", None)
    if content:
        return str(getattr(content[0], "text", content[0]))
    return str(result)


# ---------------------------------------------------------------------------
# list_cabs
# ---------------------------------------------------------------------------


async def test_list_cabs_returns_csv_with_all_cabs(boepie_client: Client[FastMCPTransport]):
    result = await boepie_client.call_tool("list_cabs", {})
    text = _text(result)
    assert "# showing" in text
    assert "cab,description" in text


async def test_list_cabs_count_header_matches_rows(
    boepie_client: Client[FastMCPTransport],
    all_cab_names: list[str],
):
    result = await boepie_client.call_tool("list_cabs", {})
    text = _text(result)
    lines = [line for line in text.splitlines() if not line.startswith("#")]
    data_rows = lines[1:]
    count_line = next(line for line in text.splitlines() if line.startswith("# showing"))
    shown = int(count_line.split()[2])
    assert shown == len(data_rows)
    assert shown == len(all_cab_names)
    for cab_name in all_cab_names:
        assert cab_name in text


async def test_list_cabs_pattern_filters_to_matching_cabs(boepie_client: Client[FastMCPTransport]):
    result = await boepie_client.call_tool("list_cabs", {"pattern": "wsclean"})
    text = _text(result)
    data_lines = [
        line for line in text.splitlines()
        if line and not line.startswith("#") and line != "cab,description"
    ]
    assert len(data_lines) == 1
    assert data_lines[0].startswith("wsclean,")


async def test_list_cabs_wildcard_pattern(
    boepie_client: Client[FastMCPTransport],
    casa_cab_names: list[str],
):
    result = await boepie_client.call_tool("list_cabs", {"pattern": "casa.*"})
    text = _text(result)
    data_lines = [
        line for line in text.splitlines()
        if line and not line.startswith("#") and line != "cab,description"
    ]
    assert len(data_lines) == len(casa_cab_names)
    for cab_name in casa_cab_names:
        assert cab_name in text


async def test_list_cabs_mid_string_wildcard_pattern(
    boepie_client: Client[FastMCPTransport],
    pfb_cab_names: list[str],
):
    result = await boepie_client.call_tool("list_cabs", {"pattern": "*pfb*"})
    text = _text(result)
    data_lines = [
        line for line in text.splitlines()
        if line and not line.startswith("#") and line != "cab,description"
    ]
    assert len(data_lines) == len(pfb_cab_names)
    for cab_name in pfb_cab_names:
        assert cab_name in text


async def test_list_cabs_single_char_wildcard_pattern(
    boepie_client: Client[FastMCPTransport],
    pfb_three_char_suffix_cab_names: list[str],
):
    result = await boepie_client.call_tool("list_cabs", {"pattern": "pfb.???"})
    text = _text(result)
    data_lines = [
        line for line in text.splitlines()
        if line and not line.startswith("#") and line != "cab,description"
    ]
    assert len(data_lines) == len(pfb_three_char_suffix_cab_names)
    for cab_name in pfb_three_char_suffix_cab_names:
        assert cab_name in text


async def test_list_cabs_character_class_pattern(
    boepie_client: Client[FastMCPTransport],
    wq_cab_names: list[str],
):
    result = await boepie_client.call_tool("list_cabs", {"pattern": "[wq]*"})
    text = _text(result)
    data_lines = [
        line for line in text.splitlines()
        if line and not line.startswith("#") and line != "cab,description"
    ]
    assert len(data_lines) == len(wq_cab_names)
    for cab_name in wq_cab_names:
        assert cab_name in text


async def test_list_cabs_negated_character_class_pattern(
    boepie_client: Client[FastMCPTransport],
    non_c_cab_names: list[str],
):
    result = await boepie_client.call_tool("list_cabs", {"pattern": "[!c]*"})
    text = _text(result)
    data_lines = [
        line for line in text.splitlines()
        if line and not line.startswith("#") and line != "cab,description"
    ]
    assert len(data_lines) == len(non_c_cab_names)
    for cab_name in non_c_cab_names:
        assert cab_name in text


async def test_list_cabs_nonexistent_pattern_returns_message(boepie_client: Client[FastMCPTransport]):
    result = await boepie_client.call_tool("list_cabs", {"pattern": "zzz_no_such_cab_xyz"})
    text = _text(result)
    assert "No cabs match" in text
    assert "zzz_no_such_cab_xyz" in text


# ---------------------------------------------------------------------------
# get_cab_docs
# ---------------------------------------------------------------------------


async def test_get_cab_docs_returns_compact_text_for_known_cab(boepie_client: Client[FastMCPTransport]):
    result = await boepie_client.call_tool("get_cab_docs", {"input": {"cab_name": "wsclean"}})
    text = _text(result)
    assert "wsclean" in text
    # to_compact() always prints the cab name first and sections for inputs/outputs
    assert "inputs" in text.lower() or "outputs" in text.lower()


async def test_get_cab_docs_unknown_cab_returns_error_string(boepie_client: Client[FastMCPTransport]):
    result = await boepie_client.call_tool("get_cab_docs", {"input": {"cab_name": "not_a_real_cab"}})
    text = _text(result)
    assert text.startswith("Error:")
    assert "not_a_real_cab" in text


async def test_get_cab_docs_params_filter_keeps_only_named_params(
    boepie_client: Client[FastMCPTransport],
    wsclean_optional_input_names: list[str],
):
    param = wsclean_optional_input_names[0]
    result = await boepie_client.call_tool(
        "get_cab_docs",
        {"input": {"cab_name": "wsclean", "params": [param]}},
    )
    text = _text(result)
    assert param in text


async def test_get_cab_docs_params_filter_excludes_other_params(
    boepie_client: Client[FastMCPTransport],
    wsclean_optional_input_names: list[str],
    wsclean_required_input_names: list[str],
):
    filter_param = wsclean_optional_input_names[0]
    excluded_param = wsclean_required_input_names[0]
    result = await boepie_client.call_tool(
        "get_cab_docs",
        {"input": {"cab_name": "wsclean", "params": [filter_param]}},
    )
    text = _text(result)
    param_section = text.split("wsclean", 1)[-1]  # skip the cab name header line
    assert filter_param in param_section
    assert f"  {excluded_param}" not in param_section


async def test_get_cab_docs_nonexistent_param_filter_returns_cab_header_only(
    boepie_client: Client[FastMCPTransport],
):
    result = await boepie_client.call_tool(
        "get_cab_docs",
        {"input": {"cab_name": "wsclean", "params": ["zzz_no_such_param"]}},
    )
    text = _text(result)
    # No error - just the cab name with no parameters listed
    assert not text.startswith("Error:")
    assert "wsclean" in text
    assert "zzz_no_such_param" not in text


# ---------------------------------------------------------------------------
# get_cab_schema
# ---------------------------------------------------------------------------


async def test_get_cab_schema_all_sections_contains_inputs_and_outputs(
    boepie_client: Client[FastMCPTransport],
):
    result = await boepie_client.call_tool(
        "get_cab_schema",
        {"input": {"cab_name": "wsclean", "section": "all"}},
    )
    text = _text(result)
    assert "wsclean inputs" in text
    assert "param,dtype" in text


async def test_get_cab_schema_inputs_section_excludes_outputs_header(
    boepie_client: Client[FastMCPTransport],
):
    result = await boepie_client.call_tool(
        "get_cab_schema",
        {"input": {"cab_name": "wsclean", "section": "inputs"}},
    )
    text = _text(result)
    assert "inputs" in text
    assert "wsclean outputs" not in text


async def test_get_cab_schema_outputs_section_excludes_inputs_header(
    boepie_client: Client[FastMCPTransport],
):
    result = await boepie_client.call_tool(
        "get_cab_schema",
        {"input": {"cab_name": "wsclean", "section": "outputs"}},
    )
    text = _text(result)
    assert "wsclean inputs" not in text


async def test_get_cab_schema_unknown_cab_returns_error_string(boepie_client: Client[FastMCPTransport]):
    result = await boepie_client.call_tool(
        "get_cab_schema",
        {"input": {"cab_name": "definitely_not_a_cab", "section": "all"}},
    )
    text = _text(result)
    assert text.startswith("Error:")
    assert "definitely_not_a_cab" in text


async def test_get_cab_schema_required_params_include_writable_column(
    boepie_client: Client[FastMCPTransport],
    wsclean_required_input_names: list[str],
):
    result = await boepie_client.call_tool(
        "get_cab_schema",
        {"input": {"cab_name": "wsclean", "section": "inputs"}},
    )
    text = _text(result)
    assert "param,dtype,writable" in text
    for param_name in wsclean_required_input_names:
        assert param_name in text


# ---------------------------------------------------------------------------
# get_cab_params
# ---------------------------------------------------------------------------


def _csv_lines(text: str) -> list[str]:
    """Strip '#' comment lines (error lines, count line), keep header + data rows."""
    return [line for line in text.splitlines() if line and not line.startswith("#")]


async def test_get_cab_params_returns_csv_with_identifier_columns(
    boepie_client: Client[FastMCPTransport],
):
    result = await boepie_client.call_tool(
        "get_cab_params",
        {
            "input": {
                "fields": ["dtype", "required"],
                "cabs": {"wsclean": {"section": "inputs", "params": ["niter"]}},
            }
        },
    )
    text = _text(result)
    header = _csv_lines(text)[0]
    assert header.startswith("cab,section,param")
    assert "dtype" in header
    assert "required" in header


async def test_get_cab_params_count_line_present(boepie_client: Client[FastMCPTransport]):
    result = await boepie_client.call_tool(
        "get_cab_params",
        {
            "input": {
                "fields": ["dtype"],
                "cabs": {"wsclean": {"section": "inputs", "params": ["niter"]}},
            }
        },
    )
    text = _text(result)
    count_line = next(line for line in text.splitlines() if line.startswith("# showing"))
    assert "1 params across 1 cabs" in count_line


async def test_get_cab_params_empty_fields_returns_all_available_columns(
    boepie_client: Client[FastMCPTransport],
):
    result = await boepie_client.call_tool(
        "get_cab_params",
        {
            "input": {
                "fields": [],
                "cabs": {"wsclean": {"section": "inputs", "params": ["niter"]}},
            }
        },
    )
    text = _text(result)
    header = _csv_lines(text)[0]
    for field in ("dtype", "info", "required", "default", "choices", "writable"):
        assert field in header


async def test_get_cab_params_data_row_contains_correct_cab_and_param(
    boepie_client: Client[FastMCPTransport],
    wsclean_optional_input_names: list[str],
):
    param = wsclean_optional_input_names[0]
    result = await boepie_client.call_tool(
        "get_cab_params",
        {
            "input": {
                "fields": ["dtype"],
                "cabs": {"wsclean": {"section": "inputs", "params": [param]}},
            }
        },
    )
    text = _text(result)
    lines = _csv_lines(text)
    assert len(lines) == 2  # header + one data row
    assert lines[1].startswith(f"wsclean,inputs,{param}")


async def test_get_cab_params_wildcard_returns_multiple_rows(
    boepie_client: Client[FastMCPTransport],
    wsclean_schema: CabSchema,
):
    result = await boepie_client.call_tool(
        "get_cab_params",
        {
            "input": {
                "fields": ["dtype"],
                "cabs": {"wsclean": {"section": "inputs", "params": ["*"]}},
            }
        },
    )
    text = _text(result)
    data_rows = _csv_lines(text)[1:]  # skip header
    assert len(data_rows) == len(wsclean_schema.inputs)


async def test_get_cab_params_no_match_pattern_produces_header_only(
    boepie_client: Client[FastMCPTransport],
):
    result = await boepie_client.call_tool(
        "get_cab_params",
        {
            "input": {
                "fields": ["dtype"],
                "cabs": {"wsclean": {"section": "inputs", "params": ["zzz_no_match_*"]}},
            }
        },
    )
    text = _text(result)
    lines = _csv_lines(text)
    assert len(lines) == 1
    assert lines[0].startswith("cab,section,param")
    count_line = next(line for line in text.splitlines() if line.startswith("# showing"))
    assert "0 params across 0 cabs" in count_line


async def test_get_cab_params_bad_cab_produces_error_line_not_column(
    boepie_client: Client[FastMCPTransport],
):
    result = await boepie_client.call_tool(
        "get_cab_params",
        {
            "input": {
                "fields": ["dtype"],
                "cabs": {"not_a_real_cab": {"section": "inputs", "params": ["ms"]}},
            }
        },
    )
    text = _text(result)
    error_line = next(line for line in text.splitlines() if line.startswith("# error"))
    assert "not_a_real_cab" in error_line
    # No phantom 'error' CSV column when every cab failed.
    header = _csv_lines(text)[0]
    assert "error" not in header.split(",")


async def test_get_cab_params_bad_cab_mixed_with_good_cab(
    boepie_client: Client[FastMCPTransport],
):
    result = await boepie_client.call_tool(
        "get_cab_params",
        {
            "input": {
                "fields": ["dtype"],
                "cabs": {
                    "wsclean": {"section": "inputs", "params": ["niter"]},
                    "not_a_real_cab": {"section": "inputs", "params": ["ms"]},
                },
            }
        },
    )
    text = _text(result)
    # Bad cab reported on its own error line, not mixed into the CSV.
    error_line = next(line for line in text.splitlines() if line.startswith("# error"))
    assert "not_a_real_cab" in error_line
    # Good cab's row still present in the table.
    data_rows = _csv_lines(text)[1:]
    assert any(row.startswith("wsclean,inputs,niter") for row in data_rows)
    assert not any("not_a_real_cab" in row for row in data_rows)


async def test_get_cab_params_multiple_patterns_deduplicated(
    boepie_client: Client[FastMCPTransport],
    wsclean_optional_input_names: list[str],
):
    # An exact name and a wildcard that also matches it should not produce a duplicate row.
    param = wsclean_optional_input_names[0]
    prefix_pattern = param[0] + "*"
    result = await boepie_client.call_tool(
        "get_cab_params",
        {
            "input": {
                "fields": ["dtype"],
                "cabs": {"wsclean": {"section": "inputs", "params": [param, prefix_pattern]}},
            }
        },
    )
    text = _text(result)
    data_rows = [
        line for line in _csv_lines(text)[1:]
        if line.startswith(f"wsclean,inputs,{param},")
    ]
    assert len(data_rows) == 1
