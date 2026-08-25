"""Tests for the cab schema layer, over the real installed stimela libraries.

These run against whatever cult-cargo is installed rather than a fixture,
because the thing worth testing is exactly the boundary boepie used to get
wrong: how stimela's own `Cab.finalize` presents a cab, versus how boepie
reads it. A fixture would have agreed with the broken implementation just as
happily as with the fixed one.

Loading the config costs about ten seconds cold, so the session-scoped
fixture below pays it once for the whole file.
"""

from __future__ import annotations

import itertools

import pytest

from boepie.pipeline.cabs import (
    CabParamSpec,
    GetCabDocsInput,
    GetCabParamsInput,
    GetCabSchemaInput,
    get_cab_docs,
    get_cab_params,
    get_cab_schema,
    list_cabs,
)
from boepie.pipeline.runner import load_cab_schema
from boepie.pipeline.stimela_config import loaded_config


@pytest.fixture(scope="session")
def library_config():
    """The configured stimela libraries, loaded once for the whole session."""
    return loaded_config()


# ---------------------------------------------------------------------------
# Nested parameter groups
#
# The regression this module exists for: boepie's own YAML reader skipped any
# parameter whose value was a mapping of mappings, calling it a "namespace
# container". Those are stimela's nested parameter groups, and `Cab.finalize`
# flattens them to dotted names. quartical came back with zero inputs.
# ---------------------------------------------------------------------------


def test_quartical_exposes_its_nested_parameter_groups(library_config):
    schema = load_cab_schema("quartical", config=library_config)
    assert schema.inputs, "quartical resolved to an empty schema"
    assert "input_ms.path" in schema.inputs


def test_a_nested_group_member_keeps_its_dtype_and_description(library_config):
    schema = load_cab_schema("quartical", config=library_config)
    param = schema.inputs["input_ms.data_column"]
    assert param.dtype == "str"
    assert param.info


def test_wsclean_exposes_its_multi_group(library_config):
    schema = load_cab_schema("wsclean", detail="obscure", config=library_config)
    assert "multi.chan" in schema.inputs


def test_a_cab_that_declares_inputs_never_projects_to_an_empty_schema(
    library_config,
):
    """No cab loses every parameter on the way through the projection.

    An empty schema was how the old reader failed - silently, and only for
    the cabs whose parameters were grouped, so a spot check would have
    missed it. The comparison is against the raw config node rather than a
    fixed list, so this keeps holding as cult-cargo changes.

    Checked at `hidden`: a cab whose only parameter is implicit
    (`astropy.test-host-cache`) is correctly empty at the default ceiling.
    A cab that declares no inputs at all (`astropy.test-internal-cache`, a
    parameterless diagnostic) is not a counterexample and is skipped.
    """
    empty = []
    for definition in library_config.cab_definitions():
        declared = library_config.config.cabs[definition.name].get("inputs")
        if not declared:
            continue
        try:
            schema = load_cab_schema(
                definition.name, detail="hidden", config=library_config
            )
        except Exception:
            # A cab stimela itself rejects is a separate concern, covered in
            # test_pipeline_stimela_config.py.
            continue
        if not schema.inputs:
            empty.append(definition.name)
    assert not empty, f"cabs that declare inputs but project to none: {empty}"


# ---------------------------------------------------------------------------
# Detail levels
# ---------------------------------------------------------------------------


def test_detail_levels_are_nested(library_config):
    levels = ["required", "optional", "implicit", "obscure", "hidden"]
    seen: list[set[str]] = [
        set(load_cab_schema("wsclean", detail=level, config=library_config).inputs)
        for level in levels
    ]
    for narrower, wider in itertools.pairwise(seen):
        assert narrower <= wider


def test_default_detail_hides_obscure_parameters(library_config):
    default = load_cab_schema("wsclean", config=library_config)
    assert "multi.chan" not in default.inputs


def test_required_detail_keeps_only_required_inputs(library_config):
    schema = load_cab_schema("wsclean", detail="required", config=library_config)
    assert schema.inputs
    assert all(param.required for param in schema.inputs.values())


# ---------------------------------------------------------------------------
# Projection details that leak into output when they go wrong
# ---------------------------------------------------------------------------


def test_an_unset_default_is_none_not_the_scabha_sentinel(library_config):
    """scabha marks 'no default' with the UNSET *class*, not None.

    Read naively it renders into MCP output as the literal text
    `<class 'scabha.basetypes.UNSET'>`.
    """
    schema = load_cab_schema("wsclean", detail="hidden", config=library_config)
    leaked = [
        name for name, param in schema.inputs.items() if "UNSET" in str(param.default)
    ]
    assert not leaked


def test_an_implicit_output_reports_the_value_it_takes(library_config):
    schema = load_cab_schema("wsclean", detail="implicit", config=library_config)
    assert schema.outputs["source-list"].implicit


def test_every_parameter_carries_a_category(library_config):
    schema = load_cab_schema("wsclean", detail="hidden", config=library_config)
    categories = {param.category for param in schema.inputs.values()}
    assert categories <= {"required", "optional", "implicit", "obscure", "hidden"}
    assert "required" in categories


# ---------------------------------------------------------------------------
# Tool-level behaviour
# ---------------------------------------------------------------------------


def test_list_cabs_reports_the_full_catalogue():
    output = list_cabs()
    assert "wsclean" in output
    assert "quartical" in output


def test_get_cab_schema_reports_an_unknown_cab_without_raising():
    output = get_cab_schema(GetCabSchemaInput(cab_name="no-such-cab"))
    assert output.startswith("Error:")
    assert "list_cabs" in output


def test_get_cab_docs_restricted_to_params_returns_only_those():
    output = get_cab_docs(
        GetCabDocsInput(cab_name="wsclean", params=["ms", "niter"])
    )
    assert "ms" in output
    assert "niter" in output
    assert "auto-threshold" not in output


def test_get_cab_docs_reaches_a_nested_param_by_its_dotted_name():
    output = get_cab_docs(
        GetCabDocsInput(cab_name="quartical", params=["input_ms.path"])
    )
    assert "input_ms.path" in output


def test_get_cab_params_reports_category_rather_than_a_null_examples_column():
    output = get_cab_params(
        GetCabParamsInput(
            fields=["dtype", "category"],
            cabs={"wsclean": CabParamSpec(section="inputs", params=["ms"])},
        )
    )
    assert "cab,section,param,dtype,category" in output
    assert "examples" not in output
    assert "wsclean,inputs,ms," in output


def test_get_cab_params_reports_a_bad_cab_without_sinking_the_batch():
    output = get_cab_params(
        GetCabParamsInput(
            fields=["dtype"],
            cabs={
                "no-such-cab": CabParamSpec(section="inputs", params=["*"]),
                "wsclean": CabParamSpec(section="inputs", params=["ms"]),
            },
        )
    )
    assert "# error: no cab 'no-such-cab'" in output
    assert "wsclean,inputs,ms" in output
