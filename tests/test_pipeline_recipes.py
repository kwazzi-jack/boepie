"""Tests for the recipe discovery and inspection tools.

`validate_recipe` and `run_recipe` are covered in test_pipeline_tools.py,
where `stimela_run` is monkeypatched. These tools do not shell out at all -
they read the same resolved config the cab tools do - so they run against
the real installed libraries plus a fixture recipe file.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from boepie.pipeline.recipe import (
    GetRecipeDocsInput,
    ListRecipesInput,
    get_recipe_docs,
    list_recipes,
)

RECIPE_YAML = """\
_include:
  - (cultcargo)wsclean.yml

demo_image:
  info: "Demo imaging recipe"
  inputs:
    ms:
      dtype: MS
      required: true
    prefix:
      dtype: str
      default: img
  steps:
    flag:
      cab: casa.flagdata
      info: "Flag the obvious"
    image:
      cab: wsclean
      params:
        ms: =recipe.ms
        prefix: =recipe.prefix
        size: [1024, 1024]
        scale: 1asec
    skipped-step:
      cab: wsclean
      skip: true

second_recipe:
  info: "Another one, to check filtering"
  steps:
    image:
      cab: wsclean
"""


@pytest.fixture
def recipe_file(tmp_path: Path) -> str:
    path = tmp_path / "demo.yml"
    path.write_text(RECIPE_YAML)
    return str(path)


# ---------------------------------------------------------------------------
# list_recipes
# ---------------------------------------------------------------------------


def test_list_recipes_finds_the_recipes_in_a_file(recipe_file: str):
    output = list_recipes(ListRecipesInput(source=recipe_file))
    assert "demo_image" in output
    assert "second_recipe" in output
    assert "Demo imaging recipe" in output


def test_list_recipes_names_the_file_each_recipe_came_from(recipe_file: str):
    """The origin column is what lets a caller find the recipe's source.

    For a file it is the path; for an installed library it is the package
    spec (`breifast.recipes`), which is also the spelling `source=` takes -
    so an agent never has to invent stimela's `::` syntax.
    """
    output = list_recipes(ListRecipesInput(source=recipe_file))
    assert f"demo_image,Demo imaging recipe,{recipe_file}" in output


def test_list_recipes_narrows_by_pattern(recipe_file: str):
    output = list_recipes(
        ListRecipesInput(source=recipe_file, pattern="second*")
    )
    assert "second_recipe" in output
    assert "demo_image" not in output


def test_list_recipes_reports_a_pattern_that_matches_nothing(recipe_file: str):
    output = list_recipes(
        ListRecipesInput(source=recipe_file, pattern="nothing-like-this")
    )
    assert "No recipes match" in output


def test_list_recipes_explains_an_empty_library_set():
    """cult-cargo ships cabs and no recipes, so this is the default state.

    stimela itself ships neither - `load_config` sets `cab_configs = []` and
    the `cargo/` directory `CAB_PATH` names is not in the wheel - so with
    only cult-cargo installed there is genuinely nothing to list.
    """
    output = list_recipes(ListRecipesInput())
    assert "No recipes available" in output
    assert "source=" in output


def test_list_recipes_reports_a_missing_file_without_raising():
    output = list_recipes(ListRecipesInput(source="/no/such/file.yml"))
    assert output.startswith("Error:")
    assert "not found" in output


# ---------------------------------------------------------------------------
# get_recipe_docs: structured
# ---------------------------------------------------------------------------


def test_get_recipe_docs_reports_the_recipe_s_own_inputs(recipe_file: str):
    output = get_recipe_docs(
        GetRecipeDocsInput(
            recipe_name="demo_image", source=recipe_file, detail="required"
        )
    )
    assert "Demo imaging recipe" in output
    assert "ms (MS)" in output


def test_get_recipe_docs_lists_steps_in_declaration_order(recipe_file: str):
    output = get_recipe_docs(
        GetRecipeDocsInput(
            recipe_name="demo_image", source=recipe_file, detail="required"
        )
    )
    assert "1. flag (cab: casa.flagdata) - Flag the obvious" in output
    assert "2. image (cab: wsclean)" in output


def test_get_recipe_docs_marks_a_skipped_step(recipe_file: str):
    output = get_recipe_docs(
        GetRecipeDocsInput(
            recipe_name="demo_image", source=recipe_file, detail="required"
        )
    )
    assert "3. skipped-step (cab: wsclean) [skip]" in output


def test_get_recipe_docs_promotes_a_step_s_unmet_requirements(recipe_file: str):
    """A finalized recipe promotes each *required* step parameter it leaves
    unset to `{step}.{param}` on its own inputs.

    That is how a caller learns what a recipe still needs from outside. The
    `image` step supplies all four of wsclean's required parameters and so
    contributes nothing; `skipped-step` supplies none and contributes all
    four. Optional step parameters are not promoted - only unmet
    requirements are, which is what keeps the recipe's schema readable.
    """
    output = get_recipe_docs(
        GetRecipeDocsInput(recipe_name="demo_image", source=recipe_file)
    )
    assert "skipped-step.prefix" in output
    assert "flag.ms" in output
    assert "image.prefix" not in output
    assert "image.niter" not in output


def test_get_recipe_docs_reports_an_unknown_recipe(recipe_file: str):
    output = get_recipe_docs(
        GetRecipeDocsInput(recipe_name="nope", source=recipe_file)
    )
    assert output.startswith("Error:")
    assert "demo_image" in output


# ---------------------------------------------------------------------------
# get_recipe_docs: raw
# ---------------------------------------------------------------------------


def test_raw_returns_the_yaml_the_user_actually_wrote(recipe_file: str):
    output = get_recipe_docs(
        GetRecipeDocsInput(
            recipe_name="demo_image", source=recipe_file, raw=True
        )
    )
    assert recipe_file in output
    assert "=recipe.ms" in output
    assert "scale: 1asec" in output


def test_raw_does_not_re_serialise_the_resolved_config(recipe_file: str):
    """The resolved node renders every unset schema field as an explicit
    null, which is both far larger than the source and not what was
    written. The file's own bytes are the answer here."""
    output = get_recipe_docs(
        GetRecipeDocsInput(
            recipe_name="demo_image", source=recipe_file, raw=True
        )
    )
    assert "dynamic_schema: null" not in output
    assert "assign_based_on" not in output
