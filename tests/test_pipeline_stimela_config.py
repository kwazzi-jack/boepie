"""Tests for driving stimela's config machinery in-process.

The interesting cases here are all failure modes of using a CLI's internals
as a library: a global that must not accumulate state between calls, a
`sys.exit` that must not reach the MCP server, and a malformed definition
that must not take the rest of the catalogue with it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from boepie.config import PIPELINE_SOURCES
from boepie.pipeline import stimela_config as stimela_config_module
from boepie.pipeline.stimela_config import (
    StimelaConfigError,
    configured_sources,
    discover_installed_sources,
    loaded_config,
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
  steps:
    flag:
      cab: casa.flagdata
      info: "Flag the obvious"
    image:
      cab: wsclean
      params:
        ms: =recipe.ms
        prefix: img
        size: [1024, 1024]
        scale: 1asec
"""

BROKEN_CAB_YAML = """\
cabs:
  boepie-test-broken:
    info: "A cab whose parameter dtype does not exist"
    command: true
    inputs:
      bad:
        dtype: not-a-real-dtype
"""


@pytest.fixture
def recipe_file(tmp_path: Path) -> Path:
    path = tmp_path / "demo.yml"
    path.write_text(RECIPE_YAML)
    return path


# ---------------------------------------------------------------------------
# The global that must not accumulate
# ---------------------------------------------------------------------------


def test_a_layered_recipe_file_does_not_leak_into_later_calls(recipe_file: Path):
    """`load_recipe_files` writes into the `stimela.CONFIG` global.

    Without the copy-and-restore in `loaded_config`, one call's recipe file
    would stay visible to every call after it - and an MCP server makes many.
    """
    layered = loaded_config(source=str(recipe_file))
    assert "demo_image" in layered.recipe_names_all()

    afterwards = loaded_config()
    assert "demo_image" not in afterwards.recipe_names_all()


def test_the_base_config_is_reused_rather_than_reloaded(recipe_file: Path):
    """The cached base is what makes the ten-second load a one-off."""
    first = loaded_config()
    loaded_config(source=str(recipe_file))
    second = loaded_config()
    assert first.config is second.config


def test_a_recipe_file_sees_the_configured_libraries_too(recipe_file: Path):
    layered = loaded_config(source=str(recipe_file))
    assert layered.has_cab("wsclean")


def test_recipe_names_distinguishes_the_caller_s_file_from_the_libraries(
    recipe_file: Path,
):
    layered = loaded_config(source=str(recipe_file))
    assert layered.recipe_names == ["demo_image"]


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------


def test_a_missing_recipe_file_raises_rather_than_exiting():
    with pytest.raises(StimelaConfigError, match="recipe file not found"):
        loaded_config(source="/no/such/recipe.yml")


def test_a_malformed_recipe_file_raises_rather_than_exiting(tmp_path: Path):
    """stimela answers a bad recipe file with `sys.exit(2)`.

    Uncaught that is a `SystemExit`, which would take the MCP server down
    instead of returning an error to the caller.
    """
    path = tmp_path / "bad.yml"
    path.write_text("demo:\n  steps: 'this should be a mapping'\n")
    with pytest.raises(StimelaConfigError):
        loaded_config(source=str(path))


def test_a_broken_cab_does_not_empty_the_catalogue(tmp_path: Path):
    """One unfinalizable cab must not cost you the other sixty.

    `list_cabs` reads the raw config node, so a cab that `Cab(...)` rejects
    still appears in the listing; only asking for its schema fails.
    """
    path = tmp_path / "broken.yml"
    path.write_text(BROKEN_CAB_YAML)
    config = loaded_config(source=str(path))

    listed = {definition.name for definition in config.cab_definitions()}
    assert "boepie-test-broken" in listed
    assert "wsclean" in listed

    with pytest.raises(StimelaConfigError, match="boepie-test-broken"):
        config.finalized_cab("boepie-test-broken")

    # The healthy cabs alongside it are untouched.
    assert config.finalized_cab("wsclean").inputs


def test_an_unknown_cab_raises_key_error_not_a_config_error():
    """Two different problems: an absent cab is the caller's mistake."""
    with pytest.raises(KeyError):
        loaded_config().finalized_cab("no-such-cab")


# ---------------------------------------------------------------------------
# Tracing a recipe back to its file
# ---------------------------------------------------------------------------


def test_a_recipe_from_a_file_can_be_traced_back_to_it(recipe_file: Path):
    config = loaded_config(source=str(recipe_file))
    assert config.recipe_source_file("demo_image") == recipe_file


def test_an_unknown_recipe_has_no_source_file(recipe_file: Path):
    config = loaded_config(source=str(recipe_file))
    assert config.recipe_source_file("not-a-recipe") is None


# ---------------------------------------------------------------------------
# Discovering installed libraries
#
# `stimela doc otherlib.recipes::thing` is a runtime lookup against whatever
# the user typed, so a server reading only a configured list is blind to a
# library sitting installed in the same venv - and an agent cannot be
# expected to invent that spelling. Discovery is what makes the bare call
# work.
# ---------------------------------------------------------------------------


def _install_fake_library(root: Path, package: str) -> None:
    """Lay out a package that looks installed, without running an installer.

    A `recipes/` subpackage holding YAML plus a `.dist-info` carrying a
    RECORD, which is what `importlib.metadata` reads to list a
    distribution's files.
    """
    recipes = root / package / "recipes"
    recipes.mkdir(parents=True)
    (root / package / "__init__.py").touch()
    (recipes / "__init__.py").touch()
    (recipes / "tron.yml").write_text(
        "cabs:\n"
        "  tron-solve:\n"
        '    info: "TRON solver"\n'
        "    command: tron\n"
        "    inputs:\n"
        "      ms: {dtype: MS, required: true}\n"
        "\n"
        "tron:\n"
        '  info: "TRON self-calibration recipe"\n'
        "  steps:\n"
        "    solve:\n"
        "      cab: tron-solve\n"
        "      params: {ms: /data/obs.ms}\n"
    )
    dist_info = root / f"{package}-0.1.0.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {package}\nVersion: 0.1.0\n"
    )
    (dist_info / "RECORD").write_text(
        f"{package}/__init__.py,,\n"
        f"{package}/recipes/__init__.py,,\n"
        f"{package}/recipes/tron.yml,,\n"
    )


def test_discovery_finds_a_package_with_a_recipes_subpackage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _install_fake_library(tmp_path, "boepiefakelib")
    monkeypatch.syspath_prepend(str(tmp_path))

    assert "boepiefakelib.recipes::" in discover_installed_sources()


def test_discovery_finds_cult_cargo_by_its_manifest():
    """cult-cargo's own marker, and the reason no configuration is needed."""
    assert "cultcargo::" in discover_installed_sources()


def test_discovery_does_not_import_the_packages_it_finds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Reading distribution metadata cannot run a third party's import side
    effects; importing every candidate to test it could."""
    _install_fake_library(tmp_path, "boepieimportcanary")
    (tmp_path / "boepieimportcanary" / "__init__.py").write_text(
        "raise AssertionError('discovery imported this package')\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    assert "boepieimportcanary.recipes::" in discover_installed_sources()
    assert "boepieimportcanary" not in sys.modules


def test_discovery_can_be_switched_off(monkeypatch: pytest.MonkeyPatch):
    """The off switch is for pinning what an agent can see across runs."""
    monkeypatch.setattr(stimela_config_module, "PIPELINE_DISCOVER", False)

    assert configured_sources() == list(PIPELINE_SOURCES)


def test_configured_sources_does_not_repeat_a_discovered_one(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(stimela_config_module, "PIPELINE_SOURCES", ["cultcargo::"])

    sources = configured_sources()

    assert sources.count("cultcargo::") == 1
