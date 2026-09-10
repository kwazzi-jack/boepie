"""Tests for driving stimela's config machinery in-process.

The interesting cases here are all failure modes of using a CLI's internals
as a library: a global that must not accumulate state between calls, a
`sys.exit` that must not reach the MCP server, and a malformed definition
that must not take the rest of the catalogue with it.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from boepie.config import PIPELINE_SOURCES
from boepie.pipeline import stimela_config as stimela_config_module
from boepie.pipeline.cabs import list_cabs
from boepie.pipeline.recipe import ListRecipesInput, list_recipes
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


# ---------------------------------------------------------------------------
# One broken library must not empty the catalogue
#
# `load_recipe_files` is handed every source's files at once and exits on
# the first one it cannot parse, so a single unparseable YAML in a package
# that merely happens to be installed beside boepie took every other
# library down with it - reported from a real venv where `pfb_imaging`
# ships a `cabs.yml` whose `_include` cannot be resolved, and `list_cabs`
# answered with an error about a file the user had never heard of.
# ---------------------------------------------------------------------------


def _install_broken_library(root: Path, package: str) -> None:
    """A package that looks installed and whose YAML cannot be loaded.

    The `_include` names a file that is not there, which is the shape the
    live failure took: resolution finds the YAML, and only loading it fails.
    """
    recipes = root / package / "recipes"
    recipes.mkdir(parents=True)
    (root / package / "__init__.py").touch()
    (recipes / "__init__.py").touch()
    (recipes / "broken.yml").write_text(
        "_include:\n"
        "  - nosuchfile.yml::cabs.nothing\n"
        "\n"
        "broken-recipe:\n"
        '  info: "a recipe that cannot load"\n'
        "  steps:\n"
        "    nope:\n"
        "      cab: nosuchcab\n"
    )
    dist_info = root / f"{package}-0.1.0.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {package}\nVersion: 0.1.0\n"
    )
    (dist_info / "RECORD").write_text(
        f"{package}/__init__.py,,\n"
        f"{package}/recipes/__init__.py,,\n"
        f"{package}/recipes/broken.yml,,\n"
    )


@pytest.fixture(scope="module")
def broken_and_healthy_libraries(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[Path]:
    """A broken library and a healthy one, both discoverable.

    Module-scoped, and so is the load it provokes: the source tuple is the
    cache key, so every test that sees these two libraries shares one
    ten-second load rather than paying for its own.
    """
    root = tmp_path_factory.mktemp("libraries")
    _install_broken_library(root, "boepiebrokenlib")
    _install_fake_library(root, "boepiehealthylib")
    sys.path.insert(0, str(root))
    try:
        yield root
    finally:
        sys.path.remove(str(root))
        stimela_config_module._BASE_CONFIGS.clear()


def test_a_broken_discovered_library_does_not_empty_the_catalogue(
    broken_and_healthy_libraries: Path,
):
    """The library that fails is the only one lost."""
    config = loaded_config()

    assert "boepiebrokenlib.recipes::" in discover_installed_sources()
    assert "wsclean" in config.cab_names()
    assert "tron-solve" in config.cab_names()
    assert "tron" in config.recipe_names_all()


def test_a_skipped_library_is_named_rather_than_dropped_in_silence(
    broken_and_healthy_libraries: Path,
):
    """A library missing from the catalogue reads as one never installed."""
    config = loaded_config()

    skipped = {item.spec for item in config.skipped_sources}
    assert skipped == {"boepiebrokenlib.recipes::"}
    reason = config.skipped_sources[0].reason
    assert "broken.yml" in reason
    assert "nosuchfile.yml" in reason


def test_the_skip_is_reported_in_the_tools_own_output(
    broken_and_healthy_libraries: Path,
):
    """`list_cabs` and `list_recipes` are where an agent learns what exists,
    so a shortened catalogue has to say that it is one."""
    assert "# skipped boepiebrokenlib.recipes::" in list_cabs()
    assert "# skipped boepiebrokenlib.recipes::" in list_recipes(ListRecipesInput())


def test_a_skip_reason_is_capped_rather_than_pasted_whole(
    broken_and_healthy_libraries: Path,
):
    """scabha ends this message with stimela's whole config search path -
    ten directories of it, repeated, into MCP output that is charged for."""
    reason = loaded_config().skipped_sources[0].reason

    assert len(reason) <= stimela_config_module._MAX_SKIP_REASON + len(" ...")
    assert reason.endswith("...")


def test_a_broken_configured_source_is_an_error_not_a_skip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """`pipeline.sources` is the user's own instruction, `cultcargo::`
    included, so a broken one there has to be seen: 16 of cult-cargo's 37
    YAML files fail under scabha rc4, and skipping them would answer "what
    can flag data" with a catalogue quietly missing every `casa.*` cab."""
    _install_broken_library(tmp_path, "boepieconfiguredlib")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(
        stimela_config_module,
        "PIPELINE_SOURCES",
        ["boepieconfiguredlib.recipes::", *PIPELINE_SOURCES],
    )
    monkeypatch.setattr(stimela_config_module, "_BASE_CONFIGS", {})

    with pytest.raises(StimelaConfigError, match="boepieconfiguredlib"):
        loaded_config()


def test_a_failed_load_is_remembered_rather_than_repeated(
    monkeypatch: pytest.MonkeyPatch,
):
    """The load costs the same ten seconds whether it succeeds or fails, and
    nothing about an installed source changes while the process runs."""
    monkeypatch.setattr(
        stimela_config_module, "PIPELINE_SOURCES", ["boepienosuchlib.recipes::"]
    )
    monkeypatch.setattr(stimela_config_module, "_BASE_CONFIGS", {})

    with pytest.raises(StimelaConfigError) as first:
        loaded_config()
    with pytest.raises(StimelaConfigError) as second:
        loaded_config()

    assert first.value is second.value


# ---------------------------------------------------------------------------
# stdout belongs to the MCP wire, and nothing else may write to it
# ---------------------------------------------------------------------------


def test_a_pipeline_tool_writes_nothing_to_stdout() -> None:
    """The MCP stdio transport *is* stdout, so a stray log line breaks it.

    stimela's rich console is built as `Console(file=sys.stdout)` and logs
    `loaded full configuration from cache` plus one `loading manifest from
    <path>` per library. Before `_log_to_stderr`, the first `list_cabs` call
    over stdio put 567 bytes of that in the JSON-RPC stream and the client
    answered with `Invalid JSON: trailing characters`.

    Asserted in a subprocess against real stimela, because the whole point is
    what a fresh server process emits before it has answered anything.
    """
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "from boepie.pipeline.cabs import list_cabs; list_cabs()",
        ],
        capture_output=True,
        text=True,
        timeout=300,
        stdin=subprocess.DEVNULL,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == "", (
        f"a pipeline tool wrote {len(completed.stdout)} bytes to stdout, which "
        f"is the MCP JSON-RPC channel:\n{completed.stdout[:500]}"
    )


def test_stimela_logging_is_pointed_at_stderr() -> None:
    """The mechanism, so a stimela release that renames it fails here first."""
    import stimela.stimelogging

    assert stimela.stimelogging.rich_console.file is sys.stderr
