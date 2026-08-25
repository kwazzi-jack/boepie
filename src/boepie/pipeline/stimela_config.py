# boepie/pipeline/stimela_config.py
"""The one place boepie talks to stimela's configuration machinery.

boepie used to merge cult-cargo's YAML files itself, with its own glob list
standing in for that package's `MANIFEST.stimela` and its own reader
walking the merged tree. That reimplemented - badly - what stimela already
does in `stimela.commands.doc`, and the two diverged: the hand-rolled reader
treated a parameter whose value was a dict of dicts as a "namespace
container" and skipped it, but those namespaces are exactly stimela's
nested parameter groups. `quartical` came back with **zero** inputs,
`cubical` with one, and `wsclean` was missing its `multi.*` group.

So the chain here is stimela's own, in stimela's own order:

    config.load_config()          -> the base config (opts, images, lib, cabs)
    resolve_recipe_files(spec)    -> a source spec to a list of YAML paths
    load_recipe_files(paths)      -> merged into `stimela.CONFIG`
    Cab(...) / Recipe(...)        -> a schema, with `.finalize()` applied

`Cab.finalize` is what flattens `input_ms: {path: ..., data_column: ...}`
into `input_ms.path`, `input_ms.data_column`, resolves `_use` inheritance,
and assigns each parameter a `ParameterCategory`. There is no shortcut to
it that is worth taking.

Four things about driving that chain in-process rather than from the CLI:

- **`stimela.VERBOSE` must exist.** Only `stimela.main` sets it, and the
  python cab flavours read it during `Cab.__post_init__`, so an in-process
  caller that skips it gets `AttributeError: module 'stimela' has no
  attribute 'VERBOSE'` from inside flavour validation.
- **`load_recipe_files` calls `sys.exit(2)`** on a bad file rather than
  raising. Uncaught, that would take the MCP server down with it, so
  `_load_sources` catches `SystemExit` and turns it back into an exception,
  recovering stimela's own message from its logger (see `_CapturedLog`).
- **`stimela.CONFIG` is a module global** that `load_recipe_files` mutates
  in place. Layering a caller's recipe file onto it would leave that file's
  cabs and recipes visible to every later call, so `loaded_config` swaps in
  a deep copy for the duration and restores the base afterwards.
- **The scabha config cache must be pointed at stimela's directory.**
  `stimela.main` sets it; scabha's own default is elsewhere, so an
  in-process caller silently keeps a second cache. Worse, a *stale* cache
  is served in preference to changed files: after cult-cargo was upgraded
  from 0.2.0 to 0.2.1 this process kept parsing the 0.2.0 definitions, and
  all 17 `casa.*` cabs kept failing on a `pre_command` key that the new
  version had already renamed. `stimela -C` clears it.
"""

from __future__ import annotations

import copy
import functools
import importlib.metadata
import logging
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import scabha.configuratt.cache
import stimela
import yaml
from stimela import config as stimela_config
from stimela.commands.run import load_recipe_files, resolve_recipe_files
from stimela.kitchen.cab import Cab
from stimela.kitchen.recipe import Recipe

from boepie.config import (
    PIPELINE_DISCOVER,
    PIPELINE_SOURCES,
    STIMELA_CONFIG_CACHE_DIR,
)


class StimelaConfigError(Exception):
    """A stimela source could not be resolved, loaded, or finalized.

    Carries stimela's own wording where it could be recovered, since that
    text names the file and the offending key.
    """


@dataclass(frozen=True)
class CabDefinition:
    """One cab as it appears in the resolved config, before finalization.

    `list_cabs` only needs the name and the blurb, and reading those off the
    raw node costs nothing - where constructing a `Cab` costs ~70ms and can
    fail outright on a malformed definition. Keeping the two apart means one
    broken cab cannot empty the whole catalogue.
    """

    name: str
    info: str


class _CapturedLog(logging.Handler):
    """Collects stimela's log records so a `sys.exit` can be explained.

    stimela reports a bad recipe file by logging the reason and then calling
    `sys.exit(2)`; the exception carries only the exit code. Attaching this
    for the duration of a load is the only way to recover what it said.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())

    def summary(self) -> str:
        return "; ".join(self.messages) if self.messages else ""


# How an installed distribution advertises that it carries stimela YAML.
# There is no entry-point convention for this - cult-cargo declares only a
# console script - so these two file-level fingerprints are all there is to
# go on. Both are read out of distribution metadata, so discovery never
# imports anything; only a source that actually matched is handed to
# `resolve_recipe_files`, which does import it.
_PACKAGE_MANIFEST = "MANIFEST.stimela"
_RECIPES_SUBPACKAGE = "recipes"
_YAML_SUFFIXES = (".yml", ".yaml")


def discover_installed_sources() -> list[str]:
    """Every stimela source the installed environment appears to provide.

    Two fingerprints, matching how stimela itself resolves a source:
    a `MANIFEST.stimela` at a package root (cult-cargo's own marker, listing
    the globs to load) becomes `<package>::`, and a `recipes/` subpackage
    holding YAML becomes `<package>.recipes::` - which is the shape
    `breifast.recipes::tron` assumes.

    This exists because `stimela doc otherlib.recipes::thing` is a *runtime*
    lookup against whatever the user typed, so a server that only ever read
    a configured list would be blind to a library sitting installed in the
    same venv. Scanning distribution metadata costs about 0.2s and, unlike
    importing every candidate, cannot execute a third party's import side
    effects.
    """
    specs: set[str] = set()
    for distribution in importlib.metadata.distributions():
        for entry in distribution.files or []:
            parts = PurePosixPath(str(entry)).parts
            if len(parts) < 2:
                continue
            if parts[-1] == _PACKAGE_MANIFEST:
                specs.add(f"{parts[0]}::")
            elif (
                len(parts) >= 3
                and parts[1] == _RECIPES_SUBPACKAGE
                and parts[-1].endswith(_YAML_SUFFIXES)
            ):
                specs.add(f"{parts[0]}.{_RECIPES_SUBPACKAGE}::")
    return sorted(specs)


def configured_sources() -> list[str]:
    """Discovered sources plus the configured ones, deduplicated.

    Configured sources come last so an explicit entry can `_use` something a
    discovered library defined. `PIPELINE_SOURCES` keeps cult-cargo in the
    list even with discovery off, so turning discovery off degrades to the
    previous behaviour rather than to an empty catalogue.
    """
    ordered = discover_installed_sources() if PIPELINE_DISCOVER else []
    seen = set(ordered)
    return ordered + [spec for spec in PIPELINE_SOURCES if spec not in seen]


def _initialise_stimela() -> None:
    """Apply the process-wide setup `stimela.main` would otherwise do."""
    # Read by the python cab flavours during Cab construction. Set
    # unconditionally: another caller may have left it True.
    stimela.VERBOSE = False
    scabha.configuratt.cache.set_cache_dir(str(STIMELA_CONFIG_CACHE_DIR))


@functools.cache
def _base_config(sources: tuple[str, ...]) -> tuple[Any, tuple[str, ...]]:
    """The resolved stimela config with `sources` merged in.

    Cached per source tuple: loading cult-cargo alone parses 36 YAML files
    and takes roughly ten seconds, which is fine once per process and not
    fine per tool call. The cache holds the config used as the *base* for
    every call - `loaded_config` copies it before anything is layered on.
    """
    _initialise_stimela()
    stimela.CONFIG = stimela_config.load_config(extra_configs=[])
    if stimela.CONFIG is None:
        raise StimelaConfigError(
            "stimela could not load its base configuration. "
            "Run 'stimela -C' to clear the config cache and try again."
        )
    # Discovered sources are loaded tolerantly: a package that merely looks
    # like a stimela library (a `recipes/` directory of unrelated YAML, say)
    # must not be able to empty the catalogue for everything else. A
    # configured source is the user's explicit instruction, so a typo there
    # stays an error.
    discovered = set(discover_installed_sources()) if PIPELINE_DISCOVER else set()
    usable: list[str] = []
    for spec in sources:
        if spec not in discovered:
            usable.append(spec)
            continue
        try:
            _resolve_source(spec)
        except StimelaConfigError as error:
            stimela.logger().warning(f"skipping discovered source {spec}: {error}")
            continue
        usable.append(spec)

    _, paths = _load_sources(usable)
    return stimela.CONFIG, tuple(paths)


def _resolve_source(spec: str) -> list[str]:
    """One source spec to the YAML paths it names, or `StimelaConfigError`."""
    try:
        resolved = resolve_recipe_files(spec, log=stimela.logger())
    except FileNotFoundError as error:
        raise StimelaConfigError(f"{spec}: {error}") from error
    if resolved is None:
        raise StimelaConfigError(
            f"'{spec}' does not name a YAML file, directory, or importable "
            f"module. Library sources look like 'cultcargo::' or "
            f"'otherlib.recipes::'; a file source needs a .yml/.yaml suffix "
            f"or a path separator."
        )
    return resolved


def _load_sources(sources: list[str]) -> tuple[list[str], list[str]]:
    """Merge each source spec into the current `stimela.CONFIG`.

    Returns the names of any recipes the sources defined, and the YAML paths
    they resolved to. Sources are loaded in the order given, since a later
    one may `_use` an earlier one's definitions - the same reason stimela
    accumulates them into a single `load_recipe_files` call.
    """
    log = stimela.logger()
    paths: list[str] = []
    for spec in sources:
        paths.extend(_resolve_source(spec))

    if not paths:
        return [], []

    captured = _CapturedLog()
    log.addHandler(captured)
    try:
        recipe_names, _ = load_recipe_files(paths)
    except SystemExit as error:
        # stimela logs the reason, then exits. Uncaught this would kill the
        # MCP server, so it becomes an exception carrying what it logged.
        detail = captured.summary() or f"stimela exited with code {error.code}"
        raise StimelaConfigError(f"failed to load {', '.join(sources)}: {detail}") from error
    finally:
        log.removeHandler(captured)
    return list(recipe_names), paths


@dataclass(frozen=True)
class LoadedConfig:
    """A resolved stimela config, plus how the caller's own file contributed.

    `recipe_names` is what distinguishes a recipe the caller just supplied
    from one that came out of a configured library, which is what lets
    `list_recipes` say where each recipe came from. `source_paths` is every
    YAML the sources resolved to, kept so a recipe can be traced back to the
    file it was written in.
    """

    config: Any
    recipe_names: list[str]
    source_paths: list[str]

    def recipe_origins(self) -> dict[str, Path]:
        """Every recipe name mapped to the YAML file that declares it.

        stimela records no provenance for recipes - `configuratt.load` is
        given `include_path="_path"` for cabs but not for `lib.recipes` - so
        the files are re-read and their top-level keys inspected. Built once
        for the whole config rather than per lookup: `list_recipes` needs an
        origin for every row, and doing it per row would re-parse every
        source file per recipe.

        A recipe reached through an `_include` from a file that is not
        itself a source has no entry, since nothing here claims it.
        """
        origins: dict[str, Path] = {}
        for candidate in self.source_paths:
            path = Path(candidate)
            try:
                document = yaml.safe_load(path.read_text(encoding="utf-8"))
            except (OSError, yaml.YAMLError):
                continue
            if not isinstance(document, dict):
                continue
            for key in document:
                if key in self.config.lib.recipes:
                    origins.setdefault(str(key), path)
        return origins

    def recipe_source_file(self, name: str) -> Path | None:
        """The YAML file one recipe was defined in, or None."""
        return self.recipe_origins().get(name)

    def cab_definitions(self) -> list[CabDefinition]:
        """Every cab in the config, by name, without constructing any."""
        cabs = self.config.get("cabs", {})
        return [
            CabDefinition(name=name, info=str(cabs[name].get("info") or ""))
            for name in sorted(cabs.keys())
        ]

    def has_cab(self, name: str) -> bool:
        return name in self.config.get("cabs", {})

    def cab_names(self) -> list[str]:
        return sorted(self.config.get("cabs", {}).keys())

    def recipe_names_all(self) -> list[str]:
        return sorted(self.config.lib.recipes.keys())

    def finalized_cab(self, name: str) -> Cab:
        """Construct and finalize one cab.

        Raises `StimelaConfigError` rather than letting scabha's own
        exception out: a malformed cab is an upstream packaging bug the
        caller can do nothing about, and the message needs to say which cab
        it was.
        """
        cabs = self.config.get("cabs", {})
        if name not in cabs:
            raise KeyError(name)
        try:
            cab = Cab(**cabs[name])
            cab.finalize(config=self.config)
        except Exception as error:
            raise StimelaConfigError(
                f"cab '{name}' has a definition stimela rejects: "
                f"{_first_line(error)}"
            ) from error
        return cab

    def finalized_recipe(self, name: str) -> Recipe:
        """Construct and finalize one recipe from `lib.recipes`."""
        recipes = self.config.lib.recipes
        if name not in recipes:
            raise KeyError(name)
        section = recipes[name]
        # stimela's own `doc` does this: a recipe defined as a top-level YAML
        # mapping has no `name` of its own, and `Recipe` requires one.
        if not section.get("name"):
            section.name = name
        try:
            recipe = Recipe(**section)
            recipe.finalize(fqname=name)
        except Exception as error:
            raise StimelaConfigError(
                f"recipe '{name}' has a definition stimela rejects: "
                f"{_first_line(error)}"
            ) from error
        return recipe


def _first_line(error: BaseException) -> str:
    """The first line of an exception's message.

    scabha's validation errors run to many lines of nested context; the
    first names the offending key, which is the part worth forwarding.
    """
    text = str(error).strip()
    return text.splitlines()[0] if text else type(error).__name__


def loaded_config(source: str | None = None) -> LoadedConfig:
    """The discovered and configured libraries, optionally with one more source.

    The base config is loaded once per process and reused. When `source` is
    given it is merged into a *copy*, so nothing it defines leaks into the
    next call - `stimela.CONFIG` is a global, and `load_recipe_files` writes
    straight into it. Layering costs about 1.5s against the base's ten, which
    is what makes a per-call source affordable at all.

    `source` takes any spelling stimela accepts: a path to a YAML file or a
    directory, or a `module::path` library spec. A path that does not exist
    is reported as such rather than being passed on, because that is the
    spelling callers get wrong and stimela's own message for it names the
    module it failed to import instead.
    """
    base, base_paths = _base_config(tuple(configured_sources()))
    if source is None:
        return LoadedConfig(config=base, recipe_names=[], source_paths=list(base_paths))

    if _looks_like_path(source) and not Path(source).exists():
        raise StimelaConfigError(f"recipe file not found: {source}")

    previous = stimela.CONFIG
    stimela.CONFIG = copy.deepcopy(base)
    try:
        recipe_names, paths = _load_sources([source])
        return LoadedConfig(
            config=stimela.CONFIG,
            recipe_names=recipe_names,
            source_paths=list(base_paths) + paths,
        )
    finally:
        stimela.CONFIG = previous


def _looks_like_path(source: str) -> bool:
    """Whether a source spec names a filesystem path rather than a library.

    `module::path` and `(module)/path` are stimela's library spellings;
    everything else is a path as far as boepie is concerned.
    """
    return "::" not in source and not source.startswith("(")


def describe_sources() -> str:
    """The configured library sources, for an error message's benefit."""
    return ", ".join(PIPELINE_SOURCES) if PIPELINE_SOURCES else "(none configured)"
