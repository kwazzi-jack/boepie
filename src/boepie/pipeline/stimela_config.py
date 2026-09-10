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

Six things about driving that chain in-process rather than from the CLI:

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
- **stimela logs to stdout, and under the MCP server stdout is the wire.**
  `stimela.stimelogging` builds its one module-level rich console as
  `Console(file=sys.stdout)`, and every INFO line - `loaded full
  configuration from cache`, `loading manifest from <path>` - goes there. On
  the stdio transport that is the JSON-RPC channel, so the first cab or
  recipe tool call interleaves log text with protocol frames and the client
  fails to parse them (`Invalid JSON: trailing characters`). `_log_to_stderr`
  points that console at stderr at import, which is where a server's
  diagnostics belong on either surface. Rebinding `.file` is stimela's own
  mechanism - `kitchen/recipe.py` swaps in a `StringIO` the same way, and
  never restores it, so nothing can put stdout back.
- **`load_recipe_files` takes every source's files at once, and exits on
  the first one it cannot parse.** That is the right shape for the CLI,
  where the user named the file, and the wrong one here, where boepie
  names them: one broken library installed beside boepie emptied the
  catalogue for every other library, and the agent was told only about a
  file it had never heard of. So a failed merged load is retried source by
  source and only the source that actually fails is dropped - loudly, in
  the tool's own output. See `_load_sources_separately`.
"""

from __future__ import annotations

import copy
import importlib.metadata
import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import scabha.configuratt.cache
import stimela
import stimela.stimelogging
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


def _log_to_stderr() -> None:
    """Send stimela's logging to stderr, before it can emit a single line.

    Called at import, which is early enough because nothing reaches stimela
    except through this module. Guarded rather than assumed: the attribute is
    stimela's, so a version that renames or drops it must not stop boepie
    importing - the cost of missing it is noisy output, not a wrong answer.
    """
    console = getattr(stimela.stimelogging, "rich_console", None)
    if console is not None:
        console.file = sys.stderr


_log_to_stderr()


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


@dataclass(frozen=True)
class SkippedSource:
    """A library left out of the catalogue, and why.

    Reported rather than dropped in silence: a library missing from the
    catalogue looks exactly like one that was never installed, and the cab
    an agent is hunting for may be precisely the one that went missing.
    """

    spec: str
    reason: str


@dataclass(frozen=True)
class _BaseConfig:
    """A loaded base config, the files behind it, and what was left out."""

    config: Any
    source_paths: tuple[str, ...]
    skipped: tuple[SkippedSource, ...]


def _is_strict(spec: str) -> bool:
    """Whether this source failing should take the whole load down with it.

    A source named in `pipeline.sources` is the user's own instruction -
    `cultcargo::` included, since it is there by default - so a broken one
    is an error they have to see: 16 of cult-cargo 0.2.1's 37 YAML files
    fail under scabha rc4, and skipping those tolerantly would answer
    "what can flag data" with a catalogue that has quietly lost all 17
    `casa.*` cabs. Anything that only turned up in discovery is a package
    that merely happens to be installed beside boepie, and one of those
    must not be able to empty the catalogue for all the others.
    """
    return spec in PIPELINE_SOURCES


# A skip reason goes into MCP output, so it is capped. scabha ends an
# `_include not found` message with stimela's whole config search path -
# ten fixed directories, about 500 characters, and repeated once per load
# attempt - where everything a reader can act on (the library, the file,
# the key, the missing include) is in the first line and a half. The full
# text is still logged to stderr.
_MAX_SKIP_REASON = 240


def _skip_reason(spec: str, error: BaseException) -> str:
    """Why a source was skipped, without repeating its own name.

    stimela's messages open with the spec it was handed, and the line
    boepie renders names it already. Collapsed onto one line, since these
    are printed one per source into an otherwise tabular payload.
    """
    reason = " ".join(str(error).split())
    for prefix in (f"{spec}: ", f"{spec} ", f"'{spec}' "):
        if reason.startswith(prefix):
            reason = reason[len(prefix) :]
            break
    if len(reason) <= _MAX_SKIP_REASON:
        return reason
    return reason[:_MAX_SKIP_REASON].rsplit(" ", 1)[0] + " ..."


def skipped_sources_note(skipped: Sequence[SkippedSource]) -> str:
    """One comment line per source left out, or an empty string.

    `list_cabs` and `list_recipes` are how an agent finds out what exists,
    so they are where a missing library has to be named. Without it the
    only signal is a cab that cannot be found, which reads as a cab that
    was never packaged rather than as a library boepie could not load.
    """
    return "".join(f"# skipped {item.spec} - {item.reason}\n" for item in skipped)


# The base config per source tuple, or the error that tuple raises. Loading
# cult-cargo alone parses 36 YAML files and takes roughly ten seconds,
# which is fine once per process and not fine per tool call - and a load
# that fails costs the same ten seconds, so the failure is remembered too.
# Nothing about an installed source changes while the process runs, so a
# second attempt could only arrive at the same answer again.
_BASE_CONFIGS: dict[tuple[str, ...], _BaseConfig | StimelaConfigError] = {}


def _base_config(sources: tuple[str, ...]) -> _BaseConfig:
    """The resolved stimela config with `sources` merged in, loaded once.

    Holds the config used as the *base* for every call - `loaded_config`
    copies it before anything is layered on.
    """
    remembered = _BASE_CONFIGS.get(sources)
    if isinstance(remembered, StimelaConfigError):
        raise remembered
    if remembered is not None:
        return remembered
    try:
        loaded = _load_base_config(sources)
    except StimelaConfigError as error:
        _BASE_CONFIGS[sources] = error
        raise
    _BASE_CONFIGS[sources] = loaded
    return loaded


def _load_base_config(sources: tuple[str, ...]) -> _BaseConfig:
    """Load every source, isolating a failure to the source that caused it."""
    _initialise_stimela()
    stimela.CONFIG = _fresh_stimela_config()
    resolved, skipped = _resolve_all(sources)
    paths = [path for _, source_paths in resolved for path in source_paths]

    try:
        _load_paths(paths)
    except StimelaConfigError:
        # One unparseable file says nothing about the other libraries, and
        # a single merged load cannot tell them apart. Start over and load
        # each source on its own, so only the broken one is lost.
        paths, isolated = _load_sources_separately(resolved)
        skipped.extend(isolated)

    return _BaseConfig(
        config=stimela.CONFIG,
        source_paths=tuple(paths),
        skipped=tuple(skipped),
    )


def _fresh_stimela_config() -> Any:
    """stimela's base config - opts, images, lib, cabs - with no sources."""
    config = stimela_config.load_config(extra_configs=[])
    if config is None:
        raise StimelaConfigError(
            "stimela could not load its base configuration. "
            "Run 'stimela -C' to clear the config cache and try again."
        )
    return config


def _resolve_all(
    sources: tuple[str, ...],
) -> tuple[list[tuple[str, list[str]]], list[SkippedSource]]:
    """Each source to the YAML files it names, dropping unusable discoveries.

    A package can look like a stimela library in its metadata and turn out
    not to be one - a `recipes/` directory of unrelated YAML, or a module
    that cannot be imported. This finds files; it deliberately does not
    prove they parse, which is what loading them is for.
    """
    resolved: list[tuple[str, list[str]]] = []
    skipped: list[SkippedSource] = []
    for spec in sources:
        try:
            source_paths = _resolve_source(spec)
        except StimelaConfigError as error:
            if _is_strict(spec):
                raise
            stimela.logger().warning(f"skipping discovered source {spec}: {error}")
            skipped.append(SkippedSource(spec=spec, reason=_skip_reason(spec, error)))
            continue
        resolved.append((spec, source_paths))
    return resolved, skipped


def _load_sources_separately(
    resolved: list[tuple[str, list[str]]],
) -> tuple[list[str], list[SkippedSource]]:
    """Load one source at a time, leaving out the ones that will not load.

    Only reached once the merged load has already failed, so the cost - one
    more base load, plus a copy of the config per source - is paid by a
    broken environment and never by a working one. The order is the order
    the sources were given: a later source reaching an earlier one's
    definitions finds them in `stimela.CONFIG`, exactly as it would have in
    the merged load.
    """
    stimela.CONFIG = _fresh_stimela_config()
    loaded: list[str] = []
    skipped: list[SkippedSource] = []
    for spec, paths in resolved:
        # `load_recipe_files` writes each recipe straight into
        # `stimela.CONFIG` as it goes, so a source that fails partway can
        # leave part of itself behind. Restoring the copy is what makes
        # dropping it complete.
        snapshot = copy.deepcopy(stimela.CONFIG)
        try:
            _load_paths(paths)
        except StimelaConfigError as error:
            stimela.CONFIG = snapshot
            if _is_strict(spec):
                raise StimelaConfigError(f"failed to load {spec}: {error}") from error
            stimela.logger().warning(f"skipping source {spec}: {error}")
            skipped.append(SkippedSource(spec=spec, reason=_skip_reason(spec, error)))
            continue
        loaded.extend(paths)
    return loaded, skipped


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


def _load_paths(paths: list[str]) -> list[str]:
    """Merge YAML files into the current `stimela.CONFIG`.

    Returns the names of any recipes they defined. Raises stimela's own
    wording, recovered from its logger, because `load_recipe_files` reports
    a bad file by logging the reason and then calling `sys.exit(2)` -
    uncaught, that would take the MCP server down with it.
    """
    if not paths:
        return []
    log = stimela.logger()
    captured = _CapturedLog()
    log.addHandler(captured)
    try:
        recipe_names, _ = load_recipe_files(paths)
    except SystemExit as error:
        detail = captured.summary() or f"stimela exited with code {error.code}"
        raise StimelaConfigError(detail) from error
    finally:
        log.removeHandler(captured)
    return list(recipe_names)


def _load_sources(sources: list[str]) -> tuple[list[str], list[str]]:
    """Resolve and merge each source spec into the current `stimela.CONFIG`.

    Sources are loaded in the order given, since a later one may `_use` an
    earlier one's definitions - the same reason stimela accumulates them
    into a single `load_recipe_files` call.
    """
    paths: list[str] = []
    for spec in sources:
        paths.extend(_resolve_source(spec))
    if not paths:
        return [], []
    try:
        return _load_paths(paths), paths
    except StimelaConfigError as error:
        raise StimelaConfigError(
            f"failed to load {', '.join(sources)}: {error}"
        ) from error


@dataclass(frozen=True)
class LoadedConfig:
    """A resolved stimela config, plus how the caller's own file contributed.

    `recipe_names` is what distinguishes a recipe the caller just supplied
    from one that came out of a configured library, which is what lets
    `list_recipes` say where each recipe came from. `source_paths` is every
    YAML the sources resolved to, kept so a recipe can be traced back to the
    file it was written in. `skipped_sources` is every library that was
    found and could not be loaded, carried this far so a tool can say so
    rather than presenting a shortened catalogue as the whole of it.
    """

    config: Any
    recipe_names: list[str]
    source_paths: list[str]
    skipped_sources: tuple[SkippedSource, ...] = ()

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
    base = _base_config(tuple(configured_sources()))
    if source is None:
        return LoadedConfig(
            config=base.config,
            recipe_names=[],
            source_paths=list(base.source_paths),
            skipped_sources=base.skipped,
        )

    if _looks_like_path(source) and not Path(source).exists():
        raise StimelaConfigError(f"recipe file not found: {source}")

    previous = stimela.CONFIG
    stimela.CONFIG = copy.deepcopy(base.config)
    try:
        recipe_names, paths = _load_sources([source])
        return LoadedConfig(
            config=stimela.CONFIG,
            recipe_names=recipe_names,
            source_paths=list(base.source_paths) + paths,
            skipped_sources=base.skipped,
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
