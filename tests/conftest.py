"""Shared pytest fixtures for boepie tests.

Cab discovery uses runner.list_cab_names() which reads cultcargo YAML files
directly - no MCP round-trip - so it is safe to use in test setup without
any dependency on the tools under test.
"""

from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import Any

import pytest
from fastmcp import Client

from boepie.config import BUNDLE_DIR_ENV_VAR
from boepie.corpus.document import write_leaf_document
from boepie.corpus.layout import full_title_filename
from boepie.corpus.schema import Source
from boepie.pipeline.runner import CabSchema, list_cab_names, load_cab_schema
from boepie.server import mcp


# ---------------------------------------------------------------------------
# Corpus fixtures
# ---------------------------------------------------------------------------


def write_corpus_document(
    collection_dir: Path,
    *,
    document_id: str,
    title: str,
    body: str,
    managed_by: str = "boepie",
    origin: str = "https://example.test/source",
    via: str = "verbatim",
    source_format: str = "markdown",
    group: str | None = None,
    assets: dict[str, bytes] | None = None,
    **blocks: Any,
) -> Path:
    """Write one document in the real corpus layout, for tests to build a
    corpus the loaders actually understand.

    Goes through `write_leaf_document` and the `schema` models rather than
    hand-rolling frontmatter, so a fixture cannot drift from what production
    writes - which is exactly how the pre-Phase-4 fixtures ended up describing
    a layout nothing read any more. `blocks` carries the collection-specific
    namespaced mapping (`bib=...` for literature, `docs=...` for docs).

    Returns the path actually written, which is one level deeper than the
    nominal target when `assets` forces a wrapped document.
    """
    source = Source(origin=origin, via=via, format=source_format)
    frontmatter: dict[str, Any] = {
        "title": title,
        "managed_by": managed_by,
        "source": source.model_dump(mode="json", by_alias=True, exclude_none=True),
    }
    frontmatter.update(blocks)

    target_dir = collection_dir / group if group else collection_dir
    document = write_leaf_document(
        target_dir / full_title_filename(title),
        document_id=document_id,
        frontmatter_fields=frontmatter,
        body=body,
        assets=assets,
    )
    return document.md_path


# ---------------------------------------------------------------------------
# Environment isolation
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_bundle_dir_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never let a developer's `BOEPIE_BUNDLE_DIR` steer bundle discovery.

    It short-circuits the walk up from the cwd, so a value set in the shell
    would silently redirect every knowledge lookup under test. Tests that
    exercise the override set it themselves.
    """
    monkeypatch.delenv(BUNDLE_DIR_ENV_VAR, raising=False)


# ---------------------------------------------------------------------------
# MCP client
# ---------------------------------------------------------------------------


@pytest.fixture
async def boepie_client():
    async with Client(mcp) as client:
        yield client


# ---------------------------------------------------------------------------
# Cab catalogue (module-scoped: cultcargo config is expensive to load once,
# so we load it once per test module and share the result)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def all_cab_names() -> list[str]:
    """Sorted list of every cab name available in the cultcargo installation."""
    names = list_cab_names()
    assert names, "No cabs found - is cultcargo installed?"
    return names


@pytest.fixture(scope="module")
def casa_cab_names(all_cab_names: list[str]) -> list[str]:
    """Cab names that belong to the casa.* namespace."""
    return [name for name in all_cab_names if name.startswith("casa.")]


@pytest.fixture(scope="module")
def non_casa_cab_names(all_cab_names: list[str]) -> list[str]:
    """Cab names outside the casa.* namespace."""
    return [name for name in all_cab_names if not name.startswith("casa.")]


@pytest.fixture(scope="module")
def pfb_cab_names(all_cab_names: list[str]) -> list[str]:
    """Cab names containing 'pfb' - used to test mid-string wildcard patterns."""
    return [name for name in all_cab_names if "pfb" in name]


@pytest.fixture(scope="module")
def pfb_three_char_suffix_cab_names(all_cab_names: list[str]) -> list[str]:
    """Cab names matching 'pfb.???' - used to test the ? single-character wildcard."""
    return fnmatch.filter(all_cab_names, "pfb.???")


@pytest.fixture(scope="module")
def wq_cab_names(all_cab_names: list[str]) -> list[str]:
    """Cab names matching '[wq]*' - used to test character-class patterns."""
    return fnmatch.filter(all_cab_names, "[wq]*")


@pytest.fixture(scope="module")
def non_c_cab_names(all_cab_names: list[str]) -> list[str]:
    """Cab names matching '[!c]*' - used to test negated character-class patterns."""
    return fnmatch.filter(all_cab_names, "[!c]*")


@pytest.fixture(scope="module")
def wsclean_schema() -> CabSchema:
    """Loaded schema for wsclean - a reliable cab with both required and optional inputs."""
    return load_cab_schema("wsclean")


@pytest.fixture(scope="module")
def wsclean_required_input_names(wsclean_schema: CabSchema) -> list[str]:
    """Names of wsclean's required input parameters."""
    return [name for name, param in wsclean_schema.inputs.items() if param.required]


@pytest.fixture(scope="module")
def wsclean_optional_input_names(wsclean_schema: CabSchema) -> list[str]:
    """Names of wsclean's optional input parameters."""
    return [name for name, param in wsclean_schema.inputs.items() if not param.required]
