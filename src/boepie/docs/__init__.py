"""Upstream-documentation acquisition: manifest + fetch/convert, run on the
end user's own machine (see `boepie.docs.fetch` for why), mirroring
`boepie.literature`."""

from __future__ import annotations

from boepie.docs.fetch import (
    PageContent,
    PageResult,
    convert_page,
    crawl_site,
    fetch_docnames,
    fetch_version,
    iter_generic_pages,
    iter_project_pages,
    iter_sphinx_pages,
    probe_discovery_mode,
)
from boepie.docs.manifest import (
    DocsProject,
    load_default_manifest,
    load_manifest,
)

__all__ = [
    "DocsProject",
    "PageContent",
    "PageResult",
    "convert_page",
    "crawl_site",
    "fetch_docnames",
    "fetch_version",
    "iter_generic_pages",
    "iter_project_pages",
    "iter_sphinx_pages",
    "load_default_manifest",
    "load_manifest",
    "probe_discovery_mode",
]
