"""ArXiv-based literature acquisition: manifest + fetch/convert, run on the
end user's own machine (see `boepie.literature.fetch` for why)."""

from __future__ import annotations

from boepie.literature.fetch import (
    FetchResult,
    convert_arxiv_html,
    fetch_paper,
    lookup_arxiv_metadata,
)
from boepie.literature.manifest import (
    ArxivPaper,
    derive_citekey,
    load_default_manifest,
    load_manifest,
    unique_citekey,
)

__all__ = [
    "ArxivPaper",
    "FetchResult",
    "convert_arxiv_html",
    "derive_citekey",
    "fetch_paper",
    "load_default_manifest",
    "load_manifest",
    "lookup_arxiv_metadata",
    "unique_citekey",
]
