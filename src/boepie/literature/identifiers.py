# src/boepie/literature/identifiers.py
"""Recognises the many ways a paper can be named.

`boepie corpus add literature` used to pass whatever string it was given
straight to arXiv's Atom API, so only the bare `2409.19750` form worked -
`arXiv:2409.19750`, `2409.19750v1.pdf`, and any abs/pdf URL all failed with
"no arXiv entry found", which reads like the paper does not exist rather than
like the identifier was not understood.

This module normalises all of those to a bare arXiv id, recognises DOIs in
their several spellings, and parses a BibTeX file into manifest-shaped
entries. Nothing here touches the network except `resolve_doi_to_arxiv`.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import httpx

_ARXIV_ATOM_API = "https://export.arxiv.org/api/query"
_ARXIV_NS = {"atom": "http://www.w3.org/2005/Atom"}
_USER_AGENT = "boepie-literature-fetch (+https://github.com/kwazzi-jack/boepie)"
_REQUEST_TIMEOUT_SECONDS = 30

# Post-2007 arXiv ids: YYMM.NNNNN, four or five digits after the dot, with an
# optional version suffix.
_MODERN_ARXIV = re.compile(r"(?<!\d)(\d{4}\.\d{4,5})(v\d+)?(?!\d)")

# Pre-2007 ids: archive[.subject-class]/YYMMNNN, e.g. astro-ph/0601234.
_LEGACY_ARXIV = re.compile(
    r"(?<![\w/])([a-z-]+(?:\.[A-Za-z]{2})?/\d{7})(v\d+)?(?![\w])"
)

# A DOI is "10." followed by a registrant code, a slash, and a suffix that
# runs to the end of the token.
_DOI = re.compile(r"(10\.\d{4,9}/[^\s\"'<>]+)")

_ARXIV_HOSTS = ("arxiv.org", "ar5iv.labs.arxiv.org", "ar5iv.org", "browse.arxiv.org")


def normalize_arxiv_id(identifier: str) -> str | None:
    """Extract a bare, unversioned arXiv id from any of its usual spellings.

    Handles `2409.19750`, `2409.19750v1`, `arXiv:2409.19750`,
    `2409.19750v1.pdf`, `https://arxiv.org/abs/2409.19750v1`,
    `https://arxiv.org/pdf/2409.19750`, the ar5iv HTML URLs, and the pre-2007
    `astro-ph/0601234` form. Returns None when nothing arXiv-shaped is found.

    The version suffix is deliberately dropped. The corpus tracks a paper, not
    a snapshot of one, and keeping the version would make `2409.19750v1` and
    `2409.19750v2` look like two different documents to duplicate detection
    while fetching near-identical text twice.
    """
    candidate = identifier.strip()
    if not candidate:
        return None

    # A URL only counts when it is actually an arXiv-family host, so an
    # unrelated page whose path happens to contain digits is not misread.
    if "://" in candidate:
        from urllib.parse import urlparse

        parsed = urlparse(candidate)
        if not any(parsed.netloc.endswith(host) for host in _ARXIV_HOSTS):
            return None
        candidate = parsed.path

    modern = _MODERN_ARXIV.search(candidate)
    if modern is not None:
        return modern.group(1)

    legacy = _LEGACY_ARXIV.search(candidate)
    if legacy is not None:
        return legacy.group(1)
    return None


def normalize_doi(identifier: str) -> str | None:
    """Extract a bare DOI from `10.x/y`, `doi:10.x/y`, or a doi.org URL."""
    candidate = identifier.strip()
    if not candidate:
        return None
    # Strip a trailing period, which sentence-cased citations often carry in.
    match = _DOI.search(candidate)
    if match is None:
        return None
    return match.group(1).rstrip(".")


def looks_like_bibtex(identifier: str) -> bool:
    return identifier.lower().endswith(".bib")


@dataclass(frozen=True)
class BibEntry:
    """One parsed BibTeX record, in manifest shape."""

    citekey: str
    title: str
    authors: str
    year: str
    doi: str | None = None
    arxiv_id: str | None = None
    # A `file = {...}` path, the convention Zotero and JabRef use to point at
    # a local PDF. Followed when present so a `.bib` export doubles as a
    # batch of documents to ingest.
    file_path: str | None = None


_BIB_ENTRY = re.compile(r"@(\w+)\s*\{\s*([^,]+),(.*?)(?=\n@|\Z)", re.S)
_BIB_FIELD = re.compile(r"(\w+)\s*=\s*(\{(?:[^{}]|\{[^{}]*\})*\}|\"[^\"]*\"|[^,\n]+)")


def _clean_bib_value(raw: str) -> str:
    value = raw.strip().rstrip(",").strip()
    if value.startswith("{") and value.endswith("}"):
        value = value[1:-1]
    elif value.startswith('"') and value.endswith('"'):
        value = value[1:-1]
    return " ".join(value.replace("{", "").replace("}", "").split())


def parse_bibtex(text: str) -> list[BibEntry]:
    """Parse a BibTeX file into entries.

    Deliberately a regex parser rather than a dependency: boepie needs five
    fields out of each entry, and `bibtexparser` is a heavy addition for that.
    Malformed entries are skipped rather than raising, so one bad record in a
    large export does not block the rest.
    """
    entries: list[BibEntry] = []
    for _entry_type, citekey, body in _BIB_ENTRY.findall(text):
        fields = {
            name.lower(): _clean_bib_value(value)
            for name, value in _BIB_FIELD.findall(body)
        }
        title = fields.get("title", "")
        if not title:
            continue

        arxiv_id = None
        for key in ("eprint", "archiveprefix", "arxivid", "arxiv"):
            if key in fields:
                arxiv_id = normalize_arxiv_id(fields[key]) or arxiv_id
        if arxiv_id is None and "url" in fields:
            arxiv_id = normalize_arxiv_id(fields["url"])

        # Zotero writes `file = {Title:/abs/path.pdf:application/pdf}`; the
        # path is the middle colon-delimited field.
        file_field = fields.get("file")
        file_path = None
        if file_field:
            parts = file_field.split(":")
            file_path = parts[1] if len(parts) >= 2 and parts[1] else parts[0]

        entries.append(
            BibEntry(
                citekey=citekey.strip(),
                title=title,
                authors=fields.get("author", ""),
                year=fields.get("year", ""),
                doi=normalize_doi(fields.get("doi", "")) if fields.get("doi") else None,
                arxiv_id=arxiv_id,
                file_path=file_path,
            )
        )
    return entries


def parse_bibtex_file(path: Path) -> list[BibEntry]:
    return parse_bibtex(path.read_text(encoding="utf-8", errors="replace"))


def resolve_doi_to_arxiv(doi: str) -> str | None:
    """Ask arXiv whether it holds a paper with this DOI.

    Many arXiv records carry the published version's DOI, so a DOI is often
    enough to reach a fetchable preprint. A miss is not an error: the paper
    may simply never have been preprinted, which the caller reports as
    needing a BYO-PDF instead.
    """
    try:
        with httpx.Client(headers={"User-Agent": _USER_AGENT}) as client:
            response = client.get(
                _ARXIV_ATOM_API,
                params={"search_query": f'doi:"{doi}"', "max_results": 1},
                timeout=_REQUEST_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
    except httpx.HTTPError:
        return None

    try:
        root = ET.fromstring(response.text)
    except ET.ParseError:
        return None

    entry = root.find("atom:entry", _ARXIV_NS)
    if entry is None:
        return None
    id_element = entry.find("atom:id", _ARXIV_NS)
    if id_element is None or not id_element.text:
        return None
    return normalize_arxiv_id(id_element.text)
