"""Fetches and converts arXiv papers to Markdown, entirely on the machine
running boepie: no paper text is ever redistributed by boepie itself, only
the small bibliographic manifest (see `boepie.literature.manifest`) naming
which arXiv ids to pull.

Tries arXiv's own native HTML5 rendering (``arxiv.org/html/<id>``, LaTeXML-
generated, available for most post-2023 submissions and being backfilled for
older ones) first, then falls back to ``ar5iv.labs.arxiv.org`` (the longer-
running LaTeXML rendering service, broader historical coverage). Both emit
the same ``ltx_*``-classed DOM, so one converter handles both. A paper with
neither (typically pre-arXiv-era, or never preprinted) is reported as
unavailable rather than silently skipped: use `scripts/corpus_to_md.py`
against your own copy of the PDF instead.

This module is a pure fetch/convert layer: `FetchResult`/`fetch_paper`/
`convert_arxiv_html`/`lookup_arxiv_metadata` write nothing to disk.
`boepie.corpus.reconcile.sync_literature` is the sole caller that turns a
`FetchResult` into an on-disk corpus document (the old `_write_paper`/
`convert_manifest` sidecar writer that used to live here is retired -
`LITERATURE_DIR` is never a redistributed asset, so nothing else needed it).
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup
from markdownify import markdownify

_ARXIV_NATIVE_HTML = "https://arxiv.org/html/{arxiv_id}"
_AR5IV_HTML = "https://ar5iv.labs.arxiv.org/html/{arxiv_id}"
_ARXIV_ATOM_API = "https://export.arxiv.org/api/query"
_ARXIV_NS = {"atom": "http://www.w3.org/2005/Atom"}

# Identify the client so arXiv/ar5iv can attribute (and rate-limit) the traffic.
_USER_AGENT = "boepie-literature-fetch (+https://github.com/kwazzi-jack/boepie)"
_REQUEST_TIMEOUT_SECONDS = 30

# The container every LaTeXML-generated page wraps its content in - identical
# on both arxiv.org's native HTML and ar5iv, since ar5iv's renderer is the same
# underlying pipeline.
_ARTICLE_SELECTOR = "article.ltx_document"

# Chrome carrying no prose: banner/nav/report-an-issue furniture either site
# adds around the LaTeXML article.
_CHROME_SELECTORS = (".ltx_page_header", ".ltx_page_footer", "nav", "script", "style")


@dataclass
class FetchResult:
    citekey: str
    markdown: str | None
    source: str  # "arxiv-html" | "ar5iv" | "unavailable"
    page_url: str | None


def _get(client: httpx.Client, url: str) -> httpx.Response | None:
    """GETs `url`, returning None (not raising) on a 404 - the expected "not
    rendered at this source" case this module has to distinguish from a real
    network failure, which is left to propagate."""
    response = client.get(url, follow_redirects=True, timeout=_REQUEST_TIMEOUT_SECONDS)
    if response.status_code == 404:
        return None
    response.raise_for_status()
    return response


def convert_arxiv_html(html: str, page_url: str) -> str:
    """LaTeXML article HTML -> Markdown, preserving equations via `alttext`.

    Every LaTeXML `<math>` element carries the original LaTeX source in its
    `alttext` attribute; replacing the element with that text (wrapped as
    inline/display math) keeps equations legible without parsing the MathML
    tree it wraps.
    """
    soup = BeautifulSoup(html, "html.parser")
    article = soup.select_one(_ARTICLE_SELECTOR)
    if article is None:
        raise ValueError("no LaTeXML article container found")

    for selector in _CHROME_SELECTORS:
        for node in article.select(selector):
            node.decompose()

    for math_element in article.find_all("math"):
        alttext = (math_element.get("alttext") or "").strip()
        is_display = math_element.get("display") == "block"
        replacement = (f"$${alttext}$$" if is_display else f"${alttext}$") if alttext else ""
        math_element.replace_with(replacement)

    # Resolve links/images against the page they came from, so the markdown
    # stays useful once it is no longer sitting next to the rest of the site.
    # Images are referenced, not downloaded: LiteratureLoader's image
    # resolution already tolerates refs with no local file (resolves to none).
    for anchor in article.select("a[href]"):
        anchor["href"] = urljoin(page_url, anchor["href"])
    for image in article.select("img[src]"):
        image["src"] = urljoin(page_url, image["src"])

    markdown = markdownify(
        str(article), heading_style="ATX",
        escape_underscores=False, escape_asterisks=False, escape_misc=False,
    )
    return re.sub(r"\n{3,}", "\n\n", markdown).strip() + "\n"


def fetch_paper(client: httpx.Client, citekey: str, arxiv_id: str) -> FetchResult:
    """Fetches and converts one paper, trying arxiv.org then ar5iv.labs.arxiv.org.

    Never raises: a transient failure on one source (or one paper) falls
    through to the next source, and finally to "unavailable", so fetching a
    17-paper manifest is not one flaky request away from aborting entirely.
    """
    for source, url_template in (("arxiv-html", _ARXIV_NATIVE_HTML), ("ar5iv", _AR5IV_HTML)):
        page_url = url_template.format(arxiv_id=arxiv_id)
        try:
            response = _get(client, page_url)
        except httpx.HTTPError:
            continue
        if response is None:
            continue
        try:
            markdown = convert_arxiv_html(response.text, page_url)
        except ValueError:
            continue
        return FetchResult(citekey=citekey, markdown=markdown, source=source, page_url=page_url)
    return FetchResult(citekey=citekey, markdown=None, source="unavailable", page_url=None)


def lookup_arxiv_metadata(arxiv_id: str) -> dict[str, str] | None:
    """Title/authors/year for `arxiv_id` from arXiv's own Atom API.

    Used by `boepie corpus add -l` so a bare arXiv id is enough to build a
    manifest entry - no title/author typing required.
    """
    with httpx.Client(headers={"User-Agent": _USER_AGENT}) as client:
        response = client.get(
            _ARXIV_ATOM_API, params={"id_list": arxiv_id}, timeout=_REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()

    root = ET.fromstring(response.text)
    entry = root.find("atom:entry", _ARXIV_NS)
    if entry is None or entry.find("atom:title", _ARXIV_NS) is None:
        return None

    title = " ".join(entry.find("atom:title", _ARXIV_NS).text.split())
    authors = " and ".join(
        author.find("atom:name", _ARXIV_NS).text.strip()
        for author in entry.findall("atom:author", _ARXIV_NS)
    )
    published = entry.find("atom:published", _ARXIV_NS).text
    return {"title": title, "authors": authors, "year": published[:4]}
