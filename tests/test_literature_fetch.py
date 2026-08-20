"""Tests for boepie.literature.fetch: the arXiv HTML->Markdown converter,
`fetch_paper`'s source fallback, and `lookup_arxiv_metadata`, with network
calls faked via httpx.MockTransport - no real request to arxiv.org/ar5iv is
ever made here. `fetch_paper` orchestration into an on-disk corpus document
is now `boepie.corpus.reconcile.sync_literature`'s job, exercised in
test_corpus_reconcile.py - this module stays a pure fetch/convert layer with
no writer of its own (the old `convert_manifest`/`_write_paper` sidecar
writer was retired).
"""

from __future__ import annotations

import httpx
import pytest

from boepie.literature.fetch import (
    convert_arxiv_html,
    fetch_paper,
    lookup_arxiv_metadata,
)

_LATEXML_HTML = """
<html><body>
<article class="ltx_document">
  <div class="ltx_page_header">Submitted to arXiv - not part of the paper</div>
  <h1 class="ltx_title">A Test Paper</h1>
  <p>Prose with inline math <math alttext="x^2"><mrow/></math> here.</p>
  <div class="ltx_equation"><math alttext="E = mc^2" display="block"><mrow/></math></div>
  <a href="other.html">a relative link</a>
  <img src="fig1.png">
  <div class="ltx_page_footer">Page 1 of 10</div>
</article>
</body></html>
"""

_NO_ARTICLE_HTML = "<html><body><p>not a LaTeXML page</p></body></html>"

_ARXIV_ATOM_ENTRY = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/1234.5678v1</id>
    <title>A Test Paper</title>
    <published>2018-03-01T00:00:00Z</published>
    <author><name>C. Tasse</name></author>
    <author><name>B. Hugo</name></author>
  </entry>
</feed>
"""

_ARXIV_ATOM_EMPTY = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/api/errors#incorrect_id_format</id>
  </entry>
</feed>
"""


# ---------------------------------------------------------------------------
# convert_arxiv_html
# ---------------------------------------------------------------------------


def test_convert_arxiv_html_strips_chrome_and_preserves_math():
    markdown = convert_arxiv_html(_LATEXML_HTML, "https://arxiv.org/html/1234.5678")

    assert "not part of the paper" not in markdown
    assert "Page 1 of 10" not in markdown
    assert "# A Test Paper" in markdown
    assert "$x^2$" in markdown
    assert "$$E = mc^2$$" in markdown


def test_convert_arxiv_html_resolves_relative_links_and_images():
    markdown = convert_arxiv_html(_LATEXML_HTML, "https://arxiv.org/html/1234.5678")

    assert "https://arxiv.org/html/other.html" in markdown
    assert "https://arxiv.org/html/fig1.png" in markdown


def test_convert_arxiv_html_raises_without_article_container():
    with pytest.raises(ValueError):
        convert_arxiv_html(_NO_ARTICLE_HTML, "https://arxiv.org/html/1234.5678")


# ---------------------------------------------------------------------------
# fetch_paper: source fallback and resilience
# ---------------------------------------------------------------------------


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_fetch_paper_prefers_native_arxiv_html():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "arxiv.org"
        return httpx.Response(200, text=_LATEXML_HTML)

    with _client(handler) as client:
        result = fetch_paper(client, "test2018", "1234.5678")

    assert result.source == "arxiv-html"
    assert result.markdown is not None
    assert "# A Test Paper" in result.markdown


def test_fetch_paper_falls_back_to_ar5iv_on_404():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "arxiv.org":
            return httpx.Response(404)
        assert request.url.host == "ar5iv.labs.arxiv.org"
        return httpx.Response(200, text=_LATEXML_HTML)

    with _client(handler) as client:
        result = fetch_paper(client, "test2018", "1234.5678")

    assert result.source == "ar5iv"
    assert result.markdown is not None


def test_fetch_paper_reports_unavailable_when_neither_source_has_it():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    with _client(handler) as client:
        result = fetch_paper(client, "test2018", "1234.5678")

    assert result.source == "unavailable"
    assert result.markdown is None


def test_fetch_paper_falls_through_a_transient_error_instead_of_raising():
    """A 500 (or connection error) on one source must not abort the whole
    fetch - it should fall through to the next source, same as a 404."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "arxiv.org":
            return httpx.Response(500)
        return httpx.Response(200, text=_LATEXML_HTML)

    with _client(handler) as client:
        result = fetch_paper(client, "test2018", "1234.5678")

    assert result.source == "ar5iv"
    assert result.markdown is not None


def test_fetch_paper_reports_unavailable_when_page_is_not_a_latexml_article():
    """A 200 response that isn't a LaTeXML article (ar5iv's own "conversion
    failed" pages do this) must not be mistaken for a successful conversion."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=_NO_ARTICLE_HTML)

    with _client(handler) as client:
        result = fetch_paper(client, "test2018", "1234.5678")

    assert result.source == "unavailable"
    assert result.markdown is None


# ---------------------------------------------------------------------------
# lookup_arxiv_metadata
# ---------------------------------------------------------------------------


def test_lookup_arxiv_metadata_parses_title_authors_year(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=_ARXIV_ATOM_ENTRY)

    real_client = httpx.Client
    monkeypatch.setattr(
        "boepie.literature.fetch.httpx.Client",
        lambda **kwargs: real_client(transport=httpx.MockTransport(handler)),
    )

    metadata = lookup_arxiv_metadata("1234.5678")

    assert metadata == {"title": "A Test Paper", "authors": "C. Tasse and B. Hugo", "year": "2018"}


def test_lookup_arxiv_metadata_returns_none_for_an_invalid_id(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=_ARXIV_ATOM_EMPTY)

    real_client = httpx.Client
    monkeypatch.setattr(
        "boepie.literature.fetch.httpx.Client",
        lambda **kwargs: real_client(transport=httpx.MockTransport(handler)),
    )

    assert lookup_arxiv_metadata("not-an-id") is None
