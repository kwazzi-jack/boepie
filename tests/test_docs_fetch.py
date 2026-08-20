"""Tests for boepie.docs.fetch: the Sphinx searchindex.js path, the generic
sitemap/crawl fallback (see the Phase 2 plan's section 3e), and the fetch/
convert orchestration - all network calls faked via httpx.MockTransport, no
real request to any upstream docs site is ever made here."""

from __future__ import annotations


import httpx
import pytest
from urllib.robotparser import RobotFileParser

from boepie.docs.fetch import (
    _derive_path_prefix,
    _discover_sitemap_urls,
    _fetch_sitemap_pages,
    _load_robots,
    _normalize_url,
    _slug_from_url,
    convert_page,
    crawl_site,
    fetch_docnames,
    fetch_version,
    iter_generic_pages,
    iter_project_pages,
    iter_sphinx_pages,
    probe_discovery_mode,
)
from boepie.docs.manifest import DocsProject

_SEARCHINDEX_JS = 'Search.setIndex({"docnames": ["index", "guide", "changelogs/1.0", "genindex"]})'

_PAGE_HTML = """
<html><body>
<div role="main">
  <h1 id="title">Title<a class="headerlink" href="#title">#</a></h1>
  <p>Prose about <code>input_ms</code> here.</p>
  <a href="other.html">a relative link</a>
  <img src="fig1.png">
  <div class="sphinxsidebar">nav chrome</div>
</div>
<div class="footer">footer chrome</div>
</body></html>
"""

_NO_CONTAINER_HTML = "<html><body><p>no main container</p></body></html>"


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


# ---------------------------------------------------------------------------
# fetch_docnames / fetch_version
# ---------------------------------------------------------------------------


def test_fetch_docnames_parses_searchindex_js():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("searchindex.js")
        return httpx.Response(200, text=_SEARCHINDEX_JS)

    with _client(handler) as client:
        docnames = fetch_docnames(client, "https://example.org/docs/", 30)

    assert docnames == ["index", "guide", "changelogs/1.0", "genindex"]


def test_fetch_docnames_raises_value_error_on_unparsable_index():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="not a searchindex payload")

    with pytest.raises(ValueError):
        with _client(handler) as client:
            fetch_docnames(client, "https://example.org/docs/", 30)


def test_fetch_version_reads_documentation_options_js():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="VAR: 1,\nVERSION: '1.2.3',\nLANG: 'en',")

    with _client(handler) as client:
        version = fetch_version(client, "https://example.org/docs/en/latest/", 30)

    assert version == "1.2.3"


def test_fetch_version_falls_back_to_url_slug_when_version_is_empty():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="VERSION: '',")

    with _client(handler) as client:
        version = fetch_version(client, "https://example.org/docs/en/latest/", 30)

    assert version == "latest"


# ---------------------------------------------------------------------------
# convert_page
# ---------------------------------------------------------------------------


def test_convert_page_strips_chrome_and_headerlinks():
    markdown = convert_page(_PAGE_HTML, "https://example.org/docs/index.html")

    assert "nav chrome" not in markdown
    assert "footer chrome" not in markdown
    assert "#" not in markdown.split("Title")[1].split("\n")[0]


def test_convert_page_resolves_relative_links_and_images():
    markdown = convert_page(_PAGE_HTML, "https://example.org/docs/index.html")

    assert "https://example.org/docs/other.html" in markdown
    assert "https://example.org/docs/fig1.png" in markdown


def test_convert_page_raises_without_main_content_container():
    with pytest.raises(ValueError):
        convert_page(_NO_CONTAINER_HTML, "https://example.org/docs/index.html")


def test_convert_page_preserves_underscored_identifiers():
    markdown = convert_page(_PAGE_HTML, "https://example.org/docs/index.html")

    assert "input_ms" in markdown
    assert "input\\_ms" not in markdown


# ---------------------------------------------------------------------------
# iter_sphinx_pages (Sphinx path)
# ---------------------------------------------------------------------------


def _sphinx_handler(request: httpx.Request) -> httpx.Response:
    url = str(request.url)
    if url.endswith("searchindex.js"):
        return httpx.Response(200, text=_SEARCHINDEX_JS)
    if url.endswith("_static/documentation_options.js"):
        return httpx.Response(200, text="VERSION: '9.9',")
    if url.endswith(".html"):
        return httpx.Response(200, text=_PAGE_HTML, headers={"content-type": "text/html"})
    return httpx.Response(404)












# ---------------------------------------------------------------------------
# 3e: discovery-mode dispatch
# ---------------------------------------------------------------------------


def test_probe_discovery_mode_prefers_sphinx():
    with _client(_sphinx_handler) as client:
        assert probe_discovery_mode(client, "https://example.org/docs/", 30) == "sphinx"


def test_probe_discovery_mode_falls_back_to_sitemap():
    urlset = (
        '<?xml version="1.0"?>'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
        '<url><loc>https://example.org/docs/a.html</loc></url>'
        "</urlset>"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url.endswith("searchindex.js"):
            return httpx.Response(404)
        if url.endswith("robots.txt"):
            return httpx.Response(404)
        if url.endswith("sitemap.xml"):
            return httpx.Response(200, text=urlset)
        return httpx.Response(404)

    with _client(handler) as client:
        assert probe_discovery_mode(client, "https://example.org/docs/", 30) == "sitemap"


def test_probe_discovery_mode_falls_back_to_crawl():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    with _client(handler) as client:
        assert probe_discovery_mode(client, "https://example.org/docs/", 30) == "crawl"


# ---------------------------------------------------------------------------
# 3e: robots.txt handling
# ---------------------------------------------------------------------------


def test_load_robots_parses_disallow_rules():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="User-agent: *\nDisallow: /private/\n")

    with _client(handler) as client:
        robots = _load_robots(client, "https://example.org", 30)

    assert robots.can_fetch("any-agent", "https://example.org/private/page.html") is False
    assert robots.can_fetch("any-agent", "https://example.org/public/page.html") is True


def test_load_robots_treats_404_as_allow_all():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    with _client(handler) as client:
        robots = _load_robots(client, "https://example.org", 30)

    assert robots.allow_all is True


def test_load_robots_treats_403_as_disallow_all():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403)

    with _client(handler) as client:
        robots = _load_robots(client, "https://example.org", 30)

    assert robots.disallow_all is True


# ---------------------------------------------------------------------------
# 3e: sitemap discovery
# ---------------------------------------------------------------------------


_URLSET_XML = (
    '<?xml version="1.0"?>'
    '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
    "<url><loc>https://example.org/docs/a.html</loc></url>"
    "<url><loc>https://example.org/docs/b.html</loc></url>"
    "</urlset>"
)

_SITEMAPINDEX_XML = (
    '<?xml version="1.0"?>'
    '<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
    "<sitemap><loc>https://example.org/sitemap1.xml</loc></sitemap>"
    "</sitemapindex>"
)


def test_discover_sitemap_urls_prefers_robots_declared_sitemap():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    with _client(handler) as client:
        robots = RobotFileParser()
        robots.parse(["User-agent: *", "Sitemap: https://example.org/custom.xml"])
        result = _discover_sitemap_urls(client, "https://example.org", robots, 30)

    assert result == ["https://example.org/custom.xml"]


def test_discover_sitemap_urls_falls_back_to_conventional_location():
    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url).endswith("sitemap.xml"):
            return httpx.Response(200, text=_URLSET_XML)
        return httpx.Response(404)

    with _client(handler) as client:
        robots = RobotFileParser()
        robots.parse(["User-agent: *"])
        result = _discover_sitemap_urls(client, "https://example.org", robots, 30)

    assert result == ["https://example.org/sitemap.xml"]


def test_discover_sitemap_urls_handles_absent_sitemap():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    with _client(handler) as client:
        robots = RobotFileParser()
        robots.parse(["User-agent: *"])
        result = _discover_sitemap_urls(client, "https://example.org", robots, 30)

    assert result == []


def test_fetch_sitemap_pages_parses_urlset():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=_URLSET_XML)

    with _client(handler) as client:
        pages = _fetch_sitemap_pages(client, ["https://example.org/sitemap.xml"], 30)

    assert pages == ["https://example.org/docs/a.html", "https://example.org/docs/b.html"]


def test_fetch_sitemap_pages_expands_sitemapindex():
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url.endswith("sitemap.xml"):
            return httpx.Response(200, text=_SITEMAPINDEX_XML)
        if url.endswith("sitemap1.xml"):
            return httpx.Response(200, text=_URLSET_XML)
        return httpx.Response(404)

    with _client(handler) as client:
        pages = _fetch_sitemap_pages(client, ["https://example.org/sitemap.xml"], 30)

    assert pages == ["https://example.org/docs/a.html", "https://example.org/docs/b.html"]


# ---------------------------------------------------------------------------
# 3e: URL normalization / slug derivation
# ---------------------------------------------------------------------------


def test_normalize_url_strips_fragment_query_and_trailing_slash():
    assert _normalize_url("https://example.org/docs/a/") == "https://example.org/docs/a"
    assert (
        _normalize_url("https://example.org/docs/a?x=1#frag")
        == "https://example.org/docs/a"
    )


def test_slug_derivation_from_a_crawled_url():
    assert _slug_from_url("https://example.org/docs/guide/setup.html", "/docs/") == "guide/setup"
    assert _slug_from_url("https://example.org/docs/", "/docs/") == "index"


# ---------------------------------------------------------------------------
# 3e: crawl bounds
# ---------------------------------------------------------------------------


_CRAWL_PAGES = {
    "https://example.org/docs/": (
        '<html><body><a href="page1.html">p1</a>'
        '<a href="/blog/post">blog</a>'
        '<a href="page2.html">p2</a></body></html>'
    ),
    "https://example.org/docs/page1.html": '<html><body><a href="page3.html">p3</a></body></html>',
    "https://example.org/docs/page2.html": "<html><body>leaf</body></html>",
    "https://example.org/docs/page3.html": "<html><body>leaf</body></html>",
}


def _crawl_handler(request: httpx.Request) -> httpx.Response:
    url = str(request.url).split("#")[0].split("?")[0]
    if url in _CRAWL_PAGES:
        return httpx.Response(200, text=_CRAWL_PAGES[url], headers={"content-type": "text/html"})
    return httpx.Response(404)


def _allow_all_robots() -> RobotFileParser:
    robots = RobotFileParser()
    robots.allow_all = True
    return robots


def test_crawl_site_stops_at_max_pages():
    with _client(_crawl_handler) as client:
        pages = crawl_site(
            client, "https://example.org/docs/", robots=_allow_all_robots(),
            delay=0.0, max_pages=2,
        )

    assert len(pages) == 2


def test_crawl_site_stops_at_max_depth():
    with _client(_crawl_handler) as client:
        pages = crawl_site(
            client, "https://example.org/docs/", robots=_allow_all_robots(),
            delay=0.0, max_depth=0,
        )

    assert pages == ["https://example.org/docs/"]


def test_crawl_site_enforces_same_origin():
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url == "https://example.org/docs/":
            return httpx.Response(
                200,
                text='<html><body><a href="https://other.org/page">off-site</a></body></html>',
                headers={"content-type": "text/html"},
            )
        if url == "https://other.org/page":
            raise AssertionError("crawl must not follow off-origin links")
        return httpx.Response(404)

    with _client(handler) as client:
        pages = crawl_site(client, "https://example.org/docs/", robots=_allow_all_robots(), delay=0.0)

    assert pages == ["https://example.org/docs/"]


def test_crawl_site_enforces_path_prefix():
    with _client(_crawl_handler) as client:
        pages = crawl_site(client, "https://example.org/docs/", robots=_allow_all_robots(), delay=0.0)

    assert not any("blog" in page for page in pages)


def test_crawl_site_path_prefix_override_widens_scope():
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url == "https://example.org/docs/":
            return httpx.Response(
                200,
                text='<html><body><a href="/blog/post">blog</a></body></html>',
                headers={"content-type": "text/html"},
            )
        if url == "https://example.org/blog/post":
            return httpx.Response(200, text="<html><body>blog leaf</body></html>",
                                   headers={"content-type": "text/html"})
        return httpx.Response(404)

    with _client(handler) as client:
        pages = crawl_site(
            client, "https://example.org/docs/", robots=_allow_all_robots(),
            path_prefix="/", delay=0.0,
        )

    assert "https://example.org/blog/post" in pages


def test_crawl_site_skips_robots_disallowed_urls():
    robots = RobotFileParser()
    robots.parse(["User-agent: *", "Disallow: /docs/page1.html"])

    with _client(_crawl_handler) as client:
        pages = crawl_site(client, "https://example.org/docs/", robots=robots, delay=0.0)

    assert "https://example.org/docs/page1.html" not in pages
    # page3 is only reachable via page1's links, so it is also unreachable.
    assert "https://example.org/docs/page3.html" not in pages


def test_crawl_site_dedups_duplicate_links():
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url).split("#")[0].split("?")[0]
        pages = {
            "https://example.org/docs/": (
                '<html><body><a href="a.html">a</a>'
                '<a href="a.html#frag">a again</a>'
                '<a href="a.html?x=1">a query</a></body></html>'
            ),
            "https://example.org/docs/a.html": "<html><body>leaf</body></html>",
        }
        if url in pages:
            return httpx.Response(200, text=pages[url], headers={"content-type": "text/html"})
        return httpx.Response(404)

    with _client(handler) as client:
        pages = crawl_site(client, "https://example.org/docs/", robots=_allow_all_robots(), delay=0.0)

    assert pages.count("https://example.org/docs/a.html") == 1












def test_derive_path_prefix_from_seed_url():
    assert _derive_path_prefix("https://example.org/docs/index.html") == "/docs/"
    assert _derive_path_prefix("https://example.org/docs/") == "/docs/"
    assert _derive_path_prefix("https://example.org/") == "/"


# ---------------------------------------------------------------------------
# iter_sphinx_pages / iter_generic_pages / iter_project_pages: the pure
# discovery/fetch/convert layer boepie.corpus.reconcile.sync_docs_project
# calls into, extracted from build_project's/build_project_generic's loop
# bodies. build_project/build_project_generic (tested above) now delegate to
# these, so this only needs to cover what a plain writer test would miss:
# the generator yields PageContent instead of writing, and failures surface
# as markdown=None + error rather than being written at all.
# ---------------------------------------------------------------------------


def test_iter_sphinx_pages_yields_page_content_without_writing(tmp_path):
    project = DocsProject(project="proj", base_url="https://example.org/docs/")

    with _client(_sphinx_handler) as client:
        pages = list(iter_sphinx_pages(client, project, 30, delay=0.0))

    # genindex is always excluded; index, guide, changelogs/1.0 remain.
    assert [page.docname for page in pages] == ["index", "guide", "changelogs/1.0"]
    assert all(page.markdown is not None for page in pages)
    assert not any(tmp_path.iterdir())


def test_iter_sphinx_pages_reports_a_failure_as_markdown_none():
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url.endswith("searchindex.js"):
            return httpx.Response(200, text='Search.setIndex({"docnames": ["good", "bad"]})')
        if url.endswith("bad.html"):
            return httpx.Response(500)
        return httpx.Response(200, text=_PAGE_HTML, headers={"content-type": "text/html"})

    project = DocsProject(project="proj", base_url="https://example.org/docs/")
    with _client(handler) as client:
        pages = {page.docname: page for page in iter_sphinx_pages(client, project, 30, delay=0.0)}

    assert pages["good"].markdown is not None
    assert pages["bad"].markdown is None
    assert pages["bad"].error is not None


def test_iter_generic_pages_yields_page_content_via_sitemap():
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url.endswith("robots.txt"):
            return httpx.Response(404)
        if url.endswith("sitemap.xml"):
            return httpx.Response(200, text=_URLSET_XML)
        if url.endswith(".html"):
            return httpx.Response(200, text=_PAGE_HTML, headers={"content-type": "text/html"})
        return httpx.Response(404)

    project = DocsProject(project="proj", base_url="https://example.org/docs/")
    with _client(handler) as client:
        pages = list(iter_generic_pages(client, project, discovery="sitemap", delay=0.0))

    assert {page.docname for page in pages} == {"a", "b"}
    assert all(page.markdown is not None for page in pages)


def test_iter_project_pages_dispatches_to_sphinx():
    project = DocsProject(project="proj", base_url="https://example.org/docs/")
    with _client(_sphinx_handler) as client:
        pages = list(iter_project_pages(client, project, delay=0.0))

    assert [page.docname for page in pages] == ["index", "guide", "changelogs/1.0"]


def test_iter_project_pages_dispatches_to_crawl_when_sphinx_and_sitemap_absent():
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url.endswith("searchindex.js"):
            return httpx.Response(404)
        if url.endswith("robots.txt"):
            return httpx.Response(404)
        if url.endswith("sitemap.xml"):
            return httpx.Response(404)
        if url == "https://example.org/docs/":
            return httpx.Response(200, text=_PAGE_HTML, headers={"content-type": "text/html"})
        return httpx.Response(404)

    project = DocsProject(project="proj", base_url="https://example.org/docs/")
    with _client(handler) as client:
        pages = list(iter_project_pages(client, project, delay=0.0))

    assert [page.docname for page in pages] == ["index"]


def test_iter_project_pages_honours_pinned_discovery():
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url.endswith("searchindex.js"):
            raise AssertionError("pinned discovery must not probe sphinx")
        if url.endswith("robots.txt"):
            return httpx.Response(404)
        if url == "https://example.org/docs/":
            return httpx.Response(200, text=_PAGE_HTML, headers={"content-type": "text/html"})
        return httpx.Response(404)

    project = DocsProject(project="proj", base_url="https://example.org/docs/", discovery="crawl")
    with _client(handler) as client:
        pages = list(iter_project_pages(client, project, delay=0.0))

    assert [page.docname for page in pages] == ["index"]
