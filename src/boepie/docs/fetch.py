"""Fetches and converts upstream documentation sites to Markdown, entirely on
the machine running boepie - the docs analogue of `boepie.literature.fetch`.

Sphinx sites publish an exact page list via their own `searchindex.js`
(`fetch_docnames`/`iter_sphinx_pages`) - the fast, authoritative path when
available. Any other doc site (mkdocs, Docusaurus, GitBook, hand-rolled) is
reached instead via its `sitemap.xml` when discoverable, or a bounded,
robots.txt-respecting link-following crawl as a last resort
(`iter_generic_pages`). `iter_project_pages` dispatches between the two:
`DocsProject.discovery` if pinned, else an auto-probe
(`probe_discovery_mode`), cheapest-and-most-authoritative source first.

Nothing here writes. This is a pure discovery/fetch/convert layer yielding
`PageContent`, and `boepie.corpus.reconcile.sync_docs_project` is its only
caller - it decides what lands on disk, in the corpus layout. A second set of
writers (`build_project`/`build_project_generic`/`build_project_auto`/
`convert_manifest`, driven by `scripts/docs_to_md.py`) used to live here and
wrote a `docs-corpus/{project}/{docname}.md` + `metadata.json` layout; it was
removed on 2026-08-20, because `DocsLoader` moved onto the corpus layout and
stopped being able to read anything that path produced - a docs corpus built
that way indexed to zero chunks.
"""
from __future__ import annotations

import json
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from fnmatch import fnmatch
from typing import Iterator
from urllib.parse import urljoin, urlsplit, urlunsplit
from urllib.robotparser import RobotFileParser

import httpx
from bs4 import BeautifulSoup
from markdownify import markdownify

from boepie.docs.manifest import DocsProject

# Identify the crawler so upstream can attribute (and block) the traffic.
_USER_AGENT = "boepie-docs-fetch (+https://github.com/kwazzi-jack/boepie)"
_REQUEST_TIMEOUT_SECONDS = 30

# Default pause between page requests - politeness towards upstream sites,
# applied on every request in the sphinx, sitemap and crawl paths alike.
_DEFAULT_FETCH_DELAY_SECONDS = 0.2

# The rendered main-content container, in preference order. Sphinx themes
# differ on the element but agree on role="main": furo/pydata render
# <article role="main">, the classic RTD theme <div role="main">.
_CONTENT_SELECTORS = ('[role="main"]', "div.document", "article")

# Chrome inside the content container that carries no prose: the per-heading
# anchor pilcrows, sidebars, breadcrumb/footer navigation, and scripts.
_CHROME_SELECTORS = (
    "a.headerlink",
    ".sphinxsidebar",
    ".related",
    ".footer",
    ".headerlinks",
    "script",
    "style",
)

# Docnames every Sphinx site generates that hold no documentation of their own.
_ALWAYS_EXCLUDE = ("genindex", "py-modindex", "search", "modindex")

# Crawl/sitemap bounds - this walks an arbitrary third-party site.
_MAX_PAGES = 300
_MAX_DEPTH = 5
_MAX_SITEMAP_FILES = 50

_SITEMAP_NS = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}

# Skipped by extension before ever being requested, mirrored by a
# Content-Type check after fetching as a second guard.
_NON_HTML_EXTENSIONS = (
    ".pdf", ".zip", ".tar", ".gz", ".tgz",
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".webp",
    ".css", ".js", ".map",
    ".woff", ".woff2", ".ttf", ".eot",
    ".mp4", ".mp3", ".wav", ".avi",
    ".xml", ".txt", ".rst",
)


@dataclass
class PageResult:
    docname: str
    ok: bool
    error: str | None = None


@dataclass
class PageContent:
    """One discovered-and-converted page, as the pure discovery/fetch/
    convert layer (`iter_sphinx_pages`/`iter_generic_pages`/
    `iter_project_pages`) yields it - `markdown` is None on a per-page
    failure (`error` set instead), mirroring `PageResult`'s ok/error split
    without committing to a write.
    """

    docname: str
    markdown: str | None
    error: str | None = None
    page_url: str | None = None



def _get(client: httpx.Client, url: str, timeout: int) -> httpx.Response:
    response = client.get(url, timeout=timeout)
    response.raise_for_status()
    return response


# ---------------------------------------------------------------------------
# Sphinx path: exact page list from searchindex.js
# ---------------------------------------------------------------------------


def fetch_docnames(client: httpx.Client, base_url: str, timeout: int) -> list[str]:
    """Every page name the site publishes, from Sphinx's own search index."""
    raw = _get(client, urljoin(base_url, "searchindex.js"), timeout).text
    match = re.search(r"Search\.setIndex\((.*)\)\s*;?\s*$", raw.strip(), re.S)
    if match is None:
        raise ValueError(f"could not parse searchindex.js at {base_url}")
    return list(json.loads(match.group(1)).get("docnames", []))


def fetch_version(client: httpx.Client, base_url: str, timeout: int) -> str:
    """The project's own version string, or the URL's version slug.

    Sphinx writes `VERSION` into `_static/documentation_options.js`, but only
    when the project sets it -- quartical and wsclean both publish an empty
    string. Falling back to the slug that actually served these pages
    ("latest") keeps the recorded provenance true rather than invented.
    """
    try:
        raw = _get(client, urljoin(base_url, "_static/documentation_options.js"), timeout).text
    except httpx.HTTPError:
        raw = ""
    match = re.search(r"VERSION:\s*['\"]([^'\"]+)['\"]", raw)
    if match is not None:
        return match.group(1)
    slug = [part for part in base_url.rstrip("/").split("/") if part]
    return slug[-1] if slug else "unknown"


def convert_page(html: str, page_url: str) -> str:
    """Rendered doc page -> markdown holding just its prose."""
    soup = BeautifulSoup(html, "html.parser")

    main = None
    for selector in _CONTENT_SELECTORS:
        main = soup.select_one(selector)
        if main is not None:
            break
    if main is None:
        raise ValueError("no main-content container found")

    for selector in _CHROME_SELECTORS:
        for node in main.select(selector):
            node.decompose()

    # Resolve links/images against the page they came from, so the markdown
    # stays useful once it is no longer sitting next to the rest of the site.
    for anchor in main.select("a[href]"):
        anchor["href"] = urljoin(page_url, anchor["href"])
    for image in main.select("img[src]"):
        image["src"] = urljoin(page_url, image["src"])

    markdown = markdownify(
        str(main),
        heading_style="ATX",
        # Identifiers are the point of these pages: escaping would turn
        # `input_ms` into `input\_ms` and put a backslash inside the very
        # tokens BM25 matches on.
        escape_underscores=False,
        escape_asterisks=False,
        escape_misc=False,
    )
    return re.sub(r"\n{3,}", "\n\n", markdown).strip() + "\n"


def _is_excluded(docname: str, exclude: tuple[str, ...]) -> bool:
    patterns = exclude + _ALWAYS_EXCLUDE
    return any(fnmatch(docname, pattern) for pattern in patterns)


def iter_sphinx_pages(
    client: httpx.Client,
    project: DocsProject,
    timeout: int,
    *,
    delay: float = _DEFAULT_FETCH_DELAY_SECONDS,
) -> Iterator[PageContent]:
    """Every page of a Sphinx project, as `PageContent`.

    Reads the exact page list out of the site's own `searchindex.js`, so
    nothing is guessed or crawled. Sleeps `delay` seconds after each
    successfully converted page.
    """
    docnames = fetch_docnames(client, project.base_url, timeout)
    wanted = [docname for docname in docnames if not _is_excluded(docname, project.exclude)]

    for docname in wanted:
        page_url = urljoin(project.base_url, f"{docname}.html")
        try:
            html = _get(client, page_url, timeout).text
            markdown = convert_page(html, page_url)
        except (httpx.HTTPError, ValueError) as error:
            yield PageContent(docname=docname, markdown=None, error=str(error), page_url=page_url)
            continue
        yield PageContent(docname=docname, markdown=markdown, page_url=page_url)
        time.sleep(delay)


# ---------------------------------------------------------------------------
# Generic path: sitemap discovery, or bounded link-following crawl
# ---------------------------------------------------------------------------


def _origin(base_url: str) -> str:
    parsed = urlsplit(base_url)
    return f"{parsed.scheme}://{parsed.netloc}"


def _derive_path_prefix(base_url: str) -> str:
    """The seed URL's directory, e.g. `.../docs/index.html` -> `/docs/`. A
    seed URL that is already the site root yields `/`, so every same-origin
    page qualifies - costs nothing for a single-purpose docs domain."""
    path = urlsplit(base_url).path
    if not path or path.endswith("/"):
        return path or "/"
    return path.rsplit("/", 1)[0] + "/"


def _normalize_url(url: str) -> str:
    """Fragment/query stripped, trailing slash collapsed, for crawl dedup."""
    parsed = urlsplit(url)
    path = parsed.path.rstrip("/") or "/"
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def _is_crawlable_link(url: str) -> bool:
    parsed = urlsplit(url)
    if parsed.scheme not in ("http", "https", ""):
        return False
    return not parsed.path.lower().endswith(_NON_HTML_EXTENSIONS)


def _in_scope(normalized_url: str, origin: str, path_prefix: str) -> bool:
    parsed = urlsplit(normalized_url)
    if f"{parsed.scheme}://{parsed.netloc}" != origin:
        return False
    return parsed.path.startswith(path_prefix)


def _slug_from_url(url: str, path_prefix: str) -> str:
    """A `{docname}`-shaped relative path for a crawled/sitemap page, derived
    from its URL relative to the crawl's `path_prefix` - mirrors what
    `fetch_docnames` gives Sphinx sites, so the generic path can reuse the
    same `{project}/{docname}.md` output layout."""
    path = urlsplit(url).path
    relative = path[len(path_prefix):] if path.startswith(path_prefix) else path.lstrip("/")
    relative = relative.strip("/")
    if relative.endswith(".html"):
        relative = relative[: -len(".html")]
    elif relative.endswith(".htm"):
        relative = relative[: -len(".htm")]
    relative = relative.strip("/")
    return relative or "index"


def _load_robots(client: httpx.Client, origin: str, timeout: int) -> RobotFileParser:
    """Fetches and parses `{origin}/robots.txt` through the module's own
    httpx.Client (not urllib.robotparser's own urlopen), hand-replicating the
    status-code handling urllib.robotparser would otherwise do itself:
    401/403 -> disallow everything, other 4xx -> allow everything, anything
    else (including an unreachable robots.txt) -> parse the body, or fail open
    if there is no body to parse."""
    parser = RobotFileParser()
    robots_url = urljoin(origin + "/", "robots.txt")
    parser.set_url(robots_url)
    try:
        response = client.get(robots_url, timeout=timeout)
    except httpx.HTTPError:
        parser.allow_all = True
        return parser

    if response.status_code in (401, 403):
        parser.disallow_all = True
    elif 400 <= response.status_code < 500:
        parser.allow_all = True
    else:
        parser.parse(response.text.splitlines())
    return parser


def _discover_sitemap_urls(
    client: httpx.Client, origin: str, robots: RobotFileParser, timeout: int
) -> list[str]:
    """Sitemap file URLs to try: robots.txt's `Sitemap:` directives if it
    declared any (trusted without a further existence check - the site
    announced them explicitly), else the conventional `/sitemap.xml`
    location, verified reachable before being trusted."""
    declared = list(robots.site_maps() or [])
    if declared:
        return declared

    fallback = urljoin(origin + "/", "sitemap.xml")
    try:
        response = client.get(fallback, timeout=timeout)
    except httpx.HTTPError:
        return []
    return [fallback] if response.status_code == 200 else []


def _fetch_sitemap_pages(client: httpx.Client, sitemap_urls: list[str], timeout: int) -> list[str]:
    """Expands sitemap file(s) into the page URLs they list, following
    `sitemapindex` nesting up to `_MAX_SITEMAP_FILES` fetched files total so a
    paginated sitemap can't loop unbounded."""
    pages: list[str] = []
    seen_files: set[str] = set()
    queue = list(sitemap_urls)

    while queue and len(seen_files) < _MAX_SITEMAP_FILES:
        url = queue.pop(0)
        if url in seen_files:
            continue
        seen_files.add(url)
        try:
            response = _get(client, url, timeout)
        except httpx.HTTPError:
            continue
        try:
            root = ET.fromstring(response.text)
        except ET.ParseError:
            continue

        tag = root.tag.rsplit("}", 1)[-1]
        if tag == "sitemapindex":
            queue.extend(
                loc.text.strip()
                for loc in root.findall(".//sm:sitemap/sm:loc", _SITEMAP_NS)
                if loc.text
            )
        elif tag == "urlset":
            pages.extend(
                loc.text.strip()
                for loc in root.findall(".//sm:url/sm:loc", _SITEMAP_NS)
                if loc.text
            )
    return pages


def probe_discovery_mode(client: httpx.Client, base_url: str, timeout: int) -> str:
    """"sphinx" | "sitemap" | "crawl", cheapest-and-most-authoritative first."""
    try:
        fetch_docnames(client, base_url, timeout)
        return "sphinx"
    except (httpx.HTTPError, ValueError):
        pass
    origin = _origin(base_url)
    robots = _load_robots(client, origin, timeout)
    if _discover_sitemap_urls(client, origin, robots, timeout):
        return "sitemap"
    return "crawl"


def crawl_site(
    client: httpx.Client,
    base_url: str,
    *,
    robots: RobotFileParser,
    path_prefix: str | None = None,
    max_pages: int = _MAX_PAGES,
    max_depth: int = _MAX_DEPTH,
    delay: float = _DEFAULT_FETCH_DELAY_SECONDS,
    timeout: int = _REQUEST_TIMEOUT_SECONDS,
    page_cache: dict[str, str] | None = None,
) -> list[str]:
    """Bounded, breadth-first, robots.txt-respecting crawl from `base_url`.

    Scope defaults to same-origin AND same-path-prefix (the prefix
    auto-derived from `base_url`'s directory unless overridden) - same-origin
    alone would sweep in a site's blog/marketing pages sitting alongside its
    docs section. Dedup is by normalized URL (fragment/query stripped,
    trailing slash collapsed). Non-HTML links are skipped by extension before
    ever being requested, and again by Content-Type after fetching. When
    `page_cache` is given, every fetched page's HTML is recorded into it
    (url -> html) so a caller converting the same pages afterwards does not
    have to re-fetch them.
    """
    origin = _origin(base_url)
    prefix = path_prefix if path_prefix is not None else _derive_path_prefix(base_url)

    # Dedup keys are normalized URLs, but every fetch uses the original
    # discovered URL - normalizing collapses a directory URL's trailing
    # slash, which is not safe to use as the literal request target.
    visited: set[str] = set()
    pages: list[str] = []
    queued: set[str] = {_normalize_url(base_url)}
    queue: list[tuple[str, int]] = [(base_url, 0)]

    while queue and len(pages) < max_pages:
        url, depth = queue.pop(0)
        key = _normalize_url(url)
        if key in visited:
            continue
        visited.add(key)

        if not robots.can_fetch(_USER_AGENT, url):
            continue
        try:
            response = _get(client, url, timeout)
        except httpx.HTTPError:
            continue

        content_type = response.headers.get("content-type", "")
        if "html" not in content_type:
            continue

        pages.append(url)
        if page_cache is not None:
            page_cache[url] = response.text
        time.sleep(delay)

        if depth >= max_depth:
            continue
        soup = BeautifulSoup(response.text, "html.parser")
        for anchor in soup.select("a[href]"):
            href = anchor.get("href")
            if not href:
                continue
            link = urljoin(url, href)
            if not _is_crawlable_link(link):
                continue
            normalized = _normalize_url(link)
            if normalized in queued or normalized in visited:
                continue
            if not _in_scope(normalized, origin, prefix):
                continue
            queued.add(normalized)
            queue.append((link, depth + 1))

    return pages


def iter_generic_pages(
    client: httpx.Client,
    project: DocsProject,
    *,
    discovery: str,
    delay: float = _DEFAULT_FETCH_DELAY_SECONDS,
    timeout: int = _REQUEST_TIMEOUT_SECONDS,
    max_pages: int = _MAX_PAGES,
    max_depth: int = _MAX_DEPTH,
) -> Iterator[PageContent]:
    """Every page of a non-Sphinx project, as `PageContent`.

    `discovery` is "sitemap" or "crawl". Sleeps `delay` seconds after each
    non-cached page fetch - a page already pulled into the crawl's own
    page_cache needs no further request, so no further sleep either.
    """
    origin = _origin(project.base_url)
    path_prefix = (
        project.path_prefix if project.path_prefix is not None
        else _derive_path_prefix(project.base_url)
    )
    robots = _load_robots(client, origin, timeout)
    page_cache: dict[str, str] = {}

    if discovery == "sitemap":
        sitemap_urls = _discover_sitemap_urls(client, origin, robots, timeout)
        page_urls = [
            url for url in _fetch_sitemap_pages(client, sitemap_urls, timeout)
            if _in_scope(_normalize_url(url), origin, path_prefix)
            and robots.can_fetch(_USER_AGENT, url)
        ]
    else:
        page_urls = crawl_site(
            client, project.base_url, robots=robots, path_prefix=path_prefix,
            max_pages=max_pages, max_depth=max_depth, delay=delay, timeout=timeout,
            page_cache=page_cache,
        )

    for page_url in page_urls:
        docname = _slug_from_url(page_url, path_prefix)
        if _is_excluded(docname, project.exclude):
            continue
        try:
            cached_html = page_cache.get(page_url)
            if cached_html is not None:
                html = cached_html
            else:
                html = _get(client, page_url, timeout).text
                time.sleep(delay)
            markdown = convert_page(html, page_url)
        except (httpx.HTTPError, ValueError) as error:
            yield PageContent(docname=docname, markdown=None, error=str(error), page_url=page_url)
            continue
        yield PageContent(docname=docname, markdown=markdown, page_url=page_url)


def iter_project_pages(
    client: httpx.Client,
    project: DocsProject,
    *,
    delay: float = _DEFAULT_FETCH_DELAY_SECONDS,
    timeout: int = _REQUEST_TIMEOUT_SECONDS,
    max_pages: int = _MAX_PAGES,
    max_depth: int = _MAX_DEPTH,
) -> Iterator[PageContent]:
    """Every page of a project, whatever its discovery mode, as `PageContent`.

    Dispatches to `iter_sphinx_pages`/`iter_generic_pages` on
    `project.discovery` if it is pinned, else on an auto-probe
    (`probe_discovery_mode`). The sole caller is
    `boepie.corpus.reconcile.sync_docs_project` - the `corpus add/fetch
    --collection docs` path.
    """
    discovery = project.discovery or probe_discovery_mode(client, project.base_url, timeout)
    if discovery == "sphinx":
        yield from iter_sphinx_pages(client, project, timeout, delay=delay)
    else:
        yield from iter_generic_pages(
            client, project, discovery=discovery, delay=delay, timeout=timeout,
            max_pages=max_pages, max_depth=max_depth,
        )
