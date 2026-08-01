# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "beautifulsoup4==4.12.3",
#     "click==8.3.1",
#     "markdownify==1.2.3",  # pinned: <1.0 lacks dl/dt/dd support (see convert_page)
#     "requests==2.32.3",
#     "rich==14.3.2",
# ]
# ///

"""
docs_to_md.py
--------------
Build the `docs-corpus/` tree that `boepie index build --collection docs`
consumes, by converting rendered upstream documentation pages to markdown.

Source is the *rendered* HTML on each project's docs site, not the `.rst`
sources in its repository. Several of the most valuable pages are generated
at Sphinx build time -- QuartiCal's whole options reference is a
`.. jinja::` template over its config schema -- so the repo sources contain
template code where the parameter documentation should be. The rendered page
is also format-agnostic: it does not care whether upstream writes RST, MyST
or mkdocs.

Pages are enumerated from each site's `searchindex.js` (`docnames`), which
Sphinx publishes for its own search box. That is the authoritative page list;
the RTD-generated `sitemap.xml` lists only version roots, and RTD's
downloadable `htmlzip` bundles are disabled for these projects.

Output layout, per `boepie.rag.loaders.DocsLoader`:

    docs-corpus/{project}/{docname}.md      # docname may be nested
    docs-corpus/{project}/metadata.json     # project, version, base_url, ...

Usage:
    uv run scripts/docs_to_md.py --all
    uv run scripts/docs_to_md.py --project stimela --out docs-corpus/
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path
from urllib.parse import urljoin

import click
import requests
from bs4 import BeautifulSoup
from markdownify import markdownify
from rich.console import Console

console = Console()

# Identify the crawler so upstream can attribute (and block) the traffic.
_USER_AGENT = "boepie-docs-to-md (+https://github.com/kwazzi-jack/boepie)"

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


@dataclass(frozen=True)
class ProjectSpec:
    """Where one project's rendered docs live, and what to leave out."""

    base_url: str
    exclude: tuple[str, ...] = field(default=())


# The curated set. cult-cargo is deliberately absent: it ships no docs site,
# and its cab YAML is already served authoritatively by boepie's own cab
# tools (L1 of the escalation ladder) -- a converted copy could only be a
# staler duplicate that contradicts the live tool.
_PROJECTS: dict[str, ProjectSpec] = {
    "stimela": ProjectSpec("https://stimela.readthedocs.io/en/latest/"),
    "quartical": ProjectSpec("https://quartical.readthedocs.io/en/latest/"),
    # ~60 of wsclean's ~96 pages are per-release changelogs: version history,
    # not usage, and they dominate the corpus by page count if kept.
    "wsclean": ProjectSpec(
        "https://wsclean.readthedocs.io/en/latest/", exclude=("changelogs/*",)
    ),
}


def _get(session: requests.Session, url: str, timeout: int) -> requests.Response:
    response = session.get(url, timeout=timeout)
    response.raise_for_status()
    return response


def fetch_docnames(session: requests.Session, base_url: str, timeout: int) -> list[str]:
    """Every page name the site publishes, from Sphinx's own search index."""
    raw = _get(session, urljoin(base_url, "searchindex.js"), timeout).text
    match = re.search(r"Search\.setIndex\((.*)\)\s*;?\s*$", raw.strip(), re.S)
    if match is None:
        raise click.ClickException(f"could not parse searchindex.js at {base_url}")
    return list(json.loads(match.group(1)).get("docnames", []))


def fetch_version(session: requests.Session, base_url: str, timeout: int) -> str:
    """The project's own version string, or the URL's version slug.

    Sphinx writes `VERSION` into `_static/documentation_options.js`, but only
    when the project sets it -- quartical and wsclean both publish an empty
    string. Falling back to the slug that actually served these pages
    ("latest") keeps the recorded provenance true rather than invented.
    """
    try:
        raw = _get(session, urljoin(base_url, "_static/documentation_options.js"), timeout).text
    except requests.RequestException:
        raw = ""
    match = re.search(r"VERSION:\s*['\"]([^'\"]+)['\"]", raw)
    if match is not None:
        return match.group(1)
    slug = [part for part in base_url.rstrip("/").split("/") if part]
    return slug[-1] if slug else "unknown"


def convert_page(html: str, page_url: str) -> str:
    """Rendered Sphinx page -> markdown holding just its prose."""
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


def build_project(
    name: str, spec: ProjectSpec, out_dir: Path, delay: float, timeout: int
) -> int:
    """Convert one project's site into `out_dir/{name}/`. Returns pages written."""
    project_dir = out_dir / name
    project_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers["User-Agent"] = _USER_AGENT

    docnames = fetch_docnames(session, spec.base_url, timeout)
    wanted = [docname for docname in docnames if not _is_excluded(docname, spec.exclude)]
    skipped = len(docnames) - len(wanted)

    console.print(
        f"[bold]{name}[/bold]: {len(wanted)} pages to fetch"
        + (f" ({skipped} excluded)" if skipped else "")
    )

    written = 0
    failures: list[str] = []
    with click.progressbar(wanted, label=f"  {name}", show_pos=True) as progress:
        for docname in progress:
            page_url = urljoin(spec.base_url, f"{docname}.html")
            try:
                html = _get(session, page_url, timeout).text
                markdown = convert_page(html, page_url)
            except (requests.RequestException, ValueError) as error:
                failures.append(f"{docname}: {error}")
                continue

            page_path = project_dir / f"{docname}.md"
            page_path.parent.mkdir(parents=True, exist_ok=True)
            page_path.write_text(markdown, encoding="utf-8")
            written += 1
            time.sleep(delay)

    metadata = {
        "project": name,
        "version": fetch_version(session, spec.base_url, timeout),
        "base_url": spec.base_url,
        "retrieved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "page_count": written,
    }
    (project_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    for failure in failures:
        console.print(f"  [yellow]skipped[/yellow] {failure}")
    console.print(
        f"  [green]{written} pages[/green] -> {project_dir} "
        f"(version={metadata['version']})"
    )
    return written


@click.command()
@click.option(
    "--project", "projects", multiple=True, type=click.Choice(sorted(_PROJECTS)),
    help="Project to convert; repeatable. Defaults to --all.",
)
@click.option("--all", "convert_all", is_flag=True, help="Convert every known project.")
@click.option(
    "--out", default="docs-corpus", show_default=True, type=click.Path(path_type=Path),
    help="Output corpus root.",
)
@click.option("--delay", default=0.2, show_default=True, type=float, help="Seconds between page requests.")
@click.option("--timeout", default=30, show_default=True, type=int, help="Per-request timeout in seconds.")
def docs_to_md(
    projects: tuple[str, ...], convert_all: bool, out: Path, delay: float, timeout: int
) -> None:
    """Convert rendered upstream docs sites into a boepie docs-corpus."""
    selected = sorted(_PROJECTS) if (convert_all or not projects) else list(projects)

    total = 0
    for name in selected:
        total += build_project(name, _PROJECTS[name], out, delay, timeout)

    console.print(
        f"\n[green]Done[/green]: {total} pages across {len(selected)} project(s) in {out}"
    )
    console.print("\nNext:\n  uv run boepie index build --collection docs")


if __name__ == "__main__":
    docs_to_md()
