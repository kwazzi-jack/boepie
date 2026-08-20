"""Docs-project manifest: which upstream documentation sites boepie converts.

Two sources merge into one list: the packaged ``default_manifest.json``
(tracked in git, ships in the wheel) and a per-machine ``user-projects.json``
sitting next to the corpus itself, appended to by a future ``corpus add
--docs`` (see the Phase 2 plan's design recommendation for it). Structurally
this mirrors ``boepie.literature.manifest`` 1:1, with one addition a citekey
never needs: a docs ``project`` name is user-supplied directly (not derived
from fetched metadata) and becomes a directory name, so it has to be
validated before it is trusted as a path component.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

_DEFAULT_MANIFEST_PATH = Path(__file__).resolve().parent / "default_manifest.json"

# A project name becomes a directory (`docs_dir/{project}/`), so it must be a
# safe path component: lowercase alnum plus `-`/`_`, no leading punctuation
# (rules out both traversal segments like "../evil" and bare "..").
_PROJECT_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


@dataclass(frozen=True)
class DocsProject:
    project: str
    base_url: str
    exclude: tuple[str, ...] = ()
    # "sphinx" | "sitemap" | "crawl" override; None = auto-probe
    # (boepie.docs.fetch.probe_discovery_mode).
    discovery: str | None = None
    # Crawl/sitemap scope override; None = derive from base_url's directory.
    path_prefix: str | None = None

    def to_dict(self) -> dict[str, str | list[str] | None]:
        return {
            "project": self.project,
            "base_url": self.base_url,
            "exclude": list(self.exclude),
            "discovery": self.discovery,
            "path_prefix": self.path_prefix,
        }


def _read_projects(path: Path) -> list[DocsProject]:
    if not path.is_file():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    projects: list[DocsProject] = []
    for entry in raw:
        fields = dict(entry)
        if "exclude" in fields:
            fields["exclude"] = tuple(fields["exclude"])
        projects.append(DocsProject(**fields))
    return projects


def load_default_manifest() -> list[DocsProject]:
    """The tracked, packaged set of doc sites boepie converts by default."""
    return _read_projects(_DEFAULT_MANIFEST_PATH)


def load_manifest(docs_dir: Path) -> list[DocsProject]:
    """Every docs project `corpus fetch` reconciles against.

    Mirrors `boepie.literature.manifest.load_manifest`: one packaged source,
    with the directory argument retained so the two reconcilers share a
    calling convention.
    """
    return load_default_manifest()
