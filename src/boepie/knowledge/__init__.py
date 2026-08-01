"""The `.boepie/` OKF knowledge bundle: lifecycle library behind `boepie knowledge`.

See ``boepie.knowledge.bundle`` for ``init_bundle``/``apply_bundle``/
``bundle_status``/``fetch_content``/``resolve_content_source``/
``append_agents_pointer``, the ``find_bundle``/``index_root_for`` pair that
locates a project's bundle and its own search index, and
``boepie.knowledge.frontmatter`` for the
YAML-frontmatter helpers they build on. CLI wiring lives elsewhere; this
package only owns the on-disk bundle and its content sources.
"""

from __future__ import annotations

from boepie.knowledge.bundle import (
    BundleManifest,
    BundleState,
    BundleStatus,
    ContentFetchResult,
    append_agents_pointer,
    apply_bundle,
    bundle_status,
    ensure_gitignore,
    fetch_content,
    find_bundle,
    index_root_for,
    init_bundle,
    is_bundle_dir,
    resolve_content_source,
)
from boepie.knowledge.frontmatter import (
    Frontmatter,
    read_frontmatter,
    read_frontmatter_file,
    write_frontmatter,
)

__all__ = [
    "BundleManifest",
    "BundleState",
    "BundleStatus",
    "ContentFetchResult",
    "append_agents_pointer",
    "apply_bundle",
    "bundle_status",
    "ensure_gitignore",
    "fetch_content",
    "find_bundle",
    "index_root_for",
    "init_bundle",
    "is_bundle_dir",
    "resolve_content_source",
    "Frontmatter",
    "read_frontmatter",
    "read_frontmatter_file",
    "write_frontmatter",
]
