"""The `.boepie/` OKF context bundle: lifecycle library behind `boepie context`.

See ``boepie.context.bundle`` for ``init_bundle``/``apply_bundle``/
``reset_bundle``/``list_source_local_files``/``bundle_status``/
``fetch_content``/``resolve_content_source``/``append_agents_pointer``,
the ``find_bundle``/``index_root_for`` pair that
locates a project's bundle and its own search index, and
``boepie.context.frontmatter`` for the
YAML-frontmatter helpers they build on. CLI wiring lives elsewhere; this
package only owns the on-disk bundle and its content sources.
"""

from __future__ import annotations

from boepie.context.bundle import (
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
    list_source_local_files,
    reset_bundle,
    resolve_content_source,
)
from boepie.context.frontmatter import (
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
    "list_source_local_files",
    "reset_bundle",
    "resolve_content_source",
    "Frontmatter",
    "read_frontmatter",
    "read_frontmatter_file",
    "write_frontmatter",
]
