"""Minimal YAML-frontmatter read/write helpers for OKF bundle files.

A bundle document is a markdown file with an optional leading YAML block
delimited by ``---`` lines. Every managed document carries the OKF fields
(``type``, ``title``, ``description``, ``tags``) plus boepie's own
``managed: boepie | human`` field that ``knowledge apply`` reads to decide
whether a file is safe to regenerate. ``index.md`` and ``log.md`` are the OKF
reserved files and carry no frontmatter at all: reading them returns an empty
mapping and the whole file as body.
"""

from __future__ import annotations

from pathlib import Path

import yaml

type Frontmatter = dict[str, str | list[str]]

_DELIMITER = "---"


def read_frontmatter(text: str) -> tuple[Frontmatter, str]:
    """Split ``text`` into its frontmatter mapping and body.

    Returns ``({}, text)`` unchanged when ``text`` has no leading ``---``
    block, or when the block is left unterminated.
    """
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].strip() != _DELIMITER:
        return {}, text

    for line_index in range(1, len(lines)):
        if lines[line_index].strip() != _DELIMITER:
            continue
        frontmatter_block = "".join(lines[1:line_index])
        body = "".join(lines[line_index + 1 :])
        parsed = yaml.safe_load(frontmatter_block) or {}
        return parsed, body.lstrip("\n")

    return {}, text


def write_frontmatter(frontmatter: Frontmatter, body: str) -> str:
    """Render ``frontmatter`` and ``body`` back into a single document string."""
    frontmatter_block = yaml.safe_dump(dict(frontmatter), sort_keys=False, default_flow_style=False)
    return f"{_DELIMITER}\n{frontmatter_block}{_DELIMITER}\n\n{body}"


def read_frontmatter_file(path: Path) -> tuple[Frontmatter, str]:
    """Convenience wrapper: read and parse a document straight from disk."""
    return read_frontmatter(path.read_text(encoding="utf-8"))
