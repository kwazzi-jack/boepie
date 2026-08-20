"""Reads and writes one corpus leaf document (bare file or asset-wrapped
directory), built entirely on `context.frontmatter`'s codec - reused
verbatim, not duplicated or modified, per the corpus-unification design's
explicit judgment call.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from boepie.context.frontmatter import read_frontmatter, write_frontmatter
from boepie.corpus.layout import WRAPPED_DOCUMENT_FILENAME, CorpusFrontmatter


@dataclass(frozen=True)
class CorpusDocument:
    id: str
    md_path: Path
    wrapper_dir: Path | None
    frontmatter: CorpusFrontmatter
    body: str


def read_document(md_path: Path) -> CorpusDocument:
    """Parse one on-disk leaf document.

    Raises `ValueError` when its frontmatter carries no `id` - the
    pre-migration signal `scripts/migrate_corpus_layout.py` uses to tell an
    old-format document (a `metadata.json` sidecar, nothing parseable as
    frontmatter yet) from one it has already converted.
    """
    frontmatter, body = read_frontmatter(md_path.read_text(encoding="utf-8"))
    document_id = frontmatter.get("id")
    if not document_id:
        raise ValueError(f"{md_path}: no 'id' in frontmatter (pre-migration document?)")

    wrapper_dir = md_path.parent if md_path.name == WRAPPED_DOCUMENT_FILENAME else None
    return CorpusDocument(
        id=str(document_id), md_path=md_path, wrapper_dir=wrapper_dir,
        frontmatter=frontmatter, body=body,
    )


def _guard_against_id_collision(leaf_path: Path, document_id: str) -> None:
    """Refuses to overwrite an existing leaf-wrapped document that carries a
    different `id`. `WRAPPED_DOCUMENT_FILENAME` being fixed means a new
    document's wrapper directory name (title-derived) could - rarely, but
    not impossibly - coincide with an unrelated existing wrapper's name; a
    same-`id` write is a legitimate refetch/update and passes through
    unchanged."""
    if not leaf_path.is_file():
        return
    existing_frontmatter, _ = read_frontmatter(leaf_path.read_text(encoding="utf-8"))
    existing_id = existing_frontmatter.get("id")
    if existing_id and existing_id != document_id:
        raise ValueError(
            f"refusing to overwrite {leaf_path} (id={existing_id}) with a "
            f"different document (id={document_id})"
        )


def write_leaf_document(
    md_path: Path,
    *,
    document_id: str,
    frontmatter_fields: dict[str, Any],
    body: str,
    assets: dict[str, bytes] | None = None,
) -> CorpusDocument:
    """Write a new (or overwrite an existing) leaf document at `md_path`.

    `md_path` is always the bare-file target (`{group}/{Title}.md`); when
    `assets` is given (image filename -> bytes), the actual write lands one
    level deeper, at `{group}/{Title}/content.md` (`WRAPPED_DOCUMENT_FILENAME`)
    alongside each asset file - the caller never has to compute that path
    itself. Without assets, `md_path` is written directly. `frontmatter_fields`
    must not already set `id`: `document_id` is the single source of truth
    for it, merged in here so the two cannot desync.
    """
    if "id" in frontmatter_fields:
        raise ValueError("frontmatter_fields must not set 'id' directly; pass document_id instead")

    full_frontmatter: CorpusFrontmatter = {"id": document_id, **frontmatter_fields}
    text = write_frontmatter(full_frontmatter, body)

    if assets:
        wrapper_dir = md_path.parent / md_path.stem
        leaf_path = wrapper_dir / WRAPPED_DOCUMENT_FILENAME
        _guard_against_id_collision(leaf_path, document_id)
        wrapper_dir.mkdir(parents=True, exist_ok=True)
        leaf_path.write_text(text, encoding="utf-8")
        for asset_name, asset_bytes in assets.items():
            asset_path = wrapper_dir / asset_name
            asset_path.parent.mkdir(parents=True, exist_ok=True)
            asset_path.write_bytes(asset_bytes)
        return CorpusDocument(
            id=document_id, md_path=leaf_path, wrapper_dir=wrapper_dir,
            frontmatter=full_frontmatter, body=body,
        )

    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(text, encoding="utf-8")
    return CorpusDocument(
        id=document_id, md_path=md_path, wrapper_dir=None,
        frontmatter=full_frontmatter, body=body,
    )


def move_leaf_document(
    document: CorpusDocument, *, target_md_path: Path, frontmatter_updates: dict[str, Any] | None = None,
) -> CorpusDocument:
    """Relocate or rename a document, keeping its `id` and body intact.

    This is what the surrogate `id` buys: the document's location and its
    filename are both free to change, and every `read_*` handle pointing at
    it stays valid because none of them address it by path.

    `target_md_path` is the bare-file target; a wrapped document (one with
    assets) moves its whole wrapper directory instead, so the assets travel
    with it. `frontmatter_updates` is merged in before writing, for the
    fields a move implies - a new `title`, or the `docs.project` that has to
    follow a docs page into a different project group.
    """
    frontmatter: CorpusFrontmatter = {**document.frontmatter, **(frontmatter_updates or {})}
    id_field = frontmatter.pop("id", document.id)
    if id_field != document.id:
        raise ValueError("a move must not change a document's id")

    if document.wrapper_dir is not None:
        # Assets live beside content.md, so the unit that moves is the
        # directory, not the file inside it.
        target_dir = target_md_path.parent / target_md_path.stem
        if target_dir.resolve() != document.wrapper_dir.resolve():
            target_dir.parent.mkdir(parents=True, exist_ok=True)
            document.wrapper_dir.rename(target_dir)
        leaf_path = target_dir / WRAPPED_DOCUMENT_FILENAME
        leaf_path.write_text(
            write_frontmatter({"id": document.id, **frontmatter}, document.body),
            encoding="utf-8",
        )
        return CorpusDocument(
            id=document.id, md_path=leaf_path, wrapper_dir=target_dir,
            frontmatter=frontmatter, body=document.body,
        )

    target_md_path.parent.mkdir(parents=True, exist_ok=True)
    target_md_path.write_text(
        write_frontmatter({"id": document.id, **frontmatter}, document.body),
        encoding="utf-8",
    )
    if target_md_path.resolve() != document.md_path.resolve():
        document.md_path.unlink()
    return CorpusDocument(
        id=document.id, md_path=target_md_path, wrapper_dir=None,
        frontmatter=frontmatter, body=document.body,
    )
