"""Generic, source-agnostic chunking of markdown / plain text.

Splits a ``Document`` into ``Chunk`` spans that retain character offsets back
into the original text (so results can point at *where* they were found),
the nearest preceding heading as a section label, and any image references
found within the span resolved to real files.

Chunking is block-aware: a ``markdown-it-py`` parse (GFM tables + dollarmath
display-math) locates atomic block boundaries -- tables, fenced code, ``$$...$$``
math, lists, headings, blockquotes -- and the packer never bisects one. Only
prose (paragraph) blocks are split, by the sliding, whitespace-snapped window
this module already used before block-awareness. The parser is reused purely
to find block boundaries; offsets and packing remain ours.
"""

from __future__ import annotations

import re
from pathlib import Path

from markdown_it import MarkdownIt
from mdit_py_plugins.dollarmath import dollarmath_plugin

from boepie.config import PROSE_CHUNK_OVERLAP, PROSE_CHUNK_SIZE
from boepie.rag.models import Chunk, Document

# Markdown image reference: ![alt](path) -> capture the path.
IMAGE_REF = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")

# ATX headings (# .. ######) at the start of a line.
_HEADING = re.compile(r"^(#{1,6})[ \t]+(.*)$", re.MULTILINE)

# Parser used only to locate atomic block boundaries -- never to render or
# to walk inline content. "gfm-like" adds table recognition on top of
# CommonMark; linkify is disabled since it needs the optional
# linkify-it-py dependency and we never consult inline tokens anyway.
# dollarmath adds a block-level ``$$...$$`` display-math token.
_MD = MarkdownIt("gfm-like").disable("linkify")
_MD.use(dollarmath_plugin)


def resolve_image_refs(text: str, base_path: str | Path | None) -> list[str]:
    """Resolve markdown image refs in ``text`` to existing files under ``base_path``.

    Tries the ref as given and under an ``images/`` subdirectory (marker writes
    files there while emitting bare filenames). Order-preserving, de-duplicated.
    """
    if base_path is None:
        return []
    base = Path(base_path)
    resolved: list[str] = []
    for ref in IMAGE_REF.findall(text):
        name = Path(ref).name
        for candidate in (base / ref, base / "images" / name):
            if candidate.exists():
                resolved.append(str(candidate))
                break
    return list(dict.fromkeys(resolved))


def chunk_document(
    document: Document,
    *,
    chunk_size: int = PROSE_CHUNK_SIZE,
    overlap: int = PROSE_CHUNK_OVERLAP,
) -> list[Chunk]:
    """Split a document into overlapping chunks, section by section."""
    text = document.text
    chunks: list[Chunk] = []

    for start, end, section in _sections(text):
        section_text = text[start:end]
        blocks = _blocks(section_text)
        # No parseable blocks (e.g. a whitespace-only section) -> fall back
        # to treating the whole span as one window, same as pre-block-aware
        # behaviour; the empty/whitespace check below still skips it.
        local_spans = _pack_blocks(blocks, section_text, chunk_size, overlap) if blocks else [(0, end - start)]
        for local_start, local_end in local_spans:
            span_start, span_end = start + local_start, start + local_end
            span = text[span_start:span_end]
            if not span.strip():
                continue
            index = len(chunks)
            chunks.append(
                Chunk(
                    id=f"{document.id}::{index}",
                    collection="",  # set by the index builder
                    document_id=document.id,
                    chunk_index=index,
                    text=span,
                    source_path=document.source_path,
                    char_start=span_start,
                    char_end=span_end,
                    section=section,
                    metadata={
                        **document.metadata,
                        "images": resolve_image_refs(span, document.base_path),
                    },
                )
            )
    return chunks


def _sections(text: str) -> list[tuple[int, int, str | None]]:
    """Yield (start, end, section_title) spans delimited by markdown headings."""
    headings = list(_HEADING.finditer(text))
    if not headings:
        return [(0, len(text), None)]

    sections: list[tuple[int, int, str | None]] = []
    # Preamble before the first heading, if any.
    if headings[0].start() > 0:
        sections.append((0, headings[0].start(), None))

    for i, match in enumerate(headings):
        end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
        title = match.group(2).strip() or None
        sections.append((match.start(), end, title))
    return sections


def _line_offsets(text: str) -> list[int]:
    """Cumulative char offset of the start of each line, plus a sentinel = len(text).

    ``offsets[i]`` is the char offset where line ``i`` (0-indexed) begins;
    ``offsets[n]`` (n = number of lines) is ``len(text)``. This is what lets
    markdown-it's ``token.map`` -- a ``[start_line, end_line)`` pair -- be
    turned into the char offsets our chunks and their invariants are built on.

    Deliberately a plain ``\\n`` scan, NOT ``str.splitlines()``: Python's
    ``splitlines`` also breaks on ``\\x0b \\x0c \\x1c \\x1d \\x1e \\x85`` etc, but
    markdown-it-py's line numbering only ever splits on ``\\n`` (after
    normalizing ``\\r\\n?`` -> ``\\n``, which doesn't change the line *count*
    since ``\\r`` stays attached to its ``\\n``-terminated line). Splitting on
    anything ``splitlines`` treats as a break but markdown-it doesn't -- most
    notably the form-feed ``\\x0c`` PDF->markdown page-break artefact -- desyncs
    ``offsets`` from markdown-it's line indices and silently truncates or
    misplaces block boundaries from that point on.
    """
    offsets = [0]
    pos = 0
    while (nl := text.find("\n", pos)) != -1:
        pos = nl + 1
        offsets.append(pos)
    if offsets[-1] != len(text):
        offsets.append(len(text))
    return offsets


def _blocks(text: str) -> list[tuple[int, int, bool]]:
    """Parse ``text`` into top-level block spans: ``(start, end, is_atomic)``.

    Only tokens at nesting level 0 are considered, and only their opening (or
    self-contained) token -- markdown-it already sets a container's ``map`` to
    span its whole content, so the matching close token (``nesting == -1``,
    always ``map is None`` in practice) adds nothing. This naturally treats an
    entire table/list/blockquote as one span, regardless of how deeply nested
    its interior is: nested tokens sit at level >= 1 and are skipped.

    Every block type is atomic (never split by the packer) except
    ``paragraph_open`` (prose), which the packer may run its sliding window
    over if oversized.
    """
    if not text.strip():
        return []
    offsets = _line_offsets(text)
    blocks: list[tuple[int, int, bool]] = []
    for token in _MD.parse(text):
        if token.level != 0 or token.nesting == -1 or token.map is None:
            continue
        start_line, end_line = token.map
        char_start = offsets[start_line]
        char_end = offsets[end_line] if end_line < len(offsets) else len(text)
        blocks.append((char_start, char_end, token.type != "paragraph_open"))
    return blocks


def _pack_blocks(
    blocks: list[tuple[int, int, bool]], text: str, chunk_size: int, overlap: int
) -> list[tuple[int, int]]:
    """Pack block spans into chunk windows without ever bisecting an atomic block.

    Adjacent blocks are merged into a run while the run still fits under
    ``chunk_size``; a run flushes into one chunk as soon as the next block
    would overflow it. A block that is oversized *by itself* is handled on
    its own, without ever growing across chunk boundaries:

    - atomic (table/code/math/list/heading/...): kept whole as its own chunk.
      One over-long chunk beats a bisected table or math block.
    - prose (paragraph): the only case split, via the pre-existing
      whitespace-snapped sliding window (``_windows``), unchanged.
    """
    windows: list[tuple[int, int]] = []
    run_start: int | None = None
    run_end = 0

    def flush() -> None:
        if run_start is not None:
            windows.append((run_start, run_end))

    for block_start, block_end, is_atomic in blocks:
        if block_end - block_start > chunk_size:
            flush()
            run_start = None
            if is_atomic:
                windows.append((block_start, block_end))
            else:
                windows.extend(_windows(block_start, block_end, text, chunk_size, overlap))
            continue
        if run_start is None:
            run_start = block_start
        elif block_end - run_start > chunk_size:
            flush()
            run_start = block_start
        run_end = block_end

    flush()
    return windows


def _windows(
    start: int, end: int, text: str, chunk_size: int, overlap: int
) -> list[tuple[int, int]]:
    """Sliding character windows over [start, end), snapped to whitespace ends."""
    if end - start <= chunk_size:
        return [(start, end)]

    step = max(1, chunk_size - overlap)
    windows: list[tuple[int, int]] = []
    cursor = start
    while cursor < end:
        stop = min(cursor + chunk_size, end)
        stop = _snap_to_boundary(text, stop, end)
        windows.append((cursor, stop))
        if stop >= end:
            break
        cursor += step
    return windows


def _snap_to_boundary(text: str, stop: int, end: int, lookahead: int = 80) -> int:
    """Extend ``stop`` to the next whitespace within ``lookahead`` to avoid cutting words."""
    limit = min(stop + lookahead, end)
    for i in range(stop, limit):
        if text[i].isspace():
            return i
    return stop
