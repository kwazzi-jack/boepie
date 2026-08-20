"""Tests for the pure, model-free parts of the RAG engine.

Chunking, image resolution, RRF fusion, and metadata filters are pure,
offline, dependency-free logic - tested directly here. End-to-end
build/query behaviour lives in test_engine.py, with a stub embedding
function so it stays offline too.
"""

from __future__ import annotations

from pathlib import Path

from boepie.rag.chunking import chunk_document, resolve_image_refs
from boepie.rag.loaders import DocsLoader, LiteratureLoader, NotesLoader
from boepie.rag.models import Chunk, Document, Filter, combine_filters
from boepie.rag.search import _reciprocal_rank_fusion
from tests.conftest import write_corpus_document

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def test_chunk_offsets_roundtrip():
    doc = Document(id="d", text="# Heading\n\n" + "word " * 1000, source_path="/x.md")
    chunks = chunk_document(doc)
    assert len(chunks) > 1  # long enough to split
    for chunk in chunks:
        # The recorded span must reproduce the chunk text exactly.
        assert doc.text[chunk.char_start : chunk.char_end] == chunk.text
        assert chunk.section == "Heading"


def test_chunk_no_headings_single_section():
    doc = Document(id="d", text="just some short prose with no headings", source_path="/x.md")
    chunks = chunk_document(doc)
    assert len(chunks) == 1
    assert chunks[0].section is None


def test_image_resolution(tmp_path):
    (tmp_path / "images").mkdir()
    (tmp_path / "images" / "fig1.jpeg").write_bytes(b"x")
    text = "See ![](fig1.jpeg) and ![](missing.jpeg)."
    resolved = resolve_image_refs(text, tmp_path)
    assert resolved == [str(tmp_path / "images" / "fig1.jpeg")]


# ---------------------------------------------------------------------------
# Block-aware chunking: atomic blocks (table/code/math/list) are never split
# ---------------------------------------------------------------------------


def test_chunk_table_stays_whole():
    """A markdown table longer than chunk_size stays in one chunk, no row split."""
    table_md = "| a | b |\n|---|---|\n" + "| 1234567890 | 1234567890 |\n" * 80
    assert len(table_md) > 1500
    doc = Document(id="d", text=table_md, source_path="/x.md")
    chunks = chunk_document(doc)
    assert len(chunks) == 1
    assert chunks[0].text == table_md
    # No table row was split across a chunk boundary.
    assert doc.text[chunks[0].char_start : chunks[0].char_end].count("|") == table_md.count("|")


def test_chunk_code_fence_stays_whole():
    """A fenced code block longer than chunk_size stays whole, no split inside the fences."""
    body = "\n".join(f"line_{i} = {i}" for i in range(200))
    code_md = f"```python\n{body}\n```\n"
    assert len(code_md) > 1500
    doc = Document(id="d", text=code_md, source_path="/x.md")
    chunks = chunk_document(doc)
    assert len(chunks) == 1
    assert chunks[0].text == code_md
    assert chunks[0].text.count("```") == 2  # open and close fence both present


def test_chunk_math_block_stays_whole():
    """A $$...$$ display-math block is never split, even with surrounding prose splits."""
    # Prose sized so a naive whitespace-only sliding window would land its
    # boundary inside the math block; block-awareness must prevent that.
    prose_before = "word " * 300
    text = (
        prose_before
        + "\n\n$$\nE = mc^2 + \\int f(x) dx\n$$\n\ntail prose after the math block.\n"
    )
    doc = Document(id="d", text=text, source_path="/x.md")
    chunks = chunk_document(doc)
    for chunk in chunks:
        assert doc.text[chunk.char_start : chunk.char_end] == chunk.text
    math_chunks = [c for c in chunks if "$$" in c.text]
    assert len(math_chunks) == 1  # open and close both land in the same chunk
    assert math_chunks[0].text.count("$$") == 2


def test_chunk_list_not_split_mid_item():
    """A list longer than chunk_size stays coherent, no item split across chunks."""
    items = "\n".join(f"- item number {i} with some padding text to add length" for i in range(40))
    list_md = items + "\n"
    assert len(list_md) > 1500
    doc = Document(id="d", text=list_md, source_path="/x.md")
    chunks = chunk_document(doc)
    assert len(chunks) == 1
    assert chunks[0].text == list_md
    assert chunks[0].text.count("- item number") == 40


def test_chunk_mixed_document_roundtrip():
    """Offset roundtrip holds across a doc mixing prose, a table, code, and math."""
    table = "| a | b |\n|---|---|\n" + "| 1234567890 | 1234567890 |\n" * 60
    code = "```python\n" + "\n".join(f"x{i} = {i}" for i in range(150)) + "\n```\n"
    math = "$$\nE = mc^2\n$$\n"
    prose = "word " * 400
    text = (
        "# Heading\n\n"
        + prose
        + "\n\n"
        + table
        + "\n"
        + code
        + "\n"
        + math
        + "\nsome trailing prose.\n"
    )
    doc = Document(id="d", text=text, source_path="/x.md")
    chunks = chunk_document(doc)
    assert len(chunks) > 1
    for chunk in chunks:
        assert doc.text[chunk.char_start : chunk.char_end] == chunk.text
        assert chunk.section == "Heading"
    # The table and the code block each land whole in a single chunk.
    assert any(table in c.text for c in chunks)
    assert any("```python" in c.text and c.text.count("```") == 2 for c in chunks)


def test_chunk_table_survives_form_feed_in_preceding_prose():
    """A \\x0c (PDF page-break artefact) before a table must not desync offsets.

    str.splitlines() treats \\x0c as a line break; markdown-it-py does not. If
    _line_offsets used splitlines(), the phantom line would desync the offset
    table from markdown-it's line numbering from that point on, truncating or
    misplacing every block after the form feed -- including the table here.
    """
    prose = "Intro paragraph.\n\n\x0cAfter a form feed.\n\n"
    table = "| a | b |\n|---|---|\n" + "\n".join(f"| {i} | {i * 2} |" for i in range(150)) + "\n"
    text = prose + table
    assert len(table) > 1500
    doc = Document(id="t", text=text, source_path="/t.md")
    chunks = chunk_document(doc)
    for chunk in chunks:
        assert doc.text[chunk.char_start : chunk.char_end] == chunk.text
    # The whole table -- header separator through the final row -- survives
    # intact in one chunk; nothing is silently dropped past the form feed.
    table_chunks = [c for c in chunks if "|---|" in c.text]
    assert len(table_chunks) == 1
    assert "| 149 | 298 |" in table_chunks[0].text


def test_chunk_long_prose_still_overlaps():
    """A prose section longer than chunk_size still splits into overlapping windows."""
    doc = Document(id="d", text="word " * 1000, source_path="/x.md")
    chunks = chunk_document(doc)
    assert len(chunks) > 1
    for chunk in chunks:
        assert doc.text[chunk.char_start : chunk.char_end] == chunk.text
    # Consecutive windows overlap, as the pre-existing sliding window does.
    for prev, nxt in zip(chunks, chunks[1:]):
        assert nxt.char_start < prev.char_end


def test_pack_blocks_merges_small_runs_then_flushes():
    """Several small paragraphs whose combined length crosses chunk_size split on a
    paragraph boundary, not mid-paragraph -- the merge-then-flush branch at
    chunking.py:212-214 (``elif block_end - run_start > chunk_size: flush(); ...``).

    Every other chunk test uses a single block that is individually oversized;
    here no single paragraph is anywhere near chunk_size, only their sum is.
    """
    paragraphs = [f"Paragraph {i} has some sample padding text right here today." for i in range(6)]
    text = "\n\n".join(paragraphs)
    para_len = len(paragraphs[0])
    # Two consecutive paragraph blocks (each ~para_len+1 chars including their
    # own trailing newline) fit comfortably; a third pushes the run over.
    chunk_size = 2 * para_len + 12
    doc = Document(id="d", text=text, source_path="/x.md")
    chunks = chunk_document(doc, chunk_size=chunk_size, overlap=0)

    assert len(chunks) > 1
    seen = {p: 0 for p in paragraphs}
    for chunk in chunks:
        # Offset roundtrip holds for every chunk produced by the packer.
        assert doc.text[chunk.char_start : chunk.char_end] == chunk.text
        # The chunk is composed of whole source paragraphs -- splitting on the
        # blank-line separator and stripping recovers exactly original
        # paragraph strings, never a fragment of one.
        for part in chunk.text.split("\n\n"):
            stripped = part.strip()
            if not stripped:
                continue
            assert stripped in seen
            seen[stripped] += 1
    # Every paragraph landed intact in exactly one chunk.
    assert all(count == 1 for count in seen.values())


def test_snap_to_boundary_falls_back_when_no_whitespace_found():
    """A single unbroken token with no whitespace anywhere forces
    ``_snap_to_boundary``'s final ``return stop`` fallback (chunking.py:247),
    cutting mid-token instead of extending to a word boundary.

    Existing prose tests use ``"word " * N`` so whitespace is always found
    within the lookahead; this text has none at all.
    """
    chunk_size, overlap = 200, 20
    text = "x" * (chunk_size + 300)  # no whitespace anywhere
    doc = Document(id="d", text=text, source_path="/x.md")
    chunks = chunk_document(doc, chunk_size=chunk_size, overlap=overlap)

    assert len(chunks) > 1
    for chunk in chunks:
        assert doc.text[chunk.char_start : chunk.char_end] == chunk.text
    # Non-final windows are exactly chunk_size long: if whitespace had been
    # found within the lookahead, the stop would have been pushed further out.
    for chunk in chunks[:-1]:
        assert len(chunk.text) == chunk_size


def test_chunk_empty_document_returns_no_chunks():
    """An empty document yields no chunks and raises nothing."""
    doc = Document(id="d", text="", source_path="/x.md")
    assert chunk_document(doc) == []


def test_chunk_whitespace_only_document_returns_no_chunks():
    """A whitespace-only document also yields no chunks (the ``if not
    span.strip(): continue`` guard, plus ``_blocks``'s empty-return fallback)."""
    doc = Document(id="d", text="   \n\n\t  \n", source_path="/x.md")
    assert chunk_document(doc) == []


def test_chunk_blockquote_stays_whole():
    """A blockquote longer than chunk_size stays in one chunk, no line split.

    Mirrors test_chunk_table_stays_whole; blockquote is named as an atomic
    block type in the module docstring but had no dedicated test.
    """
    quote_md = "\n".join(f"> Quoted line number {i} with some padding text to add length." for i in range(40)) + "\n"
    assert len(quote_md) > 1500
    doc = Document(id="d", text=quote_md, source_path="/x.md")
    chunks = chunk_document(doc)
    assert len(chunks) == 1
    assert chunks[0].text == quote_md
    assert chunks[0].text.count("> Quoted line number") == 40


# ---------------------------------------------------------------------------
# Fusion and filters
# ---------------------------------------------------------------------------


def test_rrf_prefers_agreement():
    # Index 5 is top of one list and high in the other -> should win overall.
    dense = [5, 1, 2, 3]
    bm25 = [9, 5, 8, 7]
    fused = _reciprocal_rank_fusion(dense, bm25)
    assert fused[0][0] == 5
    # Items appearing in only one list still rank, but below the agreed one.
    ranked = [idx for idx, _ in fused]
    assert set(ranked) == {5, 1, 2, 3, 9, 8, 7}


def _chunk(meta: dict) -> Chunk:
    return Chunk(
        id="x::0", collection="c", document_id="x", chunk_index=0, text="t",
        source_path="/x.md", char_start=0, char_end=1, metadata=meta,
    )


def test_filter_predicates():
    assert Filter(field="year", op="gte", value=2000).predicate()(_chunk({"year": "2014"}))
    assert not Filter(field="year", op="gte", value=2000).predicate()(_chunk({"year": "1974"}))
    assert Filter(field="author", op="contains", value="tasse").predicate()(
        _chunk({"author": "Tasse, C."})
    )
    assert Filter(field="citekey", op="eq", value="a").predicate()(_chunk({"citekey": "a"}))
    # Missing field -> excluded.
    assert not Filter(field="year", op="gte", value=2000).predicate()(_chunk({}))


def test_filter_op_in():
    """``op="in"``: value is a list, predicate true when the field is a member."""
    pred = Filter(field="tag", op="in", value=["a", "b", "c"]).predicate()
    assert pred(_chunk({"tag": "b"}))
    assert not pred(_chunk({"tag": "z"}))


def test_filter_op_lte():
    """``op="lte"``: numeric comparison, mirroring the tested ``gte`` case."""
    pred = Filter(field="year", op="lte", value=2000).predicate()
    assert pred(_chunk({"year": "1974"}))
    assert not pred(_chunk({"year": "2014"}))


def test_filter_gte_lte_string_fallback():
    """``gte``/``lte`` on non-numeric strings run the lexical-comparison
    fallback in ``_coerce_pair`` (models.py:120-122), rather than raising or
    silently comparing as equal. Every other gte/lte test uses year strings
    that parse as numbers, so this path is otherwise untested."""
    assert Filter(field="name", op="gte", value="banana").predicate()(_chunk({"name": "cherry"}))
    assert not Filter(field="name", op="gte", value="banana").predicate()(_chunk({"name": "apple"}))
    assert Filter(field="name", op="lte", value="banana").predicate()(_chunk({"name": "apple"}))
    assert not Filter(field="name", op="lte", value="banana").predicate()(_chunk({"name": "cherry"}))


def test_combine_filters_is_and():
    pred = combine_filters([
        Filter(field="year", op="gte", value=2000),
        Filter(field="citekey", op="eq", value="kalman2014"),
    ])
    assert pred(_chunk({"year": "2014", "citekey": "kalman2014"}))
    assert not pred(_chunk({"year": "2014", "citekey": "other"}))


# ---------------------------------------------------------------------------
# Corpus loaders (literature / docs / notes)
# ---------------------------------------------------------------------------
#
# All three walk the shared corpus layout (see `boepie.corpus.layout`), so the
# behaviour worth testing per loader is only what it adds on top: the
# namespaced frontmatter block it carries through to metadata, and what its
# `describe_sources` records. The walk itself - groups, wrapped documents,
# skipped bookkeeping - is covered by tests/test_corpus_layout.py.


def test_docs_loader_ids_are_surrogates_not_paths(tmp_path):
    """A document is addressed by its frontmatter `id`, never by its location.

    The old loader derived an id from `{project}/{page}`, which meant moving
    or renaming a page silently invalidated every read handle pointing at it.
    """
    corpus_dir = tmp_path / "docs-corpus"
    write_corpus_document(
        corpus_dir, document_id="aaaaaaaaa1", title="Page One", body="Content 1A",
        group="project1", docs={"project": "project1", "page": "page1"},
    )
    write_corpus_document(
        corpus_dir, document_id="aaaaaaaaa2", title="Page Two", body="Content 1B",
        group="project1", docs={"project": "project1", "page": "page2"},
    )

    documents = list(DocsLoader(corpus_dir).iter_documents())

    assert {document.id for document in documents} == {"aaaaaaaaa1", "aaaaaaaaa2"}


def test_docs_loader_carries_the_docs_block_into_metadata(tmp_path):
    """Nested, not flattened: `Filter` reaches it with a dotted path."""
    corpus_dir = tmp_path / "docs-corpus"
    write_corpus_document(
        corpus_dir, document_id="aaaaaaaaa1", title="Recipes", body="Body.",
        group="stimela",
        docs={
            "project": "stimela", "page": "recipes",
            "base_url": "https://stimela.readthedocs.io/", "version": "1.3.0",
        },
    )

    document = next(iter(DocsLoader(corpus_dir).iter_documents()))

    assert document.metadata["docs"]["project"] == "stimela"
    assert document.metadata["docs"]["version"] == "1.3.0"
    assert document.metadata["title"] == "Recipes"


def test_docs_loader_finds_pages_at_any_nesting_depth(tmp_path):
    """A group is any directory, so upstream nesting survives as groups."""
    corpus_dir = tmp_path / "docs-corpus"
    write_corpus_document(
        corpus_dir, document_id="aaaaaaaaa1", title="Params", body="Nested.",
        group="stimela/fundamentals",
        docs={"project": "stimela", "page": "fundamentals/params"},
    )

    documents = list(DocsLoader(corpus_dir).iter_documents())

    assert len(documents) == 1
    assert documents[0].metadata["docs"]["page"] == "fundamentals/params"


def test_docs_loader_body_excludes_frontmatter(tmp_path):
    """Frontmatter is metadata, not searchable prose - indexing it would put
    YAML keys into the lexical leg of every query."""
    corpus_dir = tmp_path / "docs-corpus"
    write_corpus_document(
        corpus_dir, document_id="aaaaaaaaa1", title="Recipes", body="Just the body.",
        group="stimela", docs={"project": "stimela", "page": "recipes"},
    )

    document = next(iter(DocsLoader(corpus_dir).iter_documents()))

    assert document.text.strip() == "Just the body."
    assert "managed_by" not in document.text


def test_docs_loader_source_path_points_at_the_markdown(tmp_path):
    corpus_dir = tmp_path / "docs-corpus"
    md_path = write_corpus_document(
        corpus_dir, document_id="aaaaaaaaa1", title="Recipes", body="Body.",
        group="stimela", docs={"project": "stimela", "page": "recipes"},
    )

    document = next(iter(DocsLoader(corpus_dir).iter_documents()))

    assert document.source_path == str(md_path)
    # A bare leaf has no asset directory of its own, so it must not claim its
    # enclosing group as one - that would resolve a sibling's images.
    assert document.base_path is None


def test_docs_loader_wrapped_document_gets_its_wrapper_as_base_path(tmp_path):
    corpus_dir = tmp_path / "docs-corpus"
    write_corpus_document(
        corpus_dir, document_id="aaaaaaaaa1", title="Recipes",
        body="![fig](diagram.png)", group="stimela",
        assets={"diagram.png": b"not-really-a-png"},
        docs={"project": "stimela", "page": "recipes"},
    )

    document = next(iter(DocsLoader(corpus_dir).iter_documents()))

    assert document.base_path is not None
    assert document.base_path.endswith("Recipes")
    assert document.metadata["images"] == [str(Path(document.base_path) / "diagram.png")]


def test_docs_loader_describes_each_project_from_the_pages_themselves(tmp_path):
    """Provenance comes from what was actually indexed, not a side file that
    could disagree with it."""
    corpus_dir = tmp_path / "docs-corpus"
    for index, page in enumerate(("recipes", "cabs"), start=1):
        write_corpus_document(
            corpus_dir, document_id=f"stimela000{index}", title=f"Stimela {page}",
            body="Body.", group="stimela",
            docs={
                "project": "stimela", "page": page,
                "base_url": "https://stimela.readthedocs.io/", "version": "1.3.0",
            },
        )
    write_corpus_document(
        corpus_dir, document_id="quartical01", title="Solvers", body="Body.",
        group="quartical",
        docs={"project": "quartical", "page": "solvers",
              "base_url": "https://quartical.readthedocs.io/"},
    )

    sources = DocsLoader(corpus_dir).describe_sources()

    assert sources["corpus_dir"] == "docs-corpus"
    projects = {entry["project"]: entry for entry in sources["projects"]}
    assert projects["stimela"]["page_count"] == 2
    assert projects["stimela"]["version"] == "1.3.0"
    assert projects["quartical"]["page_count"] == 1


def test_literature_loader_carries_the_bib_block(tmp_path):
    corpus_dir = tmp_path / "literature"
    write_corpus_document(
        corpus_dir, document_id="smirnov0001", title="Revisiting the RIME",
        body="Body.", origin="arxiv:1101.1111", via="ar5iv", source_format="html",
        bib={"citekey": "smirnov2011", "year": "2011", "arxiv_id": "1101.1111"},
    )

    document = next(iter(LiteratureLoader(corpus_dir).iter_documents()))

    assert document.id == "smirnov0001"
    assert document.metadata["bib"]["citekey"] == "smirnov2011"
    assert document.metadata["source"]["via"] == "ar5iv"


def test_literature_loader_describes_the_packaged_manifest(tmp_path):
    """boepie ships no converted paper text, so what a built index can record
    is the manifest naming which arXiv ids it was meant to contain - a paper
    that failed to convert leaves no other trace."""
    corpus_dir = tmp_path / "literature"
    corpus_dir.mkdir()

    sources = LiteratureLoader(corpus_dir).describe_sources()

    assert sources["corpus_dir"] == "literature"
    assert sources["default_manifest"]["entry_count"] > 0
    assert sources["default_manifest"]["citekeys"] == sorted(
        sources["default_manifest"]["citekeys"]
    )


def test_notes_loader_adds_nothing_to_the_base_schema(tmp_path):
    """Notes are the base case: no natural key, no namespaced block."""
    corpus_dir = tmp_path / "notes"
    write_corpus_document(
        corpus_dir, document_id="note0000001", title="A Test Note", body="Body.",
        managed_by="user", origin="/home/someone/note.md", via="verbatim",
        source_format="markdown",
    )

    documents = list(NotesLoader(corpus_dir).iter_documents())

    assert len(documents) == 1
    assert documents[0].id == "note0000001"
    assert documents[0].metadata["title"] == "A Test Note"
    assert documents[0].metadata["managed_by"] == "user"
    assert "slug" not in documents[0].metadata


def test_notes_loader_recurses_into_user_groups(tmp_path):
    corpus_dir = tmp_path / "notes"
    write_corpus_document(
        corpus_dir, document_id="note0000001", title="Deep Note", body="Body.",
        managed_by="user", group="calibration/subtopic",
    )

    assert len(list(NotesLoader(corpus_dir).iter_documents())) == 1


def test_corpus_loaders_skip_pre_migration_documents(tmp_path):
    """A document with no `id` predates the corpus layout. Skipping it lets a
    half-migrated corpus build what it can instead of aborting the whole run.
    """
    corpus_dir = tmp_path / "notes"
    corpus_dir.mkdir()
    (corpus_dir / "Legacy Note.md").write_text(
        "---\ntitle: Legacy\n---\n\nBody.\n", encoding="utf-8"
    )
    write_corpus_document(
        corpus_dir, document_id="note0000001", title="Migrated", body="Body.",
        managed_by="user",
    )

    documents = list(NotesLoader(corpus_dir).iter_documents())

    assert [document.id for document in documents] == ["note0000001"]


def test_notes_loader_describes_sources(tmp_path):
    corpus_dir = tmp_path / "notes"
    corpus_dir.mkdir()

    assert NotesLoader(corpus_dir).describe_sources() == {"corpus_dir": "notes"}


# ---------------------------------------------------------------------------
# Read handles: alias resolution lives in the engine, so CLI and MCP share it
# ---------------------------------------------------------------------------


def _handle_with(chunks):
    from types import SimpleNamespace

    return SimpleNamespace(chunks=chunks)


def _alias_chunk(document_id: str, metadata: dict, chunk_index: int = 0):
    return Chunk(
        id=f"{document_id}::{chunk_index}",
        collection="literature",
        document_id=document_id,
        chunk_index=chunk_index,
        text="Body.",
        source_path="/corpus/Doc.md",
        char_start=0,
        char_end=5,
        metadata=metadata,
    )


def test_alias_map_resolves_citekey_arxiv_doi_and_title():
    from boepie.rag.engine import build_alias_map

    handle = _handle_with([
        _alias_chunk("aB3dE9fGhI", {
            "title": "CubiCal",
            "bib": {"citekey": "kenyon2018", "arxiv_id": "1805.03410", "doi": "10.1/x"},
        })
    ])

    aliases = build_alias_map(handle)

    for key in ("kenyon2018", "1805.03410", "10.1/x", "CubiCal", "cubical"):
        assert aliases[key] == "aB3dE9fGhI"


def test_read_span_resolves_an_alias_case_insensitively():
    """Case-insensitivity comes from lowercasing the lookup, not from storing
    an uppercase variant of every key."""
    from boepie.rag.engine import read_span

    handle = _handle_with([
        _alias_chunk("aB3dE9fGhI", {"title": "CubiCal", "bib": {"citekey": "kenyon2018"}})
    ])

    assert read_span(handle, "KENYON2018").document_id == "aB3dE9fGhI"


def test_alias_map_resolves_a_docs_project_page_pair():
    from boepie.rag.engine import build_alias_map

    handle = _handle_with([
        _alias_chunk("doc0000001", {"title": "Params", "docs": {"project": "stimela", "page": "params"}})
    ])

    assert build_alias_map(handle)["stimela/params"] == "doc0000001"


def test_alias_map_drops_a_key_shared_by_two_documents():
    """Two papers titled "Introduction" must not make one of them silently
    win: an ambiguous alias is no alias."""
    from boepie.rag.engine import build_alias_map

    handle = _handle_with([
        _alias_chunk("aaaaaaaaa1", {"title": "Introduction"}),
        _alias_chunk("aaaaaaaaa2", {"title": "Introduction"}),
    ])

    assert "Introduction" not in build_alias_map(handle)


def test_read_span_prefers_a_literal_id_over_an_alias():
    """A real id must never be shadowed by another document's title."""
    from boepie.rag.engine import build_alias_map, read_span

    handle = _handle_with([
        _alias_chunk("realid0001", {"title": "Decoy"}),
        _alias_chunk("aaaaaaaaa2", {"title": "realid0001"}),
    ])
    assert build_alias_map(handle)["realid0001"] == "aaaaaaaaa2"

    span = read_span(handle, "realid0001")

    assert span.document_id == "realid0001"


# ---------------------------------------------------------------------------
# Group scoping: the `group` metadata and the `glob` filter over it
# ---------------------------------------------------------------------------


def test_corpus_loader_records_the_group_a_document_is_filed_under(tmp_path):
    """Recorded explicitly rather than re-derived from source_path, which is
    absolute and machine-specific - an index built on one machine has to stay
    group-filterable on another."""
    write_corpus_document(
        tmp_path, document_id="aaaaaaaaaa", title="Top Level", body="Body.\n"
    )
    write_corpus_document(
        tmp_path, document_id="bbbbbbbbbb", title="Nested", body="Body.\n",
        group="calibration/gains",
    )

    groups = {
        document.id: document.metadata["group"]
        for document in NotesLoader(tmp_path).iter_documents()
    }
    assert groups == {"aaaaaaaaaa": "", "bbbbbbbbbb": "calibration/gains"}


def test_glob_filter_selects_a_group_and_its_descendants():
    in_gains = _chunk({"group": "calibration/gains"})
    at_root = _chunk({"group": ""})

    # A group selects what is filed under it.
    assert Filter(field="group", op="glob", value="calibration").predicate()(in_gains)
    assert Filter(field="group", op="glob", value="calibration/*").predicate()(in_gains)
    # `**/` matches at any depth, including none.
    assert Filter(field="group", op="glob", value="**/gains").predicate()(in_gains)
    assert not Filter(field="group", op="glob", value="wsclean").predicate()(in_gains)
    assert not Filter(field="group", op="glob", value="calibration").predicate()(at_root)


def test_glob_filter_stops_a_single_star_at_a_separator():
    """`fnmatch`'s `*` crosses `/`, which would make `*` and `**` synonyms and
    leave no way to pin a pattern to one level."""
    from boepie.rag.models import _globstar_regex

    assert _globstar_regex("*").fullmatch("wsclean")
    assert not _globstar_regex("*").fullmatch("wsclean/deep")
    assert _globstar_regex("**").fullmatch("wsclean/deep")


def test_filter_on_a_missing_field_excludes_rather_than_raises():
    """The silent-failure mode that made the CLI's flat `year`/`project`
    filters return zero hits against nested frontmatter."""
    assert not Filter(field="bib.year", op="gte", value=2000).predicate()(
        _chunk({"year": "2014"})
    )
    assert Filter(field="bib.year", op="gte", value=2000).predicate()(
        _chunk({"bib": {"year": "2014"}})
    )
