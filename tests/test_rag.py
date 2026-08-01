"""Tests for the pure, model-free parts of the RAG engine.

Chunking, image resolution, RRF fusion, and metadata filters are pure,
offline, dependency-free logic - tested directly here. End-to-end
build/query behaviour lives in test_engine.py, with a stub embedding
function so it stays offline too.
"""

from __future__ import annotations

from boepie.rag.chunking import chunk_document, resolve_image_refs
from boepie.rag.loaders import DocsLoader, LiteratureLoader
from boepie.rag.models import Chunk, Document, Filter, combine_filters
from boepie.rag.search import _reciprocal_rank_fusion

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
# DocsLoader
# ---------------------------------------------------------------------------

import json


def test_docs_loader_document_ids(tmp_path):
    """Document IDs should be {project}/{page}."""
    corpus_dir = tmp_path / "docs-corpus"
    corpus_dir.mkdir()

    # Create two projects with pages.
    project1_dir = corpus_dir / "project1"
    project1_dir.mkdir()
    (project1_dir / "page1.md").write_text("Content 1A", encoding="utf-8")
    (project1_dir / "page2.md").write_text("Content 1B", encoding="utf-8")

    project2_dir = corpus_dir / "project2"
    project2_dir.mkdir()
    (project2_dir / "page1.md").write_text("Content 2A", encoding="utf-8")

    loader = DocsLoader(corpus_dir)
    documents = list(loader.iter_documents())

    # Should have 3 documents total.
    assert len(documents) == 3

    # Document IDs should have the correct format.
    document_ids = {doc.id for doc in documents}
    assert document_ids == {"project1/page1", "project1/page2", "project2/page1"}


def test_docs_loader_finds_nested_pages(tmp_path):
    """Pages in subdirectories must be loaded, with the nesting kept in the id.

    Upstream docs sites nest pages (stimela publishes `fundamentals/params`
    and `reference/cabdefs`), so a flat glob would silently drop the bulk of
    that corpus. The relative path is also what makes `base_url` + page +
    '.html' resolve back to the real page.
    """
    corpus_dir = tmp_path / "docs-corpus"
    project_dir = corpus_dir / "stimela"
    (project_dir / "fundamentals").mkdir(parents=True)
    (project_dir / "index.md").write_text("Top level", encoding="utf-8")
    (project_dir / "fundamentals" / "params.md").write_text("Nested", encoding="utf-8")

    documents = list(DocsLoader(corpus_dir).iter_documents())

    assert {doc.id for doc in documents} == {
        "stimela/index",
        "stimela/fundamentals/params",
    }
    nested = next(doc for doc in documents if doc.id.endswith("params"))
    assert nested.metadata["page"] == "fundamentals/params"


def test_docs_loader_metadata_merge_with_file(tmp_path):
    """Metadata from metadata.json should be merged into each document."""
    corpus_dir = tmp_path / "docs-corpus"
    corpus_dir.mkdir()

    project_dir = corpus_dir / "project1"
    project_dir.mkdir()
    (project_dir / "page1.md").write_text("Content", encoding="utf-8")
    (project_dir / "page2.md").write_text("More content", encoding="utf-8")

    metadata_file = project_dir / "metadata.json"
    metadata_file.write_text(
        json.dumps({"project": "project1", "version": "1.2.3", "base_url": "https://docs.example.com"}),
        encoding="utf-8",
    )

    loader = DocsLoader(corpus_dir)
    documents = list(loader.iter_documents())

    assert len(documents) == 2

    for doc in documents:
        assert doc.metadata["project"] == "project1"
        assert doc.metadata["version"] == "1.2.3"
        assert doc.metadata["base_url"] == "https://docs.example.com"
        # Each document should have its page name in metadata.
        assert doc.metadata["page"] in {"page1", "page2"}


def test_docs_loader_metadata_default_without_file(tmp_path):
    """Without metadata.json, should default to project name."""
    corpus_dir = tmp_path / "docs-corpus"
    corpus_dir.mkdir()

    project_dir = corpus_dir / "project1"
    project_dir.mkdir()
    (project_dir / "page1.md").write_text("Content", encoding="utf-8")

    loader = DocsLoader(corpus_dir)
    documents = list(loader.iter_documents())

    assert len(documents) == 1
    doc = documents[0]
    assert doc.metadata["project"] == "project1"
    assert doc.metadata["page"] == "page1"
    # Should not have version or base_url since metadata.json didn't exist.
    assert "version" not in doc.metadata
    assert "base_url" not in doc.metadata


def test_docs_loader_iteration_order_sorted(tmp_path):
    """Projects and pages should be iterated in sorted order."""
    corpus_dir = tmp_path / "docs-corpus"
    corpus_dir.mkdir()

    # Create projects in non-alphabetical order.
    for proj_name in ["zebra", "alpha", "beta"]:
        project_dir = corpus_dir / proj_name
        project_dir.mkdir()
        # Create pages in non-alphabetical order.
        for page_name in ["z_page", "a_page", "m_page"]:
            (project_dir / f"{page_name}.md").write_text(f"{proj_name}/{page_name}", encoding="utf-8")

    loader = DocsLoader(corpus_dir)
    documents = list(loader.iter_documents())

    # Should be 9 documents (3 projects * 3 pages).
    assert len(documents) == 9

    # Extract IDs to check order.
    document_ids = [doc.id for doc in documents]

    # Expect alphabetical order: alpha, beta, zebra (projects), then a_page, m_page, z_page (pages).
    expected_ids = [
        "alpha/a_page", "alpha/m_page", "alpha/z_page",
        "beta/a_page", "beta/m_page", "beta/z_page",
        "zebra/a_page", "zebra/m_page", "zebra/z_page",
    ]
    assert document_ids == expected_ids


def test_docs_loader_source_and_base_paths(tmp_path):
    """source_path and base_path should be set correctly."""
    corpus_dir = tmp_path / "docs-corpus"
    corpus_dir.mkdir()

    project_dir = corpus_dir / "project1"
    project_dir.mkdir()
    page_path = project_dir / "page1.md"
    page_path.write_text("Content", encoding="utf-8")

    loader = DocsLoader(corpus_dir)
    documents = list(loader.iter_documents())

    assert len(documents) == 1
    doc = documents[0]
    assert doc.source_path == str(page_path)
    assert doc.base_path == str(project_dir)


# ---------------------------------------------------------------------------
# Provenance (describe_sources)
# ---------------------------------------------------------------------------


def test_docs_loader_describes_each_project_source(tmp_path):
    """Provenance carries the upstream URL/version per project, not per page.

    A shipped docs index is a converted copy of someone else's site; without
    base_url and the retrieval date there is nothing in the tarball saying
    which snapshot it corresponds to.
    """
    corpus_dir = tmp_path / "docs-corpus"
    for name in ("quartical", "stimela"):
        project_dir = corpus_dir / name
        project_dir.mkdir(parents=True)
        (project_dir / "index.md").write_text("Content", encoding="utf-8")
        (project_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "project": name,
                    "version": "latest",
                    "base_url": f"https://{name}.readthedocs.io/en/latest/",
                    "retrieved_at": "2026-07-28T00:00:00Z",
                }
            ),
            encoding="utf-8",
        )

    sources = DocsLoader(corpus_dir).describe_sources()

    assert sources["corpus_dir"] == "docs-corpus"
    assert [project["project"] for project in sources["projects"]] == ["quartical", "stimela"]
    assert sources["projects"][1]["base_url"] == "https://stimela.readthedocs.io/en/latest/"
    assert sources["projects"][1]["retrieved_at"] == "2026-07-28T00:00:00Z"


def test_literature_loader_describes_bibliography(tmp_path):
    """The bib's checksum and entry count are recorded next to the citekeys.

    entry_count is deliberately the bib's own count rather than the number of
    converted papers, so a bib entry with no corpus directory shows up as a
    mismatch instead of vanishing.
    """
    corpus_dir = tmp_path / "stimela-corpus"
    corpus_dir.mkdir()
    bib_text = (
        "@article{smirnov2011a,\n  title = {RIME I},\n}\n"
        "@article{smirnov2011b,\n  title = {RIME II},\n}\n"
    )
    (corpus_dir / "corpus.bib").write_text(bib_text, encoding="utf-8")

    sources = LiteratureLoader(corpus_dir).describe_sources()

    assert sources["corpus_dir"] == "stimela-corpus"
    assert sources["bibtex"]["path"] == "corpus.bib"
    assert sources["bibtex"]["entry_count"] == 2
    assert len(sources["bibtex"]["sha256"]) == 64
    # Verbatim, so the shipped manifest alone reconstructs the bibliography.
    assert sources["bibtex"]["content"] == bib_text


def test_literature_loader_omits_bibliography_when_absent(tmp_path):
    """No bib file means no bibtex key - never a placeholder standing in for one."""
    corpus_dir = tmp_path / "stimela-corpus"
    corpus_dir.mkdir()

    assert "bibtex" not in LiteratureLoader(corpus_dir).describe_sources()
