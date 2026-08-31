"""Offline end-to-end tests for the embeddings-only engine.

The embedding backend is stubbed at the provider boundary
(``embedding._embed_ollama``) with a deterministic bag-of-words vector over a
fixed vocabulary, so no real Ollama/OpenAI call ever happens while the real
batching, progress reporting, cosine, BM25, and RRF-fusion code all run.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from boepie.rag import embedding as embedding_mod
from boepie.rag import engine
from boepie.rag.embedding import ModelBinding
from boepie.rag.models import Document, Filter

_VOCAB = ["kalman", "filter", "calibrate", "clean", "deconvolve", "image", "gain"]
_DOCS = {
    "kalman2014": "# Kalman calibration\n\nNonlinear Kalman filters calibrate "
    "radio interferometer gains over time using a state-space model.",
    "clean1974": "# CLEAN\n\nThe CLEAN algorithm deconvolves the dirty image by "
    "iteratively subtracting scaled copies of the point spread function.",
}

_FAKE_BINDING = ModelBinding(kind="ollama", model="fake:bow", dim=len(_VOCAB))

# Long enough to split into several overlapping chunks - exercises read_span
# windowing and the overlap-stitching in _stitch.
_LONG_DOC = "# Intro\n\n" + " ".join(f"word{i}" for i in range(1200))


class FakeLoader:
    name = "tiny"

    def iter_documents(self):
        for doc_id, text in _DOCS.items():
            yield Document(
                id=doc_id,
                text=text,
                source_path=f"/tmp/{doc_id}.md",
                metadata={"citekey": doc_id, "year": "2014" if "2014" in doc_id else "1974"},
            )


class LongLoader:
    name = "long"

    def iter_documents(self):
        yield Document(
            id="paper",
            text=_LONG_DOC,
            source_path="/tmp/paper.md",
            metadata={"citekey": "paper", "title": "Long Paper"},
        )


def _bow_vector(text: str) -> np.ndarray:
    lowered = text.lower()
    counts = np.array([float(lowered.count(w)) for w in _VOCAB], dtype=np.float32)
    norm = np.linalg.norm(counts) or 1.0
    return counts / norm


async def _fake_embed_ollama(binding: ModelBinding, texts: list[str]) -> np.ndarray:
    return np.array([_bow_vector(t) for t in texts], dtype=np.float32)


@pytest.fixture
def fake_embedding(monkeypatch):
    """Patch the provider call so build()/query() never touch a real backend."""
    monkeypatch.setattr(embedding_mod, "_embed_ollama", _fake_embed_ollama)


async def _build(tmp_path):
    return await engine.build(FakeLoader(), index_root=tmp_path, embedding=_FAKE_BINDING)


# ---------------------------------------------------------------------------
# build()
# ---------------------------------------------------------------------------


async def test_build_writes_a_self_contained_collection(tmp_path, fake_embedding):
    manifest = await _build(tmp_path)
    assert manifest.count > 0

    collection_dir = tmp_path / "tiny" / manifest.index_id
    assert (collection_dir / "manifest.json").exists()
    assert (collection_dir / "chunks.jsonl").exists()
    assert (collection_dir / "bm25").exists()
    assert (collection_dir / "embeddings.npy").exists()
    assert (tmp_path / "tiny" / "latest.json").exists()
    assert manifest.embedding_model == "fake:bow"

    matrix = np.load(collection_dir / "embeddings.npy")
    assert matrix.shape == (manifest.count, len(_VOCAB))


async def test_a_loader_that_cannot_be_fingerprinted_records_no_revision(
    tmp_path, fake_embedding
):
    """`built_from: null` means *unverifiable*, and must never be read as
    fresh. FakeLoader yields documents from memory, so there is no corpus to
    walk again and nothing this index could later be compared against."""
    manifest = await _build(tmp_path)

    assert manifest.built_from is None

    on_disk = json.loads(
        (tmp_path / "tiny" / manifest.index_id / "manifest.json").read_text(encoding="utf-8")
    )
    assert on_disk["built_from"] is None


async def test_build_records_loader_described_sources(tmp_path, fake_embedding):
    """A loader's own corpus-level provenance, and only that.

    Which documents were indexed lives in `built_from`, where it is recorded
    with the digests that make it checkable. A second copy here would be one
    more thing that could disagree with the index beside it."""

    class DescribingLoader(FakeLoader):
        def describe_sources(self):
            return {"corpus_dir": "somewhere", "projects": [{"project": "tiny"}]}

    manifest = await engine.build(
        DescribingLoader(), index_root=tmp_path, embedding=_FAKE_BINDING
    )

    assert manifest.sources is not None
    assert manifest.sources["corpus_dir"] == "somewhere"
    assert manifest.sources["projects"] == [{"project": "tiny"}]
    assert "documents" not in manifest.sources


async def test_manifest_without_sources_still_loads(tmp_path, fake_embedding):
    """Indices built before provenance existed must keep loading, not error."""
    manifest = await _build(tmp_path)
    manifest_path = tmp_path / "tiny" / manifest.index_id / "manifest.json"

    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    del data["sources"]
    manifest_path.write_text(json.dumps(data), encoding="utf-8")

    handle = await engine.load_for_query(tmp_path, "tiny", embedding=_FAKE_BINDING)
    assert handle.manifest.sources is None


async def test_build_reports_progress_up_to_total(tmp_path, fake_embedding):
    events: list[tuple[int, int]] = []
    manifest = await engine.build(
        FakeLoader(), index_root=tmp_path, embedding=_FAKE_BINDING,
        on_progress=lambda done, total: events.append((done, total)),
    )
    assert events  # called at least once
    assert events[0] == (0, manifest.count)
    assert events[-1] == (manifest.count, manifest.count)
    # done must be non-decreasing across calls
    assert all(a[0] <= b[0] for a, b in zip(events, events[1:]))


# ---------------------------------------------------------------------------
# load_for_query()
# ---------------------------------------------------------------------------


async def test_load_for_query_rejects_embedding_mismatch(tmp_path, fake_embedding):
    await _build(tmp_path)
    mismatched = ModelBinding(kind="ollama", model="something-else", dim=8)
    with pytest.raises(ValueError):
        await engine.load_for_query(tmp_path, "tiny", embedding=mismatched)


async def test_load_for_query_missing_collection_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        await engine.load_for_query(tmp_path, "nonexistent", embedding=_FAKE_BINDING)


# ---------------------------------------------------------------------------
# query()
# ---------------------------------------------------------------------------


async def test_query_bm25_finds_keyword_match(tmp_path, fake_embedding):
    await _build(tmp_path)
    handle = await engine.load_for_query(tmp_path, "tiny", embedding=_FAKE_BINDING)
    results = await engine.query(handle, "kalman calibration", mode="bm25", top_k=5)
    assert results
    assert results[0].chunk.metadata["citekey"] == "kalman2014"
    assert results[0].dense_rank is None


async def test_query_dense_uses_fake_embedding(tmp_path, fake_embedding):
    await _build(tmp_path)
    handle = await engine.load_for_query(tmp_path, "tiny", embedding=_FAKE_BINDING)
    results = await engine.query(handle, "deconvolve the image", mode="dense", top_k=5)
    assert results
    assert results[0].chunk.metadata["citekey"] == "clean1974"
    assert results[0].bm25_rank is None


async def test_query_hybrid_fuses_both_legs(tmp_path, fake_embedding):
    await _build(tmp_path)
    handle = await engine.load_for_query(tmp_path, "tiny", embedding=_FAKE_BINDING)
    results = await engine.query(handle, "calibration gains", mode="hybrid", top_k=5)
    assert results
    assert results[0].chunk.metadata["citekey"] == "kalman2014"


async def test_query_filter_excludes_by_year(tmp_path, fake_embedding):
    await _build(tmp_path)
    handle = await engine.load_for_query(tmp_path, "tiny", embedding=_FAKE_BINDING)
    results = await engine.query(
        handle, "image", mode="hybrid", top_k=5,
        filters=[Filter(field="year", op="gte", value=2000)],
    )
    assert results  # only kalman2014 (year 2014) survives the filter
    assert all(r.chunk.metadata["year"] == "2014" for r in results)


# ---------------------------------------------------------------------------
# read_span() - guided lookup
# ---------------------------------------------------------------------------


async def test_read_span_whole_document_reconstructs_text(tmp_path, fake_embedding):
    await engine.build(LongLoader(), index_root=tmp_path, embedding=_FAKE_BINDING)
    handle = await engine.load_for_query(tmp_path, "long", embedding=_FAKE_BINDING)
    assert len(handle.chunks) > 1  # actually split, so stitching is non-trivial

    span = engine.read_span(handle, "paper")  # chunk_index=None -> whole doc
    assert span.text == _LONG_DOC
    assert span.char_start == 0
    assert span.char_end == len(_LONG_DOC)


async def test_read_span_window_is_a_contiguous_slice(tmp_path, fake_embedding):
    await engine.build(LongLoader(), index_root=tmp_path, embedding=_FAKE_BINDING)
    handle = await engine.load_for_query(tmp_path, "long", embedding=_FAKE_BINDING)

    anchor = handle.chunks[len(handle.chunks) // 2].chunk_index
    span = engine.read_span(handle, "paper", chunk_index=anchor, before=1, after=1)
    assert span.chunk_start <= anchor <= span.chunk_end
    # A stitched window is exactly the original document's slice over its span.
    assert span.text == _LONG_DOC[span.char_start : span.char_end]


async def test_read_span_unknown_document_or_chunk_raises(tmp_path, fake_embedding):
    await _build(tmp_path)
    handle = await engine.load_for_query(tmp_path, "tiny", embedding=_FAKE_BINDING)
    with pytest.raises(KeyError):
        engine.read_span(handle, "does-not-exist")
    with pytest.raises(KeyError):
        engine.read_span(handle, "kalman2014", chunk_index=9999)


# ---------------------------------------------------------------------------
# search() convenience wrapper + cache
# ---------------------------------------------------------------------------


async def test_search_wrapper_round_trips_through_cache(tmp_path, fake_embedding):
    await _build(tmp_path)
    try:
        results = await engine.search(
            "kalman filters", collection="tiny", top_k=5, mode="bm25",
            index_root=tmp_path, embedding=_FAKE_BINDING,
        )
        assert results
        assert results[0].chunk.metadata["citekey"] == "kalman2014"
    finally:
        await engine.clear_cache()


# ---------------------------------------------------------------------------
# Lexical-only (embedding=None): BM25 with no embedding backend at all
# ---------------------------------------------------------------------------


async def test_build_with_no_embedding_writes_no_embeddings_file(tmp_path):
    manifest = await engine.build(FakeLoader(), index_root=tmp_path, embedding=None)
    assert manifest.count > 0
    assert manifest.index_id == "bm25"
    assert manifest.lexical_only is True
    assert manifest.embedding_kind is None
    assert manifest.embedding_model is None
    assert manifest.embedding_dim is None

    collection_dir = tmp_path / "tiny" / manifest.index_id
    assert (collection_dir / "manifest.json").exists()
    assert (collection_dir / "chunks.jsonl").exists()
    assert (collection_dir / "bm25").exists()
    assert not (collection_dir / "embeddings.npy").exists()


async def test_load_for_query_lexical_only_has_no_matrix_or_binding(tmp_path):
    await engine.build(FakeLoader(), index_root=tmp_path, embedding=None)
    handle = await engine.load_for_query(tmp_path, "tiny")
    assert handle.matrix is None
    assert handle.embedding is None
    assert handle.manifest.lexical_only is True


async def test_query_bm25_against_lexical_only_index_ranks_hits(tmp_path):
    await engine.build(FakeLoader(), index_root=tmp_path, embedding=None)
    handle = await engine.load_for_query(tmp_path, "tiny")
    results = await engine.query(handle, "kalman calibration", mode="bm25", top_k=5)
    assert results
    assert results[0].chunk.metadata["citekey"] == "kalman2014"
    assert results[0].dense_rank is None


async def test_query_hybrid_against_lexical_only_index_raises(tmp_path):
    await engine.build(FakeLoader(), index_root=tmp_path, embedding=None)
    handle = await engine.load_for_query(tmp_path, "tiny")
    with pytest.raises(ValueError, match="bm25"):
        await engine.query(handle, "kalman calibration", mode="hybrid")


async def test_query_dense_against_lexical_only_index_raises(tmp_path):
    await engine.build(FakeLoader(), index_root=tmp_path, embedding=None)
    handle = await engine.load_for_query(tmp_path, "tiny")
    with pytest.raises(ValueError, match="bm25"):
        await engine.query(handle, "kalman calibration", mode="dense")


async def test_search_wrapper_works_with_no_embedding_configured(tmp_path):
    await engine.build(FakeLoader(), index_root=tmp_path, embedding=None)
    try:
        results = await engine.search(
            "kalman filters", collection="tiny", top_k=5, mode="bm25", index_root=tmp_path,
        )
        assert results
        assert results[0].chunk.metadata["citekey"] == "kalman2014"
    finally:
        await engine.clear_cache()


# ---------------------------------------------------------------------------
# Cache invalidation
# ---------------------------------------------------------------------------


class MutableLoader:
    """A loader whose one document can be rewritten between builds."""

    name = "mutable"

    def __init__(self, text: str) -> None:
        self.text = text

    def iter_documents(self):
        yield Document(
            id="note", text=self.text, source_path="/tmp/note.md", metadata={"citekey": "note"}
        )


async def test_rebuilding_evicts_the_cached_handle(tmp_path):
    """A rebuild in the same process must be visible to the next search.

    `context apply` rebuilds and `search_context` runs in the same
    long-lived server process, so a cached pre-rebuild handle would keep
    serving deleted content indefinitely.
    """
    loader = MutableLoader("# Note\n\nThe original text mentions kalman filters.\n")
    await engine.build(loader, index_root=tmp_path, embedding=None)
    try:
        before = await engine.search(
            "text", collection="mutable", top_k=1, mode="bm25", index_root=tmp_path
        )
        assert "original" in before[0].chunk.text

        loader.text = "# Note\n\nThe replacement text mentions deconvolution.\n"
        await engine.build(loader, index_root=tmp_path, embedding=None)

        after = await engine.search(
            "text", collection="mutable", top_k=1, mode="bm25", index_root=tmp_path
        )
        assert "replacement" in after[0].chunk.text
    finally:
        await engine.clear_cache()


async def test_build_hint_names_a_real_command_per_collection(tmp_path):
    """The fix clause must name a command that actually exists."""
    with pytest.raises(FileNotFoundError) as literature_error:
        await engine.load_for_query(tmp_path, "literature")
    assert "boepie index fetch --collection literature" in str(literature_error.value)

    with pytest.raises(FileNotFoundError) as context_error:
        await engine.load_for_query(tmp_path, "context")
    message = str(context_error.value)
    assert "boepie context apply" in message
    assert "boepie index" not in message
