"""End-to-end tests for the search_literature / read_literature MCP tools.

Builds a tiny stub-embedding index (same offline pattern as test_engine.py)
and points boepie's default index lookup at it, so the tools are exercised
through the real MCP client transport without ever touching the real corpus
or a real Ollama/OpenAI backend.

Assertions target the F2/F4 output structure and payload bounds from
design/interface-spec.md, not the exact prose, which is expected to churn.
"""

from __future__ import annotations

import re

import numpy as np
import pytest
from fastmcp import Client
from fastmcp.client.transports import FastMCPTransport

from boepie.rag import embedding as embedding_mod
from boepie.rag import engine
from boepie.rag.embedding import ModelBinding
from boepie.rag.models import Document
from boepie.tools._retrieval import SHORT_SNIPPET_CHARS

_VOCAB = ["kalman", "filter", "calibrate", "clean", "deconvolve", "image", "gain"]
_BODY = (
    "Nonlinear Kalman filters calibrate radio interferometer gains over time "
    "using a state-space model. The filter propagates a gain estimate through "
    "a prediction step and a measurement update, so that solutions stay smooth "
    "across scan boundaries instead of jumping between independent solution "
    "intervals. This makes the approach attractive whenever the signal-to-noise "
    "per solution interval is too low to calibrate each interval on its own."
)
_DOCS = {"kalman2014": f"# Kalman calibration\n\n{_BODY}"}
_FAKE_BINDING = ModelBinding(kind="ollama", model="fake:bow", dim=len(_VOCAB))

# Mirrors the real corpus layout (full-title filename under the corpus root,
# surrogate id, nested `bib` block) so the source-path relativisation in
# _retrieval has its anchor to cut on and the alias map has a citekey to find.
_CORPUS_ROOT = "/home/someone/.local/share/boepie/literature-corpus"
_DOCUMENT_ID = "aB3dE9fGhI"
_CITEKEY = "kalman2014"


class _FakeLoader:
    name = "literature"

    def iter_documents(self):
        for citekey, text in _DOCS.items():
            yield Document(
                id=_DOCUMENT_ID,
                text=text,
                source_path=f"{_CORPUS_ROOT}/Fake Kalman Paper.md",
                metadata={
                    "title": "Fake Kalman Paper",
                    "managed_by": "boepie",
                    "bib": {"citekey": citekey, "year": "2014"},
                },
            )


def _bow_vector(text: str) -> np.ndarray:
    lowered = text.lower()
    counts = np.array([float(lowered.count(w)) for w in _VOCAB], dtype=np.float32)
    norm = np.linalg.norm(counts) or 1.0
    return counts / norm


async def _fake_embed_ollama(binding: ModelBinding, texts: list[str]) -> np.ndarray:
    return np.array([_bow_vector(t) for t in texts], dtype=np.float32)


def _text(result: object) -> str:
    content = getattr(result, "content", None)
    if content:
        return str(getattr(content[0], "text", content[0]))
    return str(result)


def _hit_lines(payload: str) -> list[str]:
    return [line for line in payload.splitlines() if re.match(r"^\[\d+\] ", line)]


async def _search(client, **overrides) -> str:
    payload = {"question": "kalman calibration", "mode": "bm25", **overrides}
    return _text(await client.call_tool("search_literature", {"input": payload}))


@pytest.fixture
async def literature_index(tmp_path, monkeypatch):
    """Build a stub literature index and make it the default one boepie loads."""
    monkeypatch.setattr(embedding_mod, "_embed_ollama", _fake_embed_ollama)
    await engine.build(_FakeLoader(), index_root=tmp_path, embedding=_FAKE_BINDING)

    # search_literature() calls boepie.rag.search() with no index_root/embedding
    # overrides, so it resolves the index root from engine.INDEX_DIR and the
    # embedding from default_embedding_binding() (which reads embedding module
    # globals). Patch both so the tool finds this stub index.
    monkeypatch.setattr(engine, "INDEX_DIR", tmp_path)
    monkeypatch.setattr(embedding_mod, "EMBEDDING_BINDING", _FAKE_BINDING.kind)
    monkeypatch.setattr(embedding_mod, "EMBEDDING_MODEL", _FAKE_BINDING.model)
    monkeypatch.setattr(embedding_mod, "EMBEDDING_DIM", _FAKE_BINDING.dim)

    yield
    await engine.clear_cache()


# ---------------------------------------------------------------------------
# F2: ranked hits
# ---------------------------------------------------------------------------


async def test_search_literature_emits_f2_count_line_and_numbered_blocks(
    literature_index, boepie_client: Client[FastMCPTransport]
):
    text = await _search(boepie_client)
    first = text.splitlines()[0]
    assert re.match(r'^\d+ hits for "kalman calibration" in literature$', first)
    hits = _hit_lines(text)
    assert hits
    assert int(first.split()[0]) == len(hits)


async def test_search_literature_hit_line_carries_title_and_section(
    literature_index, boepie_client: Client[FastMCPTransport]
):
    text = await _search(boepie_client)
    hit = _hit_lines(text)[0]
    assert "Fake Kalman Paper" in hit
    assert "#Kalman calibration" in hit  # section anchor appended to the title


async def test_search_literature_hit_line_carries_no_numeric_score(
    literature_index, boepie_client: Client[FastMCPTransport]
):
    # MCP output drops the score entirely; rank order is the relevance signal.
    text = await _search(boepie_client)
    hit = _hit_lines(text)[0]
    assert "score=" not in hit
    assert not re.search(r"\bbm25=|\bcos=|\brrf=", hit)


async def test_search_literature_surfaces_copy_pasteable_read_handles(
    literature_index, boepie_client: Client[FastMCPTransport]
):
    text = await _search(boepie_client)
    assert f"    read: document_id={_DOCUMENT_ID} chunk_index=" in text


async def test_search_literature_source_path_is_relative(
    literature_index, boepie_client: Client[FastMCPTransport]
):
    text = await _search(boepie_client)
    assert "source: Fake Kalman Paper.md (chars " in text
    assert _CORPUS_ROOT not in text  # no absolute path, no leaked home directory


async def test_search_literature_emits_no_banned_markdown(
    literature_index, boepie_client: Client[FastMCPTransport]
):
    text = await _search(boepie_client)
    generated = [line for line in text.splitlines() if not line.startswith("    ")]
    assert not [line for line in generated if line.startswith("#")]
    assert "---" not in text
    assert not [line for line in generated if line.startswith("_")]  # no italic footer


# ---------------------------------------------------------------------------
# Section 4: the snippet policy
# ---------------------------------------------------------------------------


async def test_search_literature_snippet_short_is_the_default_and_is_bounded(
    literature_index, boepie_client: Client[FastMCPTransport]
):
    default = await _search(boepie_client)
    explicit = await _search(boepie_client, snippet="short")
    assert default == explicit
    # The tail of the body, past the snippet cutoff, must not appear.
    assert "calibrate each interval on its own" not in default
    body_lines = [
        line
        for line in default.splitlines()
        if line.startswith("    ") and not line.lstrip().startswith(("read:", "source:"))
    ]
    assert body_lines
    for line in body_lines:
        assert len(line.strip()) <= SHORT_SNIPPET_CHARS + len(" ...")


async def test_search_literature_snippet_none_drops_bodies_and_shrinks_payload(
    literature_index, boepie_client: Client[FastMCPTransport]
):
    none_payload = await _search(boepie_client, snippet="none")
    short_payload = await _search(boepie_client, snippet="short")
    assert "Nonlinear Kalman filters" not in none_payload
    assert len(none_payload) < len(short_payload)
    # Handles survive: a caller can still escalate to read_literature.
    assert f"read: document_id={_DOCUMENT_ID}" in none_payload


async def test_search_literature_snippet_full_returns_whole_chunk(
    literature_index, boepie_client: Client[FastMCPTransport]
):
    full_payload = await _search(boepie_client, snippet="full")
    assert "calibrate each interval on its own" in full_payload
    assert len(full_payload) > len(await _search(boepie_client, snippet="short"))


# ---------------------------------------------------------------------------
# Degradation and errors
# ---------------------------------------------------------------------------


async def _raise_connection_error(binding: ModelBinding, texts: list[str]) -> np.ndarray:
    raise ConnectionError("Ollama call failed: connection refused.")


async def test_search_literature_falls_back_to_bm25_when_backend_unreachable(
    literature_index, monkeypatch, boepie_client: Client[FastMCPTransport]
):
    # Dense leg down; hybrid mode should degrade to lexical-only instead of erroring.
    monkeypatch.setattr(embedding_mod, "_embed_ollama", _raise_connection_error)
    text = await _search(boepie_client, mode="hybrid")
    note, rest = text.split("\n", 1)
    assert note.startswith("Note:") and "unreachable" in note
    assert "_" not in note  # the old note was italicised; italics are banned
    assert rest.startswith("1 hits for") or rest.startswith("0 hits for")
    assert _DOCUMENT_ID in text


async def test_search_literature_dense_mode_surfaces_backend_error(
    literature_index, monkeypatch, boepie_client: Client[FastMCPTransport]
):
    # Explicit dense request must not silently fall back to bm25.
    monkeypatch.setattr(embedding_mod, "_embed_ollama", _raise_connection_error)
    text = await _search(boepie_client, mode="dense")
    assert text.startswith("Error:")
    assert len(text.splitlines()) == 1


async def test_search_literature_no_index_reports_one_line_error_naming_a_command(
    tmp_path, monkeypatch, boepie_client: Client[FastMCPTransport]
):
    # Point INDEX_DIR at an empty dir so this is deterministic regardless of
    # whatever real index happens to be built on the dev machine.
    monkeypatch.setattr(engine, "INDEX_DIR", tmp_path)
    await engine.clear_cache()
    text = _text(
        await boepie_client.call_tool("search_literature", {"input": {"question": "anything"}})
    )
    assert text.startswith("Error:")
    assert len(text.splitlines()) == 1
    assert "boepie index build --collection literature" in text


async def test_search_literature_top_k_upper_bound_is_twenty(
    literature_index, boepie_client: Client[FastMCPTransport]
):
    with pytest.raises(Exception):
        await _search(boepie_client, top_k=21)


# ---------------------------------------------------------------------------
# F4: spans
# ---------------------------------------------------------------------------


async def test_read_literature_emits_f4_provenance_header(
    literature_index, boepie_client: Client[FastMCPTransport]
):
    text = _text(
        await boepie_client.call_tool(
            "read_literature", {"input": {"requests": [{"document_id": _DOCUMENT_ID}]}}
        )
    )
    lines = text.splitlines()
    assert re.match(
        rf"^document_id={_DOCUMENT_ID} chunks=\d+-\d+ chars=\d+-\d+$", lines[0]
    )
    assert lines[1] == "source: Fake Kalman Paper.md"
    assert lines[2] == "sections: Kalman calibration"
    assert lines[3] == ""
    assert "Nonlinear Kalman filters calibrate" in text
    assert "---" not in text


async def test_read_literature_batches_without_a_horizontal_rule(
    literature_index, boepie_client: Client[FastMCPTransport]
):
    text = _text(
        await boepie_client.call_tool(
            "read_literature",
            {
                "input": {
                    "requests": [
                        {"document_id": _DOCUMENT_ID, "chunk_index": 0, "after": 0},
                        {"document_id": _DOCUMENT_ID, "chunk_index": 0, "before": 0},
                    ]
                }
            },
        )
    )
    assert text.count(f"document_id={_DOCUMENT_ID}") == 2
    assert "---" not in text


async def test_read_literature_reports_unknown_document_on_one_line(
    literature_index, boepie_client: Client[FastMCPTransport]
):
    text = _text(
        await boepie_client.call_tool(
            "read_literature", {"input": {"requests": [{"document_id": "nope"}]}}
        )
    )
    assert text.startswith("Error:")
    assert len(text.splitlines()) == 1
    assert "search_literature" in text


# ---------------------------------------------------------------------------
# Read handles: surrogate ids, with human-writable aliases
# ---------------------------------------------------------------------------


async def test_read_literature_resolves_a_citekey_to_its_surrogate_id(
    boepie_client, literature_index
):
    """The context bundle's own paper stubs cite a citekey as the read
    handle, so a citekey has to keep working even though documents are
    addressed by an opaque id."""
    text = _text(
        await boepie_client.call_tool(
            "read_literature", {"input": {"requests": [{"document_id": _CITEKEY}]}}
        )
    )

    assert text.startswith(f"document_id={_DOCUMENT_ID}")


async def test_read_literature_resolves_a_citekey_case_insensitively(
    boepie_client, literature_index
):
    text = _text(
        await boepie_client.call_tool(
            "read_literature",
            {"input": {"requests": [{"document_id": _CITEKEY.upper()}]}},
        )
    )

    assert text.startswith(f"document_id={_DOCUMENT_ID}")


async def test_read_literature_still_rejects_an_invented_handle(
    boepie_client, literature_index
):
    text = _text(
        await boepie_client.call_tool(
            "read_literature",
            {"input": {"requests": [{"document_id": "totally-made-up"}]}},
        )
    )

    assert text.startswith("Error:")
    assert "search_literature" in text
