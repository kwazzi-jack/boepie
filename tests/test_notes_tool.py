"""Wiring tests for the search_notes / read_notes MCP tools.

The F2/F4 rendering machinery (snippet policy, score suppression, error
shapes) is already exercised end-to-end against the identical shared
`_retrieval` helpers in test_literature_tool.py; these tests just confirm the
`notes` collection is wired correctly (VIEWS entry, source_root, tool
registration), not the generic behaviour.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastmcp import Client
from fastmcp.client.transports import FastMCPTransport

from boepie.rag import embedding as embedding_mod
from boepie.rag import engine
from boepie.rag.embedding import ModelBinding
from boepie.rag.models import Document

_VOCAB = ["substitution", "syntax", "recipe", "note"]
_BODY = "Some prose about substitution syntax in a recipe note."
_DOCS = {"a-test-note": f"# A Test Note\n\n{_BODY}"}
_FAKE_BINDING = ModelBinding(kind="ollama", model="fake:bow", dim=len(_VOCAB))

# Mirrors the real corpus layout ({corpus_root}/{slug}/{slug}.md).
_CORPUS_ROOT = "/home/someone/.local/share/boepie/notes"


class _FakeLoader:
    name = "notes"

    def iter_documents(self):
        for doc_id, text in _DOCS.items():
            yield Document(
                id=doc_id,
                text=text,
                source_path=f"{_CORPUS_ROOT}/{doc_id}/{doc_id}.md",
                metadata={"slug": doc_id, "title": "A Test Note", "source": "text"},
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


@pytest.fixture
async def notes_index(tmp_path, monkeypatch):
    monkeypatch.setattr(embedding_mod, "_embed_ollama", _fake_embed_ollama)
    await engine.build(_FakeLoader(), index_root=tmp_path, embedding=_FAKE_BINDING)

    monkeypatch.setattr(engine, "INDEX_DIR", tmp_path)
    monkeypatch.setattr(embedding_mod, "EMBEDDING_BINDING", _FAKE_BINDING.kind)
    monkeypatch.setattr(embedding_mod, "EMBEDDING_MODEL", _FAKE_BINDING.model)
    monkeypatch.setattr(embedding_mod, "EMBEDDING_DIM", _FAKE_BINDING.dim)

    yield
    await engine.clear_cache()


async def test_search_notes_returns_a_hit_with_a_read_handle(
    notes_index, boepie_client: Client[FastMCPTransport]
):
    text = _text(
        await boepie_client.call_tool(
            "search_notes", {"input": {"question": "substitution syntax", "mode": "bm25"}}
        )
    )
    assert "1 hits for" in text
    assert "A Test Note" in text
    assert "read: document_id=a-test-note chunk_index=" in text


async def test_search_notes_source_path_is_relative_to_notes_dir(
    notes_index, boepie_client: Client[FastMCPTransport]
):
    text = _text(
        await boepie_client.call_tool(
            "search_notes", {"input": {"question": "substitution syntax", "mode": "bm25"}}
        )
    )
    assert "source: a-test-note/a-test-note.md" in text
    assert _CORPUS_ROOT not in text


async def test_search_notes_no_index_names_the_build_command(
    tmp_path, monkeypatch, boepie_client: Client[FastMCPTransport]
):
    monkeypatch.setattr(engine, "INDEX_DIR", tmp_path)
    await engine.clear_cache()
    text = _text(
        await boepie_client.call_tool("search_notes", {"input": {"question": "anything"}})
    )
    assert text.startswith("Error:")
    assert "boepie index build --collection notes" in text


async def test_read_notes_expands_a_hit_into_a_provenance_header(
    notes_index, boepie_client: Client[FastMCPTransport]
):
    text = _text(
        await boepie_client.call_tool(
            "read_notes", {"input": {"requests": [{"document_id": "a-test-note"}]}}
        )
    )
    lines = text.splitlines()
    assert lines[0].startswith("document_id=a-test-note")
    assert lines[1] == "source: a-test-note/a-test-note.md"
    assert "Some prose about substitution syntax" in text


async def test_read_notes_reports_unknown_document_naming_search_notes(
    notes_index, boepie_client: Client[FastMCPTransport]
):
    text = _text(
        await boepie_client.call_tool(
            "read_notes", {"input": {"requests": [{"document_id": "nope"}]}}
        )
    )
    assert text.startswith("Error:")
    assert "search_notes" in text
