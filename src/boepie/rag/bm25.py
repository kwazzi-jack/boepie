"""Lexical retrieval wrapper around bm25s.

Keeps the BM25 index aligned to the same ordered chunk list used everywhere
else, so a retrieved rank maps straight back to a chunk by integer index.
This is also the engine's pure-text path: a collection can be built with only
this index (no embeddings) for keyword-style lookup.
"""

from __future__ import annotations

from pathlib import Path

import bm25s

_STOPWORDS = "english"


class Bm25Index:
    """A persisted bm25s index over an ordered list of chunk texts."""

    def __init__(self, retriever: bm25s.BM25, count: int) -> None:
        self._retriever = retriever
        self._count = count

    @classmethod
    def build(cls, texts: list[str]) -> "Bm25Index":
        corpus_tokens = bm25s.tokenize(texts, stopwords=_STOPWORDS, show_progress=False)
        retriever = bm25s.BM25()
        retriever.index(corpus_tokens, show_progress=False)
        return cls(retriever, len(texts))

    def save(self, directory: Path | str) -> None:
        self._retriever.save(str(directory))

    @classmethod
    def load(cls, directory: Path | str, count: int) -> "Bm25Index":
        retriever = bm25s.BM25.load(str(directory), load_corpus=False)
        return cls(retriever, count)

    def retrieve(self, query: str, k: int) -> list[tuple[int, float]]:
        """Return up to ``k`` (chunk_index, score) pairs, best first."""
        if self._count == 0:
            return []
        k = min(k, self._count)
        query_tokens = bm25s.tokenize(query, stopwords=_STOPWORDS, show_progress=False)
        indices, scores = self._retriever.retrieve(
            query_tokens, k=k, show_progress=False
        )
        return [(int(i), float(s)) for i, s in zip(indices[0], scores[0])]
