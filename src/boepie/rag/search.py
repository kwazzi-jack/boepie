"""Reciprocal Rank Fusion: combines ranked index lists without normalising
their (very different) score scales.

Each chunk scores ``sum 1 / (RRF_K + rank)`` over the lists it appears in.
With only one retriever available the result is just that retriever's
ranking, so a lexical-only or dense-only query degrades cleanly. Used by
``rag.engine.query`` to fuse the dense embedding leg with the BM25 lexical
leg.
"""

from __future__ import annotations

from boepie.config import RRF_K


def _reciprocal_rank_fusion(*rank_lists: list[int]) -> list[tuple[int, float]]:
    """Fuse ranked index lists into (index, score) pairs sorted best-first."""
    scores: dict[int, float] = {}
    for ranks in rank_lists:
        for position, idx in enumerate(ranks):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (RRF_K + position + 1)
    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)


def _rank_of(idx: int, ranks: list[int]) -> int | None:
    """1-based position of ``idx`` in ``ranks``, or None if absent."""
    try:
        return ranks.index(idx) + 1
    except ValueError:
        return None
