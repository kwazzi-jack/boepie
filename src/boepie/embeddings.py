"""Embedding interface using pydantic-ai's Embedder.

Supports OpenAI and Ollama backends, configured via BOEPIE_EMBEDDING_PROVIDER.
Embeddings are generated through pydantic-ai and passed as raw vectors to
ChromaDB - keeping pydantic-ai as the single owner of embedding logic.
"""

from __future__ import annotations

from pydantic_ai import Embedder

from boepie.config import (
    EMBEDDING_PROVIDER,
    OLLAMA_EMBEDDING_MODEL,
    OPENAI_EMBEDDING_MODEL,
)

# Provider string format expected by pydantic-ai: "provider:model"
_EMBEDDER_MODELS: dict[str, str] = {
    "openai": f"openai:{OPENAI_EMBEDDING_MODEL}",
    "ollama": f"ollama:{OLLAMA_EMBEDDING_MODEL}",
}


def get_embedder(provider: str | None = None) -> Embedder:
    """Create an Embedder for the configured provider.

    Parameters
    ----------
    provider:
        Override the default provider ("openai" or "ollama").
        Falls back to BOEPIE_EMBEDDING_PROVIDER from config/env.
    """
    provider = provider or EMBEDDING_PROVIDER
    model_str = _EMBEDDER_MODELS.get(provider)
    if model_str is None:
        available = ", ".join(sorted(_EMBEDDER_MODELS))
        raise ValueError(
            f"Unknown embedding provider '{provider}'. "
            f"Available: {available}"
        )
    return Embedder(model_str)
