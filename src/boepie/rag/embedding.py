"""Embedding backends - a local ONNX model by default, or Ollama/OpenAI.

A ``ModelBinding`` names an embedding model and where to reach it;
``embed_texts`` turns a list of strings into an ``(n, dim)`` float32 matrix.
Both lifecycles use this: ``build()`` embeds every chunk once, the query path
embeds one question.

``fastembed`` is the default ``kind``: it runs a small ONNX model on CPU with
no server process, no API key, and no network beyond a one-time model
download cached locally - so `index build`/`search` work out of the box with
nothing else installed or running. ``ollama``/``openai`` remain available for
anyone already running local LLM infra or who wants a hosted API's quality.

Requests are batched and run with bounded concurrency so a large corpus embeds
in a handful of parallel calls rather than one giant request or a serial
trickle. Output row order always matches input text order, regardless of the
order batches happen to finish in.
"""

from __future__ import annotations

import asyncio
import functools
from dataclasses import dataclass
from typing import Callable, Literal

import click
import numpy as np

from boepie.config import (
    EMBEDDING_BINDING,
    EMBEDDING_CACHE_DIR,
    EMBEDDING_DIM,
    EMBEDDING_HOST,
    EMBEDDING_MODEL,
)

Kind = Literal["fastembed", "ollama", "openai"]

# Texts per provider request. Ollama and OpenAI both accept a batch; keeping it
# modest bounds the peak request size while still amortising per-call overhead.
_EMBED_BATCH = 64

# Concurrent in-flight batches when a binding doesn't set its own max_async.
_DEFAULT_MAX_ASYNC = 4


@dataclass(frozen=True)
class ModelBinding:
    """An embedding model + where to reach it.

    ``host`` doubles as an OpenAI ``base_url`` override: ``None`` means the real
    OpenAI API, a URL points at any OpenAI-compatible local server (vLLM,
    SGLang, TGI, ...). For ollama it's the daemon address.

    ``dim`` is the embedding dimension (required - it's recorded in the manifest
    and used to shape empty matrices). ``max_async`` caps concurrent requests;
    turn it down for rate-limited API keys.
    """

    kind: Kind
    model: str
    host: str | None = None
    dim: int | None = None
    max_async: int | None = None


def default_embedding_binding() -> ModelBinding:
    """The embedding binding implied by the current BOEPIE_EMBEDDING_* config."""
    return ModelBinding(EMBEDDING_BINDING, EMBEDDING_MODEL, EMBEDDING_HOST, EMBEDDING_DIM)


# fastembed's TextEmbedding loads and holds an ONNX session; keyed by model
# name so concurrent batches for the same model reuse one loaded model
# instead of re-downloading/re-initializing it per call.
_FASTEMBED_MODELS: dict[str, object] = {}


def _fastembed_model(model_name: str):
    if model_name not in _FASTEMBED_MODELS:
        from fastembed import TextEmbedding

        _FASTEMBED_MODELS[model_name] = TextEmbedding(
            model_name=model_name, cache_dir=str(EMBEDDING_CACHE_DIR)
        )
    return _FASTEMBED_MODELS[model_name]


async def _embed_fastembed(binding: ModelBinding, texts: list[str]) -> np.ndarray:
    model = _fastembed_model(binding.model)
    # TextEmbedding.embed() is a synchronous, CPU-bound ONNX call; run it off
    # the event loop so concurrent batches (bounded by max_async) can overlap.
    vectors = await asyncio.to_thread(lambda: list(model.embed(texts)))
    return np.array(vectors, dtype=np.float32)


async def _embed_ollama(binding: ModelBinding, texts: list[str]) -> np.ndarray:
    from ollama import AsyncClient

    response = await AsyncClient(host=binding.host).embed(model=binding.model, input=texts)
    return np.array(response.embeddings, dtype=np.float32)


async def _embed_openai(binding: ModelBinding, texts: list[str]) -> np.ndarray:
    from openai import AsyncOpenAI

    # base_url=None -> the real OpenAI API; a URL -> a local OpenAI-compatible
    # server. The SDK reads the API key from OPENAI_API_KEY (local servers
    # usually accept any value).
    client = AsyncOpenAI(base_url=binding.host)
    response = await client.embeddings.create(model=binding.model, input=texts)
    return np.array([item.embedding for item in response.data], dtype=np.float32)


def _embedder_for(binding: ModelBinding) -> Callable[[ModelBinding, list[str]], object]:
    if binding.kind == "fastembed":
        return _embed_fastembed
    if binding.kind == "ollama":
        return _embed_ollama
    if binding.kind == "openai":
        return _embed_openai
    raise ValueError(f"Unknown embedding binding kind '{binding.kind}'")


async def embed_texts(
    binding: ModelBinding,
    texts: list[str],
    *,
    on_progress: Callable[[int, int], None] | None = None,
) -> np.ndarray:
    """Embed ``texts`` into an ``(len(texts), dim)`` float32 matrix.

    Batches of ``_EMBED_BATCH`` run with at most ``binding.max_async`` in
    flight. ``on_progress(done, total)``, if given, is called with 0 up front
    and after each batch completes; ``done`` is monotonically non-decreasing.
    Row order matches input order.
    """
    if not texts:
        return np.zeros((0, binding.dim or 0), dtype=np.float32)

    embed_batch = _embedder_for(binding)
    batches = [texts[i : i + _EMBED_BATCH] for i in range(0, len(texts), _EMBED_BATCH)]
    results: list[np.ndarray | None] = [None] * len(batches)

    semaphore = asyncio.Semaphore(binding.max_async or _DEFAULT_MAX_ASYNC)
    progress_lock = asyncio.Lock()
    done = 0

    if on_progress is not None:
        on_progress(0, len(texts))

    async def run(index: int, batch: list[str]) -> None:
        nonlocal done
        async with semaphore:
            results[index] = await embed_batch(binding, batch)
        if on_progress is not None:
            async with progress_lock:
                done += len(batch)
                on_progress(done, len(texts))

    await asyncio.gather(*(run(i, batch) for i, batch in enumerate(batches)))
    return np.vstack(results)


# ---------------------------------------------------------------------------
# CLI plumbing: one shared decorator for the --embedding-* option group
# ---------------------------------------------------------------------------
#
# `index build` and `search` both need to construct a ModelBinding from the
# same four options, resolving and validating the host the same way. This lives
# here (next to ModelBinding) rather than in cli.py so there is a single source
# of truth for the option surface and the host rules, and no per-command drift.


def resolve_host(kind: str, host: str | None, ollama_default: str = EMBEDDING_HOST) -> str | None:
    """Resolve the effective host/base_url: an explicit value wins outright;
    otherwise ollama gets its usual local-daemon default, while fastembed and
    openai get ``None`` (fastembed ignores host entirely; ``None`` for openai
    means the real OpenAI API) - never silently reuse ollama's default URL as
    an OpenAI base_url override."""
    if host is not None:
        return host
    return ollama_default if kind == "ollama" else None


def validate_host(kind: str, host: str | None, flag: str = "--embedding-host") -> None:
    """Catch the easy mistakes: pointing ``--*-host`` at a provider name instead
    of switching ``--*-binding``, or setting a host at all for local fastembed.

    fastembed must have no host (it runs a local ONNX model, no server). ollama
    requires a URL host. openai's host is an optional base_url override - ``None``
    means the real API, but a given value must still be a URL (e.g. a local
    vLLM/SGLang/TGI). Failing fast here beats discovering it after burning real
    calls deep inside a build. Raises ``click.ClickException`` on a bad value.
    """
    if kind == "fastembed":
        if host is not None:
            raise click.ClickException(
                f"{flag} is not used for --*-binding=fastembed (it runs a local "
                f"ONNX model, no server). Omit {flag} or switch --*-binding."
            )
        return
    if host is not None and not host.startswith(("http://", "https://")):
        hint = (
            "the server address (e.g. http://localhost:11434)" if kind == "ollama"
            else "a base_url pointing at an OpenAI-compatible server (e.g. "
                 "http://localhost:8000/v1 for a local vLLM), or omit it entirely "
                 "to use the real OpenAI API"
        )
        raise click.ClickException(
            f"{flag}={host!r} doesn't look like a URL. For {kind}, {flag} must be {hint}."
        )
    if kind == "ollama" and host is None:
        raise click.ClickException(f"{flag} is required for --*-binding=ollama.")


# A zero-arg (optionally `max_async`-taking) callable the decorator injects, so
# a command that never needs a dense leg (a BM25-only collection) can simply not
# call it - and thus never triggers host resolution or validation.
EmbeddingResolver = Callable[..., "ModelBinding"]


def embedding_options(func: Callable) -> Callable:
    """Click decorator adding the shared ``--embedding-*`` option group and
    injecting a lazy ``resolve_embedding`` callable into the command.

    The command receives ``resolve_embedding(max_async=None) -> ModelBinding``
    instead of the four raw option values. Calling it resolves and validates the
    host, then builds the binding. A command targeting a BM25-only collection
    (``embedding=None``) just never calls it, so no host is required or checked.
    """

    @click.option(
        "--embedding-binding", default=EMBEDDING_BINDING, show_default=True,
        type=click.Choice(["fastembed", "ollama", "openai"]),
    )
    @click.option("--embedding-model", default=EMBEDDING_MODEL, show_default=True)
    @click.option(
        "--embedding-host", default=None,
        help=(
            f"Unused for fastembed. For ollama: server address (default "
            f"{EMBEDDING_HOST}). For openai: optional base_url override to hit a "
            f"local OpenAI-compatible server (vLLM/SGLang/TGI) instead of the real API."
        ),
    )
    @click.option("--embedding-dim", default=EMBEDDING_DIM, show_default=True, type=int)
    @functools.wraps(func)
    def wrapper(
        *args: object,
        embedding_binding: str,
        embedding_model: str,
        embedding_host: str | None,
        embedding_dim: int,
        **kwargs: object,
    ) -> object:
        def resolve(max_async: int | None = None) -> ModelBinding:
            host = resolve_host(embedding_binding, embedding_host)
            validate_host(embedding_binding, host)
            return ModelBinding(
                kind=embedding_binding, model=embedding_model, host=host,
                dim=embedding_dim, max_async=max_async,
            )

        return func(*args, resolve_embedding=resolve, **kwargs)

    return wrapper
