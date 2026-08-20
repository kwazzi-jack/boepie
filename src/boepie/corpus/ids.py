"""Surrogate document ids: short, random, frozen at creation.

Decouples a corpus document's addressing handle (its ``id``, echoed back in
every MCP read handle) from its on-disk name, which is now free to be the
full human-legible title instead of a parseable slug/citekey - see the
corpus-unification design's settled-design notes. `secrets` (stdlib) is used
rather than adding a `nanoid` dependency: the alphabet/length choice below is
the entire design surface a third-party package would add on top of it.
"""

from __future__ import annotations

import secrets
import string

_ALPHABET = string.ascii_lowercase + string.digits
_DEFAULT_LENGTH = 10

# Retry bound for unique_id's collision loop - a correctness backstop, not a
# load-bearing mechanism: at length 10 over a 36-char alphabet the collision
# probability against any realistic corpus size is astronomically small.
_MAX_GENERATION_ATTEMPTS = 100


def generate_id(length: int = _DEFAULT_LENGTH) -> str:
    """A fresh random id, no uniqueness check - see `unique_id` for that."""
    return "".join(secrets.choice(_ALPHABET) for _ in range(length))


def unique_id(existing_ids: set[str], *, length: int = _DEFAULT_LENGTH) -> str:
    """A `generate_id` result not already present in `existing_ids`."""
    for _ in range(_MAX_GENERATION_ATTEMPTS):
        candidate = generate_id(length=length)
        if candidate not in existing_ids:
            return candidate
    raise ValueError(
        f"could not generate a unique {length}-char id after "
        f"{_MAX_GENERATION_ATTEMPTS} attempts"
    )
