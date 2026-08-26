"""One shell-pattern dialect, shared by everything in boepie that matches paths.

Two callers want the same semantics for different things: ``--group`` filters
indexed documents on their group metadata (``rag.models.Filter``), and
``corpus add`` expands a pattern into the files it names
(``corpus.inputs``). They must agree, or ``--group 'quartical/*'`` and
``corpus add notes 'quartical/*'`` would select different sets from the same
words.

It lives at the package root rather than in either caller because ``rag``
imports ``corpus`` and must keep doing so; a helper in ``rag`` that ``corpus``
reached back for would close that loop.

The dialect is the .gitignore reading, which is what anybody typing it
expects: ``*`` and ``?`` stop at a separator, ``**`` crosses them, and a
leading ``**/`` is an optional run of segments so ``**/gains`` matches a
top-level ``gains`` too. Note that plain ``fnmatch`` cannot express this - its
``*`` crosses ``/`` - which is why this is a regex translation rather than a
call into the standard library.
"""

from __future__ import annotations

import functools
import re

# The characters whose presence makes an argument a pattern rather than a
# path. Kept here so callers agree on that question too: `corpus add` uses it
# to tell "you typed a pattern that matched nothing" from "you typed a path
# that does not exist", which are different mistakes with different fixes.
GLOB_METACHARACTERS = "*?["


def looks_like_pattern(candidate: str) -> bool:
    """Whether `candidate` is meant as a pattern rather than a literal path."""
    return any(character in candidate for character in GLOB_METACHARACTERS)


@functools.lru_cache(maxsize=256)
def globstar_regex(pattern: str) -> re.Pattern[str]:
    """`pattern` as a regex where `**` crosses separators and `*` does not.

    `**/` collapses to an optional run of leading segments, so `**/gains`
    matches a top-level `gains` as well as a nested one - the .gitignore
    reading, and the one anybody typing it expects.
    """
    parts: list[str] = []
    index = 0
    while index < len(pattern):
        if pattern.startswith("**/", index):
            parts.append("(?:.*/)?")
            index += 3
        elif pattern.startswith("**", index):
            parts.append(".*")
            index += 2
        elif pattern[index] == "*":
            parts.append("[^/]*")
            index += 1
        elif pattern[index] == "?":
            parts.append("[^/]")
            index += 1
        else:
            parts.append(re.escape(pattern[index]))
            index += 1
    return re.compile("".join(parts))
