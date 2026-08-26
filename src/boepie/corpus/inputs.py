"""Turning what was typed on the command line into a concrete list of files.

`corpus add` takes identifiers of several kinds - an arXiv id, a DOI, a URL, a
path, a `.bib` - and each collection resolves those differently. This module
handles only the part that is the same for all of them and needs neither the
network nor the corpus: deciding which *files* an argument names.

That narrowness is the design. A general "plan the whole batch first" phase
cannot work here, because resolving an arXiv id means fetching it and
resolving a citekey means having loaded the collection. Expanding a path or a
pattern needs neither, so it can run to completion before anything slow
starts, and everything else is left exactly where it was.

Two rules worth stating out loud:

- **Patterns are expanded here, not by the shell.** A pattern that the shell
  already expanded arrives as several existing paths and passes straight
  through; one it did not arrives as a literal string and is expanded here.
  Both routes end in the same list. This is what makes the command behave the
  same way under bash, zsh and fish - which disagree about `**` - and on
  Windows, where the shell never expands arguments for a program at all. It
  also sidesteps the argument-length ceiling: a pattern naming fifteen
  thousand files is a few dozen bytes crossing into the process, where the
  expanded list is several megabytes and fails before boepie starts.
- **A pattern matching nothing is an error, never zero files.** Quietly adding
  nothing is the one outcome that looks like success and is not.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

from boepie._glob import globstar_regex, looks_like_pattern

# Where one resolved file came from. Only `pattern` and `directory` are
# expansions; `argument` is the identifier exactly as typed, which is the
# common case and the one that must stay untouched.
type InputOrigin = Literal["argument", "pattern", "directory"]


@dataclass(frozen=True)
class ResolvedInput:
    """One concrete identifier, on its way into a collection's own resolver.

    `identifier` is what the rest of `corpus.add` sees, and for an argument
    that was not an expansion it is the original string - so an arXiv id, a
    DOI or a URL passes through this module completely unchanged.
    """

    identifier: str
    origin: InputOrigin = "argument"
    # The argument this came from, kept so a report can say which pattern
    # produced a file rather than only naming the file.
    from_argument: str = ""

    def __post_init__(self) -> None:
        if not self.from_argument:
            object.__setattr__(self, "from_argument", self.identifier)


class InputError(Exception):
    """An argument that names nothing boepie can act on.

    Raised rather than collected as a failed outcome: it means the command as
    typed cannot be carried out, and running the rest of the batch would leave
    the user to notice the gap in a summary.
    """


def _expand_pattern(pattern: str) -> list[Path]:
    """Every existing file matching `pattern`, in a stable order.

    Anchored at the pattern's own non-wildcard prefix rather than always at the
    working directory, so an absolute pattern stays absolute and a relative one
    is not silently resolved against somewhere else.
    """
    expanded = Path(pattern).expanduser()
    text = expanded.as_posix()

    segments = text.split("/")
    fixed = [
        segment for segment in _leading_literal_segments(segments) if segment != ""
    ]
    root = Path(text[:1]) if text.startswith("/") else Path(".")
    for segment in fixed:
        root = root / segment
    if not root.is_dir():
        return []

    remainder = "/".join(segments[len(fixed) + (1 if text.startswith("/") else 0) :])
    regex = globstar_regex(remainder)

    # Only `**` can match across directories, so without one the depth is fixed
    # and the search stays at that level. `notes/*.md` against a home directory
    # would otherwise walk every file beneath it to discard all but one level.
    if "**" in remainder:
        candidates = root.rglob("*")
    else:
        candidates = root.glob("/".join("*" for _ in remainder.split("/")))

    return sorted(
        candidate
        for candidate in candidates
        if candidate.is_file()
        and regex.fullmatch(candidate.relative_to(root).as_posix())
    )


def _leading_literal_segments(segments: Sequence[str]) -> list[str]:
    """The path segments before the first one containing a wildcard."""
    literal: list[str] = []
    for segment in segments:
        if looks_like_pattern(segment):
            break
        literal.append(segment)
    # The last literal segment is the file itself when nothing was a wildcard;
    # walking from its parent keeps `rglob` from having nothing to search.
    return literal[:-1] if len(literal) == len(segments) else literal


def resolve_inputs(identifiers: Sequence[str]) -> list[ResolvedInput]:
    """Expand every argument into the concrete identifiers `corpus add` acts on.

    Anything that is not a shell pattern is passed through untouched, so this
    is transparent to arXiv ids, DOIs, URLs and plain paths alike.
    """
    resolved: list[ResolvedInput] = []
    for argument in identifiers:
        # An existing path wins over pattern interpretation: a filename may
        # legitimately contain `[` or `?`, and if it is really there on disk
        # the user cannot have meant it as a pattern.
        if Path(argument).expanduser().exists():
            resolved.append(ResolvedInput(identifier=argument))
            continue

        if not looks_like_pattern(argument):
            # Not a pattern and not on disk. It may still be an arXiv id, a
            # DOI or a URL, which only the collection's own resolver can say,
            # so it passes through and fails there with a message that knows
            # what was tried.
            resolved.append(ResolvedInput(identifier=argument))
            continue

        matches = _expand_pattern(argument)
        if not matches:
            raise InputError(
                f"'{argument}' looks like a filename pattern but matched no "
                f"files. Check the path, or quote it if your shell expanded "
                f"it before boepie saw it."
            )
        resolved.extend(
            ResolvedInput(
                identifier=str(match), origin="pattern", from_argument=argument
            )
            for match in matches
        )
    return resolved
