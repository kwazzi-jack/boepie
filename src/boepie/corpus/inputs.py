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

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Sequence

from boepie._glob import globstar_regex, looks_like_pattern
from boepie.corpus.intake import is_supported_suffix

# Directory names never worth walking into. Dot-prefixed names are skipped
# separately and by rule; these are the ones that are neither hidden nor
# interesting - build output, dependency trees, caches. Skipping them by name
# is about speed, not safety: the accept-list already keeps their contents
# out, but there is no reason to walk a `node_modules` to discard every file
# in it.
_SKIPPED_DIRECTORY_NAMES = frozenset({
    "__pycache__", "node_modules", "dist", "build", "site-packages",
    "venv", "env", "target", "vendor", "coverage", "htmlcov",
})

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
    # The walked directory's own subdirectory structure, mirrored onto corpus
    # groups. Empty for anything that did not come from a directory. Relative
    # to the directory named on the command line, not including it: walking
    # `code/` files `code/gains/x.py` under `gains`, and `--group` supplies
    # any prefix above that.
    group: str = ""

    def __post_init__(self) -> None:
        if not self.from_argument:
            object.__setattr__(self, "from_argument", self.identifier)


@dataclass(frozen=True)
class SkippedInput:
    """A file a walk found and declined to take.

    Kept rather than dropped so the count can be reported: a walk that
    silently ignored half a directory would leave the user believing the
    corpus holds something it does not.
    """

    identifier: str
    reason: str


@dataclass(frozen=True)
class ResolvedInputs:
    """Everything one `corpus add` invocation will act on."""

    items: list[ResolvedInput] = field(default_factory=list)
    skipped: list[SkippedInput] = field(default_factory=list)


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


def _skip_directory(name: str) -> bool:
    """Whether a walk should decline to descend into `name`."""
    return name.startswith(".") or name in _SKIPPED_DIRECTORY_NAMES


def _walk_directory(
    root: Path, argument: str, extra_file_types: Sequence[str]
) -> tuple[list[ResolvedInput], list[SkippedInput]]:
    """Every file under `root` boepie can convert, with its group.

    `os.walk` rather than `rglob` so a pruned directory is never descended
    into at all. Symlinks are skipped, both files and directories: a link can
    point outside the tree the user named, and `boepie corpus add -n code/`
    should mean what is under `code/`, not wherever `code/vendor` aliases to.
    """
    items: list[ResolvedInput] = []
    skipped: list[SkippedInput] = []

    for current, directories, filenames in os.walk(root):
        here = Path(current)
        directories[:] = sorted(
            name
            for name in directories
            if not _skip_directory(name) and not (here / name).is_symlink()
        )
        group = here.relative_to(root).as_posix()
        group = "" if group == "." else group

        for name in sorted(filenames):
            path = here / name
            if path.is_symlink():
                skipped.append(SkippedInput(identifier=str(path), reason="symlink"))
                continue
            if not is_supported_suffix(path, extra_file_types):
                skipped.append(
                    SkippedInput(
                        identifier=str(path),
                        reason=f"unsupported file type '{path.suffix or path.name}'",
                    )
                )
                continue
            items.append(
                ResolvedInput(
                    identifier=str(path),
                    origin="directory",
                    from_argument=argument,
                    group=group,
                )
            )
    return items, skipped


def resolve_inputs(
    identifiers: Sequence[str], *, extra_file_types: Sequence[str] = ()
) -> ResolvedInputs:
    """Expand every argument into the concrete identifiers `corpus add` acts on.

    Anything that is not a directory or a shell pattern is passed through
    untouched, so this is transparent to arXiv ids, DOIs, URLs and plain paths
    alike.
    """
    resolved = ResolvedInputs()
    for argument in identifiers:
        expanded = Path(argument).expanduser()

        if expanded.is_dir():
            items, skipped = _walk_directory(expanded, argument, extra_file_types)
            if not items:
                raise InputError(
                    f"'{argument}' holds no files boepie can convert"
                    + (
                        f" ({len(skipped)} skipped as unsupported)."
                        if skipped
                        else "."
                    )
                )
            resolved.items.extend(items)
            resolved.skipped.extend(skipped)
            continue

        # An existing path wins over pattern interpretation: a filename may
        # legitimately contain `[` or `?`, and if it is really there on disk
        # the user cannot have meant it as a pattern.
        if expanded.exists():
            resolved.items.append(ResolvedInput(identifier=argument))
            continue

        if not looks_like_pattern(argument):
            # Not a pattern and not on disk. It may still be an arXiv id, a
            # DOI or a URL, which only the collection's own resolver can say,
            # so it passes through and fails there with a message that knows
            # what was tried.
            resolved.items.append(ResolvedInput(identifier=argument))
            continue

        matches = _expand_pattern(argument)
        if not matches:
            raise InputError(
                f"'{argument}' looks like a filename pattern but matched no "
                f"files. Check the path, or quote it if your shell expanded "
                f"it before boepie saw it."
            )
        resolved.items.extend(
            ResolvedInput(
                identifier=str(match), origin="pattern", from_argument=argument
            )
            for match in matches
        )
    return resolved
