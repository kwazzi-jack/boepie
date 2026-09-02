# src/boepie/corpus/review.py
"""The buffer you edit before `corpus add` writes anything.

Modelled on `git commit`: one prefilled file opened in `$EDITOR`, comments
explaining it, save to apply, empty it to cancel. Not a series of questions -
a folder of a hundred PDFs is a hundred documents to settle, and prompting per
field would be three hundred answers where a search-and-replace in an editor
is one.

**Why it exists at all.** boepie can find the identifiers printed on a paper's
first page, but it cannot know which one names *this* document: a first page
routinely carries the paper's own arXiv stamp, its journal DOI, an ADS
bibcode, and a DOI belonging to something it cites. Choosing between those is
a judgement, and a judgement recorded as a fact is how a corpus quietly fills
with wrong citekeys. So the candidates are ranked, the most likely one is
filled in, the rest are listed under it, and a person decides.

Saving the file unchanged is therefore a real answer, not a bypass: you looked
at the ranking and accepted it. `--yes` is the bypass, and it exists for
scripts.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path

import click

from boepie.literature.identifiers import PaperIdentifier, parse_identifier

REVIEWABLE_COLLECTIONS = ("literature", "notes")

# How an identifier is written in the buffer. Prefixed so its kind is visible
# without the reader having to recognise the shape - `1974A&AS...15..417H`
# says nothing to someone who has not seen a bibcode before.
_PREFIXES: dict[str, str] = {"arxiv": "arXiv:", "doi": "doi:", "bibcode": "bibcode:"}


def spell(identifier: PaperIdentifier) -> str:
    return f"{_PREFIXES[identifier.kind]}{identifier.value}"


class ReviewCancelled(Exception):
    """The buffer came back empty, which is how you abandon the whole add."""


class ReviewUnavailable(Exception):
    """There is no terminal to review in, and no `--yes` to proceed without one."""


@dataclass
class ReviewRow:
    """One document awaiting a decision."""

    path: Path
    title: str
    collection: str = "literature"
    group: str = ""
    # The chosen identifier, spelled as the buffer spells it. Prefilled with
    # the most likely candidate, or empty when there were none - in which case
    # the row is pre-set to `notes`, because a literature document without an
    # identifier is exactly what boepie refuses to write.
    identifier: str = ""
    candidates: list[PaperIdentifier] = field(default_factory=list)

    @classmethod
    def for_document(
        cls, path: Path, title: str, candidates: list[PaperIdentifier]
    ) -> ReviewRow:
        if not candidates:
            return cls(path=path, title=title, collection="notes", candidates=[])
        return cls(
            path=path,
            title=title,
            collection="literature",
            identifier=spell(candidates[0]),
            candidates=candidates,
        )

    @property
    def resolved(self) -> PaperIdentifier | None:
        return parse_identifier(self.identifier) if self.identifier else None


def _quote(value: str) -> str:
    """A TOML basic string. Paths and titles both routinely contain quotes."""
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _row_text(row: ReviewRow, problem: str | None) -> str:
    lines: list[str] = []
    if not row.candidates:
        lines += [
            "# Nothing on page one identified this document, so it has no",
            "# bibliographic identity and cannot go to literature as it stands.",
            "# Either fill in `identifier` and set collection = \"literature\",",
            "# or leave it as a note.",
        ]
    lines += [
        "[[document]]",
        f"path       = {_quote(str(row.path))}",
        f"title      = {_quote(row.title)}",
        f"collection = {_quote(row.collection)}",
        f"group      = {_quote(row.group)}",
        f"identifier = {_quote(row.identifier)}",
    ]
    if problem is not None:
        lines += [f"#            ^ {problem}"]
    others = [spell(candidate) for candidate in row.candidates[1:]]
    if others:
        lines += ["# also found on page one:"] + [f"#   {other}" for other in others]
    return "\n".join(lines)


def render(rows: list[ReviewRow], problems: dict[int, str]) -> str:
    """The buffer text for `rows`, marking any problems in place.

    Problems are rendered beside the field they are about rather than
    collected at the top, the way `visudo` does it: the edits you already made
    are still here, and the thing to fix is next to the thing that is wrong.
    """
    header = [
        f"# boepie corpus add - {len(rows)} document(s) to review",
        "#",
        "# Save and close to apply. Empty this file to cancel.",
        "#",
        "#   collection   literature | notes",
        "#   identifier   literature only: an arXiv id, DOI or ADS bibcode",
        "#   group        \"\" files the document at the collection root",
        "#",
        "# Each identifier below is the likeliest candidate boepie found, not a",
        "# decision it made. Any alternatives are listed under it. Delete a",
        "# [[document]] block to skip that file.",
    ]
    if problems:
        header = [
            f"# {len(problems)} problem(s). Fix and save, or empty this file to cancel.",
            "#",
        ] + header
    blocks = [_row_text(row, problems.get(index)) for index, row in enumerate(rows)]
    return "\n".join(header) + "\n\n" + "\n\n".join(blocks) + "\n"


def parse(text: str, rows: list[ReviewRow]) -> list[ReviewRow]:
    """Read an edited buffer back, keeping each row's candidate list.

    Matched to the original rows by `path`, because that is the one field the
    user has no reason to edit and the only stable handle a document has
    before it is written.
    """
    if not text.strip():
        raise ReviewCancelled
    parsed = tomllib.loads(text)
    documents = parsed.get("document")
    if not isinstance(documents, list) or not documents:
        raise ReviewCancelled

    by_path = {str(row.path): row for row in rows}
    edited: list[ReviewRow] = []
    for entry in documents:
        if not isinstance(entry, dict):
            continue
        original = by_path.get(str(entry.get("path", "")))
        if original is None:
            continue
        edited.append(
            replace(
                original,
                title=str(entry.get("title", original.title)),
                collection=str(entry.get("collection", original.collection)),
                group=str(entry.get("group", "")),
                identifier=str(entry.get("identifier", "")),
            )
        )
    return edited


def problems_in(rows: list[ReviewRow]) -> dict[int, str]:
    """What still stops this buffer from being applied, per row index."""
    found: dict[int, str] = {}
    for index, row in enumerate(rows):
        if row.collection not in REVIEWABLE_COLLECTIONS:
            found[index] = (
                f"'{row.collection}' is not one of "
                f"{' | '.join(REVIEWABLE_COLLECTIONS)}."
            )
        elif row.collection == "literature" and not row.identifier:
            found[index] = (
                "literature needs an arXiv id, DOI or ADS bibcode. "
                'Supply one, or set collection = "notes".'
            )
        elif row.collection == "literature" and row.resolved is None:
            found[index] = (
                f"'{row.identifier}' is not an arXiv id, DOI or ADS bibcode."
            )
    return found


def review(rows: list[ReviewRow], *, edit: bool = True) -> list[ReviewRow]:
    """Put `rows` in front of the user and return what they settled on.

    Re-opens on a validation failure rather than discarding the edits, which
    is the whole reason this is a buffer and not a prompt. `edit=False` is
    `--yes`: take the prefilled ranking as it stands, which is only safe
    because every row was prefilled with the *most likely* candidate rather
    than an arbitrary one.
    """
    if not edit:
        return rows

    text = render(rows, problems={})
    while True:
        edited_text = click.edit(text=text, extension=".toml", require_save=False)
        if edited_text is None:
            # The editor could not be opened at all; there is nothing to
            # apply and guessing on the user's behalf is what this exists to
            # avoid.
            raise ReviewCancelled
        edited = parse(edited_text, rows)
        problems = problems_in(edited)
        if not problems:
            return edited
        text = render(edited, problems)
