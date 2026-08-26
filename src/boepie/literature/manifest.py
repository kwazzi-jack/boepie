"""ArXiv paper manifest: which papers `boepie corpus fetch --collection
literature` converts.

One source only: the packaged ``default_manifest.json`` (tracked in git,
ships in the wheel). There is deliberately no per-machine user manifest -
`boepie corpus add literature` writes a `managed_by: user` document straight
to disk, and `fetch` never touches those, so a second list for the reconciler
to diff against would only be a source of truth that could drift from the
documents themselves.

Only bibliographic facts -- citekey, arxiv_id, title, authors, year, doi --
are recorded here; none of it is the paper's own text, so shipping the
default manifest carries none of the redistribution risk that shipping
converted paper Markdown would (see `boepie.literature.fetch`, which converts
each paper's HTML on the machine that will read it, not boepie's).
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

_DEFAULT_MANIFEST_PATH = Path(__file__).resolve().parent / "default_manifest.json"

# Skipped when deriving a citekey's title portion - carry no distinguishing
# meaning, unlike the corpus's own subject terms.
_CITEKEY_STOPWORDS = frozenset({
    "a", "an", "the", "of", "for", "and", "with", "in", "on", "to", "using", "at",
})

# Stands in for the surname when there is no author to take one from - a local
# PDF carries no bibliography, so the key is title-derived and reads
# `paperRadioInterferometry`. Deliberately a real word rather than a marker
# like `unknown`: it ends up in citations.
_CITEKEY_NO_AUTHOR = "paper"


@dataclass(frozen=True)
class ArxivPaper:
    citekey: str
    arxiv_id: str
    title: str
    authors: str
    year: str
    doi: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        return asdict(self)


def _read_papers(path: Path) -> list[ArxivPaper]:
    if not path.is_file():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [ArxivPaper(**entry) for entry in raw]


def load_default_manifest() -> list[ArxivPaper]:
    """The tracked, packaged set of arXiv papers boepie fetches by default."""
    return _read_papers(_DEFAULT_MANIFEST_PATH)

def load_manifest(corpus_dir: Path) -> list[ArxivPaper]:
    """Every paper `corpus fetch` reconciles against.

    Takes `corpus_dir` it no longer reads, so callers do not have to care
    whether a collection has a per-machine manifest component; none does any
    more. Kept as the reconciler's entry point rather than having callers
    reach for `load_default_manifest` directly, so reintroducing a second
    layer later would be a one-line change here.
    """
    return load_default_manifest()


def derive_citekey(authors: str, year: str, title: str) -> str:
    """A short slug in the corpus's existing style (e.g.
    `smirnovRevisitingRadioInterferometer2011`: surname + leading title words +
    year), so `boepie corpus add literature` can work from just an arXiv id. Collisions
    are handled separately by `unique_citekey`."""
    first_author = authors.split(" and ")[0].strip()
    if "," in first_author:
        # "Last, First M." - the bib/Zotero convention the default manifest's
        # own authors fields use.
        surname = first_author.split(",")[0].strip()
    else:
        # "First M. Last" - what arXiv's own Atom API returns, and what
        # `lookup_arxiv_metadata` (the real caller for `corpus add literature`) hands in.
        # A bare local file has no bibliography at all, so `authors` is empty and
        # there is no name to take a surname from; `_CITEKEY_NO_AUTHOR` below is
        # what that case falls back to.
        parts = first_author.split()
        surname = parts[-1].strip() if parts else ""
    surname = re.sub(r"[^A-Za-z]", "", surname) or _CITEKEY_NO_AUTHOR
    surname_part = surname[:1].lower() + surname[1:]

    words = [word for word in re.findall(r"[A-Za-z]+", title) if word.lower() not in _CITEKEY_STOPWORDS]
    title_part = "".join(word.capitalize() for word in words[:2])

    return f"{surname_part}{title_part}{year}"


def unique_citekey(base_citekey: str, existing_citekeys: set[str]) -> str:
    """`base_citekey`, or the first `<base><suffix>` (a, b, c, ...) not already
    taken -- mirrors the disambiguation Zotero itself applies to same-author,
    same-year citekeys (see the existing `smirnovRevisitingRadioInterferometer2011{a,b,c}`
    entries in the default manifest)."""
    if base_citekey not in existing_citekeys:
        return base_citekey
    for letter in "abcdefghijklmnopqrstuvwxyz":
        candidate = f"{base_citekey}{letter}"
        if candidate not in existing_citekeys:
            return candidate
    # Past `z` the Zotero convention has nothing more to say, and raising here
    # would abort a whole batch over one document. Title-derived keys (a folder
    # of PDFs with no bibliography) collide far more readily than the
    # author-and-year keys this was written for, so the 27th is numbered rather
    # than fatal.
    suffix = 27
    while f"{base_citekey}{suffix}" in existing_citekeys:
        suffix += 1
    return f"{base_citekey}{suffix}"
