"""Tests for boepie.literature.identifiers: recognising the many spellings a
paper can arrive as.

The bug these guard against was user-visible: `corpus add literature` passed
its argument straight to arXiv's Atom API, so `arXiv:2409.19750` and
`2409.19750v1.pdf` both failed as "no arXiv entry found" - which reads like
the paper does not exist rather than like the identifier was not understood.
"""

from __future__ import annotations

import pytest

from boepie.literature.identifiers import (
    arxiv_id_if_reference,
    looks_like_bibtex,
    normalize_arxiv_id,
    normalize_doi,
    parse_bibtex,
)


@pytest.mark.parametrize(
    "identifier",
    [
        "2409.19750",
        "2409.19750v1",
        "2409.19750v12",
        "arXiv:2409.19750",
        "arxiv:2409.19750v1",
        "2409.19750.pdf",
        "2409.19750v1.pdf",
        "https://arxiv.org/abs/2409.19750",
        "https://arxiv.org/abs/2409.19750v2",
        "https://arxiv.org/pdf/2409.19750",
        "https://arxiv.org/pdf/2409.19750v1.pdf",
        "https://ar5iv.labs.arxiv.org/html/2409.19750",
        "  2409.19750  ",
    ],
)
def test_every_modern_arxiv_spelling_normalizes_to_the_bare_id(identifier):
    assert normalize_arxiv_id(identifier) == "2409.19750"


@pytest.mark.parametrize(
    "identifier",
    ["astro-ph/0601234", "arXiv:astro-ph/0601234", "astro-ph/0601234v2"],
)
def test_pre_2007_arxiv_ids_are_recognised(identifier):
    assert normalize_arxiv_id(identifier) == "astro-ph/0601234"


def test_the_version_suffix_is_dropped():
    """The corpus tracks a paper, not a snapshot of one. Keeping the version
    would make v1 and v2 look like different documents to duplicate
    detection while fetching near-identical text twice."""
    assert normalize_arxiv_id("2409.19750v1") == normalize_arxiv_id("2409.19750v2")


@pytest.mark.parametrize(
    "identifier",
    [
        "",
        "not-an-id",
        "10.1093/mnras/sty1221",
        # A non-arXiv host whose path merely looks numeric must not match.
        "https://example.com/page/1234.56789",
        "https://doi.org/10.1093/mnras/sty1221",
    ],
)
def test_non_arxiv_identifiers_are_rejected(identifier):
    assert normalize_arxiv_id(identifier) is None


@pytest.mark.parametrize(
    "identifier",
    [
        "10.1093/mnras/sty1221",
        "doi:10.1093/mnras/sty1221",
        "https://doi.org/10.1093/mnras/sty1221",
        "https://dx.doi.org/10.1093/mnras/sty1221",
        "10.1093/mnras/sty1221.",
    ],
)
def test_doi_spellings_normalize(identifier):
    assert normalize_doi(identifier) == "10.1093/mnras/sty1221"


def test_a_bare_arxiv_id_is_not_a_doi():
    assert normalize_doi("2409.19750") is None


def test_looks_like_bibtex_is_suffix_based():
    assert looks_like_bibtex("library.bib")
    assert looks_like_bibtex("LIBRARY.BIB")
    assert not looks_like_bibtex("library.json")


_BIB = """
@article{smirnovRevisiting2011,
  title = {Revisiting the Radio Interferometer Measurement Equation},
  author = {Smirnov, O. M.},
  year = {2011},
  doi = {10.1051/0004-6361/201016082},
  eprint = {1101.1111},
  file = {Full Text:/home/someone/Zotero/storage/ABC/Smirnov 2011.pdf:application/pdf}
}

@inproceedings{kenyonCubical2018,
  title = {CubiCal: Fast Radio Interferometric Calibration},
  author = {Kenyon, J. S. and Smirnov, O. M.},
  year = {2018}
}
"""


def test_parse_bibtex_extracts_the_manifest_fields():
    entries = parse_bibtex(_BIB)

    assert [entry.citekey for entry in entries] == [
        "smirnovRevisiting2011",
        "kenyonCubical2018",
    ]
    first = entries[0]
    assert first.title == "Revisiting the Radio Interferometer Measurement Equation"
    assert first.year == "2011"
    assert first.doi == "10.1051/0004-6361/201016082"
    assert first.arxiv_id == "1101.1111"


def test_parse_bibtex_follows_the_zotero_file_field():
    """Zotero writes `file = {Label:/abs/path.pdf:mimetype}`, so exporting a
    .bib doubles as a batch of local PDFs to ingest."""
    entry = parse_bibtex(_BIB)[0]

    assert entry.file_path == "/home/someone/Zotero/storage/ABC/Smirnov 2011.pdf"


def test_parse_bibtex_skips_entries_with_no_title():
    entries = parse_bibtex("@misc{nothing,\n  author = {Nobody},\n}\n")

    assert entries == []


# The distinction this pair pins: `normalize_arxiv_id` searches, because it has
# to find an id printed inside a paper's text, while an argument a user typed
# must be an arXiv reference in its entirety. Reading a filename's date-shaped
# digits as an id answered both a real file and a mistyped path by fetching an
# unrelated paper of that number.
@pytest.mark.parametrize(
    "spelling",
    [
        "2409.19750",
        "2409.19750v1",
        "arXiv:2409.19750",
        "2409.19750v1.pdf",
        "https://arxiv.org/abs/2409.19750v1",
    ],
)
def test_a_whole_argument_that_is_an_arxiv_reference_resolves(spelling: str) -> None:
    assert arxiv_id_if_reference(spelling) == "2409.19750"


@pytest.mark.parametrize(
    "filename",
    ["my-paper-2409.19750.md", "2409.19750.md", "notes-2409.19750.txt"],
)
def test_a_filename_merely_containing_an_arxiv_id_is_not_a_reference(
    filename: str,
) -> None:
    assert arxiv_id_if_reference(filename) is None
    # Still found by the searching form, which is what reads it out of a PDF.
    assert normalize_arxiv_id(filename) == "2409.19750"


def test_the_legacy_form_survives_the_stricter_reading() -> None:
    assert arxiv_id_if_reference("astro-ph/0601234") == "astro-ph/0601234"
