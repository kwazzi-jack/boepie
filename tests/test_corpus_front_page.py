"""Reading a paper's own identifier off the page MinerU renders away.

A local PDF has no bibliographic identity of its own as far as the corpus is
concerned: its citekey has to come from its title, and duplicate detection -
which runs on `bib.arxiv_id` / `bib.doi` - has nothing to compare, so the same
paper added once as a PDF and once as an arXiv id lands twice.

The identifier *is* on the page. It is systematically absent from the
converted Markdown, because MinerU classifies the `arXiv:...` stamp as page
furniture and furniture is not body text - verified against a real
conversion, where `grep -c arxiv` over the rendered `.md` is 0 while the
content list beside it carries `arXiv:1805.03410v2 [astro-ph.IM] 15 May 2018`
as `page_aside_text`. These tests pin the recovery of that text and what is
made of it.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from boepie.corpus.add import AddOptions, add_literature
from boepie.corpus.intake import Converted, MineruResult, front_page_text
from boepie.corpus.layout import lookup_path
from boepie.context.frontmatter import read_frontmatter

# The real shape MinerU 3.x writes: a list of pages, each a list of blocks,
# each block nesting its pieces under a key named after its own type.
_ARXIV_ASIDE = "arXiv:1805.03410v2 [astro-ph.IM] 15 May 2018"


def _v2_block(kind: str, text: str) -> dict:
    return {"type": kind, "content": {f"{kind}_content": [{"type": "text", "content": text}]}}


def _write_v2(markdown_path: Path, pages: list[list[dict]]) -> None:
    markdown_path.write_text("# Body\n", encoding="utf-8")
    markdown_path.with_name(f"{markdown_path.stem}_content_list_v2.json").write_text(
        json.dumps(pages), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Pulling the text out
# ---------------------------------------------------------------------------


def test_furniture_comes_before_body_text(tmp_path: Path) -> None:
    """Ordering is the whole precedence mechanism: an identifier stamped in
    the margin should win over one mentioned in a first-page footnote's prose."""
    markdown = tmp_path / "paper.md"
    _write_v2(markdown, [[
        _v2_block("title", "A Paper"),
        _v2_block("page_aside_text", _ARXIV_ASIDE),
    ]])

    text = front_page_text(markdown)

    assert text.index(_ARXIV_ASIDE) < text.index("A Paper")


def test_only_the_first_page_is_read(tmp_path: Path) -> None:
    """A bibliography offers dozens of other people's identifiers, and every
    one of them is a wrong answer for this document."""
    markdown = tmp_path / "paper.md"
    _write_v2(markdown, [
        [_v2_block("title", "A Paper")],
        [_v2_block("paragraph", "Smirnov 2011, arXiv:1101.1764")],
    ])

    assert "1101.1764" not in front_page_text(markdown)


def test_the_older_flat_content_list_is_read_when_v2_is_absent(tmp_path: Path) -> None:
    """MinerU 2.x wrote only the flat list. It has no notion of furniture, so
    the ordering above is unavailable - but it still carries the aside."""
    markdown = tmp_path / "paper.md"
    markdown.write_text("# Body\n", encoding="utf-8")
    markdown.with_name("paper_content_list.json").write_text(
        json.dumps([
            {"type": "text", "text": _ARXIV_ASIDE, "page_idx": 0},
            {"type": "text", "text": "Other people's work", "page_idx": 1},
        ]),
        encoding="utf-8",
    )

    text = front_page_text(markdown)

    assert _ARXIV_ASIDE in text
    assert "Other people's work" not in text


def test_no_content_list_is_not_an_error(tmp_path: Path) -> None:
    markdown = tmp_path / "paper.md"
    markdown.write_text("# Body\n", encoding="utf-8")

    assert front_page_text(markdown) == ""


def test_a_malformed_content_list_is_not_an_error(tmp_path: Path) -> None:
    markdown = tmp_path / "paper.md"
    markdown.write_text("# Body\n", encoding="utf-8")
    markdown.with_name("paper_content_list_v2.json").write_text("{ not json", encoding="utf-8")

    assert front_page_text(markdown) == ""


# ---------------------------------------------------------------------------
# What the corpus does with it
# ---------------------------------------------------------------------------


_METADATA = {
    "title": "CubiCal - Fast radio interferometric calibration suite",
    "authors": "J. S. Kenyon and O. M. Smirnov",
    "year": "2018",
}


@pytest.fixture
def mineru(monkeypatch: pytest.MonkeyPatch):
    """Stub MinerU, returning a chosen front page with the markdown."""

    def install(front_page: str) -> None:
        def fake_convert(paths, *, device_mode, backend, model_source):
            return MineruResult(
                markdown={path: "# Converted\n\nBody.\n" for path in paths},
                front_page={path: front_page for path in paths},
            )

        monkeypatch.setattr("boepie.corpus.intake.mineru_available", lambda: True)
        monkeypatch.setattr("boepie.corpus.add.convert_with_mineru", fake_convert)

    return install


@pytest.fixture
def arxiv(monkeypatch: pytest.MonkeyPatch):
    """Stub arXiv's metadata API, recording what it was asked for."""
    asked: list[str] = []

    def fake_lookup(arxiv_id: str):
        asked.append(arxiv_id)
        return dict(_METADATA)

    monkeypatch.setattr("boepie.literature.fetch.lookup_arxiv_metadata", fake_lookup)
    return asked


def _pdf(tmp_path: Path, name: str = "scan.pdf") -> str:
    path = tmp_path / name
    path.write_bytes(b"%PDF-1.7 body")
    return str(path)


def _frontmatter(collection_dir: Path) -> dict:
    document = next(collection_dir.rglob("*.md"))
    return read_frontmatter(document.read_text(encoding="utf-8"))[0]


def test_a_pdf_gets_a_real_citekey_from_its_own_arxiv_stamp(
    tmp_path: Path, mineru, arxiv: list[str]
) -> None:
    """The point of the whole exercise: `kenyonCubicalSuite2018` instead of a
    key derived from whatever the title happened to be."""
    mineru(_ARXIV_ASIDE)
    corpus = tmp_path / "literature"

    outcomes = add_literature(corpus, [_pdf(tmp_path)], AddOptions())

    assert [outcome.status for outcome in outcomes] == ["added"]
    assert arxiv == ["1805.03410"]
    frontmatter = _frontmatter(corpus)
    assert lookup_path(frontmatter, "bib.arxiv_id") == "1805.03410"
    assert lookup_path(frontmatter, "bib.year") == "2018"
    assert lookup_path(frontmatter, "bib.citekey").startswith("kenyon")


def test_the_same_paper_by_pdf_and_by_arxiv_id_is_one_document(
    tmp_path: Path, mineru, arxiv: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two routes, no shared checksum and different derived citekeys - the
    recovered `bib.arxiv_id` is the only thing that can catch this."""
    mineru(_ARXIV_ASIDE)
    corpus = tmp_path / "literature"
    add_literature(corpus, [_pdf(tmp_path)], AddOptions())

    outcomes = add_literature(corpus, ["1805.03410"], AddOptions())

    assert [outcome.status for outcome in outcomes] == ["duplicate"]


def test_a_doi_on_the_page_is_recorded_even_when_it_resolves_to_nothing(
    tmp_path: Path, mineru, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Identity is worth recording on its own: it is what duplicate detection
    runs on, with or without metadata behind it."""
    mineru("Astron. Astrophys. 527, A106 (2011)\ndoi:10.1051/0004-6361/201015249")
    monkeypatch.setattr(
        "boepie.literature.identifiers.resolve_doi_to_arxiv", lambda doi: None
    )
    corpus = tmp_path / "literature"

    add_literature(corpus, [_pdf(tmp_path)], AddOptions())

    assert lookup_path(_frontmatter(corpus), "bib.doi") == "10.1051/0004-6361/201015249"


def test_a_bibcode_is_recorded_and_never_resolved(
    tmp_path: Path, mineru, arxiv: list[str]
) -> None:
    """The pre-arXiv case. Resolving one needs the ADS API and a key, so it is
    identity only - which is still enough to deduplicate on."""
    mineru("Bibcode: 1974A&AS...15..417H\nAstron. Astrophys. Suppl. 15, 417")
    corpus = tmp_path / "literature"

    add_literature(corpus, [_pdf(tmp_path)], AddOptions())

    assert lookup_path(_frontmatter(corpus), "bib.bibcode") == "1974A&AS...15..417H"
    assert arxiv == []


def test_a_page_with_no_identifier_still_adds_with_a_title_derived_citekey(
    tmp_path: Path, mineru, arxiv: list[str]
) -> None:
    """No identifier is a normal outcome, not a failure."""
    mineru("Accelerated C++\nAndrew Koenig and Barbara Moo\n")
    corpus = tmp_path / "literature"

    outcomes = add_literature(corpus, [_pdf(tmp_path)], AddOptions())

    assert [outcome.status for outcome in outcomes] == ["added"]
    frontmatter = _frontmatter(corpus)
    assert lookup_path(frontmatter, "bib.citekey")
    # Omitted entirely rather than written empty - `literature_blocks` drops
    # what it does not know, so absence is the signal.
    assert "arxiv_id" not in frontmatter["bib"]
    assert arxiv == []


def test_an_unreachable_arxiv_keeps_the_identifier_and_adds_anyway(
    tmp_path: Path, mineru, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The identifier is already in hand by the time the network is needed.
    Refusing the document because a metadata lookup failed would trade a small
    loss - no authors, no year - for a total one."""
    mineru(_ARXIV_ASIDE)

    def unreachable(arxiv_id: str):
        raise httpx.ConnectError("no network")

    monkeypatch.setattr("boepie.literature.fetch.lookup_arxiv_metadata", unreachable)
    corpus = tmp_path / "literature"

    outcomes = add_literature(corpus, [_pdf(tmp_path)], AddOptions())

    assert [outcome.status for outcome in outcomes] == ["added"]
    assert lookup_path(_frontmatter(corpus), "bib.arxiv_id") == "1805.03410"


def test_a_bib_entry_still_wins_over_the_page(
    tmp_path: Path, mineru, arxiv: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exporting a `.bib` is the documented way to keep the citekeys you
    already cite papers by; a stamp on the page must not overrule one."""
    mineru(_ARXIV_ASIDE)
    pdf = _pdf(tmp_path)
    bib = tmp_path / "library.bib"
    bib.write_text(
        "@article{myOwnKey2018,\n"
        "  title = {A Paper},\n"
        "  author = {Kenyon, J. S.},\n"
        "  year = {2018},\n"
        f"  file = {{{pdf}}},\n"
        "}\n",
        encoding="utf-8",
    )
    corpus = tmp_path / "literature"

    add_literature(corpus, [str(bib)], AddOptions())

    assert lookup_path(_frontmatter(corpus), "bib.citekey") == "myOwnKey2018"
