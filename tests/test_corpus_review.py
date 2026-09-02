"""The buffer that decides what `corpus add` writes.

Every candidate identifier boepie finds on a first page is a guess about which
one names *this* paper, and a guess recorded as a fact is how a corpus fills
with wrong citekeys. These tests pin the three things that make the buffer an
answer to that rather than a nicer prompt: the ranking is presented and not
applied, an unusable buffer comes back with the edits still in it, and there
is no silent path from "we could not tell" to a written literature document.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from boepie.corpus import review
from boepie.literature.identifiers import PaperIdentifier

ARXIV = PaperIdentifier(kind="arxiv", value="1805.03410")
DOI = PaperIdentifier(kind="doi", value="10.1093/mnras/sty1097")
BIBCODE = PaperIdentifier(kind="bibcode", value="1974A&AS...15..417H")


def _row(
    name: str = "paper.pdf",
    *,
    title: str = "Cubical Fast",
    candidates: list[PaperIdentifier] | None = None,
) -> review.ReviewRow:
    return review.ReviewRow.for_document(
        Path("/corpus") / name, title, [] if candidates is None else candidates
    )


def test_spells_each_kind_so_its_shape_need_not_be_recognised() -> None:
    assert review.spell(ARXIV) == "arXiv:1805.03410"
    assert review.spell(DOI) == "doi:10.1093/mnras/sty1097"
    assert review.spell(BIBCODE) == "bibcode:1974A&AS...15..417H"


def test_the_likeliest_candidate_is_prefilled_and_the_rest_are_offered() -> None:
    row = _row(candidates=[ARXIV, DOI, BIBCODE])
    text = review.render([row], problems={})

    assert 'identifier = "arXiv:1805.03410"' in text
    assert "# also found on page one:" in text
    assert "#   doi:10.1093/mnras/sty1097" in text
    assert "#   bibcode:1974A&AS...15..417H" in text


def test_a_lone_candidate_lists_no_alternatives() -> None:
    text = review.render([_row(candidates=[ARXIV])], problems={})

    assert "# also found on page one:" not in text


def test_a_document_with_no_identity_is_pre_set_to_notes() -> None:
    row = _row(candidates=[])

    assert row.collection == "notes"
    assert row.identifier == ""

    text = review.render([row], problems={})
    assert 'collection = "notes"' in text
    assert "cannot go to literature as it stands" in text


def test_a_saved_buffer_round_trips_unchanged() -> None:
    rows = [
        _row("one.pdf", candidates=[ARXIV, DOI]),
        _row("two.pdf", title="A Note", candidates=[]),
    ]

    edited = review.parse(review.render(rows, problems={}), rows)

    assert [row.path for row in edited] == [row.path for row in rows]
    assert [row.collection for row in edited] == ["literature", "notes"]
    assert edited[0].identifier == "arXiv:1805.03410"
    assert review.problems_in(edited) == {}


def test_parsing_keeps_the_candidates_the_buffer_never_carried() -> None:
    rows = [_row(candidates=[ARXIV, DOI])]

    edited = review.parse(review.render(rows, problems={}), rows)

    assert edited[0].candidates == [ARXIV, DOI]


def test_a_title_or_path_holding_quotes_survives_the_round_trip() -> None:
    rows = [_row('the "best" paper.pdf', title='On "Seeing" \\ Noise', candidates=[ARXIV])]

    edited = review.parse(review.render(rows, problems={}), rows)

    assert edited[0].title == 'On "Seeing" \\ Noise'


def test_choosing_a_listed_alternative_is_taken_as_written() -> None:
    rows = [_row(candidates=[ARXIV, DOI])]
    text = review.render(rows, problems={}).replace(
        'identifier = "arXiv:1805.03410"', 'identifier = "doi:10.1093/mnras/sty1097"'
    )

    edited = review.parse(text, rows)

    assert edited[0].resolved == DOI
    assert review.problems_in(edited) == {}


def test_a_deleted_block_is_dropped_rather_than_kept() -> None:
    rows = [_row("one.pdf", candidates=[ARXIV]), _row("two.pdf", candidates=[DOI])]
    text = review.render(rows, problems={})
    kept, _, _ = text.partition("[[document]]\npath       = \"/corpus/two.pdf\"")

    edited = review.parse(kept, rows)

    assert [row.path for row in edited] == [Path("/corpus/one.pdf")]


def test_an_unknown_path_is_ignored_rather_than_invented() -> None:
    rows = [_row("one.pdf", candidates=[ARXIV])]
    text = review.render(rows, problems={}) + (
        '\n[[document]]\npath = "/corpus/never-seen.pdf"\ncollection = "literature"\n'
    )

    edited = review.parse(text, rows)

    assert [row.path for row in edited] == [Path("/corpus/one.pdf")]


def test_an_emptied_buffer_cancels_the_whole_add() -> None:
    rows = [_row(candidates=[ARXIV])]

    with pytest.raises(review.ReviewCancelled):
        review.parse("   \n\n", rows)


def test_a_buffer_holding_only_comments_cancels_too() -> None:
    rows = [_row(candidates=[ARXIV])]

    with pytest.raises(review.ReviewCancelled):
        review.parse("# changed my mind\n", rows)


def test_literature_without_an_identifier_is_a_problem_naming_both_ways_out() -> None:
    rows = [_row(candidates=[])]
    rows[0].collection = "literature"

    problems = review.problems_in(rows)

    assert set(problems) == {0}
    assert "arXiv id, DOI or ADS bibcode" in problems[0]
    assert 'collection = "notes"' in problems[0]


def test_an_unparseable_identifier_is_a_problem_quoting_what_was_typed() -> None:
    rows = [_row(candidates=[ARXIV])]
    rows[0].identifier = "the second one"

    problems = review.problems_in(rows)

    assert "'the second one'" in problems[0]


def test_a_collection_that_cannot_be_added_to_here_is_a_problem() -> None:
    rows = [_row(candidates=[ARXIV])]
    rows[0].collection = "docs"

    problems = review.problems_in(rows)

    assert "'docs' is not one of literature | notes" in problems[0]


def test_notes_needs_no_identifier() -> None:
    rows = [_row(candidates=[])]

    assert review.problems_in(rows) == {}


def test_a_problem_is_marked_beside_the_field_it_is_about() -> None:
    rows = [_row(candidates=[ARXIV])]
    rows[0].identifier = ""

    text = review.render(rows, problems=review.problems_in(rows))
    lines = text.splitlines()
    identifier_line = next(
        index for index, line in enumerate(lines) if line.startswith("identifier")
    )

    assert lines[identifier_line + 1].startswith("#            ^ ")
    assert text.startswith("# 1 problem(s).")


def test_a_reopened_buffer_still_holds_the_edits_that_were_made() -> None:
    rows = [_row("one.pdf", candidates=[ARXIV]), _row("two.pdf", candidates=[DOI])]
    edited = review.parse(
        review.render(rows, problems={}).replace(
            'group      = ""', 'group      = "quartical"', 1
        ),
        rows,
    )
    edited[0].identifier = ""

    reopened = review.render(edited, review.problems_in(edited))

    assert 'group      = "quartical"' in reopened
    assert "# also found on page one:" not in reopened


def test_yes_takes_the_ranking_without_opening_an_editor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refuse(**_: object) -> str:
        raise AssertionError("--yes must not open an editor")

    monkeypatch.setattr(review.click, "edit", refuse)
    rows = [_row(candidates=[ARXIV, DOI])]

    assert review.review(rows, edit=False) == rows


def test_review_reopens_until_the_buffer_validates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[str] = []

    def edit(*, text: str, **_: object) -> str:
        seen.append(text)
        if len(seen) == 1:
            return text.replace('identifier = "arXiv:1805.03410"', 'identifier = ""')
        return text.replace('identifier = ""', 'identifier = "doi:10.1093/mnras/sty1097"')

    monkeypatch.setattr(review.click, "edit", edit)
    rows = [_row(candidates=[ARXIV])]

    settled = review.review(rows)

    assert len(seen) == 2
    assert "problem(s)" in seen[1]
    assert settled[0].resolved == DOI


def test_an_editor_that_will_not_open_cancels_rather_than_guessing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(review.click, "edit", lambda **_: None)

    with pytest.raises(review.ReviewCancelled):
        review.review([_row(candidates=[ARXIV])])


# ---------------------------------------------------------------------------
# What `corpus add literature` does with the answer
# ---------------------------------------------------------------------------


_PAGE = (
    "arXiv:1805.03410v2 [astro-ph.IM] 15 May 2018\n"
    "Astron. Astrophys. 527, A106\n"
    "doi:10.1093/mnras/sty1097\n"
)


@pytest.fixture
def mineru(monkeypatch: pytest.MonkeyPatch):
    """Stub MinerU so both passes - the two-page survey and the full
    conversion - answer with the same first page."""

    def fake_convert(paths, *, device_mode, backend, model_source, page_limit=None):
        from boepie.corpus.intake import MineruResult

        return MineruResult(
            markdown={path: "# Converted\n\nBody.\n" for path in paths},
            front_page={path: _PAGE for path in paths},
        )

    monkeypatch.setattr("boepie.corpus.intake.mineru_available", lambda: True)
    monkeypatch.setattr("boepie.corpus.add.convert_with_mineru", fake_convert)
    monkeypatch.setattr(
        "boepie.literature.fetch.lookup_arxiv_metadata",
        lambda arxiv_id: {"title": "A Paper", "authors": "J. S. Kenyon", "year": "2018"},
    )
    monkeypatch.setattr(
        "boepie.literature.identifiers.resolve_doi_to_arxiv", lambda doi: None
    )


@pytest.fixture
def editor(monkeypatch: pytest.MonkeyPatch):
    """Stub `$EDITOR`, applying one substitution to the buffer it is handed."""
    opened: list[str] = []

    def install(old: str = "", new: str = "", *, cancel: bool = False):
        def edit(*, text: str, **_: object) -> str:
            opened.append(text)
            if cancel:
                return ""
            return text.replace(old, new) if old else text

        monkeypatch.setattr("boepie.corpus.review.click.edit", edit)
        return opened

    return install


def _pdf(tmp_path: Path) -> str:
    path = tmp_path / "scan.pdf"
    path.write_bytes(b"%PDF-1.7 body")
    return str(path)


def _options(**overrides: object):
    from boepie.corpus.add import AddOptions

    return AddOptions(arxiv_delay=0, **{"can_review": True, **overrides})


def test_the_page_s_candidates_reach_the_buffer_in_rank_order(
    tmp_path: Path, mineru, editor
) -> None:
    from boepie.corpus.add import add_literature

    opened = editor()

    add_literature(tmp_path / "literature", [_pdf(tmp_path)], _options())

    assert 'identifier = "arXiv:1805.03410"' in opened[0]
    assert "#   doi:10.1093/mnras/sty1097" in opened[0]


def test_the_identifier_the_user_chose_is_the_one_recorded(
    tmp_path: Path, mineru, editor
) -> None:
    """The reason the buffer exists: boepie ranked the arXiv stamp first, and
    a person overruled it with the DOI printed beside it."""
    from boepie.corpus.add import add_literature
    from boepie.context.frontmatter import read_frontmatter
    from boepie.corpus.layout import lookup_path

    editor('identifier = "arXiv:1805.03410"', 'identifier = "doi:10.1093/mnras/sty1097"')
    corpus = tmp_path / "literature"

    outcomes = add_literature(corpus, [_pdf(tmp_path)], _options())

    assert [outcome.status for outcome in outcomes] == ["added"]
    document = next(corpus.rglob("*.md"))
    frontmatter = read_frontmatter(document.read_text(encoding="utf-8"))[0]
    assert lookup_path(frontmatter, "bib.doi") == "10.1093/mnras/sty1097"
    assert "arxiv_id" not in frontmatter["bib"]


def test_moving_a_row_to_notes_in_the_buffer_writes_it_to_notes(
    tmp_path: Path, mineru, editor
) -> None:
    from boepie.corpus.add import add_literature

    editor('collection = "literature"', 'collection = "notes"')
    corpus = tmp_path / "literature"
    notes = tmp_path / "notes"

    outcomes = add_literature(corpus, [_pdf(tmp_path)], _options(), notes_dir=notes)

    assert [outcome.status for outcome in outcomes] == ["added"]
    assert not list(corpus.rglob("*.md"))
    assert list(notes.rglob("*.md"))


def test_cancelling_the_buffer_writes_nothing(tmp_path: Path, mineru, editor) -> None:
    from boepie.corpus.add import add_literature

    editor(cancel=True)
    corpus = tmp_path / "literature"

    outcomes = add_literature(corpus, [_pdf(tmp_path)], _options())

    assert [outcome.status for outcome in outcomes] == ["skipped"]
    assert "review cancelled" in (outcomes[0].detail or "")
    assert not list(corpus.rglob("*.md"))


def test_no_terminal_and_no_yes_refuses_rather_than_choosing(
    tmp_path: Path, mineru
) -> None:
    """A pipeline has nobody to ask, and picking the top candidate silently is
    exactly the assumption the buffer exists to stop."""
    from boepie.corpus.add import add_literature
    from boepie.corpus.review import ReviewUnavailable

    with pytest.raises(ReviewUnavailable) as raised:
        add_literature(tmp_path / "literature", [_pdf(tmp_path)], _options(can_review=False))

    assert "--yes" in str(raised.value)
    assert "--identifier" in str(raised.value)


def test_an_identifier_given_on_the_command_line_opens_no_buffer(
    tmp_path: Path, mineru, editor
) -> None:
    """It is already the user's answer; asking them to confirm what they just
    typed is the ceremony this deliberately avoids."""
    from boepie.corpus.add import add_literature

    opened = editor()

    add_literature(
        tmp_path / "literature",
        [_pdf(tmp_path)],
        _options(identifier="1101.1764", can_review=False),
    )

    assert opened == []


def test_a_bare_arxiv_id_opens_no_buffer(tmp_path: Path, mineru, editor) -> None:
    """Nothing was inferred, so there is nothing to confirm."""
    from boepie.corpus.add import add_literature

    opened = editor()

    add_literature(tmp_path / "literature", ["1805.03410"], _options(can_review=False))

    assert opened == []
