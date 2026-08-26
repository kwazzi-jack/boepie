"""Expanding what was typed into the files it names.

The point of resolving patterns inside boepie rather than leaning on the shell
is that the answer stops depending on which shell you use - bash without
`globstar`, zsh and fish each read `**` differently, and PowerShell does not
expand arguments for a program at all. These tests pin the dialect boepie
commits to, independent of any of that.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from boepie.corpus.inputs import InputError, resolve_inputs


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    (tmp_path / "code" / "sub").mkdir(parents=True)
    for relative in ("code/a.py", "code/b.py", "code/sub/c.py", "code/notes.md"):
        (tmp_path / relative).write_text("body\n", encoding="utf-8")
    return tmp_path


def _identifiers(results) -> list[str]:
    return [Path(result.identifier).name for result in results]


def test_a_single_star_stays_at_one_level(tree: Path, monkeypatch) -> None:
    monkeypatch.chdir(tree)
    assert _identifiers(resolve_inputs(["code/*.py"])) == ["a.py", "b.py"]


def test_a_double_star_crosses_directories(tree: Path, monkeypatch) -> None:
    monkeypatch.chdir(tree)
    assert _identifiers(resolve_inputs(["code/**/*.py"])) == ["a.py", "b.py", "c.py"]


def test_an_already_expanded_argument_passes_through_untouched(
    tree: Path, monkeypatch
) -> None:
    """What the shell hands over when it did the expanding itself: several
    existing paths, which must not be re-interpreted."""
    monkeypatch.chdir(tree)
    results = resolve_inputs(["code/a.py", "code/b.py"])

    assert _identifiers(results) == ["a.py", "b.py"]
    assert [result.origin for result in results] == ["argument", "argument"]


@pytest.mark.parametrize(
    "identifier",
    ["2409.19750", "10.1093/mnras/stad1298", "https://example.com/page", "notes.bib"],
)
def test_a_non_path_identifier_is_left_alone(identifier: str, tmp_path: Path, monkeypatch) -> None:
    """arXiv ids, DOIs and URLs are resolved by each collection, not here.
    A DOI in particular contains a `/` and must not be read as a path."""
    monkeypatch.chdir(tmp_path)
    results = resolve_inputs([identifier])

    assert [result.identifier for result in results] == [identifier]
    assert results[0].origin == "argument"


def test_a_pattern_matching_nothing_is_an_error_not_silence(
    tree: Path, monkeypatch
) -> None:
    """The one outcome that looks like success and is not. Adding nothing
    quietly is worse than refusing."""
    monkeypatch.chdir(tree)
    with pytest.raises(InputError, match="matched no files"):
        resolve_inputs(["code/*.xyz"])


def test_a_real_file_wins_over_reading_its_name_as_a_pattern(
    tmp_path: Path, monkeypatch
) -> None:
    """`[` is legal in a filename. If the file is really there, the user
    cannot have meant the name as a pattern."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "weird[1].py").write_text("body\n", encoding="utf-8")

    results = resolve_inputs(["weird[1].py"])

    assert [result.identifier for result in results] == ["weird[1].py"]
    assert results[0].origin == "argument"


def test_a_missing_plain_path_is_left_for_the_collection_to_report(
    tmp_path: Path, monkeypatch
) -> None:
    """Not an error here: it may still be an arXiv id or a DOI, and only the
    collection's own resolver knows what it tried."""
    monkeypatch.chdir(tmp_path)
    assert [result.identifier for result in resolve_inputs(["gone.md"])] == ["gone.md"]


def test_an_expansion_remembers_the_argument_it_came_from(
    tree: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tree)
    results = resolve_inputs(["code/*.py"])

    assert {result.from_argument for result in results} == {"code/*.py"}
    assert {result.origin for result in results} == {"pattern"}


def test_an_absolute_pattern_stays_absolute(tree: Path) -> None:
    """Anchoring at the pattern's own literal prefix, not the working
    directory - resolving an absolute pattern relative to somewhere else would
    match a different set of files or none at all."""
    results = resolve_inputs([str(tree / "code" / "*.py")])

    assert _identifiers(results) == ["a.py", "b.py"]
    assert all(Path(result.identifier).is_absolute() for result in results)
