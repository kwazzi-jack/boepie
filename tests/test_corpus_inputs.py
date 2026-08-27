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


def _identifiers(resolved) -> list[str]:
    return [Path(item.identifier).name for item in resolved.items]


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
    resolved = resolve_inputs(["code/a.py", "code/b.py"])

    assert _identifiers(resolved) == ["a.py", "b.py"]
    assert [item.origin for item in resolved.items] == ["argument", "argument"]


@pytest.mark.parametrize(
    "identifier",
    ["2409.19750", "10.1093/mnras/stad1298", "https://example.com/page", "notes.bib"],
)
def test_a_non_path_identifier_is_left_alone(identifier: str, tmp_path: Path, monkeypatch) -> None:
    """arXiv ids, DOIs and URLs are resolved by each collection, not here.
    A DOI in particular contains a `/` and must not be read as a path."""
    monkeypatch.chdir(tmp_path)
    resolved = resolve_inputs([identifier])

    assert [item.identifier for item in resolved.items] == [identifier]
    assert resolved.items[0].origin == "argument"


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

    resolved = resolve_inputs(["weird[1].py"])

    assert [item.identifier for item in resolved.items] == ["weird[1].py"]
    assert resolved.items[0].origin == "argument"


def test_a_missing_plain_path_is_left_for_the_collection_to_report(
    tmp_path: Path, monkeypatch
) -> None:
    """Not an error here: it may still be an arXiv id or a DOI, and only the
    collection's own resolver knows what it tried."""
    monkeypatch.chdir(tmp_path)
    assert [result.identifier for result in resolve_inputs(["gone.md"]).items] == ["gone.md"]


def test_an_expansion_remembers_the_argument_it_came_from(
    tree: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tree)
    resolved = resolve_inputs(["code/*.py"])

    assert {item.from_argument for item in resolved.items} == {"code/*.py"}
    assert {item.origin for item in resolved.items} == {"pattern"}


def test_an_absolute_pattern_stays_absolute(tree: Path) -> None:
    """Anchoring at the pattern's own literal prefix, not the working
    directory - resolving an absolute pattern relative to somewhere else would
    match a different set of files or none at all."""
    resolved = resolve_inputs([str(tree / "code" / "*.py")])

    assert _identifiers(resolved) == ["a.py", "b.py"]
    assert all(Path(item.identifier).is_absolute() for item in resolved.items)


# ---------------------------------------------------------------------------
# Walking a directory
# ---------------------------------------------------------------------------


@pytest.fixture
def walkable(tmp_path: Path) -> Path:
    """A directory shaped like a real source tree: nested prose, a binary, a
    generated cache, a VCS directory and a symlink."""
    root = tmp_path / "code"
    (root / "gains").mkdir(parents=True)
    (root / "__pycache__").mkdir()
    (root / ".git").mkdir()
    (root / "README.md").write_text("# Top\n", encoding="utf-8")
    (root / "gains" / "README.md").write_text("# Gains\n", encoding="utf-8")
    (root / "gains" / "solve.py").write_text("x = 1\n", encoding="utf-8")
    (root / "gains" / "lib.so").write_bytes(bytes(range(256)))
    (root / "__pycache__" / "mod.pyc").write_bytes(bytes(range(256)))
    (root / ".git" / "config").write_text("secret\n", encoding="utf-8")
    (root / "linked.md").symlink_to(tmp_path / "README.md")
    return root


def test_a_directory_is_walked_recursively(walkable: Path) -> None:
    resolved = resolve_inputs([str(walkable)])

    assert _identifiers(resolved) == ["README.md", "README.md", "solve.py"]
    assert all(item.origin == "directory" for item in resolved.items)


def test_a_walk_mirrors_subdirectories_onto_groups(walkable: Path) -> None:
    """What keeps two files of the same name apart. Flattened into one group
    they would both be titled `README` with nothing to tell them apart."""
    resolved = resolve_inputs([str(walkable)])

    assert sorted(item.group for item in resolved.items) == ["", "gains", "gains"]


def test_a_walk_refuses_a_binary_rather_than_reading_it_as_text(
    walkable: Path,
) -> None:
    """The hazard the accept-list exists for: `detect_format` answers "code"
    for an unknown suffix and the encoding ladder ends in latin-1, which never
    fails, so a binary would become a document full of mojibake."""
    resolved = resolve_inputs([str(walkable)])

    assert not any(item.identifier.endswith(".so") for item in resolved.items)
    assert any(skip.identifier.endswith(".so") for skip in resolved.skipped)


def test_a_walk_never_descends_into_generated_or_vcs_directories(
    walkable: Path,
) -> None:
    """Pruned rather than skipped: nothing inside them is even looked at, so
    they appear in neither list."""
    resolved = resolve_inputs([str(walkable)])
    every = [item.identifier for item in resolved.items] + [
        skip.identifier for skip in resolved.skipped
    ]

    assert not any("__pycache__" in name for name in every)
    assert not any(".git" in name for name in every)


def test_a_walk_skips_symlinks(walkable: Path) -> None:
    """A link can point outside the tree that was named, which `corpus add
    notes code/` should not reach."""
    resolved = resolve_inputs([str(walkable)])

    assert any(skip.reason == "symlink" for skip in resolved.skipped)


def test_an_extra_file_type_is_added_to_the_accepted_list_not_substituted(
    tmp_path: Path,
) -> None:
    (tmp_path / "notebook.ipynb").write_text("{}\n", encoding="utf-8")
    (tmp_path / "note.md").write_text("# Note\n", encoding="utf-8")

    without = resolve_inputs([str(tmp_path)])
    with_extra = resolve_inputs([str(tmp_path)], extra_file_types=[".ipynb"])

    assert _identifiers(without) == ["note.md"]
    assert _identifiers(with_extra) == ["note.md", "notebook.ipynb"]


def test_a_directory_holding_nothing_convertible_is_an_error(tmp_path: Path) -> None:
    """Same reasoning as an unmatched pattern: adding nothing quietly looks
    like success."""
    (tmp_path / "image.png").write_bytes(bytes(range(256)))

    with pytest.raises(InputError, match="no files boepie can convert"):
        resolve_inputs([str(tmp_path)])
