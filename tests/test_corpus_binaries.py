"""The up-front MinerU pass: hash first, then convert in runs.

Two costs are being avoided here and they are worth stating separately,
because only one of them is about batching.

MinerU spends about twenty seconds loading its model stack before it converts
anything and then roughly sixteen seconds a document, so a process per
document turns forty-seven papers into thirty-two minutes of work that
batching does in thirteen. And a document already in the collection used to be
converted in full before its checksum was compared, so re-adding a folder paid
for every conversion twice over and discarded the second one.

MinerU itself is stubbed throughout: these pin the orchestration, not the
converter, and no model is ever loaded.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from boepie.corpus.add import AddOptions, add_notes
from boepie.corpus.intake import IntakeError, MineruResult


@pytest.fixture
def available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("boepie.corpus.intake.mineru_available", lambda: True)


@pytest.fixture
def runs(monkeypatch: pytest.MonkeyPatch) -> list[list[Path]]:
    """Record one entry per MinerU process, holding the paths it was given."""
    recorded: list[list[Path]] = []

    def fake_convert(paths, *, device_mode, backend, model_source):
        recorded.append(list(paths))
        return MineruResult(
            markdown={path: f"# {path.stem}\n\nBody.\n" for path in paths}
        )

    monkeypatch.setattr("boepie.corpus.add.convert_with_mineru", fake_convert)
    return recorded


def _pdfs(directory: Path, count: int) -> list[str]:
    """`count` distinct files that look like PDFs by name. Their bytes are
    never parsed - only hashed - so they need only differ from each other."""
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for index in range(count):
        path = directory / f"paper-{index}.pdf"
        path.write_bytes(f"%PDF-1.7 {index}".encode())
        paths.append(str(path))
    return paths


def _options(**overrides) -> AddOptions:
    return AddOptions(**overrides)


def test_one_process_converts_the_whole_batch(
    tmp_path: Path, available: None, runs: list[list[Path]]
) -> None:
    sources = _pdfs(tmp_path / "src", 4)

    outcomes = add_notes(tmp_path / "notes", sources, _options())

    assert [outcome.status for outcome in outcomes] == ["added"] * 4
    assert len(runs) == 1
    assert len(runs[0]) == 4


def test_a_batch_size_splits_the_conversion_into_runs(
    tmp_path: Path, available: None, runs: list[list[Path]]
) -> None:
    """MinerU writes nothing until a run finishes, so the run is also the unit
    of progress and the unit lost to a Ctrl-C. Chunking trades one extra model
    load per run for both."""
    sources = _pdfs(tmp_path / "src", 5)

    add_notes(tmp_path / "notes", sources, _options(mineru_batch_size=2))

    assert [len(run) for run in runs] == [2, 2, 1]


def test_a_batch_size_of_zero_converts_everything_in_one_run(
    tmp_path: Path, available: None, runs: list[list[Path]]
) -> None:
    sources = _pdfs(tmp_path / "src", 5)

    add_notes(tmp_path / "notes", sources, _options(mineru_batch_size=0))

    assert [len(run) for run in runs] == [5]


def test_a_source_already_in_the_collection_is_never_converted(
    tmp_path: Path, available: None, runs: list[list[Path]]
) -> None:
    """The whole point of hashing first. Duplicate detection already worked -
    it just used to run after the conversion it was about to discard."""
    sources = _pdfs(tmp_path / "src", 1)
    add_notes(tmp_path / "notes", sources, _options())
    runs.clear()

    outcomes = add_notes(tmp_path / "notes", sources, _options())

    assert [outcome.status for outcome in outcomes] == ["duplicate"]
    assert runs == []


def test_a_duplicate_among_new_documents_does_not_reach_the_converter(
    tmp_path: Path, available: None, runs: list[list[Path]]
) -> None:
    sources = _pdfs(tmp_path / "src", 3)
    add_notes(tmp_path / "notes", sources[:1], _options())
    runs.clear()

    outcomes = add_notes(tmp_path / "notes", sources, _options())

    assert [outcome.status for outcome in outcomes] == [
        "duplicate", "added", "added",
    ]
    assert [path.name for path in runs[0]] == ["paper-1.pdf", "paper-2.pdf"]


def test_the_same_file_named_twice_is_converted_once(
    tmp_path: Path, available: None, runs: list[list[Path]]
) -> None:
    source = _pdfs(tmp_path / "src", 1)[0]

    outcomes = add_notes(tmp_path / "notes", [source, source], _options())

    assert len(runs[0]) == 1
    assert [outcome.status for outcome in outcomes] == ["added", "duplicate"]


def test_a_missing_mineru_is_reported_once_before_anything_is_converted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, runs: list[list[Path]]
) -> None:
    """Fifty copies of the same install instructions say no more than one, and
    a batch that needs MinerU cannot get anywhere without it. Raised rather
    than collected as fifty failed outcomes, so nothing is written."""
    monkeypatch.setattr("boepie.corpus.intake.mineru_available", lambda: False)
    sources = _pdfs(tmp_path / "src", 3)

    with pytest.raises(IntakeError, match="uv sync --extra mineru"):
        add_notes(tmp_path / "notes", sources, _options())

    assert runs == []
    assert not (tmp_path / "notes").exists()


def test_a_batch_with_no_binaries_never_asks_whether_mineru_is_installed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A folder of Markdown must not be blocked by an extra it does not use."""
    monkeypatch.setattr("boepie.corpus.intake.mineru_available", lambda: False)
    note = tmp_path / "note.md"
    note.write_text("# Note\n\nBody.\n", encoding="utf-8")

    outcomes = add_notes(tmp_path / "notes", [str(note)], _options())

    assert [outcome.status for outcome in outcomes] == ["added"]


def test_a_document_mineru_produced_nothing_for_fails_alone(
    tmp_path: Path, available: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources = _pdfs(tmp_path / "src", 3)

    def fake_convert(paths, *, device_mode, backend, model_source):
        return MineruResult(
            markdown={
                path: "# Body\n" for path in paths if path.name != "paper-1.pdf"
            }
        )

    monkeypatch.setattr("boepie.corpus.add.convert_with_mineru", fake_convert)

    outcomes = add_notes(tmp_path / "notes", sources, _options())

    assert [outcome.status for outcome in outcomes] == ["added", "failed", "added"]
    assert "paper-1.pdf" in outcomes[1].detail


def test_a_run_that_failed_outright_does_not_stop_the_next_one(
    tmp_path: Path, available: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One unreadable file must not cost a folder the other forty."""
    sources = _pdfs(tmp_path / "src", 4)
    attempts: list[int] = []

    def fake_convert(paths, *, device_mode, backend, model_source):
        attempts.append(len(paths))
        if len(attempts) == 1:
            raise IntakeError("mineru failed: out of memory")
        return MineruResult(markdown={path: "# Body\n" for path in paths})

    monkeypatch.setattr("boepie.corpus.add.convert_with_mineru", fake_convert)

    outcomes = add_notes(tmp_path / "notes", sources, _options(mineru_batch_size=2))

    assert attempts == [2, 2]
    assert [outcome.status for outcome in outcomes] == [
        "failed", "failed", "added", "added",
    ]
    assert "out of memory" in outcomes[0].detail


def test_each_run_is_announced_before_it_starts(
    tmp_path: Path, available: None, runs: list[list[Path]]
) -> None:
    """A run is minutes of silence otherwise, and MinerU offers no finer
    moment to report - it writes nothing until the run is over."""
    announced: list[tuple[int, int, int]] = []
    sources = _pdfs(tmp_path / "src", 5)

    add_notes(
        tmp_path / "notes",
        sources,
        _options(mineru_batch_size=2),
        on_batch=lambda documents, number, total: announced.append(
            (documents, number, total)
        ),
    )

    assert announced == [(2, 1, 3), (2, 2, 3), (1, 3, 3)]
