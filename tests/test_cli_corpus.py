"""Tests for the `boepie corpus` CLI group (add/fetch/status/list), which
replaces the retired standalone `literature` group and `knowledge add`.

`add`/`fetch`/`status`/`list` are exercised with the underlying seams
(`add_user_paper`, `add_user_project`, `add_note`, `sync_literature`,
`sync_docs`, `collection_index`) monkeypatched at the `boepie.cli` module
level, mirroring how tests/test_cli_sync.py stubs the same kind of seam - no
real network request or filesystem write against a real corpus ever happens
here.
"""

from __future__ import annotations

import re
from pathlib import Path

import click
import pytest
from click.testing import CliRunner

from boepie import cli
from boepie.corpus.layout import IndexedDocument
from boepie.corpus.reconcile import DocsSyncResult, LiteratureSyncResult
from boepie.docs.manifest import DocsProject
from boepie.context.frontmatter import read_frontmatter
from boepie.literature.manifest import ArxivPaper
from tests.conftest import write_corpus_document

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _plain(output: str) -> str:
    """Rich's highlighter wraps numbers/punctuation (e.g. the "1" and "(s)"
    in "1 added" / "3 page(s) fetched") in their own ANSI spans even under
    CliRunner's non-tty capture, so a literal substring check on raw output
    is fragile - strip escape codes first."""
    return _ANSI_RE.sub("", output)


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def tmp_corpus_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """Point LITERATURE_DIR/DOCS_DIR/NOTES_DIR at empty tmp directories
    instead of the real machine-global defaults - hermetic, and never
    touches a developer's own fetched corpus."""
    dirs = {
        "literature": tmp_path / "literature-corpus",
        "docs": tmp_path / "docs-corpus",
        "notes": tmp_path / "notes",
    }
    monkeypatch.setattr(cli, "LITERATURE_DIR", dirs["literature"])
    monkeypatch.setattr(cli, "DOCS_DIR", dirs["docs"])
    monkeypatch.setattr(cli, "NOTES_DIR", dirs["notes"])
    return dirs


def _document(
    *, id: str = "abc1234567", natural_key: str = "smirnov2011", title: str = "A Paper", managed_by: str = "boepie",
    md_path: Path | None = None, extra: dict | None = None,
) -> IndexedDocument:
    frontmatter = {"title": title, "managed_by": managed_by, **(extra or {})}
    return IndexedDocument(
        id=id, natural_key=natural_key, md_path=md_path or Path(f"{title}.md"),
        wrapper_dir=None, frontmatter=frontmatter,
    )


# ---------------------------------------------------------------------------
# corpus add
# ---------------------------------------------------------------------------
#
# `add` writes documents straight to disk as `managed_by: user`; there is no
# manifest to intercept, so these drive the real intake and assert on what
# lands in the collection directory.


def test_corpus_add_notes_ingests_a_local_markdown_file(
    runner: CliRunner, tmp_corpus_dirs: dict[str, Path], tmp_path: Path
) -> None:
    source = tmp_path / "note.md"
    source.write_text("# My Note\n\nBody.\n", encoding="utf-8")

    result = runner.invoke(cli.cli, ["corpus", "add", "notes", str(source)])

    assert result.exit_code == 0, result.output
    assert "added" in result.output
    assert "1 added" in result.output
    written = list(tmp_corpus_dirs["notes"].glob("*.md"))
    assert [path.name for path in written] == ["My Note.md"]


def test_corpus_add_notes_takes_several_identifiers_at_once(
    runner: CliRunner, tmp_corpus_dirs: dict[str, Path], tmp_path: Path
) -> None:
    """Adding is meant to be staged like commits: several adds, one build."""
    first = tmp_path / "one.md"
    second = tmp_path / "two.md"
    first.write_text("# One\n\nFirst.\n", encoding="utf-8")
    second.write_text("# Two\n\nSecond.\n", encoding="utf-8")

    result = runner.invoke(
        cli.cli, ["corpus", "add", "notes", str(first), str(second)]
    )

    assert result.exit_code == 0, result.output
    assert "2 added" in result.output
    # The build hint is printed once for the batch, not once per document.
    assert result.output.count("index build") == 1


def test_corpus_add_notes_writes_the_nested_frontmatter_schema(
    runner: CliRunner, tmp_corpus_dirs: dict[str, Path], tmp_path: Path
) -> None:
    source = tmp_path / "note.md"
    source.write_text("# My Note\n\nBody.\n", encoding="utf-8")

    runner.invoke(cli.cli, ["corpus", "add", "notes", str(source)])

    frontmatter, _ = read_frontmatter(
        (tmp_corpus_dirs["notes"] / "My Note.md").read_text(encoding="utf-8")
    )
    assert frontmatter["managed_by"] == "user"
    assert frontmatter["source"]["from"] == str(source)
    assert frontmatter["source"]["via"] == "verbatim"
    assert frontmatter["source"]["format"] == "markdown"
    assert len(frontmatter["source"]["sha256"]) == 64
    assert "slug" not in frontmatter


def test_corpus_add_notes_warns_when_a_dotfile_title_loses_its_dot(
    runner: CliRunner, tmp_corpus_dirs: dict[str, Path], tmp_path: Path
) -> None:
    """The file on disk is not named what the title says, which is worth
    saying out loud - a dot-prefixed name would be walked as bookkeeping and
    skipped forever."""
    source = tmp_path / ".hidden-conventions.md"
    source.write_text("# .hidden-conventions\n\nBody.\n", encoding="utf-8")

    result = runner.invoke(cli.cli, ["corpus", "add", "notes", str(source)])

    assert result.exit_code == 0, result.output
    output = _plain(result.output)
    assert "looked like a dotfile name" in output
    assert (tmp_corpus_dirs["notes"] / "hidden-conventions.md").is_file()


def test_corpus_add_notes_can_suppress_the_dotfile_warning(
    runner: CliRunner, tmp_corpus_dirs: dict[str, Path], tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "CORPUS_WARN_ON_DOTFILE_TITLE", False)
    source = tmp_path / ".hidden-conventions.md"
    source.write_text("# .hidden-conventions\n\nBody.\n", encoding="utf-8")

    result = runner.invoke(cli.cli, ["corpus", "add", "notes", str(source)])

    assert result.exit_code == 0, result.output
    assert "looked like a dotfile name" not in _plain(result.output)


def test_corpus_add_notes_places_a_document_in_a_group(
    runner: CliRunner, tmp_corpus_dirs: dict[str, Path], tmp_path: Path
) -> None:
    source = tmp_path / "note.md"
    source.write_text("# Grouped\n\nBody.\n", encoding="utf-8")

    result = runner.invoke(
        cli.cli,
        ["corpus", "add", "notes", str(source), "--group", "calibration/subtopic"],
    )

    assert result.exit_code == 0, result.output
    assert (
        tmp_corpus_dirs["notes"] / "calibration" / "subtopic" / "Grouped.md"
    ).is_file()


def test_corpus_add_notes_skips_content_already_present(
    runner: CliRunner, tmp_corpus_dirs: dict[str, Path], tmp_path: Path
) -> None:
    """Detected by source checksum, so the same bytes under a different name
    are still recognised."""
    source = tmp_path / "note.md"
    source.write_text("# My Note\n\nBody.\n", encoding="utf-8")
    runner.invoke(cli.cli, ["corpus", "add", "notes", str(source)])

    result = runner.invoke(cli.cli, ["corpus", "add", "notes", str(source)])

    assert result.exit_code == 0, result.output
    assert "already present" in result.output
    assert "1 already present" in result.output
    assert len(list(tmp_corpus_dirs["notes"].glob("*.md"))) == 1


def test_corpus_add_notes_fences_a_source_file(
    runner: CliRunner, tmp_corpus_dirs: dict[str, Path], tmp_path: Path
) -> None:
    source = tmp_path / "solver.py"
    source.write_text("def solve():\n    return 1\n", encoding="utf-8")

    runner.invoke(cli.cli, ["corpus", "add", "notes", str(source)])

    text = (tmp_corpus_dirs["notes"] / "solver.py.md").read_text(encoding="utf-8")
    assert "```python" in text
    frontmatter, _ = read_frontmatter(text)
    assert frontmatter["source"]["format"] == "code"


def test_corpus_add_notes_explains_an_unreadable_binary_rather_than_leaking_a_codec_error(
    runner: CliRunner, tmp_corpus_dirs: dict[str, Path], tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A PDF used to surface a raw UnicodeDecodeError as the CLI's message."""
    monkeypatch.setattr("boepie.corpus.intake.mineru_available", lambda: False)
    source = tmp_path / "paper.pdf"
    source.write_bytes(b"%PDF-1.4\n\x8f\x8f binary\n")

    result = runner.invoke(cli.cli, ["corpus", "add", "notes", str(source)])

    assert result.exit_code == 1
    assert "codec" not in result.output
    assert "MinerU is required" in result.output
    assert "uv sync --extra mineru" in result.output


def test_corpus_add_notes_reports_an_unrecognised_identifier(
    runner: CliRunner, tmp_corpus_dirs: dict[str, Path]
) -> None:
    result = runner.invoke(cli.cli, ["corpus", "add", "notes", "not-a-thing"])

    assert result.exit_code == 1
    assert "not an existing file" in result.output


def test_corpus_add_notes_continues_past_a_failure_in_a_batch(
    runner: CliRunner, tmp_corpus_dirs: dict[str, Path], tmp_path: Path
) -> None:
    """One bad identifier must not cost the rest of the batch."""
    good = tmp_path / "good.md"
    good.write_text("# Good\n\nBody.\n", encoding="utf-8")

    result = runner.invoke(
        cli.cli, ["corpus", "add", "notes", "not-a-thing", str(good)]
    )

    assert result.exit_code == 1  # a failure still fails the run
    assert "1 added" in result.output
    assert "1 failed" in result.output
    assert (tmp_corpus_dirs["notes"] / "Good.md").is_file()


def test_corpus_add_literature_accepts_every_arxiv_id_spelling(
    runner: CliRunner, tmp_corpus_dirs: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The original bug: only the bare form was understood, and every other
    spelling failed as 'no arXiv entry found'."""
    seen: list[str] = []

    def fake_lookup(arxiv_id: str) -> dict[str, str]:
        seen.append(arxiv_id)
        return {"title": f"Paper {len(seen)}", "authors": "Doe, Jane", "year": "2024"}

    def fake_fetch(client, citekey: str, arxiv_id: str):
        from boepie.literature.fetch import FetchResult

        return FetchResult(
            citekey=citekey, markdown="# Paper\n\nBody.\n", source="ar5iv",
            page_url=f"https://ar5iv.labs.arxiv.org/html/{arxiv_id}",
        )

    monkeypatch.setattr("boepie.literature.fetch.lookup_arxiv_metadata", fake_lookup)
    monkeypatch.setattr("boepie.literature.fetch.fetch_paper", fake_fetch)

    result = runner.invoke(
        cli.cli,
        [
            "corpus", "add", "literature",
            "2409.19750",
            "arXiv:2409.19751",
            "2409.19752v1.pdf",
            "https://arxiv.org/abs/2409.19753v2",
        ],
    )

    assert result.exit_code == 0, result.output
    assert seen == ["2409.19750", "2409.19751", "2409.19752", "2409.19753"]
    assert "4 added" in result.output


def test_corpus_add_literature_writes_the_bib_block(
    runner: CliRunner, tmp_corpus_dirs: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "boepie.literature.fetch.lookup_arxiv_metadata",
        lambda arxiv_id: {
            "title": "Revisiting the RIME", "authors": "Smirnov, O. M.", "year": "2011",
        },
    )

    def fake_fetch(client, citekey: str, arxiv_id: str):
        from boepie.literature.fetch import FetchResult

        return FetchResult(
            citekey=citekey, markdown="# RIME\n\nBody.\n", source="ar5iv",
            page_url="https://ar5iv.labs.arxiv.org/html/1101.1185",
        )

    monkeypatch.setattr("boepie.literature.fetch.fetch_paper", fake_fetch)

    result = runner.invoke(
        cli.cli, ["corpus", "add", "literature", "1101.1185", "--citekey", "smirnov2011"]
    )

    assert result.exit_code == 0, result.output
    written = list(tmp_corpus_dirs["literature"].glob("*.md"))
    frontmatter, _ = read_frontmatter(written[0].read_text(encoding="utf-8"))
    assert frontmatter["bib"]["citekey"] == "smirnov2011"
    assert frontmatter["bib"]["arxiv_id"] == "1101.1185"
    assert frontmatter["bib"]["year"] == "2011"
    assert frontmatter["source"]["from"] == "arxiv:1101.1185"
    assert frontmatter["source"]["via"] == "ar5iv"


def test_corpus_add_literature_keeps_a_bib_entrys_own_citekey(
    runner: CliRunner, tmp_corpus_dirs: dict[str, Path], tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserving the citekey you already cite a paper by is the main reason
    to add a library as a .bib rather than as bare ids."""
    monkeypatch.setattr(
        "boepie.literature.fetch.lookup_arxiv_metadata",
        lambda arxiv_id: {"title": "CubiCal", "authors": "Kenyon, J. S.", "year": "2018"},
    )

    def fake_fetch(client, citekey: str, arxiv_id: str):
        from boepie.literature.fetch import FetchResult

        return FetchResult(
            citekey=citekey, markdown="# CubiCal\n\nBody.\n", source="ar5iv",
            page_url="https://ar5iv.labs.arxiv.org/html/1805.03410",
        )

    monkeypatch.setattr("boepie.literature.fetch.fetch_paper", fake_fetch)
    bib = tmp_path / "library.bib"
    bib.write_text(
        "@inproceedings{kenyonCubicalFastRadio2018,\n"
        "  title = {CubiCal},\n"
        "  author = {Kenyon, J. S.},\n"
        "  year = {2018},\n"
        "  doi = {10.1093/mnras/sty1221},\n"
        "  eprint = {1805.03410}\n}\n",
        encoding="utf-8",
    )

    result = runner.invoke(cli.cli, ["corpus", "add", "literature", str(bib)])

    assert result.exit_code == 0, result.output
    written = list(tmp_corpus_dirs["literature"].glob("*.md"))
    frontmatter, _ = read_frontmatter(written[0].read_text(encoding="utf-8"))
    assert frontmatter["bib"]["citekey"] == "kenyonCubicalFastRadio2018"
    assert frontmatter["bib"]["doi"] == "10.1093/mnras/sty1221"


def test_corpus_add_literature_detects_the_same_paper_reached_by_another_route(
    runner: CliRunner, tmp_corpus_dirs: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A paper added as a .bib entry and then as a bare arXiv id derives two
    different citekeys and has no source checksum, so only its bibliographic
    identity catches the duplicate."""
    calls: list[str] = []

    def fake_lookup(arxiv_id: str) -> dict[str, str]:
        calls.append(arxiv_id)
        return {"title": "CubiCal", "authors": "Kenyon, J. S.", "year": "2018"}

    def fake_fetch(client, citekey: str, arxiv_id: str):
        from boepie.literature.fetch import FetchResult

        return FetchResult(
            citekey=citekey, markdown="# CubiCal\n\nBody.\n", source="ar5iv",
            page_url="https://ar5iv.labs.arxiv.org/html/1805.03410",
        )

    monkeypatch.setattr("boepie.literature.fetch.lookup_arxiv_metadata", fake_lookup)
    monkeypatch.setattr("boepie.literature.fetch.fetch_paper", fake_fetch)

    runner.invoke(cli.cli, ["corpus", "add", "literature", "1805.03410"])
    calls.clear()

    result = runner.invoke(
        cli.cli, ["corpus", "add", "literature", "https://arxiv.org/abs/1805.03410v2"]
    )

    assert result.exit_code == 0, result.output
    assert "already present" in result.output
    # Short-circuits before the network round-trip, not after.
    assert calls == []


def test_corpus_add_literature_reports_an_unknown_arxiv_id(
    runner: CliRunner, tmp_corpus_dirs: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "boepie.literature.fetch.lookup_arxiv_metadata", lambda arxiv_id: None
    )

    result = runner.invoke(cli.cli, ["corpus", "add", "literature", "9999.99999"])

    assert result.exit_code == 1
    assert "no entry" in result.output.lower()


def test_corpus_add_docs_requires_a_project(
    runner: CliRunner, tmp_corpus_dirs: dict[str, Path]
) -> None:
    result = runner.invoke(
        cli.cli, ["corpus", "add", "docs", "https://example.org/docs/"]
    )

    assert result.exit_code != 0
    assert "--project" in result.output


def test_corpus_add_docs_ingests_a_local_file_under_its_project(
    runner: CliRunner, tmp_corpus_dirs: dict[str, Path], tmp_path: Path
) -> None:
    source = tmp_path / "manual.md"
    source.write_text("# Manual\n\nBody.\n", encoding="utf-8")

    result = runner.invoke(
        cli.cli, ["corpus", "add", "docs", str(source), "--project", "mytool"]
    )

    assert result.exit_code == 0, result.output
    written = tmp_corpus_dirs["docs"] / "mytool" / "Manual.md"
    assert written.is_file()
    frontmatter, _ = read_frontmatter(written.read_text(encoding="utf-8"))
    assert frontmatter["docs"]["project"] == "mytool"
    assert frontmatter["managed_by"] == "user"


# ---------------------------------------------------------------------------
# corpus move
# ---------------------------------------------------------------------------


def _add_note(runner: CliRunner, tmp_path: Path, title: str = "My Note") -> str:
    source = tmp_path / f"{title}.md"
    source.write_text(f"# {title}\n\nBody.\n", encoding="utf-8")
    runner.invoke(cli.cli, ["corpus", "add", "notes", str(source)])
    return title


def test_corpus_move_regroups_a_document_and_keeps_its_id(
    runner: CliRunner, tmp_corpus_dirs: dict[str, Path], tmp_path: Path
) -> None:
    """The whole point of the surrogate id: a document's location is free to
    change because nothing addresses it by path."""
    _add_note(runner, tmp_path)
    original, _ = read_frontmatter(
        (tmp_corpus_dirs["notes"] / "My Note.md").read_text(encoding="utf-8")
    )

    result = runner.invoke(
        cli.cli,
        ["corpus", "move", "--collection", "notes", original["id"], "--group", "calibration"],
    )

    assert result.exit_code == 0, result.output
    moved_path = tmp_corpus_dirs["notes"] / "calibration" / "My Note.md"
    assert moved_path.is_file()
    assert not (tmp_corpus_dirs["notes"] / "My Note.md").exists()
    moved, _ = read_frontmatter(moved_path.read_text(encoding="utf-8"))
    assert moved["id"] == original["id"]


def test_corpus_move_retitles_and_renames_the_file(
    runner: CliRunner, tmp_corpus_dirs: dict[str, Path], tmp_path: Path
) -> None:
    _add_note(runner, tmp_path)
    original, _ = read_frontmatter(
        (tmp_corpus_dirs["notes"] / "My Note.md").read_text(encoding="utf-8")
    )

    result = runner.invoke(
        cli.cli,
        ["corpus", "move", "--collection", "notes", original["id"], "--title", "Better Name"],
    )

    assert result.exit_code == 0, result.output
    renamed = tmp_corpus_dirs["notes"] / "Better Name.md"
    assert renamed.is_file()
    frontmatter, body = read_frontmatter(renamed.read_text(encoding="utf-8"))
    assert frontmatter["title"] == "Better Name"
    assert frontmatter["id"] == original["id"]
    assert "Body." in body


def test_corpus_move_carries_assets_with_a_wrapped_document(
    runner: CliRunner, tmp_corpus_dirs: dict[str, Path], tmp_path: Path
) -> None:
    """A document with assets moves as a directory, so its files travel."""
    written = write_corpus_document(
        tmp_corpus_dirs["notes"], document_id="wrapped0001", title="With Figure",
        body="![fig](diagram.png)", managed_by="user",
        assets={"diagram.png": b"not-really-a-png"},
    )
    assert written.parent.name == "With Figure"

    result = runner.invoke(
        cli.cli,
        ["corpus", "move", "--collection", "notes", "wrapped0001", "--group", "figures"],
    )

    assert result.exit_code == 0, result.output
    moved_dir = tmp_corpus_dirs["notes"] / "figures" / "With Figure"
    assert (moved_dir / "content.md").is_file()
    assert (moved_dir / "diagram.png").read_bytes() == b"not-really-a-png"


def test_corpus_move_keeps_a_docs_page_project_in_step_with_its_group(
    runner: CliRunner, tmp_corpus_dirs: dict[str, Path]
) -> None:
    """`docs.project` is both the natural key and what search_docs filters
    on, so it has to follow the page into a different project group."""
    write_corpus_document(
        tmp_corpus_dirs["docs"], document_id="docsmoved1", title="Guide",
        body="Body.", group="oldproject",
        docs={"project": "oldproject", "page": "guide"},
    )

    result = runner.invoke(
        cli.cli,
        ["corpus", "move", "--collection", "docs", "docsmoved1", "--group", "newproject"],
    )

    assert result.exit_code == 0, result.output
    frontmatter, _ = read_frontmatter(
        (tmp_corpus_dirs["docs"] / "newproject" / "Guide.md").read_text(encoding="utf-8")
    )
    assert frontmatter["docs"]["project"] == "newproject"


def test_corpus_move_requires_something_to_change(
    runner: CliRunner, tmp_corpus_dirs: dict[str, Path], tmp_path: Path
) -> None:
    _add_note(runner, tmp_path)
    frontmatter, _ = read_frontmatter(
        (tmp_corpus_dirs["notes"] / "My Note.md").read_text(encoding="utf-8")
    )

    result = runner.invoke(
        cli.cli, ["corpus", "move", "--collection", "notes", frontmatter["id"]]
    )

    assert result.exit_code == 1
    assert "nothing to do" in result.output


def test_corpus_move_names_a_way_to_find_the_id(
    runner: CliRunner, tmp_corpus_dirs: dict[str, Path]
) -> None:
    """`move`, `remove` and `read` all raise the same message for an unknown
    id, naming one way to go looking. They used to suggest three different
    commands, which made one failure read as three problems."""
    result = runner.invoke(
        cli.cli, ["corpus", "move", "--collection", "notes", "nosuchid99", "--group", "x"]
    )

    assert result.exit_code == 1
    assert "no document with id 'nosuchid99' in notes" in result.output
    assert "boepie corpus list --collection notes" in result.output


def test_unknown_id_reports_the_same_way_from_move_and_remove(
    runner: CliRunner, tmp_corpus_dirs: dict[str, Path]
) -> None:
    move = runner.invoke(
        cli.cli, ["corpus", "move", "--collection", "notes", "nosuchid99", "--group", "x"]
    )
    remove = runner.invoke(
        cli.cli, ["corpus", "remove", "--collection", "notes", "nosuchid99", "--yes"]
    )

    assert move.output.strip() == remove.output.strip()


# ---------------------------------------------------------------------------
# corpus remove
# ---------------------------------------------------------------------------


def test_corpus_remove_deletes_by_id(
    runner: CliRunner, tmp_corpus_dirs: dict[str, Path], tmp_path: Path
) -> None:
    source = tmp_path / "note.md"
    source.write_text("# My Note\n\nBody.\n", encoding="utf-8")
    runner.invoke(cli.cli, ["corpus", "add", "notes", str(source)])
    frontmatter, _ = read_frontmatter(
        (tmp_corpus_dirs["notes"] / "My Note.md").read_text(encoding="utf-8")
    )

    result = runner.invoke(
        cli.cli,
        ["corpus", "remove", "--collection", "notes", frontmatter["id"], "--yes"],
    )

    assert result.exit_code == 0, result.output
    assert not (tmp_corpus_dirs["notes"] / "My Note.md").exists()


def test_corpus_remove_names_the_listing_command_for_an_unknown_id(
    runner: CliRunner, tmp_corpus_dirs: dict[str, Path]
) -> None:
    result = runner.invoke(
        cli.cli, ["corpus", "remove", "--collection", "notes", "nosuchid99", "--yes"]
    )

    assert result.exit_code == 1
    assert "corpus list" in result.output


# ---------------------------------------------------------------------------
# corpus fetch
# ---------------------------------------------------------------------------


def test_corpus_fetch_literature_reports_counts(
    runner: CliRunner, tmp_corpus_dirs: dict[str, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    paper = ArxivPaper(citekey="smirnov2011", arxiv_id="1101.1185", title="X", authors="Y", year="2011")
    monkeypatch.setattr(cli, "load_literature_manifest", lambda corpus_dir: [paper])

    def fake_sync_literature(collection_dir, manifest, *, force_paths=(), delay=1.0, on_progress=None):
        return [
            LiteratureSyncResult(citekey="smirnov2011", action="added", id="abc1234567"),
            LiteratureSyncResult(citekey="gone2020", action="deleted", id="def7654321"),
        ]
    monkeypatch.setattr(cli, "sync_literature", fake_sync_literature)

    result = runner.invoke(cli.cli, ["corpus", "fetch", "--collection", "literature"])

    assert result.exit_code == 0, result.output
    output = _plain(result.output)
    assert "1 added" in output
    assert "1 deleted" in output
    assert "boepie index build --collection literature" in output


def test_corpus_fetch_literature_reports_unavailable_papers(
    runner: CliRunner, tmp_corpus_dirs: dict[str, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    paper = ArxivPaper(citekey="perkins2025", arxiv_id="2501.00000", title="X", authors="Y", year="2025")
    monkeypatch.setattr(cli, "load_literature_manifest", lambda corpus_dir: [paper])

    def fake_sync_literature(collection_dir, manifest, *, force_paths=(), delay=1.0, on_progress=None):
        return [LiteratureSyncResult(citekey="perkins2025", action="unavailable")]
    monkeypatch.setattr(cli, "sync_literature", fake_sync_literature)

    result = runner.invoke(cli.cli, ["corpus", "fetch", "--collection", "literature"])

    assert result.exit_code == 0, result.output
    assert "no HTML" in result.output
    assert "perkins2025" in result.output
    assert "boepie corpus add literature <file.pdf>" in _plain(result.output)


def test_corpus_fetch_literature_rejects_a_bad_force_target(
    runner: CliRunner, tmp_corpus_dirs: dict[str, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    paper = ArxivPaper(citekey="smirnov2011", arxiv_id="1101.1185", title="X", authors="Y", year="2011")
    monkeypatch.setattr(cli, "load_literature_manifest", lambda corpus_dir: [paper])

    def fake_sync_literature(collection_dir, manifest, *, force_paths=(), delay=1.0, on_progress=None):
        raise ValueError("no such corpus document to force-refresh: Nowhere.md")
    monkeypatch.setattr(cli, "sync_literature", fake_sync_literature)

    result = runner.invoke(
        cli.cli, ["corpus", "fetch", "--collection", "literature", "--force", "Nowhere.md"],
    )

    assert result.exit_code != 0
    assert "no such corpus document" in result.output


def test_corpus_fetch_docs_reports_counts_across_projects(
    runner: CliRunner, tmp_corpus_dirs: dict[str, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = DocsProject(project="stimela", base_url="https://stimela.readthedocs.io/en/latest/")
    monkeypatch.setattr(cli, "load_docs_manifest", lambda docs_dir: [project])

    def fake_sync_docs(collection_dir, manifest, *, force_paths=(), delay=0.2, timeout=30, on_progress=None):
        return [DocsSyncResult(project="stimela", added=2, skipped=1, refetched=0, deleted=0, failures=[])]
    monkeypatch.setattr(cli, "sync_docs", fake_sync_docs)

    result = runner.invoke(cli.cli, ["corpus", "fetch", "--collection", "docs"])

    assert result.exit_code == 0, result.output
    output = _plain(result.output)
    assert "2 added" in output
    assert "1 skipped" in output
    assert "boepie index build --collection docs" in output


def test_corpus_fetch_explains_why_notes_has_nothing_to_fetch(
    runner: CliRunner, tmp_corpus_dirs: dict[str, Path]
) -> None:
    """Accepted rather than rejected as an invalid choice: the reason is
    worth stating, and "not one of literature, docs" does not state it."""
    result = runner.invoke(cli.cli, ["corpus", "fetch", "--collection", "notes"])

    assert result.exit_code == 0, result.output
    output = _plain(result.output)
    assert "no packaged manifest" in output
    assert "corpus add notes" in output



def test_corpus_fetch_rejects_an_unknown_collection(
    runner: CliRunner, tmp_corpus_dirs: dict[str, Path]
) -> None:
    result = runner.invoke(cli.cli, ["corpus", "fetch", "--collection", "nope"])

    assert result.exit_code != 0
    assert "nope" in result.output


def test_corpus_fetch_defaults_to_the_manifest_backed_collections(
    runner: CliRunner, tmp_corpus_dirs: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """No --collection reconciles literature and docs, but never notes: notes
    have no packaged manifest to diff against."""
    reconciled: list[str] = []
    monkeypatch.setattr(
        cli, "_corpus_fetch_literature",
        lambda *args, **kwargs: reconciled.append("literature"),
    )
    monkeypatch.setattr(
        cli, "_corpus_fetch_docs", lambda *args, **kwargs: reconciled.append("docs")
    )

    result = runner.invoke(cli.cli, ["corpus", "fetch"])

    assert result.exit_code == 0, result.output
    assert reconciled == ["literature", "docs"]


def test_corpus_fetch_accepts_a_comma_separated_list(
    runner: CliRunner, tmp_corpus_dirs: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    reconciled: list[str] = []
    monkeypatch.setattr(
        cli, "_corpus_fetch_literature",
        lambda *args, **kwargs: reconciled.append("literature"),
    )
    monkeypatch.setattr(
        cli, "_corpus_fetch_docs", lambda *args, **kwargs: reconciled.append("docs")
    )

    result = runner.invoke(
        cli.cli, ["corpus", "fetch", "--collection", "docs,literature"]
    )

    assert result.exit_code == 0, result.output
    # Declared order, not the order they were typed in.
    assert reconciled == ["literature", "docs"]


# ---------------------------------------------------------------------------
# corpus status
# ---------------------------------------------------------------------------


def test_corpus_status_literature_reports_fetched_and_not_fetched(
    runner: CliRunner, tmp_corpus_dirs: dict[str, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    fetched = ArxivPaper(citekey="smirnov2011", arxiv_id="1101.1185", title="X", authors="Y", year="2011")
    missing = ArxivPaper(citekey="notyet2020", arxiv_id="2001.00001", title="X", authors="Y", year="2020")
    monkeypatch.setattr(cli, "load_literature_manifest", lambda corpus_dir: [fetched, missing])
    monkeypatch.setattr(
        cli, "collection_index",
        lambda collection_dir, *, collection, key_fields: [_document(natural_key="smirnov2011")],
    )

    result = runner.invoke(cli.cli, ["corpus", "status", "--collection", "literature"])

    assert result.exit_code == 0, result.output
    output = _plain(result.output)
    assert "1 total, 1 boepie-managed, 0 yours" in output
    assert "1 paper(s) in the manifest not fetched yet" in output
    assert "notyet2020" in output


def test_corpus_status_literature_flags_orphaned_documents(
    runner: CliRunner, tmp_corpus_dirs: dict[str, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "load_literature_manifest", lambda corpus_dir: [])
    orphan = _document(natural_key="gone2020", managed_by="boepie")
    monkeypatch.setattr(cli, "collection_index", lambda collection_dir, *, collection, key_fields: [orphan])

    result = runner.invoke(cli.cli, ["corpus", "status", "--collection", "literature"])

    assert result.exit_code == 0, result.output
    output = _plain(result.output)
    assert "no longer in the manifest" in output
    assert "gone2020" in output


def test_corpus_status_docs_reports_page_counts_per_project(
    runner: CliRunner, tmp_corpus_dirs: dict[str, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = DocsProject(project="stimela", base_url="https://stimela.readthedocs.io/en/latest/")
    monkeypatch.setattr(cli, "load_docs_manifest", lambda docs_dir: [project])
    pages = [
        _document(natural_key=f"stimela/p{index}", extra={"docs": {"project": "stimela"}})
        for index in range(3)
    ]
    monkeypatch.setattr(cli, "collection_index", lambda collection_dir, *, collection, key_fields: pages)

    result = runner.invoke(cli.cli, ["corpus", "status", "--collection", "docs"])

    assert result.exit_code == 0, result.output
    output = _plain(result.output)
    assert "3 total, 3 boepie-managed, 0 yours" in output
    assert "in step with the packaged manifest" in output


def test_corpus_status_notes_reports_a_count(
    runner: CliRunner, tmp_corpus_dirs: dict[str, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cli, "collection_index",
        lambda collection_dir, *, collection, key_fields: [
            _document(managed_by="user"),
            _document(natural_key="second-note", managed_by="user"),
        ],
    )

    result = runner.invoke(cli.cli, ["corpus", "status", "--collection", "notes"])

    assert result.exit_code == 0, result.output
    # Every collection reports the same shape, notes included.
    assert "2 total, 0 boepie-managed, 2 yours" in _plain(result.output)


# ---------------------------------------------------------------------------
# corpus list
# ---------------------------------------------------------------------------


def test_corpus_list_enumerates_documents(
    runner: CliRunner, tmp_corpus_dirs: dict[str, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    documents = [_document(id="id0000001", natural_key="smirnov2011", title="Revisiting the RIME")]
    monkeypatch.setattr(cli, "collection_index", lambda collection_dir, *, collection, key_fields: documents)

    result = runner.invoke(cli.cli, ["corpus", "list", "--collection", "literature"])

    assert result.exit_code == 0, result.output
    output = _plain(result.output)
    # Title leads: it is the only field a person recognises. The id follows
    # because it is what `read_*` and `corpus remove` take. The natural key
    # (citekey) is reconciler bookkeeping and no longer shown.
    assert output.startswith("Revisiting the RIME")
    assert "id0000001" in output
    assert "boepie" in output


def test_corpus_list_reports_an_empty_collection(
    runner: CliRunner, tmp_corpus_dirs: dict[str, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "collection_index", lambda collection_dir, *, collection, key_fields: [])

    result = runner.invoke(cli.cli, ["corpus", "list", "--collection", "notes"])

    assert result.exit_code == 0, result.output
    assert "No documents" in result.output


# ---------------------------------------------------------------------------
# group wiring / retirement of the old `literature`/`knowledge` groups
# ---------------------------------------------------------------------------


def test_corpus_group_help_lists_every_subcommand(runner: CliRunner) -> None:
    result = runner.invoke(cli.cli, ["corpus", "--help"])

    assert result.exit_code == 0
    for name in ("add", "fetch", "status", "list"):
        assert name in result.output


def test_literature_group_no_longer_exists(runner: CliRunner) -> None:
    result = runner.invoke(cli.cli, ["literature", "fetch"])

    assert result.exit_code != 0
    assert "No such command" in result.output


def test_knowledge_group_no_longer_exists(runner: CliRunner) -> None:
    result = runner.invoke(cli.cli, ["knowledge", "add", "some-identifier"])

    assert result.exit_code != 0
    assert "No such command" in result.output


# ---------------------------------------------------------------------------
# CollectionList: the comma-separated --collection selector
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "typed, expected",
    [
        ("notes", ("notes",)),
        ("docs,literature", ("literature", "docs")),
        ("literature, docs", ("literature", "docs")),
        ("all", ("literature", "docs", "notes")),
        # Repeats and `all` mixed with names both collapse to the declared set.
        ("notes,notes", ("notes",)),
        ("all,notes", ("literature", "docs", "notes")),
    ],
)
def test_collection_list_resolves_to_declared_order(typed: str, expected: tuple) -> None:
    param_type = cli.CollectionList(("literature", "docs", "notes"))
    assert param_type.convert(typed, None, None) == expected


def test_collection_list_rejects_an_unknown_name() -> None:
    param_type = cli.CollectionList(("literature", "docs", "notes"))
    with pytest.raises(click.BadParameter) as caught:
        param_type.convert("literature,bogus", None, None)
    assert "bogus" in str(caught.value)


def test_collection_list_rejects_an_empty_selection() -> None:
    param_type = cli.CollectionList(("literature", "docs", "notes"))
    with pytest.raises(click.BadParameter):
        param_type.convert(",", None, None)


# The exit code is the point of the `skipped` status, not the wording: a folder
# walk routinely turns up a file boepie declines to take (an image among the
# PDFs, a format whose converter is not installed), and that must not fail the
# command the way a genuine failure does. Nothing else in `_report_add`
# distinguishes the two, so this is the crossing test for it.
def test_corpus_add_reports_a_skip_without_failing_the_batch(
    runner: CliRunner, tmp_corpus_dirs: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    from boepie.corpus.add import AddOutcome

    def fake_add_notes(collection_dir, identifiers, options):
        return [
            AddOutcome(identifier="kept.md", status="added", title="Kept", document_id="aB3dE9fGhI"),
            AddOutcome(identifier="logo.png", status="skipped", detail="unsupported file type"),
        ]

    monkeypatch.setattr(cli, "add_notes", fake_add_notes)

    result = runner.invoke(cli.cli, ["corpus", "add", "notes", "anything"])

    assert result.exit_code == 0, result.output
    assert "skipped" in _plain(result.output)
    assert "1 skipped" in _plain(result.output)


def test_corpus_add_still_fails_the_batch_on_a_real_failure(
    runner: CliRunner, tmp_corpus_dirs: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    from boepie.corpus.add import AddOutcome

    def fake_add_notes(collection_dir, identifiers, options):
        return [AddOutcome(identifier="gone.md", status="failed", detail="does not exist")]

    monkeypatch.setattr(cli, "add_notes", fake_add_notes)

    result = runner.invoke(cli.cli, ["corpus", "add", "notes", "anything"])

    assert result.exit_code == 1, result.output


# The crossing test for input resolution: `InputError` is raised, not collected
# as an outcome, so without the CLI catching it a mistyped pattern would reach
# the user as a traceback rather than a message.
def test_corpus_add_reports_an_unmatched_pattern_as_a_clean_error(
    runner: CliRunner, tmp_corpus_dirs: dict[str, Path], tmp_path: Path
) -> None:
    (tmp_path / "code").mkdir()

    result = runner.invoke(
        cli.cli, ["corpus", "add", "notes", str(tmp_path / "code" / "*.xyz")]
    )

    assert result.exit_code == 1, result.output
    assert "matched no files" in _plain(result.output)
    assert result.exception is None or isinstance(result.exception, SystemExit)


def test_corpus_add_notes_expands_a_pattern_the_shell_did_not(
    runner: CliRunner, tmp_corpus_dirs: dict[str, Path], tmp_path: Path
) -> None:
    """A quoted pattern, or any pattern under a shell that does not expand
    arguments, arrives as one literal string and must still name its files."""
    source = tmp_path / "code"
    source.mkdir()
    (source / "a.md").write_text("# A\n\nbody\n", encoding="utf-8")
    (source / "b.md").write_text("# B\n\nbody\n", encoding="utf-8")

    result = runner.invoke(cli.cli, ["corpus", "add", "notes", str(source / "*.md")])

    assert result.exit_code == 0, result.output
    assert len(list(tmp_corpus_dirs["notes"].glob("*.md"))) == 2


# The crossing test for the folder walk: grouping, the accept-list and the
# exit code all meet here, and none of them is exercised by the unit tests in
# tests/test_corpus_inputs.py, which stop before anything is written.
def test_corpus_add_notes_walks_a_folder_into_mirrored_groups(
    runner: CliRunner, tmp_corpus_dirs: dict[str, Path], tmp_path: Path
) -> None:
    root = tmp_path / "code"
    (root / "gains").mkdir(parents=True)
    (root / "README.md").write_text("# Top\n", encoding="utf-8")
    (root / "gains" / "README.md").write_text("# Gains\n", encoding="utf-8")
    (root / "gains" / "lib.so").write_bytes(bytes(range(256)))

    result = runner.invoke(cli.cli, ["corpus", "add", "notes", str(root)])

    assert result.exit_code == 0, result.output
    notes = tmp_corpus_dirs["notes"]
    assert (notes / "Top.md").is_file()
    assert (notes / "gains" / "Gains.md").is_file()
    assert "1 skipped" in _plain(result.output)
    assert not list(notes.rglob("*.so"))


def test_corpus_add_group_prefixes_a_walked_folder_rather_than_flattening_it(
    runner: CliRunner, tmp_corpus_dirs: dict[str, Path], tmp_path: Path
) -> None:
    """Flattening would put both READMEs in one group and undo the mirroring."""
    root = tmp_path / "code"
    (root / "gains").mkdir(parents=True)
    (root / "b.md").write_text("# B\n", encoding="utf-8")
    (root / "gains" / "a.md").write_text("# A\n", encoding="utf-8")

    result = runner.invoke(
        cli.cli, ["corpus", "add", "notes", str(root), "--group", "quartical"]
    )

    assert result.exit_code == 0, result.output
    notes = tmp_corpus_dirs["notes"]
    assert (notes / "quartical" / "B.md").is_file()
    assert (notes / "quartical" / "gains" / "A.md").is_file()
