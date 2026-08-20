"""Tests for scripts/migrate_corpus_layout.py: the one-time, per-machine
conversion of an old `{name}/{name}.md` + `metadata.json` corpus (literature/
notes: one document per directory; docs: `{project}/{docname}.md` pages
flattened under one project directory) into boepie.corpus's new layout.

Loaded via importlib.util since scripts/ is dev-only tooling, not part of the
installed package - the script's own runtime dependencies (click, rich) are
already boepie's own, so no extra install is needed to exercise it directly.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path
from types import ModuleType

import pytest
from click.testing import CliRunner

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _plain(output: str) -> str:
    """Rich's highlighter wraps numbers/punctuation (e.g. the "1" and "(s)"
    in "1 document(s)") in their own ANSI spans even under CliRunner's
    non-tty capture, so a literal substring check on raw output is fragile -
    strip escape codes first."""
    return _ANSI_RE.sub("", output)

from boepie.corpus.document import read_document

_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "migrate_corpus_layout.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("migrate_corpus_layout", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


migrate = _load_script()


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _write_old_document(
    doc_dir: Path, *, name: str, markdown: str = "# Title\n\nBody.\n", metadata: dict | None = None,
) -> None:
    doc_dir.mkdir(parents=True, exist_ok=True)
    (doc_dir / f"{name}.md").write_text(markdown, encoding="utf-8")
    if metadata is not None:
        (doc_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")


# ---------------------------------------------------------------------------
# literature/notes: one document per directory
# ---------------------------------------------------------------------------


def test_migrates_a_literature_paper_directory(tmp_path):
    _write_old_document(
        tmp_path / "smirnov2011", name="smirnov2011",
        markdown="# Revisiting the RIME\n\nBody text.\n",
        metadata={
            "citekey": "smirnov2011", "title": "Revisiting the RIME", "author": "O. Smirnov",
            "year": "2011", "doi": "10.1051/example", "arxiv_id": "1101.1185",
            "source": "arxiv-html", "source_url": "https://arxiv.org/html/1101.1185",
        },
    )

    migrated = migrate._migrate_document_per_directory(tmp_path, collection="literature", dry_run=False)

    assert migrated == 1
    assert not (tmp_path / "smirnov2011").exists()
    new_paths = list(tmp_path.glob("*.md"))
    assert len(new_paths) == 1
    document = read_document(new_paths[0])
    assert document.frontmatter["bib"]["citekey"] == "smirnov2011"
    assert document.frontmatter["bib"]["doi"] == "10.1051/example"
    assert document.frontmatter["managed_by"] == "boepie"
    assert "Body text." in document.body


def test_migrates_a_notes_directory_as_user_managed(tmp_path):
    _write_old_document(
        tmp_path / "my-note", name="my-note", markdown="# My Note\n\nBody.\n",
        metadata={"slug": "my-note", "title": "My Note", "source": "text", "source_path": "/tmp/x.md"},
    )

    migrated = migrate._migrate_document_per_directory(tmp_path, collection="notes", dry_run=False)

    assert migrated == 1
    document = read_document(next(tmp_path.glob("*.md")))
    # The old `slug` is gone: notes are addressed by `id` alone.
    assert "slug" not in document.frontmatter
    assert document.frontmatter["managed_by"] == "user"
    assert document.frontmatter["source"]["via"] == "verbatim"


def test_migrates_a_document_with_assets_into_a_wrapper_directory(tmp_path):
    doc_dir = tmp_path / "with-figs"
    _write_old_document(doc_dir, name="with-figs", markdown="# With Figures\n\nSee fig1.\n")
    (doc_dir / "fig1.png").write_bytes(b"\x89PNG")

    migrate._migrate_document_per_directory(tmp_path, collection="literature", dry_run=False)

    wrapper_dirs = [path for path in tmp_path.iterdir() if path.is_dir()]
    assert len(wrapper_dirs) == 1
    assert (wrapper_dirs[0] / "fig1.png").read_bytes() == b"\x89PNG"


def test_dry_run_migrates_nothing(tmp_path):
    _write_old_document(tmp_path / "smirnov2011", name="smirnov2011")

    migrated = migrate._migrate_document_per_directory(tmp_path, collection="literature", dry_run=True)

    assert migrated == 1
    assert (tmp_path / "smirnov2011" / "smirnov2011.md").exists()
    assert list(tmp_path.glob("*.md")) == []


def test_rerun_skips_an_already_migrated_document(tmp_path):
    _write_old_document(tmp_path / "smirnov2011", name="smirnov2011")
    migrate._migrate_document_per_directory(tmp_path, collection="literature", dry_run=False)

    migrated_again = migrate._migrate_document_per_directory(tmp_path, collection="literature", dry_run=False)

    assert migrated_again == 0
    assert len(list(tmp_path.glob("*.md"))) == 1


def test_falls_back_to_directory_name_when_metadata_json_is_missing(tmp_path):
    _write_old_document(tmp_path / "bare2020", name="bare2020", markdown="No heading here.\n")

    migrated = migrate._migrate_document_per_directory(tmp_path, collection="literature", dry_run=False)

    assert migrated == 1
    document = read_document(next(tmp_path.glob("*.md")))
    assert document.frontmatter["bib"]["citekey"] == "bare2020"


# ---------------------------------------------------------------------------
# docs: {project}/{docname}.md pages, flattened under {project}/
# ---------------------------------------------------------------------------


def test_migrates_docs_pages_flattened_under_the_project(tmp_path):
    project_dir = tmp_path / "stimela"
    project_dir.mkdir()
    (project_dir / "index.md").write_text("# Stimela Docs\n\nIntro.\n", encoding="utf-8")
    (project_dir / "guide.md").write_text("# Guide\n\nHow to.\n", encoding="utf-8")
    (project_dir / "metadata.json").write_text(
        json.dumps({"base_url": "https://stimela.readthedocs.io/en/latest/", "version": "1.0"}),
        encoding="utf-8",
    )

    migrated = migrate._migrate_docs(tmp_path, dry_run=False)

    assert migrated == 2
    assert not (project_dir / "metadata.json").exists()
    pages_by_docname = {
        read_document(path).frontmatter["docs"]["page"]: read_document(path)
        for path in project_dir.glob("*.md")
    }
    assert set(pages_by_docname) == {"index", "guide"}
    assert pages_by_docname["guide"].frontmatter["docs"]["project"] == "stimela"
    assert (
        pages_by_docname["guide"].frontmatter["docs"]["base_url"]
        == "https://stimela.readthedocs.io/en/latest/"
    )


def test_migrates_a_docs_page_whose_title_collides_with_its_own_old_filename(tmp_path):
    """Regression test: a page with no markdown heading falls back to its
    docname as the title, which can make its derived new filename identical
    to its pre-migration path (e.g. docname "guide" -> "guide.md" both
    before and after). The migration must not delete the file it just wrote
    in place - see the corresponding fix in _migrate_docs."""
    project_dir = tmp_path / "stimela"
    project_dir.mkdir()
    (project_dir / "guide.md").write_text("No heading at all, just prose.\n", encoding="utf-8")

    migrated = migrate._migrate_docs(tmp_path, dry_run=False)

    assert migrated == 1
    document = read_document(project_dir / "guide.md")
    assert document.frontmatter["docs"]["page"] == "guide"
    assert "No heading at all" in document.body


def test_migrates_pages_whose_derived_filenames_collide_with_each_others_raw_paths(tmp_path):
    """Regression test: a nested page's derived title-filename can coincide
    with a *different*, not-yet-processed top-level page's still-unmigrated
    raw path (e.g. two sections both titled "Overview"). Pages are read in
    alphabetical order, so "aaa/index.md" (derived filename "zzz.md") is
    processed before the real "zzz.md" - writing the derived filename in a
    single read-then-write pass per page would silently destroy "zzz.md"'s
    content before it was ever read. _migrate_docs must read every page
    before writing any of them to avoid this."""
    project_dir = tmp_path / "stimela"
    (project_dir / "aaa").mkdir(parents=True)
    (project_dir / "aaa" / "index.md").write_text("# zzz\n\nContent from the nested page.\n", encoding="utf-8")
    (project_dir / "zzz.md").write_text("# Something Else\n\nOriginal top-level content.\n", encoding="utf-8")

    migrated = migrate._migrate_docs(tmp_path, dry_run=False)

    assert migrated == 2
    bodies = {read_document(path).body for path in project_dir.glob("*.md")}
    assert bodies == {
        "# zzz\n\nContent from the nested page.\n",
        "# Something Else\n\nOriginal top-level content.\n",
    }


def test_migrates_nested_docnames_flattened_directly_under_the_project(tmp_path):
    project_dir = tmp_path / "quartical"
    (project_dir / "changelogs").mkdir(parents=True)
    (project_dir / "changelogs" / "1.0.md").write_text("# Changelog 1.0\n\nNotes.\n", encoding="utf-8")

    migrated = migrate._migrate_docs(tmp_path, dry_run=False)

    assert migrated == 1
    assert not (project_dir / "changelogs").exists()
    new_paths = list(project_dir.glob("*.md"))
    assert len(new_paths) == 1
    document = read_document(new_paths[0])
    assert document.frontmatter["docs"]["page"] == "changelogs/1.0"


def test_docs_dry_run_leaves_nested_subdirectories_in_place(tmp_path):
    project_dir = tmp_path / "quartical"
    (project_dir / "changelogs").mkdir(parents=True)
    (project_dir / "changelogs" / "1.0.md").write_text("# Changelog 1.0\n\nNotes.\n", encoding="utf-8")

    migrated = migrate._migrate_docs(tmp_path, dry_run=True)

    assert migrated == 1
    assert (project_dir / "changelogs" / "1.0.md").exists()


def test_docs_rerun_skips_already_migrated_pages(tmp_path):
    project_dir = tmp_path / "stimela"
    project_dir.mkdir()
    (project_dir / "index.md").write_text("# Index\n\nBody.\n", encoding="utf-8")
    migrate._migrate_docs(tmp_path, dry_run=False)

    migrated_again = migrate._migrate_docs(tmp_path, dry_run=False)

    assert migrated_again == 0


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------


def test_cli_migrates_a_literature_directory(tmp_path, runner):
    _write_old_document(tmp_path / "smirnov2011", name="smirnov2011")

    result = runner.invoke(migrate.migrate_corpus_layout, ["literature", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert "migrated 1 document(s)" in _plain(result.output)
    assert list(tmp_path.glob("*.md"))


def test_cli_dry_run_flag_reports_without_writing(tmp_path, runner):
    _write_old_document(tmp_path / "smirnov2011", name="smirnov2011")

    result = runner.invoke(migrate.migrate_corpus_layout, ["literature", str(tmp_path), "--dry-run"])

    assert result.exit_code == 0, result.output
    assert "would migrate 1 document(s)" in _plain(result.output)
    assert (tmp_path / "smirnov2011" / "smirnov2011.md").exists()


def test_cli_rejects_an_unknown_collection(tmp_path, runner):
    result = runner.invoke(migrate.migrate_corpus_layout, ["images", str(tmp_path)])

    assert result.exit_code != 0
    assert "images" in result.output.lower() or "invalid" in result.output.lower()


def test_cli_rejects_a_nonexistent_directory(tmp_path, runner):
    result = runner.invoke(migrate.migrate_corpus_layout, ["literature", str(tmp_path / "does-not-exist")])

    assert result.exit_code != 0
