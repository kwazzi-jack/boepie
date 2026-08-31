"""`built_from`: refusing to serve an index whose corpus has moved on.

This is the one field in a build manifest that prevents a wrong answer rather
than describing a right one. A search hit carries a `source_path` and an
offset into a document, and `search_context` in particular hands those paths
to an agent to open with its own file tools - so an index built over a corpus
that has since changed sends the agent to text that is not there any more,
silently, with plausible-looking scores.

The distinction the whole design turns on: a document that **changed or went
away** makes the index wrong, while a document that is merely **new** makes it
incomplete. Only the first is an error. The second is the ordinary state
between a `corpus add` and the `index build` that follows it, and failing
there would break search for the entire staging workflow those two commands
were split apart to support.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from boepie.rag import engine
from boepie.rag.engine import StaleIndexError
from boepie.rag.loaders import NotesLoader
from tests.conftest import write_corpus_document

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _plain(output: str) -> str:
    return _ANSI_RE.sub("", output)


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    notes = tmp_path / "notes"
    for name in ("alpha", "beta"):
        write_corpus_document(
            notes,
            document_id=f"{name}000000"[:10],
            title=name.title(),
            body=f"# {name.title()}\n\nCalibration notes about {name} gains.\n",
            managed_by="user",
        )
    return notes


async def _build(corpus: Path, index_root: Path):
    return await engine.build(
        NotesLoader(corpus), index_root=index_root, embedding=None
    )


async def _load(index_root: Path):
    return await engine.load_for_query(index_root, "notes")


async def test_a_build_records_the_corpus_it_read(corpus: Path, tmp_path: Path):
    manifest = await _build(corpus, tmp_path / "index")

    assert manifest.built_from is not None
    assert Path(manifest.built_from["path"]) == corpus.resolve()
    assert len(manifest.built_from["documents"]) == 2


async def test_an_unchanged_corpus_loads(corpus: Path, tmp_path: Path):
    await _build(corpus, tmp_path / "index")

    handle = await _load(tmp_path / "index")

    assert handle.manifest.count > 0


async def test_an_edited_document_makes_the_index_stale(corpus: Path, tmp_path: Path):
    """The dangerous case: the index still serves the old text, under a path
    whose file now says something else."""
    await _build(corpus, tmp_path / "index")
    document = next(corpus.rglob("Alpha.md"))
    document.write_text(
        document.read_text(encoding="utf-8") + "\nRewritten entirely.\n",
        encoding="utf-8",
    )

    with pytest.raises(StaleIndexError, match="1 changed"):
        await _load(tmp_path / "index")


async def test_a_deleted_document_makes_the_index_stale(corpus: Path, tmp_path: Path):
    await _build(corpus, tmp_path / "index")
    next(corpus.rglob("Beta.md")).unlink()

    with pytest.raises(StaleIndexError, match="1 gone"):
        await _load(tmp_path / "index")


async def test_a_new_document_leaves_the_index_merely_incomplete(
    corpus: Path, tmp_path: Path
):
    """`corpus add` then search, before `index build` catches up. The new note
    will not be found, which is expected; everything else must keep working."""
    await _build(corpus, tmp_path / "index")
    write_corpus_document(
        corpus,
        document_id="gamma00000",
        title="Gamma",
        body="# Gamma\n\nAdded after the build.\n",
        managed_by="user",
    )

    handle = await _load(tmp_path / "index")

    assert handle.manifest.count > 0


async def test_the_error_names_the_command_that_fixes_it(corpus: Path, tmp_path: Path):
    """A mismatch is only actionable if it says what to run."""
    await _build(corpus, tmp_path / "index")
    next(corpus.rglob("Alpha.md")).unlink()

    with pytest.raises(StaleIndexError) as error:
        await _load(tmp_path / "index")

    assert "boepie index build --collection notes" in str(error.value)


async def test_an_absent_corpus_is_not_evidence_of_staleness(
    corpus: Path, tmp_path: Path
):
    """A prebuilt index fetched from a release has no local corpus at all. Its
    chunks are self-contained and answer queries perfectly well; refusing to
    serve them would be the worse failure."""
    import shutil

    await _build(corpus, tmp_path / "index")
    shutil.rmtree(corpus)

    handle = await _load(tmp_path / "index")

    assert handle.manifest.count > 0


async def test_an_index_built_elsewhere_is_not_checked(corpus: Path, tmp_path: Path):
    """The recorded path is another machine's. Nothing here can be compared
    against it, so the index is served rather than rejected."""
    manifest = await _build(corpus, tmp_path / "index")
    manifest_path = tmp_path / "index" / "notes" / manifest.index_id / "manifest.json"
    on_disk = json.loads(manifest_path.read_text(encoding="utf-8"))
    on_disk["built_from"]["path"] = "/build/agent/notes"
    manifest_path.write_text(json.dumps(on_disk), encoding="utf-8")

    handle = await _load(tmp_path / "index")

    assert handle.manifest.count > 0


async def test_an_index_predating_the_check_still_loads(corpus: Path, tmp_path: Path):
    """`built_from: null` means unverifiable, which is a reason to serve the
    index, not to reject it - the field did not exist when it was built."""
    manifest = await _build(corpus, tmp_path / "index")
    next(corpus.rglob("Alpha.md")).unlink()
    manifest_path = tmp_path / "index" / "notes" / manifest.index_id / "manifest.json"
    on_disk = json.loads(manifest_path.read_text(encoding="utf-8"))
    on_disk["built_from"] = None
    manifest_path.write_text(json.dumps(on_disk), encoding="utf-8")

    handle = await _load(tmp_path / "index")

    assert handle.manifest.count > 0


async def test_a_rebuild_clears_the_staleness(corpus: Path, tmp_path: Path):
    await _build(corpus, tmp_path / "index")
    next(corpus.rglob("Alpha.md")).unlink()

    await _build(corpus, tmp_path / "index")
    handle = await _load(tmp_path / "index")

    assert handle.manifest.count > 0


# ---------------------------------------------------------------------------
# What the user actually sees
# ---------------------------------------------------------------------------
#
# Every one of these was a route by which a stale index reported something
# other than staleness. They are listed separately because each surface
# swallowed it differently.


@pytest.fixture
def runner():
    from click.testing import CliRunner

    return CliRunner()


@pytest.fixture
def indexed(corpus: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A built notes index, wired into the CLI's view of the machine."""
    import asyncio

    from boepie import cli

    index_root = tmp_path / "index"
    asyncio.run(_build(corpus, index_root))
    monkeypatch.setattr(cli, "INDEX_DIR", index_root)
    monkeypatch.setattr(cli, "NOTES_DIR", corpus)
    yield corpus
    asyncio.run(engine.clear_cache())


def _make_stale(corpus: Path) -> None:
    document = next(corpus.rglob("Alpha.md"))
    document.write_text(
        document.read_text(encoding="utf-8") + "\nRewritten.\n", encoding="utf-8"
    )


def test_index_status_names_a_stale_index_and_its_fix(runner, indexed: Path) -> None:
    """The 2026-07-21 incident was undiagnosable from `status`. One row now
    answers "is anything wrong" without reading a word of the rest."""
    from boepie import cli

    _make_stale(indexed)
    result = runner.invoke(cli.cli, ["index", "status"])
    output = _plain(result.output)

    assert result.exit_code == 0
    assert "stale" in output
    assert "boepie index build --collection notes" in output


def test_index_status_says_so_when_the_index_is_in_step(runner, indexed: Path) -> None:
    from boepie import cli

    result = runner.invoke(cli.cli, ["index", "status"])

    assert "in step" in _plain(result.output)


def test_a_sweep_does_not_quietly_drop_a_stale_collection(
    runner, indexed: Path
) -> None:
    """`--collection all` omits a collection you have not indexed, which is
    fair. Omitting one whose index is stale would leave the user believing it
    was searched and had nothing to say."""
    from boepie import cli

    _make_stale(indexed)
    result = runner.invoke(cli.cli, ["search", "calibration"])

    assert result.exit_code != 0
    assert "stale" in _plain(result.output)


def test_read_reports_staleness_rather_than_an_unknown_id(
    runner, indexed: Path
) -> None:
    """`read` tries each collection in turn and treats a failure as "look in
    the next one", so this used to surface as "no document with id ..."."""
    from boepie import cli

    _make_stale(indexed)
    result = runner.invoke(cli.cli, ["read", "alpha00000"[:10]])
    output = _plain(result.output)

    assert "stale" in output
    assert "no document with id" not in output
