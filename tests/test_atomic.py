"""Writes that survive an interrupt.

The regression these exist for is concrete: `rag.engine.build` used to delete
a collection's index directory and *then* spend minutes embedding into it, so
a Ctrl-C left the collection with no index at all - worse than never having
run the command.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from boepie._atomic import replace_file, replacing_directory


def test_replace_file_writes_text_and_creates_parents(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "deeper" / "manifest.json"
    replace_file(target, json.dumps({"index_id": "bm25"}))
    assert json.loads(target.read_text(encoding="utf-8")) == {"index_id": "bm25"}


def test_replace_file_writes_bytes(tmp_path: Path) -> None:
    target = tmp_path / "figure.png"
    replace_file(target, b"\x89PNG\r\n")
    assert target.read_bytes() == b"\x89PNG\r\n"


def test_replace_file_leaves_the_previous_content_when_interrupted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A Ctrl-C between writing the temporary and renaming it into place.

    `AGENTS.md` is the case that matters: it is the user's own hand-written
    file, boepie only appends a pointer line to it, and there is no backup.
    """
    target = tmp_path / "AGENTS.md"
    target.write_text("the user's own notes\n", encoding="utf-8")

    def interrupted(*args: object, **kwargs: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr("boepie._atomic.os.replace", interrupted)
    with pytest.raises(KeyboardInterrupt):
        replace_file(target, "replacement\n")

    assert target.read_text(encoding="utf-8") == "the user's own notes\n"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["AGENTS.md"]


def test_replacing_directory_swaps_in_the_staged_contents(tmp_path: Path) -> None:
    target = tmp_path / "bm25"
    with replacing_directory(target) as staging:
        (staging / "chunks.jsonl").write_text("{}\n", encoding="utf-8")
        (staging / "sub").mkdir()
        (staging / "sub" / "vocab.json").write_text("{}\n", encoding="utf-8")

    assert (target / "chunks.jsonl").read_text(encoding="utf-8") == "{}\n"
    assert (target / "sub" / "vocab.json").exists()
    assert [path.name for path in tmp_path.iterdir()] == ["bm25"]


def test_an_interrupt_leaves_the_previous_directory_untouched(tmp_path: Path) -> None:
    """The whole point: the old index keeps serving for the entire build."""
    target = tmp_path / "bm25"
    target.mkdir()
    (target / "chunks.jsonl").write_text("the index that already worked\n", encoding="utf-8")

    with pytest.raises(KeyboardInterrupt):
        with replacing_directory(target) as staging:
            (staging / "chunks.jsonl").write_text("half a rebuild\n", encoding="utf-8")
            raise KeyboardInterrupt

    assert (target / "chunks.jsonl").read_text(encoding="utf-8") == "the index that already worked\n"
    assert [path.name for path in tmp_path.iterdir()] == ["bm25"]


def test_staging_left_by_a_kill_is_swept_by_the_next_build(tmp_path: Path) -> None:
    """Only a SIGKILL can leave one, and it would be a full copy of an index."""
    target = tmp_path / "bm25"
    orphan = tmp_path / ".bm25.staging-abcdef"
    orphan.mkdir()
    (orphan / "chunks.jsonl").write_text("abandoned\n", encoding="utf-8")

    with replacing_directory(target) as staging:
        (staging / "chunks.jsonl").write_text("{}\n", encoding="utf-8")

    assert not orphan.exists()
    assert [path.name for path in tmp_path.iterdir()] == ["bm25"]


def test_a_sweep_never_touches_a_real_sibling(tmp_path: Path) -> None:
    """Staging names are dot-prefixed; `bm25` and `bm25-other` are not."""
    target = tmp_path / "bm25"
    sibling = tmp_path / "bm25-other"
    sibling.mkdir()
    (sibling / "chunks.jsonl").write_text("a different index id\n", encoding="utf-8")

    with replacing_directory(target) as staging:
        (staging / "chunks.jsonl").write_text("{}\n", encoding="utf-8")

    assert (sibling / "chunks.jsonl").exists()


def test_an_interrupted_rebuild_leaves_the_previous_index_serving(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The end-to-end case, at the level it actually broke.

    `build` used to `rmtree` the collection directory and only then embed, so
    an interrupt during a rebuild left `latest.json` pointing at an empty
    directory and every search against that collection failing.
    """
    import asyncio

    from boepie.rag import engine
    from boepie.rag.embedding import ModelBinding
    from boepie.rag.models import Document

    class TinyLoader:
        name = "notes"

        def iter_documents(self):
            for number in range(3):
                yield Document(
                    id=f"doc{number}",
                    title=f"Document {number}",
                    text=f"a paragraph about calibration number {number}\n" * 20,
                    metadata={"group": ""},
                    source_path=f"/nowhere/doc{number}.md",
                )

    asyncio.run(engine.build(TinyLoader(), index_root=tmp_path, embedding=None))
    before = asyncio.run(engine.load_for_query(index_root=tmp_path, collection="notes"))
    assert before.chunks

    async def interrupted_embed(binding, texts, *, on_progress=None):
        raise KeyboardInterrupt

    monkeypatch.setattr(engine, "embed_texts", interrupted_embed)
    with pytest.raises(KeyboardInterrupt):
        asyncio.run(
            engine.build(
                TinyLoader(),
                index_root=tmp_path,
                embedding=ModelBinding("fastembed", "fake-model", None, 4),
                index_id="bm25",
            )
        )

    after = asyncio.run(engine.load_for_query(index_root=tmp_path, collection="notes"))
    assert [chunk.text for chunk in after.chunks] == [chunk.text for chunk in before.chunks]
