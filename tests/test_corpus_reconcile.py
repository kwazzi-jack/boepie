"""Tests for boepie.corpus.reconcile: the literature/docs manifest-diff sync
(sync_literature/sync_docs_project/sync_docs) and the shared --force-target
resolution (resolve_force_paths/normalize_force_path). Mirrors
test_bundle.py's force-path/orphan-survival test shapes, adapted to
corpus's add/skip/refetch/delete semantics.

`fetch_paper`/`iter_project_pages`/`probe_discovery_mode`/`fetch_version`
are monkeypatched at the reconcile module level - no real network request
happens here, and `boepie.corpus.reconcile`'s own `httpx.Client(...)`
context manager opens without ever making a request.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from boepie.context.frontmatter import read_frontmatter
from boepie.corpus import reconcile
from boepie.corpus.document import write_leaf_document
from boepie.corpus.layout import collection_index, lookup_path
from boepie.corpus.schema import (
    KEY_FIELDS,
    Source,
    docs_blocks,
    literature_blocks,
    parse_frontmatter,
)
from boepie.docs.fetch import PageContent
from boepie.docs.manifest import DocsProject
from boepie.literature.fetch import FetchResult
from boepie.literature.manifest import ArxivPaper


def _paper(citekey: str = "smirnov2011", arxiv_id: str = "1101.1185", title: str = "Revisiting the RIME") -> ArxivPaper:
    return ArxivPaper(citekey=citekey, arxiv_id=arxiv_id, title=title, authors="O. Smirnov", year="2011")


def _fake_fetch_paper(markdown: str = "# Paper\n\nBody.\n"):
    def fake(client, citekey, arxiv_id):
        return FetchResult(
            citekey=citekey, markdown=markdown, source="arxiv-html",
            page_url=f"https://arxiv.org/html/{arxiv_id}",
        )
    return fake


def _write_existing(md_path: Path, *, document_id: str, fields: dict, body: str = "Old body.\n"):
    return write_leaf_document(md_path, document_id=document_id, frontmatter_fields=fields, body=body)


def _source(origin: str = "arxiv:1101.1185", via: str = "arxiv-html") -> dict:
    return Source(origin=origin, via=via, format="html").model_dump(
        mode="json", by_alias=True, exclude_none=True
    )


def _lit_fields(citekey: str, title: str, managed_by: str = "boepie") -> dict:
    """An on-disk literature document in the shape the schema declares."""
    return {
        "title": title,
        "managed_by": managed_by,
        "source": _source(),
        **literature_blocks(citekey=citekey),
    }


def _docs_fields(project: str, page: str, title: str, managed_by: str = "boepie") -> dict:
    """An on-disk docs page in the shape the schema declares."""
    return {
        "title": title,
        "managed_by": managed_by,
        "source": _source(origin=f"https://example.org/{page}", via="sphinx"),
        **docs_blocks(project=project, page=page, base_url="https://example.org"),
    }


# ---------------------------------------------------------------------------
# sync_literature: add
# ---------------------------------------------------------------------------


def test_sync_literature_adds_a_new_paper(tmp_path, monkeypatch):
    monkeypatch.setattr(reconcile, "fetch_paper", _fake_fetch_paper())

    results = reconcile.sync_literature(tmp_path, [_paper()], delay=0.0)

    assert len(results) == 1 and results[0].action == "added"
    md_paths = list(tmp_path.glob("*.md"))
    assert len(md_paths) == 1
    frontmatter, body = read_frontmatter(md_paths[0].read_text(encoding="utf-8"))
    assert frontmatter["bib"]["citekey"] == "smirnov2011"
    assert frontmatter["managed_by"] == "boepie"
    assert frontmatter["source"]["via"] == "arxiv-html"
    assert frontmatter["source"]["from"] == "arxiv:1101.1185"
    assert frontmatter["id"] == results[0].id
    assert "Body." in body


def test_sync_literature_marks_unavailable_and_writes_nothing(tmp_path, monkeypatch):
    def fake(client, citekey, arxiv_id):
        return FetchResult(citekey=citekey, markdown=None, source="unavailable", page_url=None)
    monkeypatch.setattr(reconcile, "fetch_paper", fake)

    results = reconcile.sync_literature(tmp_path, [_paper()], delay=0.0)

    assert results == [reconcile.LiteratureSyncResult(citekey="smirnov2011", action="unavailable", id=None)]
    assert list(tmp_path.glob("*.md")) == []


# ---------------------------------------------------------------------------
# sync_literature: skip / force
# ---------------------------------------------------------------------------


def test_sync_literature_skips_an_existing_paper_unless_forced(tmp_path, monkeypatch):
    def must_not_be_called(client, citekey, arxiv_id):
        raise AssertionError("fetch_paper must not be called for an existing, unforced paper")
    monkeypatch.setattr(reconcile, "fetch_paper", must_not_be_called)

    existing_path = tmp_path / "Existing.md"
    _write_existing(
        existing_path, document_id="existing01",
        fields=_lit_fields("smirnov2011", "Existing"),
    )

    results = reconcile.sync_literature(tmp_path, [_paper()], delay=0.0)

    assert results == [reconcile.LiteratureSyncResult(citekey="smirnov2011", action="skipped", id="existing01")]
    assert existing_path.read_text(encoding="utf-8").count("Old body.") == 1


def test_sync_literature_force_refetches_in_place_same_id_same_path(tmp_path, monkeypatch):
    monkeypatch.setattr(reconcile, "fetch_paper", _fake_fetch_paper("# New body.\n"))

    existing_path = tmp_path / "Existing.md"
    _write_existing(
        existing_path, document_id="existing01",
        fields=_lit_fields("smirnov2011", "Existing"),
    )

    results = reconcile.sync_literature(
        tmp_path, [_paper()], force_paths=["Existing.md"], delay=0.0,
    )

    assert results == [reconcile.LiteratureSyncResult(citekey="smirnov2011", action="refetched", id="existing01")]
    frontmatter, body = read_frontmatter(existing_path.read_text(encoding="utf-8"))
    assert frontmatter["id"] == "existing01"
    assert "New body." in body
    assert list(tmp_path.glob("*.md")) == [existing_path]


def test_sync_literature_never_touches_a_source_local_paper(tmp_path, monkeypatch):
    def must_not_be_called(client, citekey, arxiv_id):
        raise AssertionError("fetch_paper must not be called for a managed_by: user paper")
    monkeypatch.setattr(reconcile, "fetch_paper", must_not_be_called)

    local_path = tmp_path / "My Local Copy.md"
    _write_existing(
        local_path, document_id="local0001",
        fields=_lit_fields("smirnov2011", "My Local Copy", managed_by="user"),
        body="Hand-edited.\n",
    )

    # Not in the manifest at all: a managed_by: user orphan must survive too.
    results = reconcile.sync_literature(tmp_path, [], delay=0.0)

    assert results == []
    assert local_path.exists()
    assert "Hand-edited." in local_path.read_text(encoding="utf-8")


def test_sync_literature_reports_a_manifest_paper_that_is_yours(tmp_path, monkeypatch):
    """The same silence as the docs case: a manifest citekey held here as
    `managed_by: user` is never fetched, and used to be reported nowhere."""
    def must_not_be_called(client, citekey, arxiv_id):
        raise AssertionError("fetch_paper must not be called for a managed_by: user paper")
    monkeypatch.setattr(reconcile, "fetch_paper", must_not_be_called)

    _write_existing(
        tmp_path / "My Local Copy.md", document_id="local0001",
        fields=_lit_fields("smirnov2011", "My Local Copy", managed_by="user"),
        body="Hand-edited.\n",
    )
    manifest = [ArxivPaper(citekey="smirnov2011", arxiv_id="1101.1764", title="RIME", authors="O. Smirnov", year="2011")]

    results = reconcile.sync_literature(tmp_path, manifest, delay=0.0)

    assert [(r.citekey, r.action) for r in results] == [("smirnov2011", "yours")]


# ---------------------------------------------------------------------------
# sync_literature: delete orphans
# ---------------------------------------------------------------------------


def test_sync_literature_deletes_a_boepie_managed_paper_dropped_from_the_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(reconcile, "fetch_paper", _fake_fetch_paper())

    orphan_path = tmp_path / "Orphaned.md"
    _write_existing(
        orphan_path, document_id="orphan001",
        fields=_lit_fields("gone2020", "Orphaned"),
    )

    results = reconcile.sync_literature(tmp_path, [], delay=0.0)

    assert results == [reconcile.LiteratureSyncResult(citekey="gone2020", action="deleted", id="orphan001")]
    assert not orphan_path.exists()


def test_sync_literature_deletes_the_wrapper_directory_of_an_orphaned_paper_with_assets(tmp_path, monkeypatch):
    monkeypatch.setattr(reconcile, "fetch_paper", _fake_fetch_paper())

    document = write_leaf_document(
        tmp_path / "With Figures.md", document_id="orphan002",
        frontmatter_fields=_lit_fields("gone2021", "With Figures"),
        body="Body.\n", assets={"fig1.png": b"data"},
    )

    reconcile.sync_literature(tmp_path, [], delay=0.0)

    assert not document.wrapper_dir.exists()


# ---------------------------------------------------------------------------
# resolve_force_paths / normalize_force_path
# ---------------------------------------------------------------------------


def test_normalize_force_path_returns_a_bare_path():
    assert reconcile.normalize_force_path("Some Title.md") == Path("Some Title.md")


def _literature_index(tmp_path):
    return collection_index(
        tmp_path, collection="literature", key_fields=KEY_FIELDS["literature"]
    )


def test_resolve_force_paths_maps_a_relative_path_to_its_natural_key(tmp_path):
    _write_existing(
        tmp_path / "Existing.md", document_id="existing01",
        fields=_lit_fields("smirnov2011", "Existing"),
    )

    resolved = reconcile.resolve_force_paths(
        ["Existing.md"], _literature_index(tmp_path), collection_dir=tmp_path,
    )

    assert resolved == frozenset({"smirnov2011"})


def test_resolve_force_paths_raises_for_a_nonexistent_target(tmp_path):
    with pytest.raises(ValueError, match="no such corpus document"):
        reconcile.resolve_force_paths(
            ["Nowhere.md"], _literature_index(tmp_path), collection_dir=tmp_path,
        )


def test_resolve_force_paths_raises_for_a_path_escaping_the_collection_dir(tmp_path):
    with pytest.raises(ValueError, match="escapes the collection directory"):
        reconcile.resolve_force_paths(
            ["../../etc/passwd"], _literature_index(tmp_path), collection_dir=tmp_path,
        )


def test_sync_literature_rejects_a_bad_force_target_before_any_fetch(tmp_path, monkeypatch):
    def must_not_be_called(client, citekey, arxiv_id):
        raise AssertionError("no fetch should happen once force-path validation has failed")
    monkeypatch.setattr(reconcile, "fetch_paper", must_not_be_called)

    with pytest.raises(ValueError, match="no such corpus document"):
        reconcile.sync_literature(tmp_path, [_paper()], force_paths=["Nowhere.md"], delay=0.0)


# ---------------------------------------------------------------------------
# sync_docs_project: add / skip / force / delete
# ---------------------------------------------------------------------------


def _stub_docs_discovery(
    monkeypatch,
    pages: list[PageContent],
    *,
    discovery: str = "sphinx",
    version: str | None = "1.0",
) -> list[str]:
    """Stub the docs fetch layer; returns the list it records requests into."""
    fetched: list[str] = []
    monkeypatch.setattr(reconcile, "probe_discovery_mode", lambda client, base_url, timeout: discovery)
    monkeypatch.setattr(reconcile, "fetch_version", lambda client, base_url, timeout: version)

    def fake_iter_project_pages(
        client, project, *, delay=0.2, timeout=30, max_pages=300, max_depth=5,
        have_page=None,
    ):
        # Honours `have_page` the way the real sphinx iterator does, so these
        # tests exercise the skip *before* the fetch rather than only the
        # bookkeeping after it. `fetched` records what was actually asked
        # for.
        for page in pages:
            if have_page is not None and have_page(page.docname):
                yield PageContent(
                    docname=page.docname, markdown=None,
                    page_url=page.page_url, not_fetched=True,
                )
                continue
            fetched.append(page.docname)
            yield page
    monkeypatch.setattr(reconcile, "iter_project_pages", fake_iter_project_pages)
    return fetched


def test_sync_docs_project_adds_new_pages(tmp_path, monkeypatch):
    _stub_docs_discovery(monkeypatch, [
        PageContent(docname="index", markdown="# Stimela Docs\n\nIntro.\n"),
        PageContent(docname="guide", markdown="# Guide\n\nHow to.\n"),
    ])
    project = DocsProject(project="stimela", base_url="https://stimela.readthedocs.io/en/latest/")

    result = reconcile.sync_docs_project(tmp_path, project, delay=0.0)

    assert (result.added, result.skipped, result.refetched, result.deleted) == (2, 0, 0, 0)
    assert (tmp_path / "stimela" / "Stimela Docs.md").exists()
    assert (tmp_path / "stimela" / "Guide.md").exists()
    frontmatter, _ = read_frontmatter((tmp_path / "stimela" / "Guide.md").read_text(encoding="utf-8"))
    assert frontmatter["docs"]["project"] == "stimela"
    assert frontmatter["docs"]["page"] == "guide"
    assert frontmatter["docs"]["base_url"] == "https://stimela.readthedocs.io/en/latest/"
    assert frontmatter["managed_by"] == "boepie"


def test_sync_docs_project_records_a_page_failure_without_writing_it(tmp_path, monkeypatch):
    _stub_docs_discovery(monkeypatch, [
        PageContent(docname="index", markdown="# Index\n"),
        PageContent(docname="broken", markdown=None, error="HTTP 500"),
    ])
    project = DocsProject(project="stimela", base_url="https://stimela.readthedocs.io/en/latest/")

    result = reconcile.sync_docs_project(tmp_path, project, delay=0.0)

    assert result.added == 1
    assert len(result.failures) == 1
    assert result.failures[0].docname == "broken"


def test_sync_docs_project_skips_an_existing_page_unless_forced(tmp_path, monkeypatch):
    _stub_docs_discovery(monkeypatch, [PageContent(docname="index", markdown="# New Index\n")])
    existing_path = tmp_path / "stimela" / "Existing Index.md"
    _write_existing(
        existing_path, document_id="existing01",
        fields=_docs_fields("stimela", "index", "Existing Index"),
    )
    project = DocsProject(project="stimela", base_url="https://stimela.readthedocs.io/en/latest/")

    result = reconcile.sync_docs_project(tmp_path, project, delay=0.0)

    assert (result.added, result.skipped, result.refetched) == (0, 1, 0)
    assert "Old body." in existing_path.read_text(encoding="utf-8")


def test_sync_docs_project_does_not_download_a_page_it_already_has(tmp_path, monkeypatch):
    """The skip has always been right about what to *write*; it decided after
    the page had been downloaded and converted, so a second `corpus sync`
    re-pulled all 98 pages and discarded every one. Measured on 2026-09-08:
    three pages, three requests on the second run.

    Nothing new records what was pulled - the documents already do, through
    `docs.project`/`docs.page`. The fix was to ask before fetching rather
    than after, which is the same defect the MinerU path had (hash first,
    convert second).
    """
    pages = [
        PageContent(docname="index", markdown="# Index\n\nIntro.\n"),
        PageContent(docname="guide", markdown="# Guide\n\nHow to.\n"),
    ]
    fetched = _stub_docs_discovery(monkeypatch, pages)
    project = DocsProject(project="stimela", base_url="https://stimela.readthedocs.io/en/latest/")

    first = reconcile.sync_docs_project(tmp_path, project, delay=0.0)
    assert (first.added, sorted(fetched)) == (2, ["guide", "index"])

    fetched.clear()
    second = reconcile.sync_docs_project(tmp_path, project, delay=0.0)

    assert second.skipped == 2
    assert second.added == 0
    assert fetched == []


def test_sync_docs_project_still_downloads_a_forced_page(tmp_path, monkeypatch):
    """--force is the way back to the old behaviour, for one page or all of
    them: it is the only thing that can pick up an upstream edit, since a
    page already on disk is otherwise never requested again."""
    pages = [
        PageContent(docname="index", markdown="# Index\n\nIntro.\n"),
        PageContent(docname="guide", markdown="# Guide\n\nHow to.\n"),
    ]
    fetched = _stub_docs_discovery(monkeypatch, pages)
    project = DocsProject(project="stimela", base_url="https://stimela.readthedocs.io/en/latest/")
    reconcile.sync_docs_project(tmp_path, project, delay=0.0)
    fetched.clear()

    result = reconcile.sync_docs_project(
        tmp_path, project, delay=0.0, force_paths=["stimela/Guide.md"]
    )

    assert result.refetched == 1
    assert result.skipped == 1
    assert fetched == ["guide"]


def test_sync_docs_project_force_refetches_in_place(tmp_path, monkeypatch):
    _stub_docs_discovery(monkeypatch, [PageContent(docname="index", markdown="# New Index\n")])
    existing_path = tmp_path / "stimela" / "Existing Index.md"
    _write_existing(
        existing_path, document_id="existing01",
        fields=_docs_fields("stimela", "index", "Existing Index"),
    )
    project = DocsProject(project="stimela", base_url="https://stimela.readthedocs.io/en/latest/")

    result = reconcile.sync_docs_project(
        tmp_path, project, force_paths=["stimela/Existing Index.md"], delay=0.0,
    )

    assert result.refetched == 1
    frontmatter, body = read_frontmatter(existing_path.read_text(encoding="utf-8"))
    assert frontmatter["id"] == "existing01"
    assert "New Index" in body


def test_sync_docs_project_never_touches_a_source_local_page(tmp_path, monkeypatch):
    _stub_docs_discovery(monkeypatch, [PageContent(docname="index", markdown="# New Index\n")])
    local_path = tmp_path / "stimela" / "Hand Written.md"
    _write_existing(
        local_path, document_id="local0001",
        fields=_docs_fields("stimela", "index", "Hand Written", managed_by="user"),
        body="Hand-edited.\n",
    )
    project = DocsProject(project="stimela", base_url="https://stimela.readthedocs.io/en/latest/")

    result = reconcile.sync_docs_project(tmp_path, project, delay=0.0)

    assert (result.added, result.skipped, result.refetched, result.deleted) == (0, 0, 0, 0)
    assert "Hand-edited." in local_path.read_text(encoding="utf-8")


def test_sync_docs_project_counts_the_pages_it_left_alone(tmp_path, monkeypatch):
    """Leaving a `managed_by: user` page alone is correct; doing it in
    silence is not. Uncounted, the page shows up nowhere: `fetch` reports
    nothing about it and `status` called its whole project "not fetched
    yet", advising a fetch that provably could not act."""
    _stub_docs_discovery(monkeypatch, [PageContent(docname="index", markdown="# New\n")])
    _write_existing(
        tmp_path / "stimela" / "Hand Written.md", document_id="local0001",
        fields=_docs_fields("stimela", "index", "Hand Written", managed_by="user"),
        body="Hand-edited.\n",
    )
    project = DocsProject(project="stimela", base_url="https://stimela.readthedocs.io/en/latest/")

    seen = []
    result = reconcile.sync_docs_project(
        tmp_path, project, delay=0.0, on_page=seen.append
    )

    assert result.yours == 1
    assert [page.action for page in seen] == ["yours"]
    assert seen[0].id == "local0001"


def test_sync_docs_project_counts_no_one_elses_pages_as_yours(tmp_path, monkeypatch):
    _stub_docs_discovery(monkeypatch, [PageContent(docname="index", markdown="# New\n")])
    project = DocsProject(project="stimela", base_url="https://stimela.readthedocs.io/en/latest/")

    result = reconcile.sync_docs_project(tmp_path, project, delay=0.0)

    assert (result.added, result.yours) == (1, 0)


def test_sync_docs_project_deletes_a_boepie_managed_page_no_longer_served(tmp_path, monkeypatch):
    _stub_docs_discovery(monkeypatch, [PageContent(docname="index", markdown="# Index\n")])
    orphan_path = tmp_path / "stimela" / "Removed Page.md"
    _write_existing(
        orphan_path, document_id="orphan001",
        fields=_docs_fields("stimela", "removed", "Removed Page"),
    )
    project = DocsProject(project="stimela", base_url="https://stimela.readthedocs.io/en/latest/")

    result = reconcile.sync_docs_project(tmp_path, project, delay=0.0)

    assert result.deleted == 1
    assert not orphan_path.exists()


# ---------------------------------------------------------------------------
# sync_docs: whole-project deletion on manifest removal
# ---------------------------------------------------------------------------


def test_sync_docs_deletes_every_boepie_page_of_a_project_removed_from_the_manifest(tmp_path, monkeypatch):
    _stub_docs_discovery(monkeypatch, [])
    boepie_page = tmp_path / "quartical" / "Old Page.md"
    _write_existing(
        boepie_page, document_id="boepie0001",
        fields=_docs_fields("quartical", "old", "Old Page"),
    )
    local_page = tmp_path / "quartical" / "My Notes.md"
    _write_existing(
        local_page, document_id="local00001",
        fields=_docs_fields("quartical", "notes", "My Notes", managed_by="user"),
        body="Keep me.\n",
    )

    # quartical is no longer in the manifest at all.
    results = reconcile.sync_docs(tmp_path, [], delay=0.0)

    assert not boepie_page.exists()
    assert local_page.exists()
    assert any(result.project == "quartical" and result.deleted == 1 for result in results)


def test_sync_docs_converges_every_manifest_project_and_reports_one_result_each(tmp_path, monkeypatch):
    monkeypatch.setattr(reconcile, "probe_discovery_mode", lambda client, base_url, timeout: "sphinx")
    monkeypatch.setattr(reconcile, "fetch_version", lambda client, base_url, timeout: "1.0")

    def fake_iter_project_pages(
        client, project, *, delay=0.2, timeout=30, max_pages=300, max_depth=5,
        have_page=None,
    ):
        yield PageContent(docname="index", markdown=f"# {project.project} index\n")
    monkeypatch.setattr(reconcile, "iter_project_pages", fake_iter_project_pages)

    projects = [
        DocsProject(project="stimela", base_url="https://stimela.readthedocs.io/en/latest/"),
        DocsProject(project="wsclean", base_url="https://wsclean.readthedocs.io/en/latest/"),
    ]

    results = reconcile.sync_docs(tmp_path, projects, delay=0.0)

    assert {result.project for result in results} == {"stimela", "wsclean"}
    assert all(result.added == 1 for result in results)


def test_sync_docs_walks_the_collection_once_regardless_of_project_count(tmp_path, monkeypatch):
    """Regression test: sync_docs_project used to call collection_index
    twice per project (once inside resolve_force_paths, once directly),
    even with no --force, so an N-project sync walked and re-parsed
    frontmatter for the whole collection 2N times. sync_docs must compute
    the index once for the whole batch and thread it through."""
    monkeypatch.setattr(reconcile, "probe_discovery_mode", lambda client, base_url, timeout: "sphinx")
    monkeypatch.setattr(reconcile, "fetch_version", lambda client, base_url, timeout: "1.0")

    def fake_iter_project_pages(
        client, project, *, delay=0.2, timeout=30, max_pages=300, max_depth=5,
        have_page=None,
    ):
        yield PageContent(docname="index", markdown=f"# {project.project} index\n")
    monkeypatch.setattr(reconcile, "iter_project_pages", fake_iter_project_pages)

    call_count = 0
    real_collection_index = reconcile.collection_index

    def counting_collection_index(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return real_collection_index(*args, **kwargs)
    monkeypatch.setattr(reconcile, "collection_index", counting_collection_index)

    projects = [
        DocsProject(project="stimela", base_url="https://stimela.readthedocs.io/en/latest/"),
        DocsProject(project="wsclean", base_url="https://wsclean.readthedocs.io/en/latest/"),
        DocsProject(project="quartical", base_url="https://quartical.readthedocs.io/en/latest/"),
    ]

    reconcile.sync_docs(tmp_path, projects, delay=0.0)

    assert call_count == 1


def test_sync_literature_walks_the_collection_once_per_call(tmp_path, monkeypatch):
    """Regression test: sync_literature had the same doubled-collection_index
    call as sync_docs_project (once inside resolve_force_paths, once
    directly)."""
    monkeypatch.setattr(reconcile, "fetch_paper", _fake_fetch_paper())

    call_count = 0
    real_collection_index = reconcile.collection_index

    def counting_collection_index(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return real_collection_index(*args, **kwargs)
    monkeypatch.setattr(reconcile, "collection_index", counting_collection_index)

    reconcile.sync_literature(tmp_path, [_paper()], delay=0.0)

    assert call_count == 1


# ---------------------------------------------------------------------------
# The fetch/read boundary
# ---------------------------------------------------------------------------
#
# What `corpus sync` writes has to satisfy the same schema every other reader
# of the corpus assumes. Nothing used to check that: `reconcile` wrote a flat
# `citekey`/`project` frontmatter and diffed it with flat key fields, so it
# agreed with itself while `corpus status`/`list`/`tree` (which key on
# `schema.KEY_FIELDS`) raised KeyError on every document it had written. These
# two tests are the crossing the old suite was missing.


def test_sync_literature_writes_frontmatter_the_schema_accepts(tmp_path, monkeypatch):
    monkeypatch.setattr(reconcile, "fetch_paper", _fake_fetch_paper())

    reconcile.sync_literature(tmp_path, [_paper()], delay=0.0)

    md_path = next(tmp_path.glob("*.md"))
    frontmatter, _ = read_frontmatter(md_path.read_text(encoding="utf-8"))
    document = parse_frontmatter("literature", frontmatter)
    assert document.bib.citekey == "smirnov2011"
    assert document.managed_by == "boepie"
    assert document.source.origin == "arxiv:1101.1185"

    # The lookup every other corpus reader performs.
    indexed = collection_index(
        tmp_path, collection="literature", key_fields=KEY_FIELDS["literature"]
    )
    assert [item.natural_key for item in indexed] == ["smirnov2011"]


def test_sync_docs_writes_frontmatter_the_schema_accepts(tmp_path, monkeypatch):
    _stub_docs_discovery(monkeypatch, [PageContent(docname="guide", markdown="# Guide\n")])
    project = DocsProject(project="stimela", base_url="https://stimela.readthedocs.io/en/latest/")

    reconcile.sync_docs_project(tmp_path, project, delay=0.0)

    md_path = next((tmp_path / "stimela").glob("*.md"))
    frontmatter, _ = read_frontmatter(md_path.read_text(encoding="utf-8"))
    document = parse_frontmatter("docs", frontmatter)
    assert (document.docs.project, document.docs.page) == ("stimela", "guide")

    indexed = collection_index(
        tmp_path, collection="docs", key_fields=KEY_FIELDS["docs"]
    )
    assert [item.natural_key for item in indexed] == ["stimela/guide"]
    # search_docs filters on this dotted path; a flat `project` matched nothing.
    assert lookup_path(frontmatter, "docs.project") == "stimela"
