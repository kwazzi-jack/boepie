"""End-to-end tests for the search_knowledge MCP tool.

Builds a tiny lexical-only (BM25, embedding=None) fixture bundle on disk -
the same offline pattern as test_engine.py's "Lexical-only" section - and
exercises search_knowledge() directly as a plain async function.

Assertions target the F2 output structure and payload bounds from
design/interface-spec.md, not the exact prose, which is expected to churn.
search_knowledge has no read_knowledge counterpart: its hits are file
locations the agent opens with its own tools, so blocks carry a source line
and no read handle.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from boepie.config import BUNDLE_DIR_ENV_VAR
from boepie.knowledge import index_root_for
from boepie.rag import engine
from boepie.rag.loaders import KnowledgeLoader
from boepie.tools._retrieval import SHORT_SNIPPET_CHARS
from boepie.tools.knowledge import SearchKnowledgeInput, search_knowledge

# Every page mentions "stimela" so a "stimela" query matches all of them -
# lets the top_k tests assert an exact hit count. Each body section is kept
# well past the snippet cutoff so truncation is exercised too.
_FRONTMATTER = "---\ntype: Concept\ntitle: {title}\nmanaged: boepie\n---\n\n"

_FIXTURE_PAGES: dict[str, tuple[str, str]] = {
    "concepts/substitution": (
        "Substitution and evaluation",
        "# Substitution\n\n"
        "Stimela recipes substitute `{recipe.var}` placeholders at runtime, "
        "pulling values from earlier step outputs or command-line overrides. "
        "Use this to wire one cab's output straight into the next cab's input "
        "without hand-copying values between steps, which keeps a long "
        "pipeline honest when an early parameter changes.\n",
    ),
    "concepts/backends": (
        "Backends",
        "# Backends\n\n"
        "A stimela recipe runs each cab through a backend - singularity, "
        "docker, or native - chosen independently of the recipe YAML itself. "
        "Switching backends never changes a cab's parameter names or "
        "defaults, only how the container is launched and where its "
        "filesystem is mounted from.\n",
    ),
    "playbooks/imaging": (
        "Imaging strategy",
        "# Imaging strategy\n\n"
        "A typical stimela imaging pipeline chains wsclean multiscale clean "
        "with successive masking rounds, tightening the auto-threshold each "
        "pass to recover faint extended emission without the deconvolution "
        "diverging on residual sidelobes near bright sources.\n",
    ),
    "cabs/wsclean": (
        "wsclean imager",
        "# wsclean cab notes\n\n"
        "wsclean is the stimela cab for wide-field imaging with w-stacking "
        "and multiscale deconvolution. Gotcha: channel-out counts must "
        "divide evenly into the selected spectral window or the run fails "
        "partway through.\n",
    ),
}


def _write_fixture_bundle(bundle_dir: Path, pages: dict[str, tuple[str, str]] | None = None) -> None:
    for document_id, (title, body) in (pages or _FIXTURE_PAGES).items():
        page_path = bundle_dir / f"{document_id}.md"
        page_path.parent.mkdir(parents=True, exist_ok=True)
        page_path.write_text(_FRONTMATTER.format(title=title) + body, encoding="utf-8")
    # The manifest is what `find_bundle` looks for, so a fixture bundle needs
    # one to be discoverable at all.
    (bundle_dir / "manifest.json").write_text(json.dumps({"bundle_version": "0.1.0"}), encoding="utf-8")


async def _build_fixture_index(bundle_dir: Path) -> None:
    """Build the bundle's own index, exactly where `init`/`apply` put it."""
    await engine.build(
        KnowledgeLoader(bundle_dir), index_root=index_root_for(bundle_dir), embedding=None
    )


def _hit_lines(payload: str) -> list[str]:
    return [line for line in payload.splitlines() if re.match(r"^\[\d+\] ", line)]


def _source_lines(payload: str) -> list[str]:
    return [line.strip() for line in payload.splitlines() if line.strip().startswith("source:")]


@pytest.fixture
async def knowledge_index(tmp_path, monkeypatch):
    """Build a fixture bundle with its own index and run from inside it.

    search_knowledge() discovers the bundle by walking up from the cwd, so
    the cwd is what points it at this fixture rather than a real project.
    """
    # Named `.boepie` so source-path relativisation has its anchor, exactly as
    # in a real project directory.
    project_dir = tmp_path / "project"
    bundle_dir = project_dir / ".boepie"
    _write_fixture_bundle(bundle_dir)
    await _build_fixture_index(bundle_dir)

    monkeypatch.chdir(project_dir)

    yield bundle_dir
    await engine.clear_cache()


# ---------------------------------------------------------------------------
# F2: ranked hits
# ---------------------------------------------------------------------------


async def test_search_knowledge_emits_f2_count_line_and_numbered_blocks(knowledge_index):
    result = await search_knowledge(SearchKnowledgeInput(question="wsclean multiscale imaging"))
    first = result.splitlines()[0]
    assert re.match(r'^\d+ hits for "wsclean multiscale imaging" in knowledge$', first)
    hits = _hit_lines(result)
    assert hits
    assert int(first.split()[0]) == len(hits)


async def test_search_knowledge_hit_line_carries_title_and_section(knowledge_index):
    result = await search_knowledge(SearchKnowledgeInput(question="wsclean multiscale imaging"))
    hit = _hit_lines(result)[0]
    assert "#" in hit  # section anchor appended to the title


async def test_search_knowledge_hit_line_carries_no_numeric_score(knowledge_index):
    # The score is not for the agent: rank order (the "[1] [2] [3]" numbering)
    # is the relevance signal. Raw scores are uninterpretable on a small model
    # and invite miscomparison across collections - see design/phase-2.md item 3.
    result = await search_knowledge(SearchKnowledgeInput(question="wsclean multiscale imaging"))
    hit = _hit_lines(result)[0]
    assert "score=" not in hit
    assert not re.search(r"\bbm25=|\bcos=|\brrf=", hit)


async def test_search_knowledge_source_line_is_an_openable_bundle_path(knowledge_index):
    result = await search_knowledge(SearchKnowledgeInput(question="wsclean multiscale imaging"))
    sources = _source_lines(result)
    assert sources
    for line in sources:
        path = line.removeprefix("source:").split("(")[0].strip()
        assert path.startswith(".boepie/") and path.endswith(".md")
        assert (knowledge_index.parent / path).exists()
    assert str(knowledge_index) not in result  # no absolute path


async def test_search_knowledge_never_returns_index_md(tmp_path, monkeypatch):
    """index.md is the navigation entry point the agent reads first, not a
    search answer: it must be excluded from the corpus even when a query matches
    its text verbatim."""
    project_dir = tmp_path / "project"
    bundle_dir = project_dir / ".boepie"
    _write_fixture_bundle(bundle_dir)
    # A unique token that lives only in index.md - if the page were indexed this
    # query would hit it and nothing else.
    (bundle_dir / "index.md").write_text(
        "# Bundle index\n\nEscalation ladder zzquux routing table.\n", encoding="utf-8"
    )
    await _build_fixture_index(bundle_dir)
    monkeypatch.chdir(project_dir)
    try:
        result = await search_knowledge(SearchKnowledgeInput(question="zzquux routing"))
        # index.md is never a hit, even though its text matches the query.
        assert not [line for line in _source_lines(result) if "index.md" in line]
        # Its unique token never surfaces in a hit body either. The count line
        # echoes the raw question, so check only the blocks below it.
        assert "zzquux" not in "\n".join(result.splitlines()[1:])
    finally:
        await engine.clear_cache()


async def test_search_knowledge_emits_no_read_handle(knowledge_index):
    # There is no read_knowledge tool - the source line is the handle.
    result = await search_knowledge(SearchKnowledgeInput(question="stimela"))
    assert "read:" not in result


async def test_search_knowledge_emits_no_banned_markdown_or_footer(knowledge_index):
    result = await search_knowledge(SearchKnowledgeInput(question="stimela"))
    generated = [line for line in result.splitlines() if not line.startswith("    ")]
    assert not [line for line in generated if line.startswith(("#", "_"))]
    assert "---" not in result
    # The escalation ladder is taught once by the server instructions block;
    # repeating it per call is what the footer rule exists to prevent.
    assert "search_literature" not in result
    assert "search_docs" not in result


async def test_search_knowledge_top_k_respected(knowledge_index):
    result_one = await search_knowledge(SearchKnowledgeInput(question="stimela", top_k=1))
    assert len(_hit_lines(result_one)) == 1

    result_many = await search_knowledge(SearchKnowledgeInput(question="stimela", top_k=4))
    assert len(_hit_lines(result_many)) == 4


async def test_search_knowledge_top_k_upper_bound_is_twenty(knowledge_index):
    # Aligned with search_literature / search_docs; the old cap was 10.
    assert SearchKnowledgeInput(question="x", top_k=20).top_k == 20
    with pytest.raises(ValueError):
        SearchKnowledgeInput(question="x", top_k=21)


# ---------------------------------------------------------------------------
# Section 4: the snippet policy
# ---------------------------------------------------------------------------


async def test_search_knowledge_snippet_short_is_the_default_and_is_bounded(knowledge_index):
    default = await search_knowledge(SearchKnowledgeInput(question="wsclean multiscale imaging"))
    explicit = await search_knowledge(
        SearchKnowledgeInput(question="wsclean multiscale imaging", snippet="short")
    )
    assert default == explicit
    assert "the run fails partway through" not in default  # past the cutoff
    body_lines = [
        line
        for line in default.splitlines()
        if line.startswith("    ") and not line.lstrip().startswith("source:")
    ]
    assert body_lines
    for line in body_lines:
        assert len(line.strip()) <= SHORT_SNIPPET_CHARS + len(" ...")


async def test_search_knowledge_snippet_none_drops_bodies_and_shrinks_payload(knowledge_index):
    none_payload = await search_knowledge(
        SearchKnowledgeInput(question="wsclean multiscale imaging", snippet="none")
    )
    short_payload = await search_knowledge(
        SearchKnowledgeInput(question="wsclean multiscale imaging", snippet="short")
    )
    assert len(none_payload) < len(short_payload)
    assert _source_lines(none_payload)  # locations survive


async def test_search_knowledge_snippet_full_returns_whole_chunk(knowledge_index):
    full_payload = await search_knowledge(
        SearchKnowledgeInput(question="wsclean multiscale imaging", snippet="full")
    )
    assert "the run fails partway through" in full_payload


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


async def test_search_knowledge_without_any_bundle_says_init(tmp_path, monkeypatch):
    """No `.boepie/` above the cwd: the fix is to create one."""
    empty_dir = tmp_path / "not-a-project"
    empty_dir.mkdir()
    monkeypatch.chdir(empty_dir)
    await engine.clear_cache()
    try:
        result = await search_knowledge(SearchKnowledgeInput(question="anything"))
        assert result.startswith("Error:")
        assert len(result.splitlines()) == 1
        # The fix clause must name a command that really exists.
        assert "boepie knowledge init" in result
        assert "boepie knowledge apply" not in result
    finally:
        await engine.clear_cache()


async def test_search_knowledge_with_bundle_but_no_index_says_apply(tmp_path, monkeypatch):
    """A bundle that was never indexed is a different failure with a
    different fix: the bundle exists, only derived state is missing."""
    project_dir = tmp_path / "project"
    bundle_dir = project_dir / ".boepie"
    _write_fixture_bundle(bundle_dir)  # no index built
    monkeypatch.chdir(project_dir)
    await engine.clear_cache()
    try:
        result = await search_knowledge(SearchKnowledgeInput(question="anything"))
        assert result.startswith("Error:")
        assert len(result.splitlines()) == 1
        assert "boepie knowledge apply" in result
    finally:
        await engine.clear_cache()


# ---------------------------------------------------------------------------
# Bundle discovery
# ---------------------------------------------------------------------------


async def test_search_knowledge_finds_the_bundle_from_a_nested_subdirectory(
    knowledge_index, monkeypatch
):
    """An agent working in a subdirectory still gets the project's bundle."""
    nested_dir = knowledge_index.parent / "recipes" / "imaging"
    nested_dir.mkdir(parents=True)
    monkeypatch.chdir(nested_dir)
    await engine.clear_cache()

    result = await search_knowledge(SearchKnowledgeInput(question="wsclean multiscale imaging"))
    assert not result.startswith("Error:")
    assert _hit_lines(result)


async def test_bundle_dir_env_override_wins_over_the_walk(tmp_path, knowledge_index, monkeypatch):
    """BOEPIE_BUNDLE_DIR is the escape hatch for a server launched elsewhere:
    it must beat whatever the cwd would have found."""
    other_project = tmp_path / "other"
    other_bundle = other_project / ".boepie"
    _write_fixture_bundle(
        other_bundle,
        {"concepts/only-here": ("Only here", "# Only here\n\nA page about quasar absorption.\n")},
    )
    await _build_fixture_index(other_bundle)

    # cwd is the fixture project (which has no such page), but the override
    # points at the other bundle.
    monkeypatch.setenv(BUNDLE_DIR_ENV_VAR, str(other_bundle))
    await engine.clear_cache()

    result = await search_knowledge(SearchKnowledgeInput(question="quasar"))
    assert "only-here" in result
    assert "wsclean" not in result


async def test_two_projects_do_not_see_each_others_pages(tmp_path, monkeypatch):
    """The regression this whole change exists for: with a machine-global
    index root, initialising project B clobbered project A's index and A's
    searches started returning B's chunks under A-relative paths."""
    project_a = tmp_path / "a"
    project_b = tmp_path / "b"
    bundle_a = project_a / ".boepie"
    bundle_b = project_b / ".boepie"
    _write_fixture_bundle(
        bundle_a,
        {"concepts/alpha": ("Alpha", "# Alpha\n\nProject alpha documents pulsar timing.\n")},
    )
    _write_fixture_bundle(
        bundle_b,
        {"concepts/beta": ("Beta", "# Beta\n\nProject beta documents quasar absorption.\n")},
    )

    await _build_fixture_index(bundle_a)
    await _build_fixture_index(bundle_b)  # would previously have wiped A's index

    assert (index_root_for(bundle_a) / "knowledge" / "bm25" / "manifest.json").exists()
    assert (index_root_for(bundle_b) / "knowledge" / "bm25" / "manifest.json").exists()

    try:
        monkeypatch.chdir(project_a)
        await engine.clear_cache()
        result_a = await search_knowledge(SearchKnowledgeInput(question="pulsar quasar"))
        assert "alpha" in result_a
        assert "beta" not in result_a

        monkeypatch.chdir(project_b)
        await engine.clear_cache()
        result_b = await search_knowledge(SearchKnowledgeInput(question="pulsar quasar"))
        assert "beta" in result_b
        assert "alpha" not in result_b
    finally:
        await engine.clear_cache()
