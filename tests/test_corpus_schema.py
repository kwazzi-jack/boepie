"""Tests for boepie.corpus.schema: the one frontmatter declaration every
collection extends, and its round-trip through the YAML codec."""

from __future__ import annotations

import pytest
import yaml
from pydantic import ValidationError

from boepie.context.frontmatter import read_frontmatter, write_frontmatter
from boepie.corpus.schema import (
    KEY_FIELDS,
    Bibliography,
    DocsDocument,
    DocsPage,
    LiteratureDocument,
    NoteDocument,
    Source,
    dump_frontmatter,
    parse_frontmatter,
)


def _source(**overrides) -> Source:
    fields = {"from": "arxiv:1101.1185", "via": "ar5iv", "format": "html"}
    fields.update(overrides)
    return Source.model_validate(fields)


def test_from_is_exposed_under_its_yaml_name():
    """`from` is a Python keyword, so the field is `origin` in code and
    `from` on disk."""
    source = _source()

    assert source.origin == "arxiv:1101.1185"
    assert dump_frontmatter(
        NoteDocument(id="a", title="t", managed_by="user", source=source)
    )["source"]["from"] == "arxiv:1101.1185"


def test_a_note_adds_nothing_to_the_base_schema():
    """Notes are the base case: no natural key, no namespaced block."""
    note = NoteDocument(id="a", title="t", managed_by="user", source=_source())

    assert set(dump_frontmatter(note)) == {"id", "title", "managed_by", "source"}
    assert KEY_FIELDS["notes"] == ()


def test_literature_and_docs_extend_the_same_base():
    assert issubclass(LiteratureDocument, NoteDocument.__mro__[1])
    assert issubclass(DocsDocument, NoteDocument.__mro__[1])


def test_round_trips_through_the_yaml_frontmatter_codec():
    """`context.frontmatter` needed no changes to carry nesting: safe_load
    and safe_dump handle nested mappings natively."""
    document = LiteratureDocument(
        id="aB3dE9fGhI",
        title="Revisiting the RIME",
        managed_by="boepie",
        source=_source(sha256="0" * 64),
        bib=Bibliography(citekey="smirnov2011", year="2011", arxiv_id="1101.1185"),
    )

    text = write_frontmatter(dump_frontmatter(document), "Body.\n")
    frontmatter, body = read_frontmatter(text)

    assert body == "Body.\n"
    assert parse_frontmatter("literature", frontmatter) == document


def test_dump_omits_unset_optionals_rather_than_writing_nulls():
    """A note ingested from a local file has no `original`; saying so by
    omission is smaller and more honest than writing a null."""
    note = NoteDocument(id="a", title="t", managed_by="user", source=_source())

    dumped = dump_frontmatter(note)

    assert "original" not in dumped["source"]
    assert "sha256" not in dumped["source"]


def test_the_timestamp_serialises_as_a_string_yaml_can_write():
    note = NoteDocument(id="a", title="t", managed_by="user", source=_source())

    dumped = dump_frontmatter(note)

    assert isinstance(dumped["source"]["at"], str)
    # safe_dump refuses objects it has no representer for, so this is the
    # assertion that actually protects the write path.
    assert yaml.safe_dump(dumped)


def test_an_unknown_managed_by_value_is_rejected():
    with pytest.raises(ValidationError):
        NoteDocument(id="a", title="t", managed_by="human", source=_source())


def test_an_unknown_conversion_via_is_rejected():
    with pytest.raises(ValidationError):
        _source(via="telepathy")


def test_a_literature_document_without_a_citekey_is_rejected():
    with pytest.raises(ValidationError):
        parse_frontmatter(
            "literature",
            {
                "id": "a", "title": "t", "managed_by": "user",
                "source": {"from": "x", "via": "verbatim", "format": "markdown"},
                "bib": {},
            },
        )


def test_key_fields_are_dotted_paths_into_the_namespaced_blocks():
    assert KEY_FIELDS["literature"] == ("bib.citekey",)
    assert KEY_FIELDS["docs"] == ("docs.project", "docs.page")


def test_docs_crawl_scope_is_optional():
    """A page added from a local file has no crawl scope, and should not be
    made to invent one."""
    page = DocsPage(project="stimela", page="index")

    assert page.crawl is None
