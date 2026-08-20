"""Tests for boepie.corpus.ids: the surrogate document id generator."""

from __future__ import annotations

import pytest

from boepie.corpus.ids import _ALPHABET, generate_id, unique_id


def test_generate_id_default_length_and_alphabet():
    identifier = generate_id()
    assert len(identifier) == 10
    assert all(character in _ALPHABET for character in identifier)


def test_generate_id_honours_a_custom_length():
    assert len(generate_id(length=6)) == 6


def test_generate_id_is_not_deterministic():
    # Astronomically unlikely to collide by chance at length 10; a failure
    # here almost certainly means secrets.choice stopped being random.
    assert generate_id() != generate_id()


def test_unique_id_returns_a_fresh_id_when_none_taken():
    identifier = unique_id(set())
    assert len(identifier) == 10


def test_unique_id_avoids_existing_ids(monkeypatch):
    calls = iter(["aaaaaaaaaa", "aaaaaaaaaa", "bbbbbbbbbb"])
    monkeypatch.setattr("boepie.corpus.ids.generate_id", lambda length=10: next(calls))

    identifier = unique_id({"aaaaaaaaaa"})

    assert identifier == "bbbbbbbbbb"


def test_unique_id_raises_after_exhausting_attempts(monkeypatch):
    monkeypatch.setattr("boepie.corpus.ids.generate_id", lambda length=10: "aaaaaaaaaa")

    with pytest.raises(ValueError, match="could not generate a unique"):
        unique_id({"aaaaaaaaaa"})
