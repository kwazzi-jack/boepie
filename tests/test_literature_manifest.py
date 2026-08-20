"""Tests for boepie.literature.manifest: the packaged manifest `corpus fetch`
reconciles against, and the citekey derivation `corpus add literature` uses."""

from __future__ import annotations

from boepie.literature.manifest import (
    derive_citekey,
    load_default_manifest,
    unique_citekey,
)


def test_load_default_manifest_has_the_verified_arxiv_papers():
    papers = load_default_manifest()

    assert len(papers) >= 15
    citekeys = {paper.citekey for paper in papers}
    assert "smirnovRevisitingRadioInterferometer2011" in citekeys
    for paper in papers:
        assert paper.arxiv_id
        assert paper.title
        assert paper.year.isdigit()


def test_derive_citekey_matches_the_corpus_convention():
    citekey = derive_citekey(
        "Smirnov, O. M.", "2011",
        "Revisiting the radio interferometer measurement equation. I. A full-sky Jones formalism",
    )

    assert citekey == "smirnovRevisitingRadio2011"


def test_derive_citekey_uses_first_author_only():
    citekey = derive_citekey("Tasse, C. and Hugo, B.", "2018", "Faceting for direction-dependent spectral deconvolution")

    assert citekey.startswith("tasse")


def test_unique_citekey_disambiguates_on_collision():
    existing = {"smirnov2011", "smirnov2011a"}

    assert unique_citekey("smirnov2011", existing) == "smirnov2011b"


def test_unique_citekey_passes_through_when_no_collision():
    assert unique_citekey("freshcitekey2020", {"other2020"}) == "freshcitekey2020"
