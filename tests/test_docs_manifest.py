"""Tests for boepie.docs.manifest: the packaged manifest `corpus fetch`
reconciles against."""

from __future__ import annotations

from boepie.docs.manifest import DocsProject, load_default_manifest


def test_load_default_manifest_has_the_curated_projects():
    projects = load_default_manifest()

    names = {project.project for project in projects}
    assert names == {"stimela", "quartical", "wsclean"}
    for project in projects:
        assert project.project
        assert project.base_url
        assert project.base_url.startswith("https://")


def test_to_dict_omits_or_nulls_discovery_and_path_prefix_when_unset():
    project = DocsProject(project="mycab", base_url="https://example.org/docs/")

    as_dict = project.to_dict()

    assert as_dict.get("discovery") is None
    assert as_dict.get("path_prefix") is None
