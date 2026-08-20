"""Tests for the pipeline MCP tools: validate_recipe, run_recipe.

stimela_run is monkeypatched throughout - these are unit tests of the tools'
formatting and input handling, not of stimela itself. Assertions target the
F5 sentinel vocabulary and the SCHEMA-OK disclosure required by
design/interface-spec.md section 3, not exact prose.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from boepie.pipeline.runner import RunResult
from boepie.pipeline import recipe as pipeline
from boepie.pipeline.recipe import (
    RunRecipeInput,
    ValidateRecipeInput,
    run_recipe,
    validate_recipe,
)


def _ok_result(stdout: str = "dry run was requested, exiting\n") -> RunResult:
    return RunResult(command=["stimela"], stdout=stdout, stderr="", returncode=0)


def _failed_result(stderr: str = "unknown cab 'bogus'\n") -> RunResult:
    return RunResult(command=["stimela"], stdout="", stderr=stderr, returncode=1)


@pytest.fixture
def recipe_file(tmp_path: Path) -> Path:
    path = tmp_path / "image.yml"
    path.write_text("test_recipe:\n  info: throwaway\n")
    return path


# ---------------------------------------------------------------------------
# ValidateRecipeInput: mutually-exclusive source validator
# ---------------------------------------------------------------------------


def test_validate_recipe_input_rejects_neither_source():
    with pytest.raises(ValidationError, match="exactly one"):
        ValidateRecipeInput()


def test_validate_recipe_input_rejects_both_sources():
    with pytest.raises(ValidationError, match="exactly one"):
        ValidateRecipeInput(recipe_path="image.yml", yaml_content="a: b")


def test_validate_recipe_input_accepts_recipe_path_only():
    ValidateRecipeInput(recipe_path="image.yml")


def test_validate_recipe_input_accepts_yaml_content_only():
    ValidateRecipeInput(yaml_content="a: b")


# ---------------------------------------------------------------------------
# RunRecipeInput: backend is a closed choice
# ---------------------------------------------------------------------------


def test_run_recipe_input_rejects_unknown_backend():
    with pytest.raises(ValidationError):
        RunRecipeInput(recipe_path="image.yml", backend="docker")


def test_run_recipe_input_accepts_known_backends():
    for backend in ("native", "singularity", "kube"):
        RunRecipeInput(recipe_path="image.yml", backend=backend)


# ---------------------------------------------------------------------------
# validate_recipe: sentinel vocabulary and the "not checked" disclosure
# ---------------------------------------------------------------------------


def test_validate_recipe_success_is_schema_ok_not_valid(
    monkeypatch: pytest.MonkeyPatch, recipe_file: Path
):
    monkeypatch.setattr(pipeline, "stimela_run", lambda **_: _ok_result())
    monkeypatch.setattr(pipeline, "find_recipe_logs", lambda _: [])

    result = validate_recipe(ValidateRecipeInput(recipe_path=str(recipe_file)))

    lines = result.splitlines()
    assert lines[0] == "SCHEMA-OK"
    assert "VALID" not in result


def test_validate_recipe_success_discloses_what_was_not_checked(
    monkeypatch: pytest.MonkeyPatch, recipe_file: Path
):
    monkeypatch.setattr(pipeline, "stimela_run", lambda **_: _ok_result())
    monkeypatch.setattr(pipeline, "find_recipe_logs", lambda _: [])

    result = validate_recipe(ValidateRecipeInput(recipe_path=str(recipe_file)))

    assert "checked: cab names, parameter names, required parameters" in result
    assert (
        "not checked: input file existence, {substitution} resolvability, "
        "container availability" in result
    )


def test_validate_recipe_failure_is_failed_not_invalid(
    monkeypatch: pytest.MonkeyPatch, recipe_file: Path
):
    monkeypatch.setattr(pipeline, "stimela_run", lambda **_: _failed_result())
    monkeypatch.setattr(pipeline, "find_recipe_logs", lambda _: [])

    result = validate_recipe(ValidateRecipeInput(recipe_path=str(recipe_file)))

    assert result.startswith("FAILED")
    assert "INVALID" not in result


def test_validate_recipe_recipe_path_kept_relative_as_given(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, recipe_file: Path
):
    monkeypatch.setattr(pipeline, "stimela_run", lambda **_: _ok_result())
    monkeypatch.setattr(pipeline, "find_recipe_logs", lambda _: [])
    monkeypatch.chdir(tmp_path)

    result = validate_recipe(ValidateRecipeInput(recipe_path="image.yml"))

    assert "recipe: image.yml" in result
    assert str(tmp_path) not in result


def test_validate_recipe_missing_file_is_one_line_error():
    result = validate_recipe(ValidateRecipeInput(recipe_path="/no/such/recipe.yml"))
    assert result.startswith("Error:")
    assert "\n" not in result


def test_validate_recipe_yaml_content_mode_success(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(pipeline, "stimela_run", lambda **_: _ok_result())

    result = validate_recipe(ValidateRecipeInput(yaml_content="test_recipe:\n  info: x\n"))

    assert result.splitlines()[0] == "SCHEMA-OK"
    assert "not checked:" in result


# ---------------------------------------------------------------------------
# run_recipe: sentinel vocabulary
# ---------------------------------------------------------------------------


def test_run_recipe_success_is_ok_not_success(
    monkeypatch: pytest.MonkeyPatch, recipe_file: Path
):
    monkeypatch.setattr(pipeline, "stimela_run", lambda **_: _ok_result("done\n"))
    monkeypatch.setattr(pipeline, "find_recipe_logs", lambda _: [])

    result = run_recipe(RunRecipeInput(recipe_path=str(recipe_file)))

    assert result.splitlines()[0] == "OK"
    assert "SUCCESS" not in result


def test_run_recipe_failure_is_failed(monkeypatch: pytest.MonkeyPatch, recipe_file: Path):
    monkeypatch.setattr(pipeline, "stimela_run", lambda **_: _failed_result())
    monkeypatch.setattr(pipeline, "find_recipe_logs", lambda _: [])

    result = run_recipe(RunRecipeInput(recipe_path=str(recipe_file)))

    assert result.startswith("FAILED")


def test_run_recipe_missing_file_is_one_line_error():
    result = run_recipe(RunRecipeInput(recipe_path="/no/such/recipe.yml"))
    assert result.startswith("Error:")
    assert "\n" not in result


def test_run_recipe_reports_log_paths(monkeypatch: pytest.MonkeyPatch, recipe_file: Path):
    log_path = recipe_file.parent / "log-test.txt"
    monkeypatch.setattr(pipeline, "stimela_run", lambda **_: _ok_result())
    monkeypatch.setattr(pipeline, "find_recipe_logs", lambda _: [log_path])

    result = run_recipe(RunRecipeInput(recipe_path=str(recipe_file)))

    assert str(log_path) in result
