"""Tests for the `boepie config` CLI commands (create, path, show, get, set).

Isolates BOEPIE_CONFIG_DIR to a tmp_path, mirroring how tests/test_settings.py
isolates the same env var - never touches a developer's real config file - and
clears the schema's own BOEPIE_* variables so a developer's environment cannot
change a result.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from boepie import cli, settings


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture(autouse=True)
def _isolated_config_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    config_dir = tmp_path / "config"
    monkeypatch.setenv(settings.CONFIG_DIR_ENV_VAR, str(config_dir))
    for key in settings.known_keys():
        monkeypatch.delenv(settings.env_var_for(key), raising=False)
    return config_dir


# ---------------------------------------------------------------------------
# config create
# ---------------------------------------------------------------------------


def test_config_create_writes_a_full_default_file(
    runner: CliRunner, _isolated_config_dir: Path
) -> None:
    result = runner.invoke(cli.cli, ["config", "create"])

    assert result.exit_code == 0, result.output
    config_file = _isolated_config_dir / "config.toml"
    assert config_file.is_file()
    raw = config_file.read_text(encoding="utf-8")
    assert "[embedding]" in raw
    assert 'binding = "fastembed"' in raw


def test_config_create_refuses_an_existing_file(runner: CliRunner) -> None:
    runner.invoke(cli.cli, ["config", "create"])
    settings.set_value("embedding.binding", "ollama")

    result = runner.invoke(cli.cli, ["config", "create"])

    assert result.exit_code != 0
    assert "--force" in result.output
    # The refusal has to actually protect the file.
    assert settings.get("embedding.binding") == "ollama"


def test_config_create_force_overwrites(runner: CliRunner) -> None:
    runner.invoke(cli.cli, ["config", "create"])
    settings.set_value("embedding.binding", "ollama")

    result = runner.invoke(cli.cli, ["config", "create", "--force"])

    assert result.exit_code == 0, result.output
    assert settings.get("embedding.binding") == "fastembed"


def test_config_create_output_does_not_change_resolved_settings(runner: CliRunner) -> None:
    """Creating the file must be a no-op for behaviour - it only makes the
    defaults explicit."""
    before = settings.load().model_dump()

    result = runner.invoke(cli.cli, ["config", "create"])

    assert result.exit_code == 0, result.output
    assert settings.load().model_dump() == before


# ---------------------------------------------------------------------------
# config path
# ---------------------------------------------------------------------------


def test_config_path_prints_the_config_file_path(
    runner: CliRunner, _isolated_config_dir: Path
) -> None:
    result = runner.invoke(cli.cli, ["config", "path"])

    assert result.exit_code == 0, result.output
    assert str(_isolated_config_dir / "config.toml") in result.output


def test_config_path_warns_when_the_file_does_not_exist(runner: CliRunner) -> None:
    result = runner.invoke(cli.cli, ["config", "path"])

    assert result.exit_code == 0, result.output
    assert "No config file yet" in result.output
    assert "boepie config create" in result.output


def test_config_path_does_not_warn_once_the_file_exists(runner: CliRunner) -> None:
    runner.invoke(cli.cli, ["config", "create"])

    result = runner.invoke(cli.cli, ["config", "path"])

    assert "No config file yet" not in result.output


# ---------------------------------------------------------------------------
# config show
# ---------------------------------------------------------------------------


def test_config_show_prints_section_headers_and_defaults(runner: CliRunner) -> None:
    result = runner.invoke(cli.cli, ["config", "show"])

    assert result.exit_code == 0, result.output
    # Regression guard: rich.Console.print() parses '[embedding]' as markup
    # (an unknown style tag) unless the command disables it - this must
    # actually appear in the rendered output, not be silently swallowed.
    assert "[embedding]" in result.output
    assert 'binding = "fastembed"' in result.output


def test_config_show_says_when_no_file_backs_the_values(runner: CliRunner) -> None:
    """The complaint this answers: `config show` printing a full config gives
    no hint that nothing on disk backs it."""
    result = runner.invoke(cli.cli, ["config", "show"])

    assert result.exit_code == 0, result.output
    assert "No config file yet" in result.output
    assert "# default" in result.output


def test_config_show_marks_file_backed_values(runner: CliRunner) -> None:
    settings.set_value("embedding.binding", "ollama")

    result = runner.invoke(cli.cli, ["config", "show"])

    assert result.exit_code == 0, result.output
    assert 'binding = "ollama"  # file' in result.output
    assert "No config file yet" not in result.output


def test_config_show_names_the_variable_behind_an_env_backed_value(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("BOEPIE_RETRIEVAL_DEFAULT_TOP_K", "42")

    result = runner.invoke(cli.cli, ["config", "show"])

    assert result.exit_code == 0, result.output
    assert "default_top_k = 42  # BOEPIE_RETRIEVAL_DEFAULT_TOP_K" in result.output


def test_config_show_no_sources_prints_plain_toml(runner: CliRunner) -> None:
    result = runner.invoke(cli.cli, ["config", "show", "--no-sources"])

    assert result.exit_code == 0, result.output
    assert "#" not in result.output
    assert 'binding = "fastembed"' in result.output


# ---------------------------------------------------------------------------
# config get
# ---------------------------------------------------------------------------


def test_config_get_prints_the_resolved_value(runner: CliRunner) -> None:
    result = runner.invoke(cli.cli, ["config", "get", "mineru.backend"])

    assert result.exit_code == 0, result.output
    assert result.output.strip() == "pipeline"


def test_config_get_source_reports_the_layer(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("BOEPIE_MINERU_BACKEND", "vlm")

    result = runner.invoke(cli.cli, ["config", "get", "mineru.backend", "--source"])

    assert result.exit_code == 0, result.output
    assert result.output.strip() == "vlm (BOEPIE_MINERU_BACKEND)"


def test_config_get_rejects_an_unknown_key(runner: CliRunner) -> None:
    result = runner.invoke(cli.cli, ["config", "get", "nonsense.key"])

    assert result.exit_code != 0
    assert "unknown config key" in result.output


# ---------------------------------------------------------------------------
# config set
# ---------------------------------------------------------------------------


def test_config_set_writes_and_get_reads_it_back(runner: CliRunner) -> None:
    set_result = runner.invoke(cli.cli, ["config", "set", "literature.prefer_pdf", "true"])
    assert set_result.exit_code == 0, set_result.output

    get_result = runner.invoke(cli.cli, ["config", "get", "literature.prefer_pdf"])
    assert get_result.output.strip() == "True"


@pytest.mark.parametrize(
    ("key", "raw_value", "expected"),
    [
        ("literature.prefer_pdf", "true", True),
        ("sync.check_interval_days", "14", 14),
        ("literature.fetch_delay", "2.5", 2.5),
        ("mineru.device_mode", "cuda", "cuda"),
    ],
)
def test_config_set_coerces_value_types(
    runner: CliRunner, key: str, raw_value: str, expected: object
) -> None:
    result = runner.invoke(cli.cli, ["config", "set", key, raw_value])

    assert result.exit_code == 0, result.output
    assert settings.get(key) == expected


def test_config_set_rejects_an_unknown_key(runner: CliRunner) -> None:
    result = runner.invoke(cli.cli, ["config", "set", "nonsense.key", "1"])

    assert result.exit_code != 0
    assert "unknown config key" in result.output


def test_config_set_rejects_a_value_the_schema_refuses(runner: CliRunner) -> None:
    """The bad value must be caught before it reaches the file, or every
    later command fails on a file the user cannot easily connect to this."""
    result = runner.invoke(cli.cli, ["config", "set", "retrieval.default_mode", "nonsense"])

    assert result.exit_code != 0
    assert "retrieval.default_mode" in result.output
    assert not settings.config_file_exists()


def test_config_set_enforces_field_constraints(runner: CliRunner) -> None:
    result = runner.invoke(cli.cli, ["config", "set", "retrieval.default_top_k", "0"])

    assert result.exit_code != 0
    assert settings.get("retrieval.default_top_k") == 5


def test_config_set_notes_that_it_created_the_file(runner: CliRunner) -> None:
    result = runner.invoke(cli.cli, ["config", "set", "literature.prefer_pdf", "true"])

    assert result.exit_code == 0, result.output
    assert "boepie config create" in result.output


def test_config_set_warns_when_an_env_var_shadows_the_write(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Silently writing a value the environment overrides is exactly the
    confusion `config show`'s sources exist to prevent."""
    monkeypatch.setenv("BOEPIE_EMBEDDING_BINDING", "openai")

    result = runner.invoke(cli.cli, ["config", "set", "embedding.binding", "ollama"])

    assert result.exit_code == 0, result.output
    # Substrings chosen to survive rich's line wrapping of the prose between them.
    assert "BOEPIE_EMBEDDING_BINDING" in result.output
    assert "'openai'" in result.output
    assert settings.get("embedding.binding") == "openai"


def test_config_set_preserves_a_bracketed_value_through_rich_markup(runner: CliRunner) -> None:
    """Regression guard for the same markup-escaping issue as config show:
    the confirmation message must not swallow a literal '[' in the value."""
    result = runner.invoke(
        cli.cli, ["config", "set", "instructions.custom", "prefer [wsclean] over casa.clean"]
    )

    assert result.exit_code == 0, result.output
    assert "prefer [wsclean] over casa.clean" in result.output
    assert settings.get("instructions.custom") == "prefer [wsclean] over casa.clean"


# ---------------------------------------------------------------------------
# help / structure
# ---------------------------------------------------------------------------


def test_config_group_help_lists_subcommands(runner: CliRunner) -> None:
    result = runner.invoke(cli.cli, ["config", "--help"])

    assert result.exit_code == 0
    for subcommand in ("create", "path", "show", "get", "set"):
        assert subcommand in result.output


def test_top_level_help_lists_config(runner: CliRunner) -> None:
    result = runner.invoke(cli.cli, ["--help"])

    assert result.exit_code == 0
    assert "config" in result.output
