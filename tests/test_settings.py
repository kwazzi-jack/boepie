"""Tests for `boepie.settings`: the config schema and its three layers.

Every test isolates BOEPIE_CONFIG_DIR to a tmp_path - never touches a
developer's real ~/.config/boepie - and clears any BOEPIE_* variable the
schema knows about, so a developer's own environment cannot change a result.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from boepie import settings


@pytest.fixture(autouse=True)
def _isolated_config_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    config_dir = tmp_path / "config"
    monkeypatch.setenv(settings.CONFIG_DIR_ENV_VAR, str(config_dir))
    for key in settings.known_keys():
        monkeypatch.delenv(settings.env_var_for(key), raising=False)
    return config_dir


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def test_config_path_is_config_dir_slash_config_toml(_isolated_config_dir: Path) -> None:
    assert settings.config_path() == _isolated_config_dir / "config.toml"


def test_config_dir_falls_back_to_platformdirs_without_the_env_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(settings.CONFIG_DIR_ENV_VAR, raising=False)
    assert settings.config_dir().name == "boepie"


def test_config_file_exists_tracks_the_file(_isolated_config_dir: Path) -> None:
    assert settings.config_file_exists() is False
    settings.create()
    assert settings.config_file_exists() is True


# ---------------------------------------------------------------------------
# env_var_for
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("embedding.binding", "BOEPIE_EMBEDDING_BINDING"),
        ("retrieval.default_top_k", "BOEPIE_RETRIEVAL_DEFAULT_TOP_K"),
        ("corpus.warn_on_dotfile_title", "BOEPIE_CORPUS_WARN_ON_DOTFILE_TITLE"),
    ],
)
def test_env_var_for_derives_the_variable_name(key: str, expected: str) -> None:
    assert settings.env_var_for(key) == expected


# ---------------------------------------------------------------------------
# Layering: env > file > default
# ---------------------------------------------------------------------------


def test_defaults_apply_when_no_file_exists() -> None:
    assert settings.get("embedding.binding") == "fastembed"
    assert settings.get("retrieval.default_top_k") == 5
    assert settings.get("mineru.device_mode") == "auto"


def test_a_partial_file_overrides_only_its_own_keys(_isolated_config_dir: Path) -> None:
    _isolated_config_dir.mkdir(parents=True)
    (_isolated_config_dir / "config.toml").write_text(
        '[embedding]\nbinding = "ollama"\n', encoding="utf-8"
    )

    assert settings.get("embedding.binding") == "ollama"
    # A sibling key in the same section, untouched in the file, keeps its default.
    assert settings.get("embedding.model") == "BAAI/bge-small-en-v1.5"
    # An entirely different section is unaffected.
    assert settings.get("mineru.device_mode") == "auto"


def test_an_env_var_beats_the_file(monkeypatch: pytest.MonkeyPatch) -> None:
    settings.set_value("embedding.binding", "ollama")
    monkeypatch.setenv("BOEPIE_EMBEDDING_BINDING", "openai")

    assert settings.get("embedding.binding") == "openai"


def test_an_env_var_reaches_a_key_whose_name_contains_underscores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression guard for the reason ExactNameEnvSource exists: pydantic-
    settings' own env_nested_delimiter='_' splits this variable into
    retrieval.default.top.k, silently leaving the default in place."""
    monkeypatch.setenv("BOEPIE_RETRIEVAL_DEFAULT_TOP_K", "42")

    assert settings.get("retrieval.default_top_k") == 42


def test_an_unrelated_boepie_env_var_is_not_absorbed_into_a_section(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """BOEPIE_LITERATURE_DIR is a path setting, deliberately env-only and not
    part of the schema; the delimiter-splitting source would read it as
    literature.dir."""
    monkeypatch.setenv("BOEPIE_LITERATURE_DIR", "/tmp/somewhere")

    assert settings.get("literature.prefer_pdf") is False


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("1", True), ("true", True), ("yes", True), ("on", True), ("0", False), ("false", False)],
)
def test_env_bools_coerce(monkeypatch: pytest.MonkeyPatch, raw: str, expected: bool) -> None:
    monkeypatch.setenv("BOEPIE_LITERATURE_PREFER_PDF", raw)

    assert settings.get("literature.prefer_pdf") is expected


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_an_invalid_file_value_raises_config_error(_isolated_config_dir: Path) -> None:
    _isolated_config_dir.mkdir(parents=True)
    (_isolated_config_dir / "config.toml").write_text(
        '[retrieval]\ndefault_mode = "rubbish"\n', encoding="utf-8"
    )

    with pytest.raises(settings.ConfigError) as caught:
        settings.load()

    message = str(caught.value)
    assert "retrieval.default_mode" in message
    # The message has to name the file, since editing it is the only fix.
    assert str(settings.config_path()) in message


def test_a_malformed_toml_file_raises_config_error(_isolated_config_dir: Path) -> None:
    """A hand-edited file can have a syntax error, not just a bad value -
    `config create` writes a file expecting it to be edited by hand."""
    _isolated_config_dir.mkdir(parents=True)
    (_isolated_config_dir / "config.toml").write_text(
        '[embedding\nbinding = "ollama"\n', encoding="utf-8"
    )

    with pytest.raises(settings.ConfigError, match="not valid TOML"):
        settings.load()


def test_an_invalid_env_value_raises_config_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BOEPIE_MINERU_BACKEND", "nonsense")

    with pytest.raises(settings.ConfigError) as caught:
        settings.load()

    assert "BOEPIE_MINERU_BACKEND" in str(caught.value)


# ---------------------------------------------------------------------------
# parse_value
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("key", "raw", "expected"),
    [
        ("literature.prefer_pdf", "true", True),
        ("sync.check_interval_days", "14", 14),
        ("literature.fetch_delay", "2.5", 2.5),
        ("mineru.device_mode", "cuda", "cuda"),
        ("instructions.custom", "prefer wsclean", "prefer wsclean"),
    ],
)
def test_parse_value_coerces_to_the_declared_type(key: str, raw: str, expected: object) -> None:
    assert settings.parse_value(key, raw) == expected


def test_parse_value_rejects_a_value_outside_a_literal() -> None:
    with pytest.raises(settings.ConfigError, match="retrieval.default_mode"):
        settings.parse_value("retrieval.default_mode", "nonsense")


def test_parse_value_enforces_field_constraints() -> None:
    """A bare TypeAdapter over the annotation would accept 0 - the gt=0 lives
    on the field, so validation has to go through the section model."""
    with pytest.raises(settings.ConfigError, match="greater than 0"):
        settings.parse_value("retrieval.default_top_k", "0")


def test_parse_value_rejects_a_non_numeric_int() -> None:
    with pytest.raises(settings.ConfigError):
        settings.parse_value("sync.check_interval_days", "soon")


# ---------------------------------------------------------------------------
# set_value
# ---------------------------------------------------------------------------


def test_set_value_creates_the_config_dir_and_file(_isolated_config_dir: Path) -> None:
    assert not _isolated_config_dir.exists()

    settings.set_value("literature.prefer_pdf", True)

    assert settings.config_path().is_file()
    assert settings.get("literature.prefer_pdf") is True


def test_set_value_preserves_other_keys_and_comments(_isolated_config_dir: Path) -> None:
    """The whole reason for tomlkit over a plain rewrite: a `set` on one key
    must not clobber a user's own comments or other settings."""
    _isolated_config_dir.mkdir(parents=True)
    (_isolated_config_dir / "config.toml").write_text(
        '# my own note\n[embedding]\nbinding = "ollama"\n', encoding="utf-8"
    )

    settings.set_value("literature.prefer_pdf", True)

    raw = settings.config_path().read_text(encoding="utf-8")
    assert "# my own note" in raw
    assert settings.get("embedding.binding") == "ollama"
    assert settings.get("literature.prefer_pdf") is True


def test_set_value_overwrites_an_existing_key() -> None:
    settings.set_value("sync.check_interval_days", 7)
    settings.set_value("sync.check_interval_days", 14)

    assert settings.get("sync.check_interval_days") == 14


def test_set_value_keeps_the_generated_files_comments(_isolated_config_dir: Path) -> None:
    settings.create()

    settings.set_value("embedding.binding", "ollama")

    raw = settings.config_path().read_text(encoding="utf-8")
    assert "# boepie configuration." in raw
    assert "# Embedding model, named as the chosen binding names it." in raw
    assert settings.get("embedding.binding") == "ollama"


# ---------------------------------------------------------------------------
# create / render_default_config
# ---------------------------------------------------------------------------


def test_create_writes_every_known_key(_isolated_config_dir: Path) -> None:
    created = settings.create()

    raw = created.read_text(encoding="utf-8")
    for key in settings.known_keys():
        assert raw.count(f"\n{key.partition('.')[2]} = ") >= 1, key
    for section in ("embedding", "retrieval", "literature", "corpus", "mineru"):
        assert f"[{section}]" in raw


def test_create_writes_values_equal_to_the_defaults(_isolated_config_dir: Path) -> None:
    """The generated file must be a no-op: loading it has to produce exactly
    what the built-in defaults produce, or `config create` would change
    boepie's behaviour just by existing."""
    before = settings.load().model_dump()

    settings.create()

    assert settings.load().model_dump() == before


def test_create_refuses_to_overwrite_without_force(_isolated_config_dir: Path) -> None:
    settings.create()
    settings.set_value("embedding.binding", "ollama")

    with pytest.raises(FileExistsError):
        settings.create()

    assert settings.get("embedding.binding") == "ollama"


def test_create_with_force_replaces_the_file(_isolated_config_dir: Path) -> None:
    settings.create()
    settings.set_value("embedding.binding", "ollama")

    settings.create(force=True)

    assert settings.get("embedding.binding") == "fastembed"


def test_render_default_config_documents_each_key(_isolated_config_dir: Path) -> None:
    rendered = settings.render_default_config()

    assert "# Precedence is: environment variable > this file > built-in default." in rendered
    # Every key's description is carried into the file as a comment, so the
    # file is the reference and there is no second place to keep in step.
    for key in settings.known_keys():
        description = settings.field_for(key).description
        assert description, key
        assert description.split()[0] in rendered, key


# ---------------------------------------------------------------------------
# resolve_settings: the provenance behind `config show`
# ---------------------------------------------------------------------------


def test_resolve_settings_reports_default_when_nothing_is_set() -> None:
    by_key = {item.key: item for item in settings.resolve_settings()}

    assert by_key["embedding.binding"].source == "default"
    assert by_key["embedding.binding"].value == "fastembed"


def test_resolve_settings_reports_file_and_env_separately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings.set_value("embedding.binding", "ollama")
    monkeypatch.setenv("BOEPIE_MINERU_BACKEND", "vlm")

    by_key = {item.key: item for item in settings.resolve_settings()}

    assert by_key["embedding.binding"].source == "file"
    assert by_key["mineru.backend"].source == "env"
    assert by_key["mineru.backend"].env_var == "BOEPIE_MINERU_BACKEND"
    assert by_key["retrieval.default_mode"].source == "default"


def test_resolve_settings_covers_every_known_key() -> None:
    assert [item.key for item in settings.resolve_settings()] == settings.known_keys()


# ---------------------------------------------------------------------------
# known_keys / field_for
# ---------------------------------------------------------------------------


def test_known_keys_covers_every_section_field() -> None:
    keys = settings.known_keys()

    assert "embedding.binding" in keys
    assert "mineru.device_mode" in keys
    assert "instructions.custom" in keys
    assert len(keys) == len(set(keys))


def test_field_for_rejects_an_unknown_key() -> None:
    with pytest.raises(KeyError):
        settings.field_for("nonsense.key")


# ---------------------------------------------------------------------------
# List-valued settings
#
# `pipeline.sources` is the first key that is not a scalar. TOML holds it as
# an array, but a shell argument and an environment variable can only be one
# string, so both of those layers split on commas before pydantic sees them.
# ---------------------------------------------------------------------------


def test_is_list_setting_distinguishes_lists_from_scalars() -> None:
    assert settings.is_list_setting("pipeline.sources")
    assert not settings.is_list_setting("sync.auto_sync")
    assert not settings.is_list_setting("embedding.binding")


def test_pipeline_sources_defaults_to_cultcargo() -> None:
    assert settings.get("pipeline.sources") == ["cultcargo::"]


def test_a_list_setting_splits_a_comma_separated_env_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BOEPIE_PIPELINE_SOURCES", "cultcargo::, otherlib.recipes::")

    assert settings.load().pipeline.sources == ["cultcargo::", "otherlib.recipes::"]


def test_a_list_setting_env_var_drops_empty_items(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BOEPIE_PIPELINE_SOURCES", "cultcargo::,,")

    assert settings.load().pipeline.sources == ["cultcargo::"]


def test_parse_value_splits_a_list_setting() -> None:
    parsed = settings.parse_value("pipeline.sources", "cultcargo::,otherlib::")

    assert parsed == ["cultcargo::", "otherlib::"]


def test_set_value_round_trips_a_list_setting(_isolated_config_dir: Path) -> None:
    settings.set_value("pipeline.sources", ["cultcargo::", "otherlib::"])

    assert settings.get("pipeline.sources") == ["cultcargo::", "otherlib::"]
    raw = settings.config_path().read_text(encoding="utf-8")
    assert "otherlib::" in raw


def test_a_generated_config_file_renders_a_list_setting_as_an_array(
    _isolated_config_dir: Path,
) -> None:
    settings.create()

    raw = settings.config_path().read_text(encoding="utf-8")
    assert 'sources = ["cultcargo::"]' in raw
