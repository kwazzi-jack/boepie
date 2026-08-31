# boepie/settings.py
"""The user-editable config file and the layering over it.

One pydantic-settings model, `BoepieSettings`, declares every key boepie
understands: its type, its built-in default, and a one-line description.
That single declaration is what `boepie.config` reads constants from, what
`boepie config show|get|set` validates against, and what `boepie config
create` renders into a commented TOML file - so a new setting is added in
exactly one place.

Resolution is three layers, highest first:

    BOEPIE_<SECTION>_<KEY> environment variable
    ~/.config/boepie/config.toml   (or $BOEPIE_CONFIG_DIR/config.toml)
    the field's default

Two deliberate departures from stock pydantic-settings:

- **Reading env vars.** `env_nested_delimiter="_"` splits on every
  underscore, so `BOEPIE_RETRIEVAL_DEFAULT_TOP_K` resolves as
  `retrieval.default.top.k` and never reaches `retrieval.default_top_k` -
  most of boepie's keys contain an underscore, and the failure is silent
  (the default is used, and the bad value is never validated). It also
  swallows unrelated variables: `BOEPIE_LITERATURE_DIR`, a path setting that
  is deliberately env-only, would be read as `literature.dir`.
  `ExactNameEnvSource` therefore looks up one exact variable name per known
  field instead of splitting.
- **Writing.** pydantic-settings only reads. `set_value` writes through
  `tomlkit` so a `set` on one key round-trips the file without clobbering
  the user's own comments or formatting on another.
"""

from __future__ import annotations

import os
import textwrap
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from tomllib import TOMLDecodeError
from typing import Any, Literal, get_origin

import tomlkit
from platformdirs import user_config_dir
from pydantic import BaseModel, Field, ValidationError
from pydantic.fields import FieldInfo
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

CONFIG_DIR_ENV_VAR = "BOEPIE_CONFIG_DIR"
CONFIG_FILENAME = "config.toml"
ENV_VAR_PREFIX = "BOEPIE_"

type SettingSource = Literal["env", "file", "default"]


# ---------------------------------------------------------------------------
# Where the file lives
# ---------------------------------------------------------------------------


def config_dir() -> Path:
    """The directory the config file lives in - `BOEPIE_CONFIG_DIR`, or the
    platform's config directory (e.g. `~/.config/boepie` on Linux)."""
    override = os.environ.get(CONFIG_DIR_ENV_VAR)
    return Path(override).expanduser() if override else Path(user_config_dir("boepie"))


def config_path() -> Path:
    return config_dir() / CONFIG_FILENAME


def config_file_exists() -> bool:
    return config_path().is_file()


def env_var_for(key: str) -> str:
    """The environment variable overriding a dotted key, e.g.
    `'embedding.binding'` -> `'BOEPIE_EMBEDDING_BINDING'`."""
    return ENV_VAR_PREFIX + key.replace(".", "_").upper()


# ---------------------------------------------------------------------------
# The schema: one section per nested model, one key per field
# ---------------------------------------------------------------------------


class EmbeddingSettings(BaseModel):
    """Which embedding backend builds and queries indices."""

    binding: Literal["fastembed", "ollama", "openai"] = Field(
        "fastembed",
        description=(
            "'fastembed' runs a small ONNX model locally on CPU - no service, "
            "API key or GPU. 'ollama' and 'openai' need their own infra."
        ),
    )
    model: str = Field(
        "BAAI/bge-small-en-v1.5",
        description="Embedding model, named as the chosen binding names it.",
    )
    host: str = Field(
        "http://localhost:11434",
        description="Base URL for the 'ollama' binding. Ignored by the others.",
    )
    dim: int = Field(
        384,
        gt=0,
        description=(
            "Embedding dimension, recorded in each index manifest. Must match "
            "the model above - the default is BAAI/bge-small-en-v1.5's."
        ),
    )


class RetrievalSettings(BaseModel):
    """Defaults for the search_*/read_* MCP tools and `boepie search`."""

    default_top_k: int = Field(
        5, gt=0, description="How many hits a search returns when the caller does not say."
    )
    default_mode: Literal["hybrid", "dense", "bm25"] = Field(
        "hybrid", description="'hybrid' fuses dense and lexical; the others use one retriever."
    )
    default_snippet: Literal["none", "short", "full"] = Field(
        "short", description="How much of each hit's chunk body to render."
    )


class LiteratureSettings(BaseModel):
    """Fetching papers for the literature corpus."""

    prefer_pdf: bool = Field(
        False,
        description=(
            "Resolve each paper's DOI and prefer an open-access published PDF "
            "(converted with MinerU, needs local compute) over the arXiv HTML "
            "rendering. Declared but not consumed yet."
        ),
    )
    fetch_delay: float = Field(
        1.0,
        ge=0,
        description="Seconds between paper fetches - politeness, not an API requirement.",
    )


class CorpusSettings(BaseModel):
    """Shared behaviour across the literature, docs and notes corpora."""

    warn_on_dotfile_title: bool = Field(
        True,
        description=(
            "Warn when an added document's title would have produced a "
            "filename starting with '.' before the leading dot is stripped."
        ),
    )
    extra_file_types: list[str] = Field(
        default=[],
        description=(
            "Extra file extensions a folder walk should accept, on top of "
            "the formats boepie already converts (e.g. '.ipynb'). Added to "
            "that list, not a replacement for it. Anything not accepted is "
            "skipped and counted, never read as text - the encoding fallback "
            "would otherwise turn a binary into a document of mojibake."
        ),
    )

    keep_original: bool = Field(
        False,
        description=(
            "Keep the source bytes (the PDF, DOCX, ...) alongside the "
            "converted Markdown, so a document can be re-converted later "
            "without refetching. Off by default because originals are large."
        ),
    )


class MineruSettings(BaseModel):
    """How `corpus add` converts PDF, DOCX, PPTX and XLSX sources.

    MinerU is an optional extra (`uv sync --extra mineru`), driven through
    its CLI so its model stack is never imported into a boepie process that
    is not converting anything.
    """

    device_mode: Literal["auto", "cpu", "cuda", "mps", "npu", "musa", "mlu"] = Field(
        "auto",
        description=(
            "Passed through to MinerU's MINERU_DEVICE_MODE. 'auto' leaves the "
            "variable unset so MinerU detects the device itself."
        ),
    )
    backend: Literal["pipeline", "vlm-engine", "hybrid-engine"] = Field(
        "pipeline",
        description=(
            "'pipeline' is fast and uses no LLM; 'vlm-engine' adds a "
            "vision-language pass for figures and complex layouts; "
            "'hybrid-engine' does both. MinerU's two '*-http-client' backends "
            "are deliberately absent: they offload to a server whose URL "
            "boepie has no setting for."
        ),
    )
    batch_size: int = Field(
        8,
        ge=0,
        description=(
            "How many documents one MinerU process converts. MinerU spends "
            "about 20 seconds loading its models before converting anything, "
            "so batching amortises that; but it writes nothing until a run "
            "finishes, so a large run shows no progress and keeps nothing if "
            "interrupted. 0 converts the whole batch in a single run."
        ),
    )
    model_source: Literal["auto", "huggingface", "modelscope", "local"] = Field(
        "auto",
        description=(
            "Where MinerU loads model weights from. 'auto' leaves "
            "MINERU_MODEL_SOURCE unset, which is how MinerU wants auto "
            "detection expressed - it rejects the literal string 'auto'."
        ),
    )


class IngestionSettings(BaseModel):
    """How an ambiguous `corpus add` identifier is classified. Not consumed yet."""

    use_mcp_sampling: bool = Field(
        True,
        description=(
            "Allow an agent-hosted LLM call (MCP sampling, no separate API "
            "key) to pick a collection when the heuristic cannot."
        ),
    )
    default_collection: Literal["literature", "docs", "notes"] = Field(
        "notes",
        description=(
            "Where an identifier nothing could classify lands. 'notes' is the "
            "only collection that takes a bare identifier and is always "
            "managed_by: user - the other two are manifest-reconciled, and 'docs' "
            "needs a project and base_url rather than a single document."
        ),
    )


class PipelineSettings(BaseModel):
    """Which stimela libraries the cab and recipe tools read."""

    sources: list[str] = Field(
        default=["cultcargo::"],
        description=(
            "stimela sources to load, in stimela's own 'module::path' or "
            "'(module)/path' spelling - 'cultcargo::' means every YAML that "
            "package's MANIFEST.stimela names. Plain file and directory "
            "paths work too. Loaded once per process and cached, so this is "
            "for installed libraries; a recipe file you are working on goes "
            "to a recipe tool's recipe_file argument instead."
        ),
    )


class SyncSettings(BaseModel):
    """Staleness nudges. boepie never self-updates or schedules OS-level jobs."""

    auto_sync: bool = Field(
        False, description="Run a stale sync automatically instead of only printing a nudge."
    )
    check_interval_days: int = Field(
        7, gt=0, description="Days since the last sync before it counts as stale."
    )
    check_boepie_version: bool = Field(
        True, description="Check GitHub for a newer boepie and print an upgrade nudge."
    )


class InstructionsSettings(BaseModel):
    """Your own standing preferences, surfaced to the agent."""

    custom: str = Field(
        "",
        description=(
            "Free text, surfaced through its own doorway - never spliced into "
            "the MCP server's boepie-authored instructions block."
        ),
    )


def is_list_setting(key: str) -> bool:
    """Whether a dotted key holds a list rather than a scalar.

    List-valued keys are a TOML array in the file, but a single string in an
    environment variable or a `config set` argument, so both of those layers
    have to split before pydantic sees the value.
    """
    return get_origin(field_for(key).annotation) is list


def _split_list_value(raw: str) -> list[str]:
    """Split a comma-separated setting into its items, dropping empties.

    Comma rather than another separator to match the `--collection`
    convention the CLI already uses, and because a whitespace split would
    mangle the one thing these lists actually hold - paths.
    """
    return [item.strip() for item in raw.split(",") if item.strip()]


class ExactNameEnvSource(PydanticBaseSettingsSource):
    """Reads one exact `BOEPIE_<SECTION>_<KEY>` variable per known field.

    See this module's docstring for why the stock nested-delimiter source
    cannot be used here.
    """

    def get_field_value(self, field: Any, field_name: str) -> tuple[Any, str, bool]:
        # Required by the base class, but only meaningful for sources that
        # resolve field-by-field; this one builds the whole mapping at once.
        raise NotImplementedError

    def __call__(self) -> dict[str, Any]:
        values: dict[str, Any] = {}
        for section, key in _iter_schema_keys(self.settings_cls):
            dotted = f"{section}.{key}"
            raw = os.environ.get(env_var_for(dotted))
            if raw is None:
                continue
            parsed = _split_list_value(raw) if is_list_setting(dotted) else raw
            values.setdefault(section, {})[key] = parsed
        return values


class BoepieSettings(BaseSettings):
    """Every user-tunable setting. Constructing this applies all three layers."""

    model_config = SettingsConfigDict(extra="ignore")

    embedding: EmbeddingSettings = EmbeddingSettings()
    retrieval: RetrievalSettings = RetrievalSettings()
    literature: LiteratureSettings = LiteratureSettings()
    corpus: CorpusSettings = CorpusSettings()
    mineru: MineruSettings = MineruSettings()
    ingestion: IngestionSettings = IngestionSettings()
    pipeline: PipelineSettings = PipelineSettings()
    sync: SyncSettings = SyncSettings()
    instructions: InstructionsSettings = InstructionsSettings()

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Highest precedence first. The TOML path is resolved here, per
        # instantiation, rather than in `model_config`: BOEPIE_CONFIG_DIR can
        # change after import (the test suite repoints it per test), and a
        # path frozen at class-creation time would ignore that.
        return (
            init_settings,
            ExactNameEnvSource(settings_cls),
            TomlConfigSettingsSource(settings_cls, toml_file=config_path()),
        )


def _iter_schema_keys(
    settings_cls: type[BaseSettings] = BoepieSettings,
) -> Iterator[tuple[str, str]]:
    """Every (section, key) pair the schema declares, in declaration order."""
    for section, section_field in settings_cls.model_fields.items():
        section_model = section_field.annotation
        if section_model is None or not issubclass(section_model, BaseModel):
            continue
        for key in section_model.model_fields:
            yield section, key


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


class ConfigError(Exception):
    """A config file or environment variable holds a value the schema rejects."""


def _describe_validation_error(error: ValidationError) -> str:
    path = config_path()
    lines = [f"invalid boepie configuration ({path}):"]
    for detail in error.errors():
        key = ".".join(str(part) for part in detail["loc"])
        variable = env_var_for(key) if key.count(".") == 1 else None
        origin = f" (or unset {variable})" if variable else ""
        lines.append(f"  {key}: {detail['msg']}{origin}")
    lines.append(f"Edit {path} to fix it, or delete the file to fall back to defaults.")
    return "\n".join(lines)


def load() -> BoepieSettings:
    """The fully resolved settings: env vars over the config file over defaults.

    Raises `ConfigError` - naming the offending key, or the syntax error's
    position - rather than letting pydantic's or tomllib's own traceback
    surface. The cause is almost always something the user typed into a file
    they can go and edit, and `config create` writes that file expecting
    them to.
    """
    try:
        return BoepieSettings()
    except ValidationError as error:
        raise ConfigError(_describe_validation_error(error)) from error
    except TOMLDecodeError as error:
        path = config_path()
        raise ConfigError(
            f"{path} is not valid TOML: {error}\n"
            f"Fix the syntax, or delete the file to fall back to defaults."
        ) from error


def known_keys() -> list[str]:
    """Every dotted key the schema defines, e.g. `'embedding.binding'`."""
    return [f"{section}.{key}" for section, key in _iter_schema_keys()]


def field_for(key: str) -> FieldInfo:
    """The pydantic `FieldInfo` behind a dotted key, for its type and description."""
    section, _, name = key.partition(".")
    section_field = BoepieSettings.model_fields.get(section)
    if section_field is None or not name:
        raise KeyError(key)
    section_model = section_field.annotation
    if section_model is None or name not in section_model.model_fields:
        raise KeyError(key)
    return section_model.model_fields[name]


def get(key: str) -> Any:
    """The resolved value at a dotted key, e.g. `get("embedding.binding")`."""
    section, _, name = key.partition(".")
    return getattr(getattr(load(), section), name)


@dataclass(frozen=True)
class ResolvedSetting:
    key: str
    value: Any
    source: SettingSource
    env_var: str
    description: str


def resolve_settings() -> list[ResolvedSetting]:
    """Every key's resolved value plus which layer supplied it.

    `boepie config show` exists to answer "why is boepie behaving this way",
    which a bare value cannot: the same `binding = "ollama"` means something
    different depending on whether it came from the file the user is looking
    at or from a variable in their shell.
    """
    settings = load()
    from_env = ExactNameEnvSource(BoepieSettings)()
    from_file = TomlConfigSettingsSource(BoepieSettings, toml_file=config_path())()

    resolved: list[ResolvedSetting] = []
    for section, name in _iter_schema_keys():
        key = f"{section}.{name}"
        if name in from_env.get(section, {}):
            source: SettingSource = "env"
        elif name in from_file.get(section, {}):
            source = "file"
        else:
            source = "default"
        resolved.append(
            ResolvedSetting(
                key=key,
                value=getattr(getattr(settings, section), name),
                source=source,
                env_var=env_var_for(key),
                description=field_for(key).description or "",
            )
        )
    return resolved


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


type SettingValue = bool | int | float | str | list[str]


def parse_value(key: str, raw: str) -> SettingValue:
    """Validates a `config set` string against the key's declared type.

    Validation runs through the section's own model rather than the field's
    bare annotation, so the field's constraints apply too - `set
    retrieval.default_top_k 0` has to fail on `gt=0` here, at the point the
    user can still see what they typed, not be written to the file and
    rejected on the next import. Every field has a default, so validating
    the section with just this one key set is well defined.

    A list-valued key is given as one comma-separated argument and split
    before validation, since a shell argument cannot be a TOML array.
    """
    section, _, name = key.partition(".")
    section_model = BoepieSettings.model_fields[section].annotation
    if section_model is None or not issubclass(section_model, BaseModel):
        raise KeyError(key)
    supplied = _split_list_value(raw) if is_list_setting(key) else raw
    try:
        validated = section_model.model_validate({name: supplied})
    except ValidationError as error:
        detail = "; ".join(item["msg"] for item in error.errors())
        raise ConfigError(f"invalid value for {key}: {detail}") from error

    value = getattr(validated, name)
    if isinstance(value, list):
        return [str(item) for item in value]
    if not isinstance(value, bool | int | float | str):
        raise ConfigError(f"{key} does not hold a value TOML can store")
    return value


def set_value(key: str, value: SettingValue) -> None:
    """Writes `value` at the dotted path `key`, creating the file and its
    parent table as needed. Round-tripped via tomlkit, so other keys and any
    comments already in the file survive untouched."""
    document = _load_document()
    section, _, name = key.partition(".")
    if section not in document or not isinstance(document.get(section), dict):
        document[section] = tomlkit.table()
    document[section][name] = value  # pyright: ignore[reportIndexIssue]

    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(tomlkit.dumps(document), encoding="utf-8")


def _load_document() -> tomlkit.TOMLDocument:
    path = config_path()
    if not path.is_file():
        return tomlkit.document()
    return tomlkit.parse(path.read_text(encoding="utf-8"))


_FILE_HEADER = """\
boepie configuration.

Every key boepie understands is listed below at its built-in default, so
this file doubles as the reference for what can be tuned. Delete a key to
fall back to boepie's default for it.

Precedence is: environment variable > this file > built-in default. The
variable for any key is BOEPIE_<SECTION>_<KEY> upper-cased, so
embedding.binding is overridden by BOEPIE_EMBEDDING_BINDING.

Run `boepie config show` to see each key's resolved value and which of those
three layers it actually came from.\
"""


def render_default_config() -> str:
    """The full config file as text: every key at its default, with each
    section's and key's description as comments."""
    lines = [f"# {line}".rstrip() for line in _FILE_HEADER.splitlines()]

    for section, section_field in BoepieSettings.model_fields.items():
        section_model = section_field.annotation
        if section_model is None or not issubclass(section_model, BaseModel):
            continue
        lines.append("")
        if section_model.__doc__:
            lines.append(f"# {' '.join(section_model.__doc__.split())}")
        lines.append(f"[{section}]")
        for name, field in section_model.model_fields.items():
            if field.description:
                lines.extend(_wrapped_comment(field.description))
            rendered = tomlkit.item(field.get_default()).as_string()
            lines.append(f"{name} = {rendered}")
    return "\n".join(lines) + "\n"


def _wrapped_comment(text: str, width: int = 74) -> list[str]:
    return [f"# {line}" for line in textwrap.wrap(" ".join(text.split()), width=width)]


def create(*, force: bool = False) -> Path:
    """Writes a fresh config file containing every key at its default.

    Raises `FileExistsError` rather than overwriting unless `force` - the
    file is the user's, and may hold settings nothing else records.
    """
    path = config_path()
    if path.is_file() and not force:
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_default_config(), encoding="utf-8")
    return path
