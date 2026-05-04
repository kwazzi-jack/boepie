"""MSInfo: lightweight metadata for a MeasurementSet.

Ported from snoepie.typings.measurement_set. Provides structured access to
observational, array, frequency, temporal, spatial, polarization, data
quality, UV coverage, file system, and software metadata for a single MS.
The display groups drive the section-filtering used by the MCP tools.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import cache
from pathlib import Path
from typing import Annotated, Any, Callable, ClassVar, Self

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
    model_validator,
)

from boepie.ms.extraction import extract_ms_metadata


@dataclass(frozen=True)
class FieldMeta:
    display_name: str


class _BoepieModel(BaseModel):
    @classmethod
    @cache
    def field_metadata(cls) -> dict[str, FieldMeta | None]:
        return {
            name: next((entry for entry in field.metadata if isinstance(entry, FieldMeta)), None)
            for name, field in cls.model_fields.items()
        }


# -- Per-field formatters for markdown rendering --
# Each formatter receives the raw attribute value and returns either a
# string (the rendered value) or None (skip this field). Anything not in
# this dict falls through to _format_generic, so this stays data-driven
# rather than a nested if-else.

def _fmt_freq_hz(value: float) -> str:
    return f"{value:.6g} Hz"


def _fmt_freq_range(value: tuple[float, float]) -> str:
    return f"{value[0] / 1e9:.3f} GHz to {value[1] / 1e9:.3f} GHz"


def _fmt_channel_width(value: float) -> str:
    return f"{value / 1e3:.3f} kHz"


def _fmt_size_mb(value: float) -> str:
    return f"{value:.1f} MB"


def _fmt_fraction(value: float) -> str:
    return f"{value:.1%}"


def _fmt_baseline_range(value: tuple[float, float]) -> str:
    return f"{value[0]:.1f} m to {value[1]:.1f} m"


def _fmt_time_range(value: tuple[datetime, datetime]) -> str:
    return f"{value[0].isoformat()} to {value[1].isoformat()}"


def _fmt_spw_info(value: list[dict[str, Any]]) -> str | None:
    if not value:
        return None
    return "\n" + "\n".join(
        f"  - spw {entry['spw_id']}: {entry['n_channels']} ch, "
        f"{entry['freq_start'] / 1e9:.3f}–{entry['freq_end'] / 1e9:.3f} GHz"
        for entry in value
    )


_FIELD_FORMATTERS: dict[str, Callable[[Any], str | None]] = {
    "central_frequency": _fmt_freq_hz,
    "total_bandwidth": _fmt_freq_hz,
    "channel_width": _fmt_channel_width,
    "frequency_range": _fmt_freq_range,
    "baseline_lengths": _fmt_baseline_range,
    "time_range": _fmt_time_range,
    "size_on_disk_mb": _fmt_size_mb,
    "flagged_fraction": _fmt_fraction,
    "unflagged_fraction": _fmt_fraction,
    "spectral_window_info": _fmt_spw_info,
}


def _format_generic(value: Any) -> str | None:
    """Fallback renderer for values without a dedicated formatter."""
    if value is None:
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list):
        if not value:
            return None
        return ", ".join(str(item) for item in value)
    if isinstance(value, tuple):
        return ", ".join(str(item) for item in value)
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


class MSInfo(_BoepieModel):
    """Structured metadata snapshot of a MeasurementSet.

    Populate via :meth:`from_path`, which opens the MS with python-casacore,
    reads subtables, samples the main table, and closes everything before
    returning. All fields are frozen after construction.
    """

    model_config = ConfigDict(
        frozen=True,
        validate_assignment=True,
        arbitrary_types_allowed=True,
        str_strip_whitespace=True,
    )

    __display_groups__: ClassVar[list[dict[str, Any]]] = [
        {
            "title": "Observation",
            "key": "observation",
            "fields": [
                "path", "telescope", "observer", "project",
                "observation_date", "observation_id",
            ],
        },
        {
            "title": "Array",
            "key": "array",
            "fields": ["number_of_antennas", "antenna_names", "number_of_baselines"],
        },
        {
            "title": "Frequency",
            "key": "frequency",
            "fields": [
                "central_frequency", "observing_frequency_ghz",
                "channel_width", "total_bandwidth", "bandwidth_mhz",
                "frequency_range", "number_of_channels",
                "number_of_spectral_windows", "is_multi_spw",
                "spectral_window_info",
            ],
        },
        {
            "title": "Data Dimensions",
            "key": "data_dimensions",
            "fields": [
                "number_of_rows", "number_of_polarizations",
                "number_of_timesteps", "total_visibilities",
            ],
        },
        {
            "title": "Temporal",
            "key": "temporal",
            "fields": [
                "time_range", "integration_time",
                "total_observation_time", "observation_duration",
            ],
        },
        {
            "title": "Spatial / Pointing",
            "key": "spatial",
            "fields": [
                "phase_center_ra", "phase_center_dec",
                "field_names", "number_of_fields", "is_multi_field",
            ],
        },
        {
            "title": "Polarization",
            "key": "polarization",
            "fields": ["polarization_types", "stokes_types"],
        },
        {
            "title": "Data Columns",
            "key": "data_columns",
            "fields": [
                "has_data", "has_corrected_data", "has_model_data",
                "available_data_columns", "nonzero_data_columns",
                "data_column_info", "main_table_columns",
            ],
        },
        {
            "title": "Data Quality",
            "key": "data_quality",
            "fields": ["flagged_fraction", "unflagged_fraction", "flag_summary_by_antenna"],
        },
        {
            "title": "UV Coverage",
            "key": "uv_coverage",
            "fields": [
                "baseline_lengths", "uv_range",
                "longest_baseline_km", "angular_resolution_arcsec",
            ],
        },
        {
            "title": "Scan / State",
            "key": "scan_state",
            "fields": ["number_of_scans", "scan_intents", "state_ids"],
        },
        {
            "title": "File Info",
            "key": "file_info",
            "fields": ["size_on_disk_mb", "creation_date", "last_modified"],
        },
        {
            "title": "Software",
            "key": "software",
            "fields": ["casa_version", "history_summary"],
        },
    ]

    # === Observational ===
    path: Annotated[Path, Field(description="Absolute path to the MS directory"), FieldMeta(display_name="Path")]
    telescope: Annotated[str, Field(min_length=1, description="Telescope or array name"), FieldMeta(display_name="Telescope")]
    observer: Annotated[str | None, Field(default=None), FieldMeta(display_name="Observer")]
    project: Annotated[str | None, Field(default=None), FieldMeta(display_name="Project")]
    observation_date: Annotated[datetime | None, Field(default=None), FieldMeta(display_name="Observation Date")]
    observation_id: Annotated[str | None, Field(default=None), FieldMeta(display_name="Observation ID")]

    # === Array ===
    number_of_antennas: Annotated[int, Field(ge=0), FieldMeta(display_name="Number of Antennas")]
    antenna_names: Annotated[list[str], Field(default_factory=list), FieldMeta(display_name="Antenna Names")]
    antenna_positions: Annotated[list[tuple[float, float, float]] | None, Field(default=None), FieldMeta(display_name="Antenna Positions")]
    array_center: Annotated[tuple[float, float, float] | None, Field(default=None), FieldMeta(display_name="Array Center")]

    # === Data dimensions ===
    number_of_rows: Annotated[int, Field(ge=0), FieldMeta(display_name="Number of Rows")]
    number_of_channels: Annotated[int, Field(ge=0), FieldMeta(display_name="Number of Channels")]
    number_of_polarizations: Annotated[int, Field(ge=0, le=4), FieldMeta(display_name="Number of Polarizations")]
    number_of_baselines: Annotated[int, Field(ge=0), FieldMeta(display_name="Number of Baselines")]
    number_of_timesteps: Annotated[int, Field(default=0, ge=0), FieldMeta(display_name="Number of Timesteps")]
    number_of_spectral_windows: Annotated[int, Field(ge=0), FieldMeta(display_name="Number of Spectral Windows")]

    # === Frequency ===
    central_frequency: Annotated[float, Field(ge=0.0), FieldMeta(display_name="Central Frequency")]
    channel_width: Annotated[float, Field(ge=0.0), FieldMeta(display_name="Channel Width")]
    total_bandwidth: Annotated[float, Field(ge=0.0), FieldMeta(display_name="Total Bandwidth")]
    frequency_range: Annotated[tuple[float, float], Field(), FieldMeta(display_name="Frequency Range")]
    spectral_window_info: Annotated[list[dict[str, Any]], Field(default_factory=list), FieldMeta(display_name="Spectral Window Info")]

    # === Temporal ===
    time_range: Annotated[tuple[datetime, datetime] | None, Field(default=None), FieldMeta(display_name="Time Range")]
    integration_time: Annotated[float | None, Field(default=None, ge=0.0), FieldMeta(display_name="Integration Time")]
    total_observation_time: Annotated[float | None, Field(default=None, ge=0.0), FieldMeta(display_name="Total Observation Time")]

    # === Spatial / Pointing ===
    phase_center_ra: Annotated[float, Field(ge=0.0, lt=360.0), FieldMeta(display_name="Phase Center RA")]
    phase_center_dec: Annotated[float, Field(ge=-90.0, le=90.0), FieldMeta(display_name="Phase Center Dec")]
    field_names: Annotated[list[str], Field(default_factory=list), FieldMeta(display_name="Field Names")]
    number_of_fields: Annotated[int, Field(default=1, ge=0), FieldMeta(display_name="Number of Fields")]

    # === Polarization ===
    polarization_types: Annotated[list[str], Field(default_factory=list), FieldMeta(display_name="Polarization Types")]
    stokes_types: Annotated[list[str] | None, Field(default=None), FieldMeta(display_name="Stokes Types")]

    # === Data columns ===
    has_data: Annotated[bool, Field(default=True), FieldMeta(display_name="Has DATA")]
    has_corrected_data: Annotated[bool, Field(default=False), FieldMeta(display_name="Has CORRECTED_DATA")]
    has_model_data: Annotated[bool, Field(default=False), FieldMeta(display_name="Has MODEL_DATA")]
    available_data_columns: Annotated[list[str], Field(default_factory=list), FieldMeta(display_name="Available Data Columns")]
    nonzero_data_columns: Annotated[list[str], Field(default_factory=list), FieldMeta(display_name="Non-zero Data Columns")]
    data_column_info: Annotated[list[dict[str, Any]], Field(default_factory=list), FieldMeta(display_name="Data Column Info")]
    main_table_columns: Annotated[list[str], Field(default_factory=list), FieldMeta(display_name="Main Table Columns")]

    # === Data quality ===
    flagged_fraction: Annotated[float, Field(default=0.0, ge=0.0, le=1.0), FieldMeta(display_name="Flagged Fraction")]
    flag_summary_by_antenna: Annotated[dict[str, float] | None, Field(default=None), FieldMeta(display_name="Flag Summary by Antenna")]

    # === UV coverage ===
    baseline_lengths: Annotated[tuple[float, float] | None, Field(default=None), FieldMeta(display_name="Baseline Lengths")]
    uv_range: Annotated[tuple[float, float] | None, Field(default=None), FieldMeta(display_name="UV Range")]

    # === Scan / State ===
    number_of_scans: Annotated[int | None, Field(default=None, ge=0), FieldMeta(display_name="Number of Scans")]
    scan_intents: Annotated[list[str], Field(default_factory=list), FieldMeta(display_name="Scan Intents")]
    state_ids: Annotated[list[int], Field(default_factory=list), FieldMeta(display_name="State IDs")]

    # === File info ===
    size_on_disk_mb: Annotated[float | None, Field(default=None, ge=0.0), FieldMeta(display_name="Size on Disk (MB)")]
    creation_date: Annotated[datetime | None, Field(default=None), FieldMeta(display_name="Creation Date")]
    last_modified: Annotated[datetime | None, Field(default=None), FieldMeta(display_name="Last Modified")]

    # === Software ===
    casa_version: Annotated[str | None, Field(default=None), FieldMeta(display_name="CASA Version")]
    history_summary: Annotated[list[str], Field(default_factory=list), FieldMeta(display_name="History Summary")]

    # -- Validators --

    @field_validator("path", mode="before")
    @classmethod
    def _validate_path(cls, v: Any) -> Path:
        if isinstance(v, (str, Path)):
            return Path(v).absolute()
        raise ValueError(f"path must be str or Path, got {type(v)}")

    @field_validator("frequency_range")
    @classmethod
    def _validate_frequency_range(cls, v: tuple[float, float]) -> tuple[float, float]:
        if v[0] > v[1]:
            raise ValueError(f"min frequency {v[0]} > max frequency {v[1]}")
        return v

    @field_validator("time_range")
    @classmethod
    def _validate_time_range(
        cls, v: tuple[datetime, datetime] | None
    ) -> tuple[datetime, datetime] | None:
        if v is not None and v[0] > v[1]:
            raise ValueError(f"start time {v[0]} > end time {v[1]}")
        return v

    @field_validator("baseline_lengths")
    @classmethod
    def _validate_baseline_lengths(
        cls, v: tuple[float, float] | None
    ) -> tuple[float, float] | None:
        if v is not None and v[0] > v[1]:
            raise ValueError(f"min baseline {v[0]} > max baseline {v[1]}")
        return v

    @model_validator(mode="after")
    def _validate_consistency(self) -> Self:
        length_checks = [
            ("polarization_types", self.polarization_types, "number_of_polarizations", self.number_of_polarizations),
            ("antenna_names", self.antenna_names, "number_of_antennas", self.number_of_antennas),
            ("field_names", self.field_names, "number_of_fields", self.number_of_fields),
        ]
        for list_name, list_value, count_name, count_value in length_checks:
            if list_value and len(list_value) != count_value:
                raise ValueError(
                    f"{list_name} length ({len(list_value)}) != {count_name} ({count_value})"
                )
        return self

    # -- Factory --

    @classmethod
    def from_path(cls, ms_path: str | Path) -> Self:
        parsed = Path(ms_path).absolute()
        if not parsed.exists():
            raise FileNotFoundError(f"Cannot find MS at '{parsed}'")
        if not parsed.is_dir():
            raise ValueError(f"MS path must be a directory: '{parsed}'")
        return cls(**extract_ms_metadata(parsed))

    # -- Computed fields --

    @computed_field
    @property
    def observing_frequency_ghz(self) -> float:
        return self.central_frequency / 1e9

    @computed_field
    @property
    def bandwidth_mhz(self) -> float:
        return self.total_bandwidth / 1e6

    @computed_field
    @property
    def observation_duration(self) -> timedelta | None:
        return self.time_range[1] - self.time_range[0] if self.time_range else None

    @computed_field
    @property
    def longest_baseline_km(self) -> float | None:
        return self.baseline_lengths[1] / 1000.0 if self.baseline_lengths else None

    @computed_field
    @property
    def angular_resolution_arcsec(self) -> float | None:
        if not self.baseline_lengths or not self.central_frequency:
            return None
        max_baseline = self.baseline_lengths[1]
        if max_baseline <= 0:
            return None
        wavelength = 3e8 / self.central_frequency
        return float(np.degrees(wavelength / max_baseline) * 3600)

    @computed_field
    @property
    def is_multi_field(self) -> bool:
        return self.number_of_fields > 1

    @computed_field
    @property
    def is_multi_spw(self) -> bool:
        return self.number_of_spectral_windows > 1

    @computed_field
    @property
    def total_visibilities(self) -> int:
        return self.number_of_rows * self.number_of_channels * self.number_of_polarizations

    @computed_field
    @property
    def unflagged_fraction(self) -> float:
        return 1.0 - self.flagged_fraction

    # -- Display --

    def summary(self) -> dict[str, Any]:
        """Quick-glance dict of the most useful fields."""
        return {
            "path": str(self.path),
            "telescope": self.telescope,
            "observation_date": self.observation_date,
            "n_antennas": self.number_of_antennas,
            "n_fields": self.number_of_fields,
            "n_spectral_windows": self.number_of_spectral_windows,
            "field_names": self.field_names,
            "time_range": self.time_range,
            "frequency_ghz": self.observing_frequency_ghz,
            "bandwidth_mhz": self.bandwidth_mhz,
            "data_columns": self.available_data_columns,
            "nonzero_data_columns": self.nonzero_data_columns,
            "data_column_info": self.data_column_info,
            "size_mb": self.size_on_disk_mb,
            "flagged_fraction": f"{self.flagged_fraction:.1%}",
        }

    def to_markdown(self, sections: list[str] | None = None) -> str:
        """Render selected display groups as markdown.

        ``sections`` is a list of group keys (see ``__display_groups__``).
        ``None`` renders all groups.
        """
        selected = set(sections) if sections else None
        lines: list[str] = [f"# MeasurementSet: `{self.path.name}`", ""]

        for group in self.__display_groups__:
            if selected is not None and group["key"] not in selected:
                continue
            lines.append(f"## {group['title']}")
            for field_name in group["fields"]:
                rendered = self._render_field(field_name)
                if rendered is None:
                    continue
                lines.append(f"- **{self._display_name(field_name)}**: {rendered}")
            lines.append("")

        return "\n".join(lines).rstrip() + "\n"

    def _display_name(self, field_name: str) -> str:
        meta = self.field_metadata().get(field_name)
        return meta.display_name if meta else field_name.replace("_", " ").title()

    def _render_field(self, field_name: str) -> str | None:
        value = getattr(self, field_name, None)
        formatter = _FIELD_FORMATTERS.get(field_name)
        if formatter is not None and value is not None:
            return formatter(value)
        return _format_generic(value)
