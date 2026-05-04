"""Extract metadata from CASA MeasurementSets via python-casacore.

Reads observation/antenna/spectral window/field/polarization subtables and
samples the main table for time, UV, flag, and scan information. Returns a
plain ``dict`` consumable by :class:`boepie.ms.metadata.MSInfo`.

Adapted from snoepie.utilities.measurement_set, with the astropy-based MJD
conversion replaced by plain ``datetime`` arithmetic so boepie does not need
to depend on astropy.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
from casacore.tables import table


# MJD epoch: 1858-11-17 00:00:00 UTC. MeasurementSets store times as
# "seconds since the MJD epoch", so converting is a flat offset and does not
# need astropy.
_MJD_EPOCH = datetime(1858, 11, 17, tzinfo=UTC)


def mjd_seconds_to_datetime(mjd_seconds: float) -> datetime:
    """Convert seconds-since-MJD-epoch to a UTC-aware ``datetime``."""
    return _MJD_EPOCH + timedelta(seconds=mjd_seconds)


def get_directory_size_mb(path: Path) -> float:
    """Return the total size of ``path`` on disk in megabytes."""
    total_size = 0
    for dirpath, _dirnames, filenames in os.walk(path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if os.path.exists(filepath):
                total_size += os.path.getsize(filepath)
    return total_size / (1024 * 1024)


# CASA polarization codes from Stokes.h
_POL_MAP = {
    1: "I", 2: "Q", 3: "U", 4: "V",
    5: "RR", 6: "RL", 7: "LR", 8: "LL",
    9: "XX", 10: "XY", 11: "YX", 12: "YY",
}


def pol_type_to_string(pol_type: int) -> str:
    """Map a CASA polarization integer code to a string like ``'XX'``."""
    return _POL_MAP.get(pol_type, f"POL_{pol_type}")


def _get_data_column_info(tb: Any, n_rows: int) -> list[dict[str, Any]]:
    """Inspect all main-table columns and return info for those that are complex visibility arrays.

    For each qualifying column, records:
      - name: column name
      - dtype: casacore value type ('complex' or 'dcomplex')
      - shape: [n_channels, n_polarizations] from the column descriptor or a sample cell
      - is_nonzero: True if at least one sampled value is non-zero
    """
    _COMPLEX_TYPES = {"complex", "fcomplex", "dcomplex"}
    sample_rows = min(10, n_rows) if n_rows > 0 else 0
    result: list[dict[str, Any]] = []

    for colname in tb.colnames():
        try:
            descriptor = tb.getcoldesc(colname)
            value_type = str(descriptor.get("valueType", "")).lower()
            if value_type not in _COMPLEX_TYPES:
                continue
            if descriptor.get("ndim", 0) < 1:
                continue

            # Prefer the fixed shape from the descriptor; fall back to reading a cell.
            shape: list[int] | None = None
            if "shape" in descriptor:
                shape = list(descriptor["shape"])
            elif n_rows > 0:
                try:
                    shape = list(tb.getcell(colname, 0).shape)
                except Exception:
                    pass

            is_nonzero = False
            if sample_rows > 0:
                try:
                    sample_data = tb.getcol(colname, startrow=0, nrow=sample_rows)
                    is_nonzero = bool(np.any(sample_data != 0))
                except Exception:
                    pass

            result.append({
                "name": colname,
                "dtype": value_type,
                "shape": shape,
                "is_nonzero": is_nonzero,
            })
        except Exception:
            pass

    return result


def extract_ms_metadata(ms_path: Path) -> dict[str, Any]:
    """Extract metadata from a MeasurementSet into a flat dict.

    Opens the main table plus each subtable once, samples the main table
    for time/UV/flag info, and attaches filesystem stats. The result is
    designed to be passed straight into ``MSInfo(**metadata)``.
    """
    metadata: dict[str, Any] = {"path": ms_path}

    tb = table(str(ms_path), ack=False)
    try:
        metadata["number_of_rows"] = tb.nrows()

        colnames = tb.colnames()
        metadata["main_table_columns"] = list(colnames)
        metadata["has_data"] = "DATA" in colnames
        metadata["has_corrected_data"] = "CORRECTED_DATA" in colnames
        metadata["has_model_data"] = "MODEL_DATA" in colnames

        col_info = _get_data_column_info(tb, metadata["number_of_rows"])
        metadata["available_data_columns"] = [entry["name"] for entry in col_info]
        metadata["nonzero_data_columns"] = [entry["name"] for entry in col_info if entry["is_nonzero"]]
        metadata["data_column_info"] = col_info

        metadata.update(read_observation_table(ms_path))
        metadata.update(read_antenna_table(ms_path))
        metadata.update(read_spectral_window_table(ms_path))
        metadata.update(read_field_table(ms_path))
        metadata.update(read_polarization_table(ms_path))

        if metadata["number_of_rows"] > 0:
            metadata.update(extract_time_info(tb))
            metadata.update(extract_uv_info(tb))
            metadata.update(extract_flag_info(tb, metadata["number_of_rows"]))
            metadata.update(extract_scan_info(tb))

        n_ant = metadata.get("number_of_antennas", 0)
        metadata["number_of_baselines"] = n_ant * (n_ant - 1) // 2
    finally:
        tb.close()

    metadata["size_on_disk_mb"] = get_directory_size_mb(ms_path)
    stat = ms_path.stat()
    metadata["last_modified"] = datetime.fromtimestamp(stat.st_mtime, tz=UTC)
    metadata["creation_date"] = datetime.fromtimestamp(stat.st_ctime, tz=UTC)

    try:
        metadata.update(read_history_table(ms_path))
    except Exception:
        pass  # HISTORY subtable is optional

    return metadata


def read_observation_table(ms_path: Path) -> dict[str, Any]:
    obs_path = ms_path / "OBSERVATION"
    empty = {
        "telescope": "Unknown",
        "observer": None,
        "project": None,
        "observation_date": None,
        "observation_id": None,
    }
    if not obs_path.exists():
        return empty

    tb = table(str(obs_path), ack=False)
    try:
        if tb.nrows() == 0:
            return empty
        result: dict[str, Any] = {}
        cols = tb.colnames()
        result["telescope"] = tb.getcol("TELESCOPE_NAME")[0] if "TELESCOPE_NAME" in cols else "Unknown"
        result["observer"] = tb.getcol("OBSERVER")[0] if "OBSERVER" in cols else None
        result["project"] = tb.getcol("PROJECT")[0] if "PROJECT" in cols else None
        if "TIME_RANGE" in cols:
            time_range = tb.getcol("TIME_RANGE")[0]
            result["observation_date"] = mjd_seconds_to_datetime(float(time_range[0]))
        else:
            result["observation_date"] = None
        result["observation_id"] = "0"
        return result
    finally:
        tb.close()


def read_antenna_table(ms_path: Path) -> dict[str, Any]:
    ant_path = ms_path / "ANTENNA"
    if not ant_path.exists():
        return {
            "number_of_antennas": 0,
            "antenna_names": [],
            "antenna_positions": None,
            "array_center": None,
        }

    tb = table(str(ant_path), ack=False)
    try:
        n_antennas = tb.nrows()
        antenna_names: list[str] = list(tb.getcol("NAME")) if "NAME" in tb.colnames() else []

        antenna_positions: list[tuple[float, float, float]] | None = None
        array_center: tuple[float, float, float] | None = None
        if "POSITION" in tb.colnames():
            positions = tb.getcol("POSITION")
            antenna_positions = [tuple(pos) for pos in positions]
            array_center = tuple(np.mean(positions, axis=0))

        return {
            "number_of_antennas": n_antennas,
            "antenna_names": antenna_names,
            "antenna_positions": antenna_positions,
            "array_center": array_center,
        }
    finally:
        tb.close()


def read_spectral_window_table(ms_path: Path) -> dict[str, Any]:
    spw_path = ms_path / "SPECTRAL_WINDOW"
    if not spw_path.exists():
        return {
            "number_of_spectral_windows": 0,
            "number_of_channels": 0,
            "central_frequency": 0.0,
            "channel_width": 0.0,
            "total_bandwidth": 0.0,
            "frequency_range": (0.0, 0.0),
            "spectral_window_info": [],
        }

    tb = table(str(spw_path), ack=False)
    try:
        n_spw = tb.nrows()
        spw_info_list: list[dict[str, Any]] = []
        all_freqs: list[float] = []

        for spw_id in range(n_spw):
            chan_freq = tb.getcell("CHAN_FREQ", spw_id)
            chan_width = tb.getcell("CHAN_WIDTH", spw_id)
            n_channels = len(chan_freq)
            spw_info_list.append(
                {
                    "spw_id": spw_id,
                    "n_channels": n_channels,
                    "freq_start": float(chan_freq[0]),
                    "freq_end": float(chan_freq[-1]),
                    "chan_width": float(np.abs(chan_width[0])),
                }
            )
            all_freqs.extend(chan_freq)

        freq_arr = np.array(all_freqs) if all_freqs else np.array([0.0])
        freq_min = float(np.min(freq_arr))
        freq_max = float(np.max(freq_arr))

        if n_spw > 0:
            first_spw = spw_info_list[0]
            central_frequency = float(tb.getcell("REF_FREQUENCY", 0))
            channel_width = first_spw["chan_width"]
            total_bandwidth = float(tb.getcell("TOTAL_BANDWIDTH", 0))
            number_of_channels = first_spw["n_channels"]
        else:
            central_frequency = 0.0
            channel_width = 0.0
            total_bandwidth = 0.0
            number_of_channels = 0

        return {
            "number_of_spectral_windows": n_spw,
            "number_of_channels": number_of_channels,
            "central_frequency": central_frequency,
            "channel_width": channel_width,
            "total_bandwidth": total_bandwidth,
            "frequency_range": (freq_min, freq_max),
            "spectral_window_info": spw_info_list,
        }
    finally:
        tb.close()


def read_field_table(ms_path: Path) -> dict[str, Any]:
    field_path = ms_path / "FIELD"
    if not field_path.exists():
        return {
            "number_of_fields": 0,
            "field_names": [],
            "phase_center_ra": 0.0,
            "phase_center_dec": 0.0,
        }

    tb = table(str(field_path), ack=False)
    try:
        n_fields = tb.nrows()
        field_names = list(tb.getcol("NAME")) if "NAME" in tb.colnames() else []

        if n_fields > 0 and "PHASE_DIR" in tb.colnames():
            phase_dir = tb.getcell("PHASE_DIR", 0).flatten().tolist()
            # Map RA from [-180, 180] to [0, 360) so pydantic's ge=0 lt=360 holds.
            ra_deg = float(np.degrees(phase_dir[0])) % 360.0
            dec_deg = float(np.degrees(phase_dir[1]))
        else:
            ra_deg = 0.0
            dec_deg = 0.0

        return {
            "number_of_fields": n_fields,
            "field_names": field_names,
            "phase_center_ra": ra_deg,
            "phase_center_dec": dec_deg,
        }
    finally:
        tb.close()


def read_polarization_table(ms_path: Path) -> dict[str, Any]:
    pol_path = ms_path / "POLARIZATION"
    empty = {
        "number_of_polarizations": 0,
        "polarization_types": [],
        "stokes_types": None,
    }
    if not pol_path.exists():
        return empty

    tb = table(str(pol_path), ack=False)
    try:
        if tb.nrows() == 0:
            return empty
        corr_type = tb.getcol("CORR_TYPE")[0]
        pol_types = [pol_type_to_string(int(pt)) for pt in corr_type]
        stokes = {"I", "Q", "U", "V"}
        stokes_types = pol_types if all(pt in stokes for pt in pol_types) else None
        return {
            "number_of_polarizations": len(corr_type),
            "polarization_types": pol_types,
            "stokes_types": stokes_types,
        }
    finally:
        tb.close()


def extract_time_info(tb: Any) -> dict[str, Any]:
    times = tb.getcol("TIME")
    unique_times = np.unique(times)
    time_min = mjd_seconds_to_datetime(float(np.min(times)))
    time_max = mjd_seconds_to_datetime(float(np.max(times)))
    integration_time = float(tb.getcell("INTERVAL", 0)) if "INTERVAL" in tb.colnames() else None
    total_obs_time = float(np.max(times) - np.min(times))
    return {
        "number_of_timesteps": len(unique_times),
        "time_range": (time_min, time_max),
        "integration_time": integration_time,
        "total_observation_time": total_obs_time,
    }


def extract_uv_info(tb: Any) -> dict[str, Any]:
    n_rows = tb.nrows()
    sample_size = min(10000, n_rows)
    if "UVW" not in tb.colnames():
        return {"baseline_lengths": None, "uv_range": None}

    row_indices = np.linspace(0, n_rows - 1, sample_size, dtype=int)
    uvw = tb.getcol("UVW", startrow=int(row_indices[0]), nrow=len(row_indices))
    baseline_lengths_m = np.sqrt(uvw[:, 0] ** 2 + uvw[:, 1] ** 2)
    return {
        "baseline_lengths": (float(np.min(baseline_lengths_m)), float(np.max(baseline_lengths_m))),
        "uv_range": None,
    }


def extract_flag_info(tb: Any, n_rows: int) -> dict[str, Any]:
    if "FLAG" not in tb.colnames():
        return {"flagged_fraction": 0.0}
    sample_size = min(1000, n_rows)
    row_indices = np.linspace(0, n_rows - 1, sample_size, dtype=int)
    flags = tb.getcol("FLAG", startrow=int(row_indices[0]), nrow=len(row_indices))
    return {"flagged_fraction": float(np.mean(flags))}


def extract_scan_info(tb: Any) -> dict[str, Any]:
    result: dict[str, Any] = {"scan_intents": []}
    if "SCAN_NUMBER" in tb.colnames():
        scans = tb.getcol("SCAN_NUMBER")
        result["number_of_scans"] = int(len(np.unique(scans)))
    else:
        result["number_of_scans"] = None
    if "STATE_ID" in tb.colnames():
        states = tb.getcol("STATE_ID")
        result["state_ids"] = np.unique(states).tolist()
    else:
        result["state_ids"] = []
    return result


def read_history_table(ms_path: Path) -> dict[str, Any]:
    hist_path = ms_path / "HISTORY"
    if not hist_path.exists():
        return {"history_summary": [], "casa_version": None}

    tb = table(str(hist_path), ack=False)
    try:
        history_summary: list[str] = []
        casa_version: str | None = None
        n_hist = min(10, tb.nrows())
        if n_hist > 0 and "MESSAGE" in tb.colnames():
            messages = tb.getcol("MESSAGE", startrow=max(0, tb.nrows() - n_hist), nrow=n_hist)
            history_summary = [msg for msg in messages if msg.strip()]
            for msg in messages:
                lowered = msg.lower()
                if "version" in lowered or "casa" in lowered:
                    casa_version = msg.strip()
                    break
        return {"history_summary": history_summary, "casa_version": casa_version}
    finally:
        tb.close()
