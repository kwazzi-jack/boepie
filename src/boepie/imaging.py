import subprocess as sbp
from collections.abc import Sequence
from typing import Any

from fastmcp.tools import tool  # Import the standalone decorator
from pydantic import BaseModel

from boepie.utilities import cli_sanitise


class WSCleanInput(BaseModel):
    # Required inputs
    ms: Sequence[str]  # Measurement set(s)
    prefix: str  # Prefix of output products
    size: int | tuple[int, int]  # Image size in pixels
    scale: str | float  # Angular pixel size

    # Optional inputs
    column: str | None = "DATA"
    nchan: int | None = None
    deconvolution_channels: int | None = None
    channel_range: Sequence[int] | None = None
    pol: str | Sequence[str] | None = None
    join_polarizations: bool | None = None
    link_polarizations: str | Sequence[str] | None = None
    intervals_out: int | None = None
    threads: int | None = None
    make_psf_only: bool | None = None
    weight: str | tuple[str, float] | None = None
    multiscale: bool | None = None
    multiscale_scales: Sequence[int] | None = None
    multiscale_scale_bias: float | None = None
    taper_gaussian: str | None = None
    niter: int | None = None
    nmiter: int | None = None
    fits_mask: str | None = None
    threshold: float | None = None
    auto_threshold: float | None = None
    auto_mask: float | None = None
    local_rms: bool | None = None
    local_rms_window: int | None = None
    local_rms_method: str | None = None
    gain: float | None = None
    mgain: float | None = None
    baseline_averaging: float | None = None
    join_channels: bool | None = None
    fit_spectral_pol: int | None = None
    fit_beam: bool | None = None
    elliptical_beam: bool | None = None
    padding: float | None = None
    nwlayers: int | None = None
    nwlayers_factor: float | None = None
    save_source_list: bool | None = None
    store_imaging_weights: bool | None = None
    parallel_deconvolution: int | None = None
    parallel_gridding: int | None = None
    parallel_reordering: int | None = None
    no_reorder: bool | None = None
    reorder: bool | None = None
    predict: bool | None = None
    no_update_model_required: bool | None = None
    continue_: bool | None = None
    subtract_model: bool | None = None
    use_wgridder: bool | None = None
    log_time: bool | None = None
    interval: Sequence[int] | None = None
    no_dirty: bool | None = None
    make_psf: bool | None = None
    simulate_noise: float | str | None = None
    taper_inner_tukey: float | None = None
    minuv_l: float | None = None
    maxuv_l: float | None = None
    minuvw_m: float | None = None
    maxuvw_m: float | None = None

    def to_cmd(self) -> list[str]:
        result = ["cultcargo::wsclean"]
        args = self.model_dump()

        for arg_name, arg_value in args.items():
            if arg_value is not None:
                result.append(f"{cli_sanitise(arg_name)}={arg_value}")

        return result


class RunResult(BaseModel):
    command: str
    stdout: str
    stderr: str
    return_code: int


def stimela_run(commands: list[str]) -> RunResult:
    final_commands = ["uv", "run", "stimela", "run"] + commands
    process = sbp.Popen(final_commands, stdout=sbp.PIPE, stderr=sbp.PIPE, text=True)
    stdout, stderr = process.communicate()
    return RunResult(
        command=" ".join(final_commands),
        stdout=stdout,
        stderr=stderr,
        return_code=process.returncode,
    )


def run_wsclean(input_args: WSCleanInput) -> RunResult:
    return stimela_run(input_args.to_cmd())
