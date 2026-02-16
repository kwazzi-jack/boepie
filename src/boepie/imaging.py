# imaging.py
import subprocess as sbp
from collections.abc import Sequence
from typing import Annotated, Literal

from fastmcp.tools import tool
from pydantic import BaseModel, Field, field_validator

from boepie.utilities import cli_sanitise


class WSCleanInput(BaseModel):
    """Input parameters for WSClean imager (https://wsclean.readthedocs.io)"""

    # Required inputs
    ms: Annotated[Sequence[str], Field(description="Measurement set(s)")]

    prefix: Annotated[str, Field(description="Prefix of output products")]

    size: Annotated[
        list[int],
        Field(
            description="Image size in pixels ([size] for square or [width, height])",
            min_length=1,
            max_length=2,
        ),
    ]

    scale: Annotated[str | float, Field(description="Angular pixel size")]

    # Optional inputs with descriptions from docs
    column: Annotated[
        str | None, Field(default="DATA", description="Data column to image")
    ] = "DATA"

    nchan: Annotated[int | None, Field(default=None, description="Channels out")] = None

    deconvolution_channels: Annotated[
        int | None, Field(default=None, description="Channels to use in deconvolution")
    ] = None

    channel_range: Annotated[
        Sequence[int] | None, Field(default=None, description="Channel range to image")
    ] = None

    pol: Annotated[
        str | Sequence[str] | None,
        Field(default=None, description="Polarizations to image"),
    ] = None

    join_polarizations: Annotated[
        bool | None, Field(default=None, description="Join polarizations")
    ] = None

    link_polarizations: Annotated[
        str | Sequence[str] | None,
        Field(default=None, description="Link polarizations"),
    ] = None

    intervals_out: Annotated[
        int | None,
        Field(default=None, description="Number of time intervals to output"),
    ] = None

    threads: Annotated[
        int | None, Field(default=None, description="Number of threads")
    ] = None

    make_psf_only: Annotated[
        bool | None, Field(default=None, description="Make PSF only")
    ] = None

    weight: Annotated[
        str | list[str | float] | None,
        Field(
            default=None,
            description="Weighting: natural, uniform (default), briggs. When using Briggs, add the robustness parameter as a number",
        ),
    ] = None

    multiscale: Annotated[
        bool | None, Field(default=None, description="Use multiscale clean")
    ] = None

    multiscale_scales: Annotated[
        Sequence[int] | None,
        Field(default=None, description="Scales for multiscale clean"),
    ] = None

    multiscale_scale_bias: Annotated[
        float | None, Field(default=None, description="Scale bias for multiscale clean")
    ] = None

    taper_gaussian: Annotated[
        str | None, Field(default=None, description="Gaussian taper")
    ] = None

    niter: Annotated[
        int | None, Field(default=None, description="Number of minor clean iterations")
    ] = None

    nmiter: Annotated[
        int | None,
        Field(default=None, description="Max number of major clean iterations"),
    ] = None

    fits_mask: Annotated[
        str | None, Field(default=None, description="FITS mask file")
    ] = None

    threshold: Annotated[
        float | None, Field(default=None, description="Cleaning threshold")
    ] = None

    auto_threshold: Annotated[
        float | None, Field(default=None, description="Auto threshold")
    ] = None

    auto_mask: Annotated[float | None, Field(default=None, description="Auto mask")] = (
        None
    )

    local_rms: Annotated[
        bool | None, Field(default=None, description="Use local RMS")
    ] = None

    local_rms_window: Annotated[
        int | None, Field(default=None, description="Local RMS window size")
    ] = None

    local_rms_method: Annotated[
        Literal["rms", "rms-with-min"] | None,
        Field(default=None, description="Local RMS method (rms or rms-with-min)"),
    ] = None

    gain: Annotated[float | None, Field(default=None, description="Clean gain")] = None

    mgain: Annotated[
        float | None, Field(default=None, description="Major iteration gain")
    ] = None

    baseline_averaging: Annotated[
        float | None, Field(default=None, description="Baseline averaging")
    ] = None

    join_channels: Annotated[
        bool | None, Field(default=None, description="Join channels")
    ] = None

    fit_spectral_pol: Annotated[
        int | None, Field(default=None, description="Fit spectral polynomial order")
    ] = None

    fit_beam: Annotated[bool | None, Field(default=None, description="Fit beam")] = None

    elliptical_beam: Annotated[
        bool | None, Field(default=None, description="Use elliptical beam")
    ] = None

    padding: Annotated[
        float | None, Field(default=None, description="Padding factor")
    ] = None

    nwlayers: Annotated[
        int | None, Field(default=None, description="Number of w-layers")
    ] = None

    nwlayers_factor: Annotated[
        float | None, Field(default=None, description="W-layers factor")
    ] = None

    save_source_list: Annotated[
        bool | None, Field(default=None, description="Save source list")
    ] = None

    store_imaging_weights: Annotated[
        bool | None, Field(default=None, description="Store imaging weights")
    ] = None

    parallel_deconvolution: Annotated[
        int | None, Field(default=None, description="Parallel deconvolution threads")
    ] = None

    parallel_gridding: Annotated[
        int | None, Field(default=None, description="Parallel gridding threads")
    ] = None

    parallel_reordering: Annotated[
        int | None, Field(default=None, description="Parallel reordering threads")
    ] = None

    no_reorder: Annotated[
        bool | None, Field(default=None, description="Disable reordering")
    ] = None

    reorder: Annotated[
        bool | None, Field(default=None, description="Enable reordering")
    ] = None

    predict: Annotated[bool | None, Field(default=None, description="Predict mode")] = (
        None
    )

    no_update_model_required: Annotated[
        bool | None, Field(default=None, description="No update model required")
    ] = None

    continue_: Annotated[
        bool | None,
        Field(default=None, description="Continue from previous run", alias="continue"),
    ] = None

    subtract_model: Annotated[
        bool | None, Field(default=None, description="Subtract model")
    ] = None

    use_wgridder: Annotated[
        bool | None, Field(default=None, description="Use W-gridder")
    ] = None

    log_time: Annotated[bool | None, Field(default=None, description="Log time")] = None

    interval: Annotated[
        Sequence[int] | None, Field(default=None, description="Time intervals to image")
    ] = None

    no_dirty: Annotated[
        bool | None, Field(default=None, description="Don't make dirty image")
    ] = None

    make_psf: Annotated[bool | None, Field(default=None, description="Make PSF")] = None

    simulate_noise: Annotated[
        float | str | None, Field(default=None, description="Simulate noise")
    ] = None

    taper_inner_tukey: Annotated[
        float | None, Field(default=None, description="Inner Tukey taper")
    ] = None

    minuv_l: Annotated[
        float | None, Field(default=None, description="Minimum UV in lambda")
    ] = None

    maxuv_l: Annotated[
        float | None, Field(default=None, description="Maximum UV in lambda")
    ] = None

    minuvw_m: Annotated[
        float | None, Field(default=None, description="Minimum UVW in meters")
    ] = None

    maxuvw_m: Annotated[
        float | None, Field(default=None, description="Maximum UVW in meters")
    ] = None

    # Validators
    @field_validator("size")
    @classmethod
    def validate_size(cls, v):
        """Validate size is either int or list of 2 ints"""
        if isinstance(v, list):
            if len(v) != 2:
                raise ValueError(
                    "Size list must have exactly 2 elements [width, height]"
                )
            if not all(isinstance(x, int) for x in v):
                raise ValueError("Size list elements must be integers")
        return v

    @field_validator("weight")
    @classmethod
    def validate_weight(cls, v):
        """Validate weight is proper format"""
        if isinstance(v, list):
            if len(v) != 2:
                raise ValueError("Weight list must be [weighting_type, robustness]")
            if not isinstance(v[0], str):
                raise ValueError("First element must be weighting type (string)")
            if not isinstance(v[1], (int, float)):
                raise ValueError("Second element must be robustness parameter (number)")
        elif isinstance(v, str):
            valid = ["natural", "uniform", "briggs"]
            if v not in valid:
                raise ValueError(f"Weight must be one of {valid}")
        return v

    def to_cmd(self) -> list[str]:
        result = ["cultcargo::wsclean"]
        args = self.model_dump(by_alias=True)

        for arg_name, arg_value in args.items():
            if arg_value is not None:
                result.append(f"{cli_sanitise(arg_name)}={arg_value}")

        return result


class RunResult(BaseModel):
    """Result of running a stimela command"""

    command: str
    stdout: str
    stderr: str
    return_code: int


def stimela_run(commands: list[str]) -> RunResult:
    final_commands = ["stimela", "run"] + commands
    process = sbp.Popen(final_commands, stdout=sbp.PIPE, stderr=sbp.PIPE, text=True)
    stdout, stderr = process.communicate()
    return RunResult(
        command=" ".join(final_commands),
        stdout=stdout,
        stderr=stderr,
        return_code=process.returncode,
    )


def run_wsclean(input_args: WSCleanInput) -> RunResult:
    """
    Runs WSClean software for radio interferometric imaging.

    WSClean is an imager for radio astronomy that creates images from measurement sets.
    See https://wsclean.readthedocs.io for full documentation.
    """
    return stimela_run(input_args.to_cmd())
