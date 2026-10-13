"""Waveform configuration and generation for LED calibration."""

from dataclasses import dataclass

import numpy as np

@dataclass(frozen=True)
class CalibrationWaveformConfig:
    """Configuration for a five-phase LED calibration waveform."""

    dark_before_s: float = 1.0
    ramp_up_s: float = 10.0
    hold_s: float = 1.0
    ramp_down_s: float = 10.0
    dark_after_s: float = 1.0
    max_voltage: float = 10.0
    sample_rate_hz: float = 1000.0

    def __post_init__(self):
        if not 0 < self.max_voltage <= 10:
            raise ValueError("Maximum voltage must be between 0 and 10 V")

        if self.sample_rate_hz <= 0:
            raise ValueError("Sample rate must be positive")

        for name in (
            "dark_before_s",
            "ramp_up_s",
            "hold_s",
            "ramp_down_s",
            "dark_after_s",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class BleachWaveformConfig:
    """Timing configuration for a bleaching exposure."""

    duration_s: float
    sample_rate_hz: float = 1000.0
    pre_dark_s: float = 0.1
    post_dark_s: float = 0.05

    def __post_init__(self):
        if not np.isfinite(self.duration_s) or self.duration_s <= 0:
            raise ValueError("duration must be positive")

        if not np.isfinite(self.sample_rate_hz) or self.sample_rate_hz <= 0:
            raise ValueError("sample_rate must be positive")

        if not np.isfinite(self.pre_dark_s) or self.pre_dark_s < 0:
            raise ValueError("pre_dark_s cannot be negative")

        if not np.isfinite(self.post_dark_s) or self.post_dark_s <= 0:
            raise ValueError("post_dark_s must be positive")


def generate_bleach_waveform(
    config: BleachWaveformConfig,
    control_mode: str,
    command_voltage: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a rectangular analog or digital bleach waveform."""

    if control_mode not in ("analog", "digital"):
        raise ValueError("Invalid control_mode")

    if control_mode == "analog":
        if command_voltage is None:
            raise ValueError("Analog bleach requires command_voltage")

        if not np.isfinite(command_voltage) or not 0 <= command_voltage <= 10:
            raise ValueError("command_voltage must be between 0 and 10 V")

    elif command_voltage is not None:
        raise ValueError("Digital bleach requires command_voltage=None")

    fs = config.sample_rate_hz

    n_pre = round(config.pre_dark_s * fs)
    n_on = round(config.duration_s * fs)
    n_post = round(config.post_dark_s * fs)

    if n_on < 1:
        raise ValueError("duration is too short for sample_rate")

    if n_post < 1:
        raise ValueError("post_dark_s is too short for sample_rate")

    if control_mode == "digital":
        output = np.concatenate([
            np.zeros(n_pre, dtype=bool),
            np.ones(n_on, dtype=bool),
            np.zeros(n_post, dtype=bool),
        ])
    else:
        output = np.concatenate([
            np.zeros(n_pre, dtype=float),
            np.full(n_on, command_voltage, dtype=float),
            np.zeros(n_post, dtype=float),
        ])

    time_s = np.arange(len(output)) / fs

    return time_s, output


def generate_calibration_waveform(
    config: CalibrationWaveformConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a five-phase LED calibration waveform.

    Returns
    -------
    time_s : np.ndarray
        Sample times in seconds.
    voltage : np.ndarray
        LED command voltage (V).
    """
    fs = config.sample_rate_hz

    def samples(duration_s):
        return round(duration_s * fs)

    n_dark_before = samples(config.dark_before_s)
    n_ramp_up = samples(config.ramp_up_s)
    n_hold = samples(config.hold_s)
    n_ramp_down = samples(config.ramp_down_s)
    n_dark_after = samples(config.dark_after_s)

    if min(
        n_dark_before,
        n_ramp_up,
        n_hold,
        n_ramp_down,
        n_dark_after,
    ) < 1:
        raise ValueError("Each waveform phase must contain at least one sample")

    max_v = config.max_voltage

    voltage = np.concatenate([
        np.zeros(n_dark_before),
        np.linspace(0, max_v, n_ramp_up, endpoint=False),
        np.full(n_hold, max_v),
        np.linspace(max_v, 0, n_ramp_down, endpoint=False),
        np.zeros(n_dark_after),
    ])

    time_s = np.arange(len(voltage)) / fs

    return time_s, voltage
