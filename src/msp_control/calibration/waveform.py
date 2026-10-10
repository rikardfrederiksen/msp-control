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
