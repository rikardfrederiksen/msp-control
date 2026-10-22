"""Processing of raw LED calibration measurements."""

from dataclasses import dataclass

import numpy as np

from msp_control.calibration.acquisition import RawCalibrationAcquisition, RawDigitalCalibrationAcquisition
from msp_control.calibration.photodiode import (
    voltage_to_current,
    current_to_power,
    power_to_photon_flux_density,
)
from msp_control.calibration.waveform import CalibrationWaveformConfig


@dataclass
class AnalogCalibrationRamps:
    """Separate ascending and descending LED calibration measurements."""

    voltage_up: np.ndarray
    monitor_up: np.ndarray
    flux_up: np.ndarray

    voltage_down: np.ndarray
    monitor_down: np.ndarray
    flux_down: np.ndarray

    dark_voltage: float


def process_analog_calibration(
    raw: RawCalibrationAcquisition,
    config: CalibrationWaveformConfig,
    wavelength_nm: float,
    amplifier_range: str,
    illuminated_area_um2: float,
) -> AnalogCalibrationRamps:
    """Convert analog calibration ramps to photon flux density."""

    fs = config.sample_rate_hz

    n_dark_before = round(config.dark_before_s * fs)
    n_ramp_up = round(config.ramp_up_s * fs)
    n_hold = round(config.hold_s * fs)
    n_ramp_down = round(config.ramp_down_s * fs)
    n_dark_after = round(config.dark_after_s * fs)

    expected_samples = (
        n_dark_before
        + n_ramp_up
        + n_hold
        + n_ramp_down
        + n_dark_after
    )

    if len(raw.time_s) != expected_samples:
        raise ValueError("Raw calibration length does not match waveform")

    if any(
        len(array) != expected_samples
        for array in (
            raw.command_voltage,
            raw.monitor_voltage,
            raw.photodiode_voltage,
        )
    ):
        raise ValueError("Raw calibration arrays have different lengths")

    # Estimate dark offset from both dark periods.
    dark_samples = np.concatenate([
        raw.photodiode_voltage[:n_dark_before],
        raw.photodiode_voltage[-n_dark_after:],
    ])

    dark_voltage = float(np.mean(dark_samples))

    # Locate the two ramps.
    up_start = n_dark_before
    up_end = up_start + n_ramp_up

    down_start = up_end + n_hold
    down_end = down_start + n_ramp_down

    def calculate_flux(voltage):
        current = voltage_to_current(
            voltage,
            amplifier_range,
            dark_voltage=dark_voltage,
        )

        power = current_to_power(current, wavelength_nm)

        return power_to_photon_flux_density(
            power,
            wavelength_nm,
            illuminated_area_um2,
        )

    return AnalogCalibrationRamps(
        voltage_up=raw.command_voltage[up_start:up_end].copy(),
        monitor_up=raw.monitor_voltage[up_start:up_end].copy(),
        flux_up=calculate_flux(
            raw.photodiode_voltage[up_start:up_end]
        ),
        voltage_down=raw.command_voltage[down_start:down_end].copy(),
        monitor_down=raw.monitor_voltage[down_start:down_end].copy(),
        flux_down=calculate_flux(
            raw.photodiode_voltage[down_start:down_end]
        ),
        dark_voltage=dark_voltage,
    )


def process_digital_calibration(
    raw: RawDigitalCalibrationAcquisition,
    wavelength_nm: float,
    amplifier_range: str,
    illuminated_area_um2: float,
) -> tuple[float, float]:
    """
    Calculate the calibrated ON-state output of a digital LED.

    Returns
    -------
    monitor_voltage_on : float
        Mean LED monitor voltage during illumination.
    photon_flux_density_on : float
        Mean photon flux density (photons/µm²/s).
    """

    state = np.asarray(raw.command_state)

    if state.dtype.kind != "b":
        raise ValueError("Digital command state must be Boolean")

    if state.ndim != 1:
        raise ValueError("Digital command state must be one-dimensional")

    n_samples = len(state)

    if any(
        np.asarray(array).shape != (n_samples,)
        for array in (
            raw.time_s,
            raw.monitor_voltage,
            raw.photodiode_voltage,
        )
    ):
        raise ValueError("Digital calibration arrays have inconsistent lengths")

    if not np.any(state):
        raise ValueError("Calibration contains no ON samples")

    if not np.any(~state):
        raise ValueError("Calibration contains no OFF samples")

    if not (
        np.all(np.isfinite(raw.monitor_voltage))
        and np.all(np.isfinite(raw.photodiode_voltage))
    ):
        raise ValueError("Calibration measurements must be finite")

    # Estimate dark offset from all OFF samples.
    dark_voltage = float(np.mean(raw.photodiode_voltage[~state]))

    # Mean photodiode and LED monitor signals during illumination.
    photodiode_voltage_on = float(
        np.mean(raw.photodiode_voltage[state])
    )

    monitor_voltage_on = float(
        np.mean(raw.monitor_voltage[state])
    )

    current = voltage_to_current(
        photodiode_voltage_on,
        amplifier_range,
        dark_voltage=dark_voltage,
    )

    power = current_to_power(current, wavelength_nm)

    photon_flux_density_on = float(
        power_to_photon_flux_density(
            power,
            wavelength_nm,
            illuminated_area_um2,
        )
    )

    if photon_flux_density_on < 0:
        raise ValueError(
            "Mean ON-state photon flux is negative after dark subtraction"
        )

    return monitor_voltage_on, photon_flux_density_on

def combine_analog_ramps(
    ramps: AnalogCalibrationRamps,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Combine ascending and descending calibration ramps.

    Returns
    -------
    voltage : np.ndarray
        Increasing LED command voltages.
    monitor_voltage : np.ndarray
        Mean LED monitor voltage.
    photon_flux_density : np.ndarray
        Mean photon flux density.
    """

    voltage = ramps.voltage_up.copy()

    # Descending measurements in increasing voltage order.
    voltage_down = ramps.voltage_down[::-1]
    monitor_down = ramps.monitor_down[::-1]
    flux_down = ramps.flux_down[::-1]

    # Interpolate descending measurements onto ascending voltage grid.
    monitor_down_interp = np.interp(
        voltage,
        voltage_down,
        monitor_down,
    )

    flux_down_interp = np.interp(
        voltage,
        voltage_down,
        flux_down,
    )

    monitor_voltage = (
        ramps.monitor_up + monitor_down_interp
    ) / 2

    photon_flux_density = (
        ramps.flux_up + flux_down_interp
    ) / 2

    return voltage, monitor_voltage, photon_flux_density
