from dataclasses import dataclass
from datetime import datetime

import numpy as np


@dataclass
class AnalogLEDCalibration:
    """Calibration of an analog voltage-controlled LED."""
    
    led_channel: int
    wavelength_nm: float

    amplifier_range: str
    amplifier_gain: float
    photodiode_responsivity: float
    illuminated_area_um2: float

    time_s: np.ndarray
    command_voltage: np.ndarray
    monitor_voltage: np.ndarray
    photodiode_voltage: np.ndarray

    calibration_voltage: np.ndarray
    calibration_monitor_voltage: np.ndarray
    photon_flux_density: np.ndarray

    def __post_init__(self):
        """Validate the analog LED calibration data."""

        raw_arrays = (
            self.time_s,
            self.command_voltage,
            self.monitor_voltage,
            self.photodiode_voltage,
        )

        calibration_arrays = (
            self.calibration_voltage,
            self.calibration_monitor_voltage,
            self.photon_flux_density,
        )

        # All raw recordings must have the same number of samples.
        if len(set(map(len, raw_arrays))) != 1:
            raise ValueError("Analog LED raw arrays have different lengths")

        # All processed calibration arrays must have the same length.
        if len(set(map(len, calibration_arrays))) != 1:
            raise ValueError("Analog LED calibration arrays have different lengths")

        if len(self.calibration_voltage) < 2:
            raise ValueError("Analog LED calibration requires at least two points")

        # Reject NaN and infinite values.
        for array in raw_arrays + calibration_arrays:
            if not np.all(np.isfinite(array)):
                raise ValueError("Analog LED calibration contains non-finite values")

        # Command voltages must be strictly increasing for interpolation.
        if not np.all(np.diff(self.calibration_voltage) > 0):
            raise ValueError(
                "Analog LED calibration voltages must be strictly increasing"
            )

        if np.any(self.photon_flux_density < 0):
            raise ValueError("Photon flux density cannot be negative")

        if self.wavelength_nm <= 0 or not np.isfinite(self.wavelength_nm):
            raise ValueError("LED wavelength must be positive and finite")

        if (
            self.illuminated_area_um2 <= 0
            or not np.isfinite(self.illuminated_area_um2)
        ):
            raise ValueError("Illuminated area must be positive and finite")

        if self.amplifier_gain <= 0 or not np.isfinite(self.amplifier_gain):
            raise ValueError(
                "Photodiode amplifier gain must be positive and finite"
            )

        if (
            self.photodiode_responsivity <= 0
            or not np.isfinite(self.photodiode_responsivity)
        ):
            raise ValueError(
                "Photodiode responsivity must be positive and finite"
            )


@dataclass
class DigitalLEDCalibration:
    """Calibration of a digitally gated LED."""

    led_channel: int
    wavelength_nm: float

    amplifier_range: str
    amplifier_gain: float
    photodiode_responsivity: float
    illuminated_area_um2: float

    time_s: np.ndarray
    command_state: np.ndarray
    monitor_voltage: np.ndarray
    photodiode_voltage: np.ndarray

    monitor_voltage_on: float
    photon_flux_density_on: float


    def __post_init__(self):
        """Validate the digital LED calibration data."""

        raw_arrays = (
            self.time_s,
            self.command_state,
            self.monitor_voltage,
            self.photodiode_voltage,
        )

        if len(set(map(len, raw_arrays))) != 1:
            raise ValueError("Digital LED raw arrays have different lengths")

        if len(self.time_s) == 0:
            raise ValueError("Digital LED calibration has no samples")

        for array in (
            self.time_s,
            self.monitor_voltage,
            self.photodiode_voltage,
        ):
            if not np.all(np.isfinite(array)):
                raise ValueError(
                    "Digital LED calibration contains non-finite values"
                )

        if not np.issubdtype(
            np.asarray(self.command_state).dtype, np.bool_
        ):
            raise ValueError("Digital LED command_state must be boolean")

        if not np.isfinite(self.monitor_voltage_on):
            raise ValueError("Digital LED monitor voltage must be finite")

        if not np.isfinite(self.photon_flux_density_on):
            raise ValueError("Digital LED photon flux must be finite")

        if self.photon_flux_density_on < 0:
            raise ValueError("Digital LED photon flux cannot be negative")

        if self.wavelength_nm <= 0 or not np.isfinite(self.wavelength_nm):
            raise ValueError("LED wavelength must be positive and finite")

        if (
            self.illuminated_area_um2 <= 0
            or not np.isfinite(self.illuminated_area_um2)
        ):
            raise ValueError("Illuminated area must be positive and finite")

        if self.amplifier_gain <= 0 or not np.isfinite(self.amplifier_gain):
            raise ValueError("Photodiode amplifier gain must be positive and finite")

        if (
            self.photodiode_responsivity <= 0
            or not np.isfinite(self.photodiode_responsivity)
        ):
            raise ValueError(
                "Photodiode responsivity must be positive and finite"
            )


@dataclass
class LightCalibration:
    """Calibration of the MSP illumination sources."""

    calibration_id: str
    created_at: datetime

    analog_led: AnalogLEDCalibration
    digital_led: DigitalLEDCalibration
