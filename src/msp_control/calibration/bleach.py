from dataclasses import dataclass
import numpy as np

from msp_control.calibration.model import (
    AnalogLEDCalibration,
    DigitalLEDCalibration,
)


@dataclass(frozen=True)
class BleachEstimate:
    photon_flux_density: float
    effective_photon_flux_density: float
    fraction_bleached: float


def estimate_bleach(
    calibration: AnalogLEDCalibration | DigitalLEDCalibration,
    duration_s: float,
    nd_filter: float,
    lambda_max_nm: float = 500.0,
    command_voltage: float | None = None,
) -> BleachEstimate:
    """Estimate photon flux density and fraction of pigment bleached."""

    if duration_s < 0:
        raise ValueError("Bleach duration cannot be negative")

    if nd_filter < 0:
        raise ValueError("ND filter cannot be negative")

    photon_flux_density = calibrated_photon_flux_density(
        calibration,
        command_voltage=command_voltage,
    )

    effective_flux = effective_photon_flux_density(
        photon_flux_density=photon_flux_density,
        nd_filter=nd_filter,
        wavelength_nm=calibration.wavelength_nm,
        lambda_max_nm=lambda_max_nm,
    )

    fraction_bleached = estimated_bleach_fraction(
        effective_flux_density=effective_flux,
        duration_s=duration_s,
    )

    return BleachEstimate(
        photon_flux_density=photon_flux_density * 10 ** (-nd_filter),
        effective_photon_flux_density=effective_flux,
        fraction_bleached=fraction_bleached,
    )


PHOTOSENSITIVITY = 5.7e-9


def effective_photon_flux_density(
    photon_flux_density: float,
    nd_filter: float,
    wavelength_nm: float,
    lambda_max_nm: float = 500.0,
) -> float:
    """Calculate photon flux density weighted by A1 sensitivity."""

    if photon_flux_density < 0:
        raise ValueError("Photon flux density cannot be negative")

    if nd_filter < 0:
        raise ValueError("ND filter cannot be negative")

    relative_sensitivity = relative_a1_sensitivity(
        wavelength_nm,
        lambda_max_nm,
        beta_band=True,
    )

    return float(
        photon_flux_density
        * 10 ** (-nd_filter)
        * relative_sensitivity
    )


def estimated_bleach_fraction(
    effective_flux_density: float,
    duration_s: float,
) -> float:
    """Estimate the fraction of A1 pigment bleached."""

    if effective_flux_density < 0:
        raise ValueError("Effective photon flux density cannot be negative")

    if duration_s < 0:
        raise ValueError("Bleach duration cannot be negative")

    return float(
        -np.expm1(
            -PHOTOSENSITIVITY * effective_flux_density * duration_s
        )
    )


def a1_sensitivity(
    wavelength_nm: float,
    lambda_max_nm: float = 500.0,
    beta_band: bool = True,
) -> float:
    """A1 pigment sensitivity, optionally including the beta band."""

    x = lambda_max_nm / wavelength_nm

    a = 0.8795 + 0.0459 * np.exp(
        -((lambda_max_nm - 300) ** 2) / 11940
    )

    alpha = 1.0 / (
        np.exp(69.7 * (a - x))
        + np.exp(28 * (0.922 - x))
        + np.exp(-14.9 * (1.104 - x))
        + 0.674
    )

    if not beta_band:
        return float(alpha)

    b_beta = -40.5 + 0.195 * lambda_max_nm
    lambda_beta = 189 + 0.315 * lambda_max_nm

    beta = 0.26 * np.exp(
        -((wavelength_nm - lambda_beta) / b_beta) ** 2
    )

    return float(alpha + beta)


def relative_a1_sensitivity(
    wavelength_nm: float,
    lambda_max_nm: float = 500.0,
    beta_band: bool = True,
) -> float:
    """A1 sensitivity relative to its value at lambda_max."""

    sensitivity = a1_sensitivity(
        wavelength_nm,
        lambda_max_nm,
        beta_band=beta_band,
    )

    peak_sensitivity = a1_sensitivity(
        lambda_max_nm,
        lambda_max_nm,
        beta_band=beta_band,
    )

    return sensitivity / peak_sensitivity


def calibrated_photon_flux_density(
    calibration: AnalogLEDCalibration | DigitalLEDCalibration,
    command_voltage: float | None = None,
    n_points: int = 5,
) -> float:
    """Return photon flux density before ND attenuation.

    For analog LEDs, fit a local linear regression to the
    calibration points nearest the requested command voltage.
    """

    if isinstance(calibration, DigitalLEDCalibration):
        return float(calibration.photon_flux_density_on)

    if not isinstance(calibration, AnalogLEDCalibration):
        raise TypeError("Unsupported LED calibration type")

    if command_voltage is None:
        raise ValueError("Analog LED requires a command voltage")

    voltages = calibration.calibration_voltage
    flux = calibration.photon_flux_density

    if not voltages[0] <= command_voltage <= voltages[-1]:
        raise ValueError("Command voltage is outside the calibration range")

    if n_points < 2:
        raise ValueError("At least two calibration points are required")

    n = min(n_points, len(voltages))

    indices = np.argsort(
        np.abs(voltages - command_voltage)
    )[:n]

    slope, intercept = np.polyfit(
        voltages[indices],
        flux[indices],
        1,
    )

    return float(max(0.0, slope * command_voltage + intercept))



