import pytest

import numpy as np

from msp_control.calibration.bleach import (
    relative_a1_sensitivity,
    calibrated_photon_flux_density,
    estimated_bleach_fraction,
    effective_photon_flux_density,
    estimate_bleach,
)
from msp_control.calibration.model import (
    AnalogLEDCalibration,
    DigitalLEDCalibration,
)



def test_relative_a1_sensitivity_at_peak():
    assert relative_a1_sensitivity(500.0, 500.0) == pytest.approx(1.0)


def test_relative_a1_sensitivity_off_peak():
    sensitivity = relative_a1_sensitivity(617.0, 500.0)

    assert 0.0 < sensitivity < 1.0


def test_beta_band_increases_short_wavelength_sensitivity():
    alpha_only = relative_a1_sensitivity(
        350.0, 500.0, beta_band=False
    )

    alpha_beta = relative_a1_sensitivity(
        350.0, 500.0, beta_band=True
    )

    assert alpha_beta > alpha_only


def test_analog_calibrated_flux():
    calibration = object.__new__(AnalogLEDCalibration)
    calibration.calibration_voltage = np.array([0, 1, 2, 3, 4], dtype=float)
    calibration.photon_flux_density = np.array([0, 10, 20, 30, 40], dtype=float)

    result = calibrated_photon_flux_density(
        calibration,
        command_voltage=2.5,
    )

    assert result == pytest.approx(25.0)


def test_digital_calibrated_flux():
    calibration = object.__new__(DigitalLEDCalibration)
    calibration.photon_flux_density_on = 12345.0

    result = calibrated_photon_flux_density(calibration)

    assert result == pytest.approx(12345.0)


def test_effective_flux_at_pigment_peak():
    result = effective_photon_flux_density(
        photon_flux_density=1e8,
        nd_filter=0.0,
        wavelength_nm=500.0,
        lambda_max_nm=500.0,
    )

    assert result == pytest.approx(1e8)


def test_nd_attenuation():
    result = effective_photon_flux_density(
        photon_flux_density=1e8,
        nd_filter=2.0,
        wavelength_nm=500.0,
        lambda_max_nm=500.0,
    )

    assert result == pytest.approx(1e6)


def test_estimated_bleach_fraction():
    result = estimated_bleach_fraction(
        effective_flux_density=1e8,
        duration_s=1.0,
    )

    expected = 1 - np.exp(-5.7e-9 * 1e8)

    assert result == pytest.approx(expected)


def test_estimate_bleach_analog():
    calibration = object.__new__(AnalogLEDCalibration)

    calibration.wavelength_nm = 500.0
    calibration.calibration_voltage = np.array(
        [0, 1, 2, 3, 4], dtype=float
    )
    calibration.photon_flux_density = np.array(
        [0, 1e8, 2e8, 3e8, 4e8], dtype=float
    )

    result = estimate_bleach(
        calibration=calibration,
        duration_s=1.0,
        nd_filter=1.0,
        lambda_max_nm=500.0,
        command_voltage=2.0,
    )

    assert result.photon_flux_density == pytest.approx(2e7)
    assert result.effective_photon_flux_density == pytest.approx(2e7)

    expected_fraction = 1 - np.exp(-5.7e-9 * 2e7)
    assert result.fraction_bleached == pytest.approx(expected_fraction)
