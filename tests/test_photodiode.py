import pytest

from msp_control.calibration.photodiode import (
    PDA200C_GAINS,
    voltage_to_current,
    responsivity_at_wavelength,
    current_to_power,
    power_to_photon_flux_density,
)


@pytest.mark.parametrize(
    "range_label, expected_gain",
    [
        ("nA 1", 1e8),
        ("nA 2", 1e7),
        ("uA 1", 1e6),
        ("uA 2", 1e5),
        ("uA 3", 1e4),
        ("mA 1", 1e3),
    ],
)
def test_pda200c_gains(range_label, expected_gain):
    assert PDA200C_GAINS[range_label] == expected_gain


def test_voltage_to_current():
    # PDA200C set to nA 2 (10^7 V/A).
    # A 0.5 V output corresponds to 50 nA.
    current = voltage_to_current(
        voltage=0.5,
        range_label="nA 2",
    )

    assert current == pytest.approx(50e-9)


def test_voltage_to_current_with_dark_baseline():
    # PDA200C set to nA 2 (10^7 V/A).
    # Dark output: 0.02 V
    # Illuminated output: 0.52 V
    # Net photocurrent: 50 nA

    current = voltage_to_current(
        voltage=0.52,
        range_label="nA 2",
        dark_voltage=0.02,
    )

    assert current == pytest.approx(50e-9)


def test_responsivity_interpolation():
    # Exact calibration point: 400 nm, 0.1672 A/W.
    assert responsivity_at_wavelength(400) == pytest.approx(0.1672)

    # Halfway between 400 and 405 nm.
    # Expected linear interpolation:
    # (0.1672 + 0.1723) / 2 = 0.16975 A/W.
    assert responsivity_at_wavelength(402.5) == pytest.approx(
        0.16975
    )


def test_responsivity_rejects_out_of_range():
    with pytest.raises(ValueError, match="outside the calibrated range"):
        responsivity_at_wavelength(300)

    with pytest.raises(ValueError, match="outside the calibrated range"):
        responsivity_at_wavelength(1000)


def test_current_to_power():
    # At 400 nm, responsivity is 0.1672 A/W.
    # A photocurrent of 16.72 nA corresponds to 100 nW.

    power = current_to_power(
        current=16.72e-9,
        wavelength_nm=400,
    )

    assert power == pytest.approx(100e-9)


def test_power_to_photon_flux_density():
    # 100 nW at 500 nm, distributed over 1000 µm².

    flux = power_to_photon_flux_density(
        power_w=100e-9,
        wavelength_nm=500,
        area_um2=1000,
    )

    # Photon energy at 500 nm is approximately 3.973e-19 J.
    # 100 nW corresponds to approximately 2.517e11 photons/s.
    # Over 1000 µm²: 2.517e8 photons/µm²/s.

    assert flux == pytest.approx(2.517e8, rel=1e-3)
