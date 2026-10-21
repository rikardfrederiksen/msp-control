import numpy as np

from msp_control.calibration.acquisition import RawCalibrationAcquisition, RawDigitalCalibrationAcquisition
from msp_control.calibration.processing import process_analog_calibration, process_digital_calibration
from msp_control.calibration.waveform import (
    CalibrationWaveformConfig,
    generate_calibration_waveform,
)
from msp_control.calibration.photodiode import (
    PDA200C_GAINS,
    responsivity_at_wavelength,
)


def test_process_analog_calibration():
    """Verify ramp extraction, dark subtraction, and photon flux."""

    config = CalibrationWaveformConfig(
        dark_before_s=0.1,
        ramp_up_s=0.2,
        hold_s=0.1,
        ramp_down_s=0.2,
        dark_after_s=0.1,
        max_voltage=5.0,
        sample_rate_hz=100.0,
    )

    wavelength_nm = 617.0
    amplifier_range = "uA 1"
    area_um2 = 10000.0
    dark_voltage = 0.05

    time_s, command_voltage = generate_calibration_waveform(config)

    # Simulate a photodiode signal proportional to command voltage.
    photodiode_voltage = dark_voltage + 0.1 * command_voltage

    raw = RawCalibrationAcquisition(
        time_s=time_s,
        command_voltage=command_voltage,
        monitor_voltage=0.2 * command_voltage,
        photodiode_voltage=photodiode_voltage,
    )

    result = process_analog_calibration(
        raw,
        config,
        wavelength_nm=wavelength_nm,
        amplifier_range=amplifier_range,
        illuminated_area_um2=area_um2,
    )

    # Verify the dark offset.
    assert np.isclose(result.dark_voltage, dark_voltage)

    # Verify that both ramps were extracted correctly.
    assert len(result.voltage_up) == 20
    assert len(result.voltage_down) == 20

    np.testing.assert_allclose(
        result.voltage_up,
        command_voltage[10:30],
    )

    np.testing.assert_allclose(
        result.voltage_down,
        command_voltage[40:60],
    )

    # Independently calculate the expected photon flux.
    gain = PDA200C_GAINS[amplifier_range]
    responsivity = responsivity_at_wavelength(wavelength_nm)

    h = 6.62607015e-34
    c = 299792458

    photon_energy = h * c / (wavelength_nm * 1e-9)

    expected_flux = (
        (0.1 * command_voltage)
        / gain
        / responsivity
        / photon_energy
        / area_um2
    )

    np.testing.assert_allclose(
        result.flux_up,
        expected_flux[10:30],
        rtol=1e-7,
        atol=1e-6,
    )

    np.testing.assert_allclose(
        result.flux_down,
        expected_flux[40:60],
        rtol=1e-7,
        atol=1e-6,
    )


def test_analog_calibration_preserves_hysteresis():
    """Ascending and descending ramps must remain independent."""

    config = CalibrationWaveformConfig(
        dark_before_s=0.1,
        ramp_up_s=0.2,
        hold_s=0.1,
        ramp_down_s=0.2,
        dark_after_s=0.1,
        max_voltage=5.0,
        sample_rate_hz=100.0,
    )

    time_s, command_voltage = generate_calibration_waveform(config)

    # Simulate a different photodiode response during ramp-down.
    photodiode_voltage = 0.05 + 0.1 * command_voltage

    # Ramp-down occupies samples 40:60.
    photodiode_voltage[40:60] = (
        0.05 + 0.12 * command_voltage[40:60]
    )

    raw = RawCalibrationAcquisition(
        time_s=time_s,
        command_voltage=command_voltage,
        monitor_voltage=0.2 * command_voltage,
        photodiode_voltage=photodiode_voltage,
    )

    result = process_analog_calibration(
        raw,
        config,
        wavelength_nm=617.0,
        amplifier_range="uA 1",
        illuminated_area_um2=10000.0,
    )

    # Compare the two ramps at matching command voltages.
    # The descending ramp is stored in decreasing voltage order.
    flux_down_ascending = np.interp(
        result.voltage_up,
        result.voltage_down[::-1],
        result.flux_down[::-1],
    )

    # At voltages shared by both ramps, descending flux should be higher.
    # Exclude zero and the highest point to avoid endpoint effects.
    assert np.all(
        flux_down_ascending[1:-1] > result.flux_up[1:-1]
    )


def test_process_digital_calibration():
    """Verify digital LED dark subtraction and ON-state photon flux."""

    command_state = np.array(
        [False, False, True, True, True, False, False],
        dtype=bool,
    )

    raw = RawDigitalCalibrationAcquisition(
        time_s=np.arange(7) / 1000.0,
        command_state=command_state,
        monitor_voltage=np.array(
            [0.0, 0.0, 2.0, 2.1, 1.9, 0.0, 0.0]
        ),
        photodiode_voltage=np.array(
            [0.05, 0.05, 0.55, 0.55, 0.55, 0.05, 0.05]
        ),
    )

    monitor_on, flux_on = process_digital_calibration(
        raw,
        wavelength_nm=505.0,
        amplifier_range="uA 1",
        illuminated_area_um2=10000.0,
    )

    assert np.isclose(monitor_on, 2.0)

    gain = PDA200C_GAINS["uA 1"]
    responsivity = responsivity_at_wavelength(505.0)

    h = 6.62607015e-34
    c = 299792458

    photon_energy = h * c / (505.0e-9)

    expected_flux = (
        0.5 / gain / responsivity / photon_energy / 10000.0
    )

    assert np.isclose(flux_on, expected_flux)
