from datetime import datetime

import numpy as np

from msp_control.calibration.acquisition import (
    RawCalibrationAcquisition,
    RawDigitalCalibrationAcquisition,
)
from msp_control.calibration.assembly import assemble_light_calibration
from msp_control.calibration.storage import (
    write_calibration,
    read_calibration,
)
from msp_control.calibration.waveform import (
    CalibrationWaveformConfig,
    BleachWaveformConfig,
    generate_calibration_waveform,
    generate_bleach_waveform,
)


def test_light_calibration_round_trip(tmp_path):
    """Assemble, save, and reload a complete LED calibration."""

    analog_config = CalibrationWaveformConfig(
        dark_before_s=0.1,
        ramp_up_s=0.2,
        hold_s=0.1,
        ramp_down_s=0.2,
        dark_after_s=0.1,
        max_voltage=5.0,
        sample_rate_hz=100.0,
    )

    analog_time, analog_command = generate_calibration_waveform(
        analog_config
    )

    analog_raw = RawCalibrationAcquisition(
        time_s=analog_time,
        command_voltage=analog_command,
        monitor_voltage=0.2 * analog_command,
        photodiode_voltage=0.05 + 0.1 * analog_command,
    )

    digital_config = BleachWaveformConfig(
        duration_s=0.2,
        sample_rate_hz=100.0,
        pre_dark_s=0.1,
        post_dark_s=0.1,
    )

    digital_time, digital_state = generate_bleach_waveform(
        digital_config,
        control_mode="digital",
        command_voltage=None,
    )

    digital_raw = RawDigitalCalibrationAcquisition(
        time_s=digital_time,
        command_state=digital_state,
        monitor_voltage=2.0 * digital_state.astype(float),
        photodiode_voltage=0.05 + 0.5 * digital_state.astype(float),
    )

    created_at = datetime(2026, 10, 21, 12, 0, 0)

    calibration = assemble_light_calibration(
        calibration_id="test_calibration",
        created_at=created_at,
        analog_raw=analog_raw,
        digital_raw=digital_raw,
        analog_config=analog_config,
        analog_wavelength_nm=617.0,
        digital_wavelength_nm=505.0,
        analog_amplifier_range="uA 1",
        digital_amplifier_range="uA 1",
        analog_area_um2=10000.0,
        digital_area_um2=10000.0,
    )

    # Verify assembly.
    assert calibration.calibration_id == "test_calibration"
    assert calibration.analog_led.led_channel == 1
    assert calibration.digital_led.led_channel == 2

    assert np.all(calibration.analog_led.photon_flux_density >= 0)
    assert calibration.digital_led.photon_flux_density_on > 0

    # Save and reload.
    filename = tmp_path / "light_calibration.h5"

    write_calibration(filename, calibration)
    loaded = read_calibration(filename)

    assert loaded.calibration_id == calibration.calibration_id
    assert loaded.created_at == calibration.created_at

    np.testing.assert_allclose(
        loaded.analog_led.command_voltage,
        calibration.analog_led.command_voltage,
    )

    np.testing.assert_allclose(
        loaded.analog_led.calibration_voltage,
        calibration.analog_led.calibration_voltage,
    )

    np.testing.assert_allclose(
        loaded.analog_led.photon_flux_density,
        calibration.analog_led.photon_flux_density,
    )

    np.testing.assert_array_equal(
        loaded.digital_led.command_state,
        calibration.digital_led.command_state,
    )

    assert np.isclose(
        loaded.digital_led.photon_flux_density_on,
        calibration.digital_led.photon_flux_density_on,
    )
