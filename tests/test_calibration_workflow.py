from datetime import datetime
from unittest.mock import Mock

import pytest
import numpy as np

from msp_control.calibration.acquisition import (
    RawCalibrationAcquisition,
    RawDigitalCalibrationAcquisition,
)
from msp_control.calibration.model import LightCalibration
from msp_control.calibration.waveform import (
    CalibrationWaveformConfig,
    BleachWaveformConfig,
    generate_calibration_waveform,
    generate_bleach_waveform,
)
from msp_control.calibration.workflow import (
    LightCalibrationConfig,
    LightCalibrationWorkflow,
)


def test_light_calibration_workflow():
    analog_config = CalibrationWaveformConfig(
        dark_before_s=0.1,
        ramp_up_s=0.2,
        hold_s=0.1,
        ramp_down_s=0.2,
        dark_after_s=0.1,
        max_voltage=5.0,
        sample_rate_hz=100.0,
    )

    digital_config = BleachWaveformConfig(
        duration_s=0.2,
        sample_rate_hz=100.0,
        pre_dark_s=0.1,
        post_dark_s=0.1,
    )

    analog_time, analog_command = generate_calibration_waveform(
        analog_config
    )

    digital_time, digital_command = generate_bleach_waveform(
        digital_config,
        control_mode="digital",
        command_voltage=None,
    )

    controller = Mock()

    controller.acquire_analog.return_value = RawCalibrationAcquisition(
        time_s=analog_time,
        command_voltage=analog_command,
        monitor_voltage=0.2 * analog_command,
        photodiode_voltage=0.05 - 0.1 * analog_command,
    )

    controller.acquire_digital.return_value = RawDigitalCalibrationAcquisition(
        time_s=digital_time,
        command_state=digital_command,
        monitor_voltage=2.0 * digital_command.astype(float),
        photodiode_voltage=0.05 - 0.5 * digital_command.astype(float),
    )

    config = LightCalibrationConfig(
        analog_waveform=analog_config,
        digital_waveform=digital_config,
        analog_wavelength_nm=617.0,
        digital_wavelength_nm=505.0,
        analog_amplifier_range="uA 1",
        digital_amplifier_range="uA 1",
        analog_area_um2=10000.0,
        digital_area_um2=10000.0,
    )

    workflow = LightCalibrationWorkflow(controller)

    created_at = datetime(2026, 10, 21, 12, 0)

    calibration = workflow.acquire(
        config=config,
        calibration_id="test_workflow",
        created_at=created_at,
    )

    controller.acquire_analog.assert_called_once_with(analog_config)
    controller.acquire_digital.assert_called_once_with(digital_config)

    assert isinstance(calibration, LightCalibration)
    assert calibration.calibration_id == "test_workflow"
    assert calibration.created_at == created_at

    assert calibration.analog_led.wavelength_nm == 617.0
    assert calibration.digital_led.wavelength_nm == 505.0

    assert len(calibration.analog_led.calibration_voltage) > 0
    assert np.all(calibration.analog_led.photon_flux_density >= 0)

    assert calibration.digital_led.photon_flux_density_on > 0




def test_analog_acquisition_failure():
    """A failed analog acquisition must prevent digital acquisition."""

    controller = Mock()
    controller.acquire_analog.side_effect = RuntimeError(
        "Analog acquisition failed"
    )

    workflow = LightCalibrationWorkflow(controller)

    config = LightCalibrationConfig(
        analog_waveform=CalibrationWaveformConfig(),
        digital_waveform=BleachWaveformConfig(duration_s=1.0),
        analog_wavelength_nm=617.0,
        digital_wavelength_nm=505.0,
        analog_amplifier_range="uA 1",
        digital_amplifier_range="uA 1",
        analog_area_um2=10000.0,
        digital_area_um2=10000.0,
    )

    with pytest.raises(RuntimeError, match="Analog acquisition failed"):
        workflow.acquire(
            config=config,
            calibration_id="failed_calibration",
            created_at=datetime(2026, 10, 21, 12, 0),
        )

    controller.acquire_analog.assert_called_once()
    controller.acquire_digital.assert_not_called()


def test_light_calibration_workflow_repeated_acquisitions():
    """Repeated acquisitions are averaged before calibration processing."""

    analog_config = CalibrationWaveformConfig(
        dark_before_s=0.1,
        ramp_up_s=0.2,
        hold_s=0.1,
        ramp_down_s=0.2,
        dark_after_s=0.1,
        max_voltage=5.0,
        sample_rate_hz=100.0,
    )

    digital_config = BleachWaveformConfig(
        duration_s=0.2,
        sample_rate_hz=100.0,
        pre_dark_s=0.1,
        post_dark_s=0.1,
    )

    analog_time, analog_command = generate_calibration_waveform(
        analog_config
    )

    digital_time, digital_command = generate_bleach_waveform(
        digital_config,
        control_mode="digital",
        command_voltage=None,
    )

    controller = Mock()

    # Three analog acquisitions with known offsets.
    controller.acquire_analog.side_effect = [
        RawCalibrationAcquisition(
            time_s=analog_time,
            command_voltage=analog_command,
            monitor_voltage=0.2 * analog_command + offset,
            photodiode_voltage=0.05 - 0.1 * analog_command + offset,
        )
        for offset in (0.0, 0.01, 0.02)
    ]

    # Two digital acquisitions with known offsets.
    controller.acquire_digital.side_effect = [
        RawDigitalCalibrationAcquisition(
            time_s=digital_time,
            command_state=digital_command,
            monitor_voltage=2.0 * digital_command.astype(float) + offset,
            photodiode_voltage=0.05 - 0.5 * digital_command.astype(float) + offset,
        )
        for offset in (0.0, 0.02)
    ]

    config = LightCalibrationConfig(
        analog_waveform=analog_config,
        digital_waveform=digital_config,
        analog_wavelength_nm=617.0,
        digital_wavelength_nm=505.0,
        analog_amplifier_range="uA 1",
        digital_amplifier_range="uA 1",
        analog_area_um2=10000.0,
        digital_area_um2=10000.0,
        analog_repeats=3,
        digital_repeats=2,
    )

    workflow = LightCalibrationWorkflow(controller)

    progress = []

    def record_progress(led, current, total):
        progress.append((led, current, total))

    calibration = workflow.acquire(
        config=config,
        calibration_id="repeated_calibration",
        created_at=datetime(2026, 10, 21, 12, 0),
        progress_callback=record_progress,
    )

    assert progress == [
        ("analog", 1, 3),
        ("analog", 2, 3),
        ("analog", 3, 3),
        ("digital", 1, 2),
        ("digital", 2, 2),
    ]
    
    # Verify the number of hardware acquisitions.
    assert controller.acquire_analog.call_count == 3
    assert controller.acquire_digital.call_count == 2

    analog = calibration.analog_led
    digital = calibration.digital_led

    # Verify repetition counts.
    assert analog.n_repeats == 3
    assert digital.n_repeats == 2

    # Verify averaged analog signals.
    np.testing.assert_allclose(
        analog.monitor_voltage,
        0.2 * analog_command + 0.01,
    )

    np.testing.assert_allclose(
        analog.photodiode_voltage,
        0.05 - 0.1 * analog_command + 0.01,
    )

    # Three offsets: 0, 0.01, 0.02 -> sample SD = 0.01.
    np.testing.assert_allclose(
        analog.monitor_voltage_sd,
        0.01,
    )

    np.testing.assert_allclose(
        analog.photodiode_voltage_sd,
        0.01,
    )

    # Verify averaged digital signals.
    np.testing.assert_allclose(
        digital.monitor_voltage,
        2.0 * digital_command.astype(float) + 0.01,
    )

    np.testing.assert_allclose(
        digital.photodiode_voltage,
        0.05 - 0.5 * digital_command.astype(float) + 0.01,
    )

    # Two offsets: 0, 0.02 -> sample SD = 0.02 / sqrt(2).
    expected_digital_sd = 0.02 / np.sqrt(2)

    np.testing.assert_allclose(
        digital.monitor_voltage_sd,
        expected_digital_sd,
    )

    np.testing.assert_allclose(
        digital.photodiode_voltage_sd,
        expected_digital_sd,
    )

    # Confirm that calibration processing still produces valid results.
    assert len(analog.calibration_voltage) > 0
    assert np.all(analog.photon_flux_density >= 0)
    assert digital.photon_flux_density_on > 0


def test_calibration_progress_callback():
    """Progress callback reports each acquisition in order."""

    controller = Mock()

    config = LightCalibrationConfig(
        analog_waveform=CalibrationWaveformConfig(),
        digital_waveform=BleachWaveformConfig(duration_s=1.0),
        analog_wavelength_nm=617.0,
        digital_wavelength_nm=505.0,
        analog_amplifier_range="uA 1",
        digital_amplifier_range="uA 1",
        analog_area_um2=10000.0,
        digital_area_um2=10000.0,
        analog_repeats=3,
        digital_repeats=2,
    )

    progress = []

    def record_progress(led, current, total):
        progress.append((led, current, total))

    # Stop after the first acquisition attempt. We only want to
    # verify that the callback runs before hardware acquisition.
    controller.acquire_analog.side_effect = RuntimeError("Test stop")

    workflow = LightCalibrationWorkflow(controller)

    with pytest.raises(RuntimeError, match="Test stop"):
        workflow.acquire(
            config=config,
            calibration_id="progress_test",
            created_at=datetime(2026, 10, 21, 12, 0),
            progress_callback=record_progress,
        )

    assert progress == [
        ("analog", 1, 3),
    ]
