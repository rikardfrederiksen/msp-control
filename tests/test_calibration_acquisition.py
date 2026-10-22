from unittest.mock import Mock, call

import numpy as np
import pytest

from msp_control.calibration.acquisition import CalibrationController
from msp_control.calibration.waveform import CalibrationWaveformConfig, BleachWaveformConfig


def test_acquire_analog():
    """Analog calibration acquires two synchronized channels."""

    daq = Mock()
    controller = CalibrationController(daq)

    config = CalibrationWaveformConfig(
        dark_before_s=0.1,
        ramp_up_s=0.2,
        hold_s=0.1,
        ramp_down_s=0.2,
        dark_after_s=0.1,
        max_voltage=5.0,
        sample_rate_hz=100.0,
    )

    samples = 70

    monitor = np.linspace(0, 1, samples)
    photodiode = np.linspace(0, 2, samples)

    daq.read_bleach_monitor.return_value = np.vstack(
        [monitor, photodiode]
    )

    result = controller.acquire_analog(config)

    daq.configure_bleach_analog_output.assert_called_once()

    daq.configure_bleach_monitor.assert_called_once_with(
        led_channel=1,
        samples=samples,
        sample_rate_hz=100.0,
        include_photodiode=True,
    )

    daq.start_bleach_tasks.assert_called_once()
    daq.wait_for_bleach_output.assert_called_once()

    np.testing.assert_array_equal(
        result.monitor_voltage, monitor
    )
    np.testing.assert_array_equal(
        result.photodiode_voltage, photodiode
    )

    assert len(result.time_s) == samples
    assert len(result.command_voltage) == samples
    assert result.command_voltage.max() == 5.0

    daq.stop_bleach_analog_output.assert_called_once()
    daq.close.assert_called_once()


def test_acquire_analog_failure_cleanup():
    """The LED must be switched off if acquisition fails."""

    daq = Mock()
    controller = CalibrationController(daq)

    config = CalibrationWaveformConfig()

    daq.read_bleach_monitor.side_effect = RuntimeError(
        "Simulated DAQ failure"
    )

    with pytest.raises(RuntimeError, match="Simulated DAQ failure"):
        controller.acquire_analog(config)

    daq.stop_bleach_analog_output.assert_called_once()
    daq.close.assert_called_once()


def test_acquire_analog_operation_order():
    """Verify DAQ configuration, acquisition, and shutdown order."""

    daq = Mock()
    controller = CalibrationController(daq)

    config = CalibrationWaveformConfig(
        dark_before_s=0.1,
        ramp_up_s=0.2,
        hold_s=0.1,
        ramp_down_s=0.2,
        dark_after_s=0.1,
        sample_rate_hz=100.0,
    )

    samples = 70
    daq.read_bleach_monitor.return_value = np.zeros((2, samples))

    controller.acquire_analog(config)

    expected = [
        call.configure_bleach_analog_output(
            waveform=daq.configure_bleach_analog_output.call_args.kwargs[
                "waveform"
            ],
            sample_rate_hz=100.0,
        ),
        call.configure_bleach_monitor(
            led_channel=1,
            samples=samples,
            sample_rate_hz=100.0,
            include_photodiode=True,
        ),
        call.start_bleach_tasks(),
        call.read_bleach_monitor(
            samples=samples,
            timeout=5.7,
        ),
        call.wait_for_bleach_output(timeout=5.7),
        call.stop_bleach_analog_output(),
        call.close(),
    ]

    assert [c[0] for c in daq.mock_calls] == [
        c[0] for c in expected
    ]


def test_acquire_digital():
    """Digital calibration acquires two synchronized channels."""

    daq = Mock()
    controller = CalibrationController(daq)

    config = BleachWaveformConfig(
        duration_s=0.5,
        pre_dark_s=0.1,
        post_dark_s=0.1,
        sample_rate_hz=100.0,
    )

    samples = 70

    monitor = np.linspace(0, 1, samples)
    photodiode = np.linspace(0, 2, samples)

    daq.read_bleach_monitor.return_value = np.vstack(
        [monitor, photodiode]
    )

    result = controller.acquire_digital(config)

    daq.configure_bleach_digital_output.assert_called_once()

    daq.configure_bleach_monitor.assert_called_once_with(
        led_channel=2,
        samples=samples,
        sample_rate_hz=100.0,
        include_photodiode=True,
    )

    daq.start_bleach_tasks.assert_called_once()
    daq.wait_for_bleach_output.assert_called_once()

    np.testing.assert_array_equal(
        result.monitor_voltage, monitor
    )
    np.testing.assert_array_equal(
        result.photodiode_voltage, photodiode
    )

    assert len(result.time_s) == samples
    assert result.command_state.dtype == np.bool_
    assert np.count_nonzero(result.command_state) == 50
    assert not result.command_state[0]
    assert not result.command_state[-1]

    daq.stop_bleach_digital_output.assert_called_once()
    daq.close.assert_called_once()


def test_acquire_digital_failure_cleanup():
    """The digital LED must be switched off if acquisition fails."""

    daq = Mock()
    controller = CalibrationController(daq)

    config = BleachWaveformConfig(duration_s=0.5)

    daq.read_bleach_monitor.side_effect = RuntimeError(
        "Simulated digital DAQ failure"
    )

    with pytest.raises(RuntimeError, match="Simulated digital DAQ failure"):
        controller.acquire_digital(config)

    daq.stop_bleach_digital_output.assert_called_once()
    daq.close.assert_called_once()


def test_sequential_analog_digital_acquisitions():
    """The same DAQ can be used for successive LED calibrations."""

    daq = Mock()
    controller = CalibrationController(daq)

    analog_config = CalibrationWaveformConfig(
        dark_before_s=0.1,
        ramp_up_s=0.2,
        hold_s=0.1,
        ramp_down_s=0.2,
        dark_after_s=0.1,
        sample_rate_hz=100.0,
    )

    digital_config = BleachWaveformConfig(
        duration_s=0.5,
        pre_dark_s=0.1,
        post_dark_s=0.1,
        sample_rate_hz=100.0,
    )

    # Both waveforms contain 70 samples.
    daq.read_bleach_monitor.return_value = np.zeros((2, 70))

    controller.acquire_analog(analog_config)
    controller.acquire_digital(digital_config)

    # Both acquisitions use the same DAQ instance.
    assert daq.configure_bleach_analog_output.call_count == 1
    assert daq.configure_bleach_digital_output.call_count == 1

    # Each acquisition performs its own cleanup.
    daq.stop_bleach_analog_output.assert_called_once()
    daq.stop_bleach_digital_output.assert_called_once()
    assert daq.close.call_count == 2

    # Verify that analog cleanup completes before digital configuration.
    calls = [c[0] for c in daq.mock_calls]

    assert calls.index("stop_bleach_analog_output") < calls.index(
        "configure_bleach_digital_output"
    )

    # Verify that each acquisition closes the DAQ after stopping its output.
    assert calls == [
        "configure_bleach_analog_output",
        "configure_bleach_monitor",
        "start_bleach_tasks",
        "read_bleach_monitor",
        "wait_for_bleach_output",
        "stop_bleach_analog_output",
        "close",
        "configure_bleach_digital_output",
        "configure_bleach_monitor",
        "start_bleach_tasks",
        "read_bleach_monitor",
        "wait_for_bleach_output",
        "stop_bleach_digital_output",
        "close",
    ]
