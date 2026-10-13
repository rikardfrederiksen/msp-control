from unittest.mock import MagicMock

import numpy as np
import pytest

from msp_control.bleach import BleachController
from msp_control.calibration.waveform import BleachWaveformConfig


def test_digital_bleach_acquisition():
    daq = MagicMock()
    controller = BleachController(daq)

    config = BleachWaveformConfig(
        duration_s=1.0,
        sample_rate_hz=1000.0,
        pre_dark_s=0.1,
        post_dark_s=0.05,
    )

    daq.read_bleach_monitor.return_value = np.zeros(1150)

    result = controller.acquire(
        led_channel=2,
        config=config,
        command_voltage=None,
    )

    daq.configure_bleach_digital_output.assert_called_once()
    daq.configure_bleach_monitor.assert_called_once_with(
        led_channel=2,
        samples=1150,
        sample_rate_hz=1000.0,
    )

    daq.start_bleach_tasks.assert_called_once()

    daq.read_bleach_monitor.assert_called_once()

    daq.wait_for_bleach_output.assert_called_once()

    daq.stop_bleach_digital_output.assert_called_once()

    assert len(result.time_s) == 1150
    np.testing.assert_array_equal(
        result.monitor_voltage,
        np.zeros(1150),
    )


def test_digital_bleach_cleanup_on_monitor_failure():
    daq = MagicMock()
    controller = BleachController(daq)

    config = BleachWaveformConfig(
        duration_s=1.0,
        sample_rate_hz=1000.0,
        pre_dark_s=0.1,
        post_dark_s=0.05,
    )

    daq.read_bleach_monitor.side_effect = RuntimeError(
        "Monitor acquisition failed"
    )

    with pytest.raises(
        RuntimeError,
        match="Monitor acquisition failed",
    ):
        controller.acquire(
            led_channel=2,
            config=config,
            command_voltage=None,
        )

    daq.stop_bleach_digital_output.assert_called_once()
    daq.close.assert_called_once()

    # Do not wait for normal completion after acquisition fails.
    daq.wait_for_bleach_output.assert_not_called()

def test_digital_bleach_cleanup_on_start_failure():
    daq = MagicMock()
    controller = BleachController(daq)

    config = BleachWaveformConfig(
        duration_s=1.0,
        sample_rate_hz=1000.0,
        pre_dark_s=0.1,
        post_dark_s=0.05,
    )

    daq.start_bleach_tasks.side_effect = RuntimeError(
        "Bleach task startup failed"
    )

    with pytest.raises(
        RuntimeError,
        match="Bleach task startup failed",
    ):
        controller.acquire(
            led_channel=2,
            config=config,
            command_voltage=None,
        )

    daq.stop_bleach_digital_output.assert_called_once()
    daq.close.assert_called_once()

    daq.read_bleach_monitor.assert_not_called()
    daq.wait_for_bleach_output.assert_not_called()
