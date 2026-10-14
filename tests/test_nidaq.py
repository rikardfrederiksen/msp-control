from unittest.mock import MagicMock, patch

from nidaqmx.constants import (
    AcquisitionType,
    Edge,
    TerminalConfiguration,
)

from msp_control.hardware.nidaq import NIDaq, NIDaqConfig
import pytest

import numpy as np

def test_nidaq_config():
    config = NIDaqConfig()

    assert config.pmt_physical_channel == "Dev1/ai0"
    assert config.sample_clock_source == "/Dev1/PFI1"
    assert config.start_trigger_source == "/Dev1/PFI2"
    assert config.shutter_physical_channel == "Dev1/port0/line0"
    assert config.ir_physical_channel == "Dev1/port0/line1"

    assert config.pmt_min_v == 0.0
    assert config.pmt_max_v == 5.0
    assert config.nominal_sample_rate == 10_000.0
    assert config.read_timeout == 10.0

def test_configure_acquisition():
    config = NIDaqConfig()

    with patch("msp_control.hardware.nidaq.nidaqmx.Task") as task_class:
        task = MagicMock()
        task_class.return_value = task

        daq = NIDaq(config)

        daq.configure_ai()
        daq.configure_timing(1448)
        daq.configure_start_trigger()

        task.ai_channels.add_ai_voltage_chan.assert_called_once_with(
            "Dev1/ai0",
            terminal_config=TerminalConfiguration.DIFF,
            min_val=0.0,
            max_val=5.0,
        )

        task.timing.cfg_samp_clk_timing.assert_called_once_with(
            rate=10_000.0,
            source="/Dev1/PFI1",
            active_edge=Edge.RISING,
            sample_mode=AcquisitionType.FINITE,
            samps_per_chan=1448,
        )

        task.triggers.start_trigger.cfg_dig_edge_start_trig.assert_called_once_with(
            trigger_source="/Dev1/PFI2",
            trigger_edge=Edge.RISING,
        )

        daq.close()

        task.close.assert_called_once()
        assert daq._ai_task is None

def test_configure_timing_requires_ai():
    daq = NIDaq(NIDaqConfig())

    try:
        daq.configure_timing(1448)
    except RuntimeError:
        pass
    else:
        raise AssertionError("Expected RuntimeError")


def test_configure_timing_rejects_invalid_sample_count():
    with patch("msp_control.hardware.nidaq.nidaqmx.Task"):
        daq = NIDaq(NIDaqConfig())
        daq.configure_ai()

        try:
            daq.configure_timing(0)
        except ValueError:
            pass
        else:
            raise AssertionError("Expected ValueError")

def test_configure_acquisition():
    config = NIDaqConfig()

    with patch("msp_control.hardware.nidaq.nidaqmx.Task") as task_class:
        task = MagicMock()
        task_class.return_value = task

        daq = NIDaq(config)

        daq.configure_ai()
        daq.configure_timing(1448)
        daq.configure_start_trigger()

        task.ai_channels.add_ai_voltage_chan.assert_called_once_with(
            "Dev1/ai0",
            terminal_config=TerminalConfiguration.DIFF,
            min_val=0.0,
            max_val=5.0,
        )

        task.timing.cfg_samp_clk_timing.assert_called_once_with(
            rate=10_000.0,
            source="/Dev1/PFI1",
            active_edge=Edge.RISING,
            sample_mode=AcquisitionType.FINITE,
            samps_per_chan=1448,
        )

        task.triggers.start_trigger.cfg_dig_edge_start_trig.assert_called_once_with(
            trigger_source="/Dev1/PFI2",
            trigger_edge=Edge.RISING,
        )

        daq.close()

        task.close.assert_called_once()
        assert daq._ai_task is None

def test_configure_timing_requires_ai():
    daq = NIDaq(NIDaqConfig())

    with pytest.raises(RuntimeError):
        daq.configure_timing(1448)


def test_configure_timing_rejects_invalid_sample_count():
    with patch("msp_control.hardware.nidaq.nidaqmx.Task"):
        daq = NIDaq(NIDaqConfig())
        daq.configure_ai()

        with pytest.raises(ValueError):
            daq.configure_timing(0)

        daq.close()

def test_start():
    with patch("msp_control.hardware.nidaq.nidaqmx.Task") as task_class:
        task = MagicMock()
        task_class.return_value = task

        daq = NIDaq(NIDaqConfig())
        daq.configure_ai()
        daq.start()

        task.start.assert_called_once_with()

        daq.close()


def test_read_sweep():
    with patch("msp_control.hardware.nidaq.nidaqmx.Task") as task_class:
        task = MagicMock()
        task_class.return_value = task
        task.read.return_value = [0.1, 0.2, 0.3]

        daq = NIDaq(NIDaqConfig())
        daq.configure_ai()

        data = daq.read_sweep(3)

        task.read.assert_called_once_with(
            number_of_samples_per_channel=3,
            timeout=10.0,
        )

        assert data == [0.1, 0.2, 0.3]

        daq.close()

def test_start_requires_ai():
    daq = NIDaq(NIDaqConfig())

    with pytest.raises(RuntimeError):
        daq.start()


def test_read_sweep_requires_ai():
    daq = NIDaq(NIDaqConfig())

    with pytest.raises(RuntimeError):
        daq.read_sweep(181)

def test_shutter_control():
    daq = NIDaq(NIDaqConfig())

    with patch.object(daq, "_write_digital_line") as write_line:
        daq.open_shutter()
        write_line.assert_called_with(
            "Dev1/port0/line0",
            True,
        )

        daq.close_shutter()
        write_line.assert_called_with(
            "Dev1/port0/line0",
            False,
        )


def test_ir_control():
    daq = NIDaq(NIDaqConfig())

    with patch.object(daq, "_write_digital_line") as write_line:
        daq.ir_on()
        write_line.assert_called_with(
            "Dev1/port0/line1",
            True,
        )

        daq.ir_off()
        write_line.assert_called_with(
            "Dev1/port0/line1",
            False,
        )

def test_write_digital_line():
    with patch("msp_control.hardware.nidaq.nidaqmx.Task") as task_class:
        task = MagicMock()
        task_class.return_value.__enter__.return_value = task

        daq = NIDaq(NIDaqConfig())
        daq._write_digital_line("Dev1/port0/line0", True)

        task.do_channels.add_do_chan.assert_called_once_with(
            "Dev1/port0/line0"
        )
        task.write.assert_called_once_with(True)


def test_configure_bleach_digital_output():
    config = NIDaqConfig()

    waveform = np.array(
        [False, False, True, True, True, False],
        dtype=bool,
    )

    with patch("msp_control.hardware.nidaq.nidaqmx.Task") as task_class:
        task = MagicMock()
        task_class.return_value = task

        daq = NIDaq(config)

        daq.configure_bleach_digital_output(
            waveform=waveform,
            sample_rate_hz=1000.0,
        )

        task.do_channels.add_do_chan.assert_called_once_with(
            "Dev1/port0/line2",
        )

        task.timing.cfg_samp_clk_timing.assert_called_once_with(
            rate=1000.0,
            source="/Dev1/ai/SampleClock",
            active_edge=Edge.RISING,
            sample_mode=AcquisitionType.FINITE,
            samps_per_chan=6,
        )


        task.write.assert_called_once_with(
           [False, False, True, True, True, False],
           auto_start=False,
        )

        task.start.assert_not_called()

        daq.close()
       
        task.close.assert_called_once()


def test_configure_bleach_monitor():
    config = NIDaqConfig()

    with patch("msp_control.hardware.nidaq.nidaqmx.Task") as task_class:
        task = MagicMock()
        task_class.return_value = task

        daq = NIDaq(config)

        daq.configure_bleach_monitor(
            led_channel=2,
            samples=1000,
            sample_rate_hz=1000.0,
        )

        task.ai_channels.add_ai_voltage_chan.assert_called_once_with(
            "Dev1/ai6",
            terminal_config=TerminalConfiguration.DIFF,
        )

        task.timing.cfg_samp_clk_timing.assert_called_once_with(
            rate=1000.0,
            sample_mode=AcquisitionType.FINITE,
            samps_per_chan=1000,
        )

        task.start.assert_not_called()

        daq.close()

        task.close.assert_called_once()


def test_start_bleach_tasks():
    daq = NIDaq(NIDaqConfig())

    daq._bleach_do_task = MagicMock()
    daq._bleach_ai_task = MagicMock()

    calls = []

    daq._bleach_do_task.start.side_effect = (
        lambda: calls.append("digital")
    )
    daq._bleach_ai_task.start.side_effect = (
        lambda: calls.append("analog")
    )

    daq.start_bleach_tasks()

    assert calls == ["digital", "analog"]

    daq._bleach_do_task.start.assert_called_once_with()
    daq._bleach_ai_task.start.assert_called_once_with()


def test_read_bleach_monitor():
    daq = NIDaq(NIDaqConfig())

    daq._bleach_ai_task = MagicMock()
    daq._bleach_ai_task.read.return_value = [
        0.0, 0.1, 2.5, 2.5, 0.0
    ]

    data = daq.read_bleach_monitor(
        samples=5,
        timeout=2.0,
    )

    daq._bleach_ai_task.read.assert_called_once_with(
        number_of_samples_per_channel=5,
        timeout=2.0,
    )

    np.testing.assert_array_equal(
        data,
        [0.0, 0.1, 2.5, 2.5, 0.0],
    )


def test_wait_for_bleach_output():
    daq = NIDaq(NIDaqConfig())
    daq._bleach_do_task = MagicMock()

    daq.wait_for_bleach_output(timeout=12.0)

    daq._bleach_do_task.wait_until_done.assert_called_once_with(
        timeout=12.0,
    )


def test_wait_for_bleach_output_requires_task():
    daq = NIDaq(NIDaqConfig())

    with pytest.raises(
        RuntimeError,
        match="Bleach output is not configured",
    ):
        daq.wait_for_bleach_output(timeout=2.0)


def test_stop_bleach_digital_output():
    daq = NIDaq(NIDaqConfig())
    daq._bleach_do_task = MagicMock()

    calls = []

    daq._bleach_do_task.stop.side_effect = (
        lambda: calls.append("stop")
    )
    daq._bleach_do_task.close.side_effect = (
        lambda: calls.append("close")
    )

    with patch.object(daq, "_write_digital_line") as write_line:
        write_line.side_effect = (
            lambda channel, value: calls.append("off")
        )

        daq.stop_bleach_digital_output()

        write_line.assert_called_once_with(
            "Dev1/port0/line2",
            False,
        )

    assert calls == ["stop", "close", "off"]
    assert daq._bleach_do_task is None


def test_stop_bleach_digital_output_when_stop_fails():
    daq = NIDaq(NIDaqConfig())
    daq._bleach_do_task = MagicMock()

    task = daq._bleach_do_task
    task.stop.side_effect = RuntimeError("DAQ stop failed")

    with patch.object(daq, "_write_digital_line") as write_line:
        with pytest.raises(RuntimeError, match="DAQ stop failed"):
            daq.stop_bleach_digital_output()

        task.close.assert_called_once()

        write_line.assert_called_once_with(
            "Dev1/port0/line2",
            False,
        )

    assert daq._bleach_do_task is None


def test_configure_bleach_analog_output():
    config = NIDaqConfig(device="Dev1")
    daq = NIDaq(config)

    waveform = np.array([0.0, 2.5, 5.0, 2.5, 0.0])

    with patch("msp_control.hardware.nidaq.nidaqmx.Task") as task_class:
        task = task_class.return_value

        daq.configure_bleach_analog_output(
            waveform=waveform,
            sample_rate_hz=1000.0,
        )

        task.ao_channels.add_ao_voltage_chan.assert_called_once_with(
            "Dev1/ao1",
            min_val=0.0,
            max_val=10.0,
        )

        task.timing.cfg_samp_clk_timing.assert_called_once_with(
            rate=1000.0,
            source="/Dev1/ai/SampleClock",
            active_edge=Edge.RISING,
            sample_mode=AcquisitionType.FINITE,
            samps_per_chan=5,
        )

        task.write.assert_called_once_with(
            waveform.tolist(),
            auto_start=False,
        )


def test_start_bleach_tasks_with_analog_output():
    daq = NIDaq(NIDaqConfig(device="Dev1"))

    analog_task = MagicMock()
    monitor_task = MagicMock()

    daq._bleach_ao_task = analog_task
    daq._bleach_ai_task = monitor_task

    daq.start_bleach_tasks()

    analog_task.start.assert_called_once_with()
    monitor_task.start.assert_called_once_with()


def test_wait_for_bleach_analog_output():
    daq = NIDaq(NIDaqConfig(device="Dev1"))

    analog_task = MagicMock()
    daq._bleach_ao_task = analog_task

    daq.wait_for_bleach_output(timeout=2.0)

    analog_task.wait_until_done.assert_called_once_with(
        timeout=2.0
    )


def test_stop_bleach_analog_output():
    daq = NIDaq(NIDaqConfig(device="Dev1"))

    analog_task = MagicMock()
    daq._bleach_ao_task = analog_task

    with patch("msp_control.hardware.nidaq.nidaqmx.Task") as task_class:
        zero_task = task_class.return_value.__enter__.return_value

        daq.stop_bleach_analog_output()

        analog_task.stop.assert_called_once_with()
        analog_task.close.assert_called_once_with()

        zero_task.ao_channels.add_ao_voltage_chan.assert_called_once_with(
            "Dev1/ao1",
            min_val=0.0,
            max_val=10.0,
        )

        zero_task.write.assert_called_once_with(0.0)

        assert daq._bleach_ao_task is None


def test_stop_bleach_analog_output_when_stop_fails():
    daq = NIDaq(NIDaqConfig(device="Dev1"))

    analog_task = MagicMock()
    analog_task.stop.side_effect = RuntimeError("Stop failed")
    daq._bleach_ao_task = analog_task

    with patch("msp_control.hardware.nidaq.nidaqmx.Task") as task_class:
        zero_task = task_class.return_value.__enter__.return_value

        with pytest.raises(RuntimeError, match="Stop failed"):
            daq.stop_bleach_analog_output()

        # The original waveform task must still be closed.
        analog_task.close.assert_called_once_with()

        # The analog output must still be explicitly set to 0 V.
        zero_task.ao_channels.add_ao_voltage_chan.assert_called_once_with(
            "Dev1/ao1",
            min_val=0.0,
            max_val=10.0,
        )
        zero_task.write.assert_called_once_with(0.0)

        # The task reference must be cleared.
        assert daq._bleach_ao_task is None


def test_cannot_configure_analog_and_digital_bleach_outputs():
    daq = NIDaq(NIDaqConfig(device="Dev1"))

    daq._bleach_do_task = MagicMock()

    analog_waveform = np.array(
        [0.0, 2.5, 5.0, 2.5, 0.0]
    )

    with patch("msp_control.hardware.nidaq.nidaqmx.Task") as task_class:
        with pytest.raises(
            RuntimeError,
            match="Bleach output is already configured",
        ):
            daq.configure_bleach_analog_output(
                waveform=analog_waveform,
                sample_rate_hz=1000.0,
            )

        # Configuration should be rejected before creating a task.
        task_class.assert_not_called()

    daq._bleach_do_task.stop.assert_not_called()
    daq._bleach_do_task.close.assert_not_called()
    assert daq._bleach_ao_task is None


def test_cannot_configure_digital_bleach_when_analog_exists():
    daq = NIDaq(NIDaqConfig(device="Dev1"))

    # Simulate an existing analog output task.
    daq._bleach_ao_task = MagicMock()

    digital_waveform = np.array(
        [False, True, True, False],
        dtype=bool,
    )

    with patch("msp_control.hardware.nidaq.nidaqmx.Task") as task_class:
        with pytest.raises(
            RuntimeError,
            match="Bleach output is already configured",
        ):
            daq.configure_bleach_digital_output(
                waveform=digital_waveform,
                sample_rate_hz=1000.0,
            )

        # Reject the operation before creating another task.
        task_class.assert_not_called()

    # The existing analog task must remain untouched.
    daq._bleach_ao_task.stop.assert_not_called()
    daq._bleach_ao_task.close.assert_not_called()
    assert daq._bleach_do_task is None
