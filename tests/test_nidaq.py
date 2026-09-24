from unittest.mock import MagicMock, patch

from nidaqmx.constants import (
    AcquisitionType,
    Edge,
    TerminalConfiguration,
)

from msp_control.hardware.nidaq import NIDaq, NIDaqConfig
import pytest

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
