from unittest.mock import MagicMock, call

import numpy as np

from msp_control.acquisition import AcquisitionController
from msp_control.config import ScanConfig

import pytest

def test_acquire():
    daq = MagicMock()
    optoscan = MagicMock()

    config = ScanConfig(
        start_nm=360,
        end_nm=720,
        step_nm=2,
        step_time_ms=2,
        input_slit_nm=4,
        output_slit_nm=4,
        dark_scans=2,
        data_scans=5,
    )

    sweeps = [
        [float(i)] * config.wavelength_points
        for i in range(config.cycles)
    ]

    daq.read_sweep.side_effect = sweeps

    controller = AcquisitionController(
        daq=daq,
        optoscan=optoscan,
    )

    result = controller.acquire(config)

    assert len(result.dark) == 2
    assert len(result.data) == 5
    assert result.transition.shape == (181,)

    np.testing.assert_array_equal(
        result.dark[0],
        np.zeros(181),
    )

    np.testing.assert_array_equal(
        result.dark[1],
        np.ones(181),
    )

    np.testing.assert_array_equal(
        result.transition,
        np.full(181, 2.0),
    )

    np.testing.assert_array_equal(
        result.data[0],
        np.full(181, 3.0),
    )

    np.testing.assert_array_equal(
        result.data[-1],
        np.full(181, 7.0),
    )

def test_acquire_hardware_sequence():
    hardware = MagicMock()

    daq = hardware.daq
    optoscan = hardware.optoscan

    config = ScanConfig(
        start_nm=360,
        end_nm=720,
        step_nm=2,
        step_time_ms=2,
        input_slit_nm=4,
        output_slit_nm=4,
        dark_scans=2,
        data_scans=5,
    )

    daq.read_sweep.side_effect = [
        [float(i)] * config.wavelength_points
        for i in range(config.cycles)
    ]

    controller = AcquisitionController(
        daq=daq,
        optoscan=optoscan,
    )

    controller.acquire(config)

    assert hardware.mock_calls == [
        call.daq.close_shutter(),
        call.daq.ir_off(),

        call.optoscan.configure_scan(config),

        call.daq.configure_ai(),
        call.daq.configure_timing(config.samples),
        call.daq.configure_start_trigger(),
        call.daq.start(),

        call.optoscan.start_scan(),

        call.daq.read_sweep(config.wavelength_points),
        call.daq.read_sweep(config.wavelength_points),

        call.daq.open_shutter(),

        call.daq.read_sweep(config.wavelength_points),
        call.daq.read_sweep(config.wavelength_points),
        call.daq.read_sweep(config.wavelength_points),
        call.daq.read_sweep(config.wavelength_points),
        call.daq.read_sweep(config.wavelength_points),
        call.daq.read_sweep(config.wavelength_points),

        call.optoscan.wait_for_scan_complete(),

        call.daq.close_shutter(),
        call.daq.ir_on(),
        call.daq.close(),
    ]
def test_acquire_restores_hardware_after_failure():
    daq = MagicMock()
    optoscan = MagicMock()

    config = ScanConfig(
        start_nm=360,
        end_nm=720,
        step_nm=2,
        step_time_ms=2,
        input_slit_nm=4,
        output_slit_nm=4,
        dark_scans=2,
        data_scans=5,
    )

    # Simulate a DAQ failure during acquisition.
    daq.read_sweep.side_effect = RuntimeError("DAQ failure")

    controller = AcquisitionController(
        daq=daq,
        optoscan=optoscan,
    )

    with pytest.raises(RuntimeError, match="DAQ failure"):
        controller.acquire(config)

    daq.close_shutter.assert_called()
    daq.ir_on.assert_called_once()
    daq.close.assert_called_once()

    assert daq.mock_calls[-3:] == [
        call.close_shutter(),
        call.ir_on(),
        call.close(),
    ]

def make_config():
    return ScanConfig(
        start_nm=360,
        end_nm=720,
        step_nm=2,
        step_time_ms=2,
        input_slit_nm=4,
        output_slit_nm=4,
        dark_scans=2,
        data_scans=5,
    )


def test_acquire():
    config = make_config()

    daq = MagicMock()
    optoscan = MagicMock()

    sweeps = [
        [0.0] * 181,  # dark 0
        [0.1] * 181,  # dark 1
        [0.2] * 181,  # transition
        [1.0] * 181,  # data 0
        [1.1] * 181,  # data 1
        [1.2] * 181,  # data 2
        [1.3] * 181,  # data 3
        [1.4] * 181,  # data 4
    ]

    daq.read_sweep.side_effect = sweeps

    controller = AcquisitionController(daq, optoscan)
    result = controller.acquire(config)

    assert len(result.dark) == 2
    assert len(result.data) == 5

    np.testing.assert_allclose(result.dark[0], 0.0)
    np.testing.assert_allclose(result.dark[1], 0.1)
    np.testing.assert_allclose(result.transition, 0.2)
    np.testing.assert_allclose(result.data[0], 1.0)
    np.testing.assert_allclose(result.data[4], 1.4)

    daq.configure_timing.assert_called_once_with(1448)
    daq.read_sweep.assert_has_calls([call(181)] * 8)
    assert daq.read_sweep.call_count == 8

    daq.open_shutter.assert_called_once_with()

    optoscan.configure_scan.assert_called_once_with(config)
    optoscan.start_scan.assert_called_once_with()
    optoscan.wait_for_scan_complete.assert_called_once_with()

    daq.close.assert_called_once_with()
    daq.ir_off.assert_called_once_with()
    daq.ir_on.assert_called_once_with()

def test_acquire_restores_safe_state_after_read_failure():
    config = make_config()

    daq = MagicMock()
    optoscan = MagicMock()

    daq.read_sweep.side_effect = [
        [0.0] * 181,
        [0.1] * 181,
        [0.2] * 181,
        RuntimeError("DAQ read failed"),
    ]

    controller = AcquisitionController(daq, optoscan)

    with pytest.raises(RuntimeError, match="DAQ read failed"):
        controller.acquire(config)

    assert daq.close_shutter.call_count == 2
    daq.ir_on.assert_called_once_with()
    daq.close.assert_called_once_with()
