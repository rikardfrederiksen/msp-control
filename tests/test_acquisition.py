from unittest.mock import MagicMock, call

import numpy as np

from msp_control.acquisition import AcquisitionController
from msp_control.hardware.optoscan import ScanConfig

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
