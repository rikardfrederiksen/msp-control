import numpy as np
import pytest

from msp_control.calibration.acquisition import RawCalibrationAcquisition, RawDigitalCalibrationAcquisition
from msp_control.calibration.averaging import average_analog_acquisitions, average_digital_acquisitions


def test_average_analog_acquisitions():
    """Verify samplewise mean, SD, and repetition count."""

    time = np.array([0.0, 1.0, 2.0])
    command = np.array([0.0, 5.0, 10.0])

    acquisitions = [
        RawCalibrationAcquisition(
            time_s=time,
            command_voltage=command,
            monitor_voltage=np.array([1.0, 2.0, 3.0]),
            photodiode_voltage=np.array([10.0, 20.0, 30.0]),
        ),
        RawCalibrationAcquisition(
            time_s=time,
            command_voltage=command,
            monitor_voltage=np.array([2.0, 3.0, 4.0]),
            photodiode_voltage=np.array([20.0, 30.0, 40.0]),
        ),
        RawCalibrationAcquisition(
            time_s=time,
            command_voltage=command,
            monitor_voltage=np.array([3.0, 4.0, 5.0]),
            photodiode_voltage=np.array([30.0, 40.0, 50.0]),
        ),
    ]

    result = average_analog_acquisitions(acquisitions)

    assert result.n_repeats == 3

    np.testing.assert_array_equal(
        result.mean.time_s,
        time,
    )
    np.testing.assert_array_equal(
        result.mean.command_voltage,
        command,
    )

    np.testing.assert_allclose(
        result.mean.monitor_voltage,
        [2.0, 3.0, 4.0],
    )
    np.testing.assert_allclose(
        result.mean.photodiode_voltage,
        [20.0, 30.0, 40.0],
    )

    np.testing.assert_allclose(
        result.monitor_voltage_sd,
        [1.0, 1.0, 1.0],
    )
    np.testing.assert_allclose(
        result.photodiode_voltage_sd,
        [10.0, 10.0, 10.0],
    )


def test_average_single_analog_acquisition():
    """A single acquisition has no estimable sample SD."""

    acquisition = RawCalibrationAcquisition(
        time_s=np.array([0.0, 1.0, 2.0]),
        command_voltage=np.array([0.0, 5.0, 10.0]),
        monitor_voltage=np.array([1.0, 2.0, 3.0]),
        photodiode_voltage=np.array([10.0, 20.0, 30.0]),
    )

    result = average_analog_acquisitions([acquisition])

    assert result.n_repeats == 1
    assert result.monitor_voltage_sd is None
    assert result.photodiode_voltage_sd is None

    np.testing.assert_array_equal(
        result.mean.monitor_voltage,
        acquisition.monitor_voltage,
    )
    np.testing.assert_array_equal(
        result.mean.photodiode_voltage,
        acquisition.photodiode_voltage,
    )


def test_average_analog_mismatched_waveforms():
    """Acquisitions with different commands must not be averaged."""

    time = np.array([0.0, 1.0, 2.0])

    first = RawCalibrationAcquisition(
        time_s=time,
        command_voltage=np.array([0.0, 5.0, 10.0]),
        monitor_voltage=np.array([1.0, 2.0, 3.0]),
        photodiode_voltage=np.array([10.0, 20.0, 30.0]),
    )

    second = RawCalibrationAcquisition(
        time_s=time,
        command_voltage=np.array([0.0, 6.0, 10.0]),
        monitor_voltage=np.array([2.0, 3.0, 4.0]),
        photodiode_voltage=np.array([20.0, 30.0, 40.0]),
    )

    with pytest.raises(ValueError, match="command waveforms"):
        average_analog_acquisitions([first, second])


def test_average_digital_acquisitions():
    """Verify digital acquisition means and sample SDs."""

    time = np.array([0.0, 1.0, 2.0])
    command = np.array([0, 1, 0])

    acquisitions = [
        RawDigitalCalibrationAcquisition(
            time_s=time,
            command_state=command,
            monitor_voltage=np.array([1.0, 2.0, 3.0]),
            photodiode_voltage=np.array([10.0, 20.0, 30.0]),
        ),
        RawDigitalCalibrationAcquisition(
            time_s=time,
            command_state=command,
            monitor_voltage=np.array([2.0, 3.0, 4.0]),
            photodiode_voltage=np.array([20.0, 30.0, 40.0]),
        ),
        RawDigitalCalibrationAcquisition(
            time_s=time,
            command_state=command,
            monitor_voltage=np.array([3.0, 4.0, 5.0]),
            photodiode_voltage=np.array([30.0, 40.0, 50.0]),
        ),
    ]

    result = average_digital_acquisitions(acquisitions)

    assert result.n_repeats == 3

    np.testing.assert_array_equal(result.mean.time_s, time)
    np.testing.assert_array_equal(result.mean.command_state, command)

    np.testing.assert_allclose(
        result.mean.monitor_voltage,
        [2.0, 3.0, 4.0],
    )
    np.testing.assert_allclose(
        result.mean.photodiode_voltage,
        [20.0, 30.0, 40.0],
    )

    np.testing.assert_allclose(
        result.monitor_voltage_sd,
        [1.0, 1.0, 1.0],
    )
    np.testing.assert_allclose(
        result.photodiode_voltage_sd,
        [10.0, 10.0, 10.0],
    )


def test_average_single_digital_acquisition():
    """A single acquisition preserves the data and has no sample SD."""

    acquisition = RawDigitalCalibrationAcquisition(
        time_s=np.array([0.0, 1.0, 2.0]),
        command_state=np.array([0, 1, 0]),
        monitor_voltage=np.array([1.0, 2.0, 3.0]),
        photodiode_voltage=np.array([10.0, 20.0, 30.0]),
    )

    result = average_digital_acquisitions([acquisition])

    assert result.n_repeats == 1
    assert result.monitor_voltage_sd is None
    assert result.photodiode_voltage_sd is None

    np.testing.assert_array_equal(
        result.mean.monitor_voltage,
        acquisition.monitor_voltage,
    )
    np.testing.assert_array_equal(
        result.mean.photodiode_voltage,
        acquisition.photodiode_voltage,
    )


def test_average_digital_mismatched_commands():
    """Different digital command states must not be averaged."""

    time = np.array([0.0, 1.0, 2.0])

    first = RawDigitalCalibrationAcquisition(
        time_s=time,
        command_state=np.array([0, 1, 0]),
        monitor_voltage=np.array([1.0, 2.0, 3.0]),
        photodiode_voltage=np.array([10.0, 20.0, 30.0]),
    )

    second = RawDigitalCalibrationAcquisition(
        time_s=time,
        command_state=np.array([0, 1, 1]),
        monitor_voltage=np.array([2.0, 3.0, 4.0]),
        photodiode_voltage=np.array([20.0, 30.0, 40.0]),
    )

    with pytest.raises(ValueError, match="command states"):
        average_digital_acquisitions([first, second])
