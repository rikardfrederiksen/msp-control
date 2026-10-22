"""Average repeated LED calibration acquisitions."""

from dataclasses import dataclass

import numpy as np

from msp_control.calibration.acquisition import (
    RawCalibrationAcquisition,
    RawDigitalCalibrationAcquisition,
    )


@dataclass
class AveragedAnalogAcquisition:
    """Mean analog acquisition and samplewise variability."""

    mean: RawCalibrationAcquisition
    monitor_voltage_sd: np.ndarray | None
    photodiode_voltage_sd: np.ndarray | None
    n_repeats: int


def average_analog_acquisitions(
    acquisitions: list[RawCalibrationAcquisition],
) -> AveragedAnalogAcquisition:
    """Average repeated analog LED acquisitions."""

    if not acquisitions:
        raise ValueError("At least one acquisition is required")

    reference = acquisitions[0]

    for acquisition in acquisitions[1:]:
        if not np.array_equal(acquisition.time_s, reference.time_s):
            raise ValueError("Acquisition time axes do not match")

        if not np.array_equal(
            acquisition.command_voltage,
            reference.command_voltage,
        ):
            raise ValueError("Acquisition command waveforms do not match")

    monitor = np.stack(
        [acquisition.monitor_voltage for acquisition in acquisitions]
    )
    photodiode = np.stack(
        [acquisition.photodiode_voltage for acquisition in acquisitions]
    )

    mean = RawCalibrationAcquisition(
        time_s=reference.time_s.copy(),
        command_voltage=reference.command_voltage.copy(),
        monitor_voltage=np.mean(monitor, axis=0),
        photodiode_voltage=np.mean(photodiode, axis=0),
    )

    n_repeats = len(acquisitions)

    if n_repeats > 1:
        monitor_sd = np.std(monitor, axis=0, ddof=1)
        photodiode_sd = np.std(photodiode, axis=0, ddof=1)
    else:
        monitor_sd = None
        photodiode_sd = None

    return AveragedAnalogAcquisition(
        mean=mean,
        monitor_voltage_sd=monitor_sd,
        photodiode_voltage_sd=photodiode_sd,
        n_repeats=n_repeats,
    )


@dataclass
class AveragedDigitalAcquisition:
    """Mean digital acquisition and samplewise variability."""

    mean: RawDigitalCalibrationAcquisition
    monitor_voltage_sd: np.ndarray | None
    photodiode_voltage_sd: np.ndarray | None
    n_repeats: int


def average_digital_acquisitions(
    acquisitions: list[RawDigitalCalibrationAcquisition],
) -> AveragedDigitalAcquisition:
    """Average repeated digital LED acquisitions."""

    if not acquisitions:
        raise ValueError("At least one acquisition is required")

    reference = acquisitions[0]

    for acquisition in acquisitions[1:]:
        if not np.array_equal(acquisition.time_s, reference.time_s):
            raise ValueError("Acquisition time axes do not match")

        if not np.array_equal(
            acquisition.command_state,
            reference.command_state,
        ):
            raise ValueError("Acquisition command states do not match")

    monitor = np.stack(
        [acquisition.monitor_voltage for acquisition in acquisitions]
    )
    photodiode = np.stack(
        [acquisition.photodiode_voltage for acquisition in acquisitions]
    )

    mean = RawDigitalCalibrationAcquisition(
        time_s=reference.time_s.copy(),
        command_state=reference.command_state.copy(),
        monitor_voltage=np.mean(monitor, axis=0),
        photodiode_voltage=np.mean(photodiode, axis=0),
    )

    n_repeats = len(acquisitions)

    if n_repeats > 1:
        monitor_sd = np.std(monitor, axis=0, ddof=1)
        photodiode_sd = np.std(photodiode, axis=0, ddof=1)
    else:
        monitor_sd = None
        photodiode_sd = None

    return AveragedDigitalAcquisition(
        mean=mean,
        monitor_voltage_sd=monitor_sd,
        photodiode_voltage_sd=photodiode_sd,
        n_repeats=n_repeats,
    )
