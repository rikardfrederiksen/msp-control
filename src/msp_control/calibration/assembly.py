"""Assemble processed LED measurements into calibration objects."""

from datetime import datetime

import numpy as np

from msp_control.calibration.acquisition import (
    RawCalibrationAcquisition,
    RawDigitalCalibrationAcquisition,
)
from msp_control.calibration.model import (
    AnalogLEDCalibration,
    DigitalLEDCalibration,
    LightCalibration,
)
from msp_control.calibration.photodiode import (
    PDA200C_GAINS,
    responsivity_at_wavelength,
)
from msp_control.calibration.processing import (
    process_analog_calibration,
    process_digital_calibration,
    combine_analog_ramps,
)
from msp_control.calibration.waveform import CalibrationWaveformConfig


def assemble_light_calibration(
    calibration_id: str,
    created_at: datetime,
    analog_raw: RawCalibrationAcquisition,
    digital_raw: RawDigitalCalibrationAcquisition,
    analog_config: CalibrationWaveformConfig,
    analog_wavelength_nm: float,
    digital_wavelength_nm: float,
    analog_amplifier_range: str,
    digital_amplifier_range: str,
    analog_area_um2: float,
    digital_area_um2: float,
    analog_repeats: int = 1,
    digital_repeats: int = 1,
    analog_monitor_sd: np.ndarray | None = None,
    analog_photodiode_sd: np.ndarray | None = None,
    digital_monitor_sd: np.ndarray | None = None,
    digital_photodiode_sd: np.ndarray | None = None,
) -> LightCalibration:
    """Construct a complete light calibration from raw measurements."""

    # Process analog calibration.
    ramps = process_analog_calibration(
        analog_raw,
        analog_config,
        wavelength_nm=analog_wavelength_nm,
        amplifier_range=analog_amplifier_range,
        illuminated_area_um2=analog_area_um2,
    )

    voltage, monitor, flux = combine_analog_ramps(ramps)

    # Remove negative values caused by dark-subtraction noise
    # from the final calibration curve only.
    flux = np.maximum(flux, 0.0)

    analog_led = AnalogLEDCalibration(
        led_channel=1,
        wavelength_nm=analog_wavelength_nm,
        amplifier_range=analog_amplifier_range,
        amplifier_gain=PDA200C_GAINS[analog_amplifier_range],
        photodiode_responsivity=responsivity_at_wavelength(
            analog_wavelength_nm
        ),
        illuminated_area_um2=analog_area_um2,
        time_s=analog_raw.time_s.copy(),
        command_voltage=analog_raw.command_voltage.copy(),
        monitor_voltage=analog_raw.monitor_voltage.copy(),
        photodiode_voltage=analog_raw.photodiode_voltage.copy(),
        calibration_voltage=voltage,
        calibration_monitor_voltage=monitor,
        photon_flux_density=flux,
        n_repeats=analog_repeats,
        monitor_voltage_sd=analog_monitor_sd,
        photodiode_voltage_sd=analog_photodiode_sd,
    )

    # Process digital calibration.
    monitor_on, flux_on = process_digital_calibration(
        digital_raw,
        wavelength_nm=digital_wavelength_nm,
        amplifier_range=digital_amplifier_range,
        illuminated_area_um2=digital_area_um2,
    )

    digital_led = DigitalLEDCalibration(
        led_channel=2,
        wavelength_nm=digital_wavelength_nm,
        amplifier_range=digital_amplifier_range,
        amplifier_gain=PDA200C_GAINS[digital_amplifier_range],
        photodiode_responsivity=responsivity_at_wavelength(
            digital_wavelength_nm
        ),
        illuminated_area_um2=digital_area_um2,
        time_s=digital_raw.time_s.copy(),
        command_state=digital_raw.command_state.copy(),
        monitor_voltage=digital_raw.monitor_voltage.copy(),
        photodiode_voltage=digital_raw.photodiode_voltage.copy(),
        monitor_voltage_on=monitor_on,
        photon_flux_density_on=flux_on,
        n_repeats=digital_repeats,
        monitor_voltage_sd=digital_monitor_sd,
        photodiode_voltage_sd=digital_photodiode_sd,
    )

    return LightCalibration(
        calibration_id=calibration_id,
        created_at=created_at,
        analog_led=analog_led,
        digital_led=digital_led,
    )
