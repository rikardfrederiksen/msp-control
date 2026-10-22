"""Coordinate acquisition and processing of LED calibrations."""

from dataclasses import dataclass
from datetime import datetime

from collections.abc import Callable

from msp_control.calibration.acquisition import CalibrationController
from msp_control.calibration.assembly import assemble_light_calibration
from msp_control.calibration.model import LightCalibration
from msp_control.calibration.waveform import (
    CalibrationWaveformConfig,
    BleachWaveformConfig,
)
from msp_control.calibration.averaging import (
    average_analog_acquisitions,
    average_digital_acquisitions,
)

@dataclass(frozen=True)
class LightCalibrationConfig:
    """Parameters for a complete light calibration."""

    analog_waveform: CalibrationWaveformConfig
    digital_waveform: BleachWaveformConfig

    analog_wavelength_nm: float
    digital_wavelength_nm: float

    analog_amplifier_range: str
    digital_amplifier_range: str

    analog_area_um2: float
    digital_area_um2: float

    analog_repeats: int = 1
    digital_repeats: int = 1

    def __post_init__(self):
        if self.analog_repeats < 1:
            raise ValueError("analog_repeats must be at least 1")

        if self.digital_repeats < 1:
            raise ValueError("digital_repeats must be at least 1")


class LightCalibrationWorkflow:
    """Acquire and assemble a complete calibration."""

    def __init__(self, controller: CalibrationController):
        self.controller = controller

    def acquire(
        self,
        config: LightCalibrationConfig,
        calibration_id: str,
        created_at: datetime,
        progress_callback: Callable[[str, int, int], None] | None = None,
    ) -> LightCalibration:
        """Acquire, average, and process both LED calibrations."""

        analog_acquisitions = []

        for i in range(config.analog_repeats):
            if progress_callback is not None:
                progress_callback("analog", i + 1, config.analog_repeats)

            analog_acquisitions.append(
                self.controller.acquire_analog(config.analog_waveform)
            )

        analog_average = average_analog_acquisitions(
            analog_acquisitions
        )

        # Acquire repeated digital LED measurements.
        digital_acquisitions = []

        for i in range(config.digital_repeats):
            if progress_callback is not None:
                progress_callback("digital", i + 1, config.digital_repeats)

            digital_acquisitions.append(
                self.controller.acquire_digital(config.digital_waveform)
            )

        digital_average = average_digital_acquisitions(
            digital_acquisitions
        )

        # Process averaged signals and retain repetition statistics.
        return assemble_light_calibration(
            calibration_id=calibration_id,
            created_at=created_at,
            analog_raw=analog_average.mean,
            digital_raw=digital_average.mean,
            analog_config=config.analog_waveform,
            analog_wavelength_nm=config.analog_wavelength_nm,
            digital_wavelength_nm=config.digital_wavelength_nm,
            analog_amplifier_range=config.analog_amplifier_range,
            digital_amplifier_range=config.digital_amplifier_range,
            analog_area_um2=config.analog_area_um2,
            digital_area_um2=config.digital_area_um2,
            analog_repeats=analog_average.n_repeats,
            digital_repeats=digital_average.n_repeats,
            analog_monitor_sd=analog_average.monitor_voltage_sd,
            analog_photodiode_sd=analog_average.photodiode_voltage_sd,
            digital_monitor_sd=digital_average.monitor_voltage_sd,
            digital_photodiode_sd=digital_average.photodiode_voltage_sd,
        )
