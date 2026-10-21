"""Hardware acquisition for LED calibration."""

from dataclasses import dataclass

import numpy as np

from msp_control.hardware.nidaq import NIDaq
from msp_control.calibration.waveform import (
    CalibrationWaveformConfig,
    BleachWaveformConfig,
    generate_calibration_waveform,
    generate_bleach_waveform,
)


@dataclass
class RawCalibrationAcquisition:
    """Raw synchronized LED calibration measurements."""

    time_s: np.ndarray
    command_voltage: np.ndarray
    monitor_voltage: np.ndarray
    photodiode_voltage: np.ndarray


@dataclass
class RawDigitalCalibrationAcquisition:
    """Raw synchronized measurements from a digitally controlled LED."""

    time_s: np.ndarray
    command_state: np.ndarray
    monitor_voltage: np.ndarray
    photodiode_voltage: np.ndarray


class CalibrationController:
    """Coordinate LED calibration waveform output and acquisition."""

    def __init__(self, daq: NIDaq):
        self.daq = daq

    def acquire_analog(
        self,
        config: CalibrationWaveformConfig,
    ) -> RawCalibrationAcquisition:
        """Execute an analog LED calibration waveform."""

        time_s, waveform = generate_calibration_waveform(config)

        samples = len(waveform)
        timeout = samples / config.sample_rate_hz + 5.0

        acquisition_error = None

        try:
            self.daq.configure_bleach_analog_output(
                waveform=waveform,
                sample_rate_hz=config.sample_rate_hz,
            )

            self.daq.configure_bleach_monitor(
                led_channel=1,
                samples=samples,
                sample_rate_hz=config.sample_rate_hz,
                include_photodiode=True,
            )

            self.daq.start_bleach_tasks()

            data = self.daq.read_bleach_monitor(
                samples=samples,
                timeout=timeout,
            )

            self.daq.wait_for_bleach_output(timeout=timeout)

            data = np.asarray(data, dtype=float)

            if data.shape != (2, samples):
                raise ValueError(
                    f"Expected two calibration channels, got {data.shape}"
                )

            monitor_voltage = data[0]
            photodiode_voltage = data[1]

        except Exception as exc:
            acquisition_error = exc
            raise

        finally:
            try:
                self.daq.stop_bleach_analog_output()

            except Exception as shutdown_error:
                if acquisition_error is None:
                    raise
                acquisition_error.add_note(
                    f"Calibration shutdown also failed: {shutdown_error}"
                )

            finally:
                try:
                    self.daq.close()
                except Exception as close_error:
                    if acquisition_error is None:
                        raise
                    acquisition_error.add_note(
                        f"DAQ cleanup also failed: {close_error}"
                    )

        return RawCalibrationAcquisition(
            time_s=time_s,
            command_voltage=waveform,
            monitor_voltage=monitor_voltage,
            photodiode_voltage=photodiode_voltage,
        )


    def acquire_digital(
        self,
        config: BleachWaveformConfig,
    ) -> RawDigitalCalibrationAcquisition:
        """Execute a digital LED calibration and record both inputs."""

        time_s, waveform = generate_bleach_waveform(
            config,
            control_mode="digital",
            command_voltage=None,
        )

        samples = len(waveform)
        timeout = samples / config.sample_rate_hz + 5.0

        acquisition_error = None

        try:
            self.daq.configure_bleach_digital_output(
                waveform=waveform,
                sample_rate_hz=config.sample_rate_hz,
            )

            self.daq.configure_bleach_monitor(
                led_channel=2,
                samples=samples,
                sample_rate_hz=config.sample_rate_hz,
                include_photodiode=True,
            )

            self.daq.start_bleach_tasks()

            data = self.daq.read_bleach_monitor(
                samples=samples,
                timeout=timeout,
            )

            self.daq.wait_for_bleach_output(timeout=timeout)

            data = np.asarray(data, dtype=float)

            if data.shape != (2, samples):
                raise ValueError(
                    f"Expected two calibration channels, got {data.shape}"
                )

            monitor_voltage = data[0]
            photodiode_voltage = data[1]

        except Exception as exc:
            acquisition_error = exc
            raise

        finally:
            try:
                self.daq.stop_bleach_digital_output()

            except Exception as shutdown_error:
                if acquisition_error is None:
                    raise
                acquisition_error.add_note(
                    f"Digital calibration shutdown also failed: {shutdown_error}"
                )

            finally:
                try:
                    self.daq.close()
                except Exception as close_error:
                    if acquisition_error is None:
                        raise
                    acquisition_error.add_note(
                        f"DAQ cleanup also failed: {close_error}"
                    )

        return RawDigitalCalibrationAcquisition(
            time_s=time_s,
            command_state=waveform,
            monitor_voltage=monitor_voltage,
            photodiode_voltage=photodiode_voltage,
        )
