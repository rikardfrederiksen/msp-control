from dataclasses import dataclass

import numpy as np

from msp_control.hardware.nidaq import NIDaq
from msp_control.calibration.waveform import (
    BleachWaveformConfig,
    generate_bleach_waveform,
)


@dataclass
class RawBleachAcquisition:
    """Raw LED monitor recording from a bleaching exposure."""

    time_s: np.ndarray
    monitor_voltage: np.ndarray


class BleachController:
    """Coordinate LED output and monitor acquisition."""

    def __init__(self, daq: NIDaq):
        self.daq = daq

    def acquire(
        self,
        led_channel: int,
        config: BleachWaveformConfig,
        command_voltage: float | None = None,
    ) -> RawBleachAcquisition:
        """Execute a bleaching exposure and record its LED monitor."""

        if led_channel == 1:
            control_mode = "analog"
        elif led_channel == 2:
            control_mode = "digital"
        else:
            raise ValueError("LED channel must be 1 or 2")

        time_s, waveform = generate_bleach_waveform(
            config,
            control_mode=control_mode,
            command_voltage=command_voltage,
        )

        samples = len(waveform)
        timeout = samples / config.sample_rate_hz + 5.0

        acquisition_error = None

        try:
            if control_mode == "analog":
                self.daq.configure_bleach_analog_output(
                    waveform=waveform,
                    sample_rate_hz=config.sample_rate_hz,
                )
            else:
                self.daq.configure_bleach_digital_output(
                    waveform=waveform,
                    sample_rate_hz=config.sample_rate_hz,
                )

            self.daq.configure_bleach_monitor(
                led_channel=led_channel,
                samples=samples,
                sample_rate_hz=config.sample_rate_hz,
            )

            self.daq.start_bleach_tasks()

            monitor_voltage = self.daq.read_bleach_monitor(
                samples=samples,
                timeout=timeout,
            )

            self.daq.wait_for_bleach_output(timeout=timeout)

        except Exception as exc:
            acquisition_error = exc
            raise

        finally:
            try:
                if control_mode == "analog":
                    self.daq.stop_bleach_analog_output()
                else:
                    self.daq.stop_bleach_digital_output()

            except Exception as shutdown_error:
                if acquisition_error is None:
                    raise
                acquisition_error.add_note(
                    f"Bleach shutdown also failed: {shutdown_error}"
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

        return RawBleachAcquisition(
            time_s=time_s,
            monitor_voltage=np.asarray(monitor_voltage, dtype=float),
        )
