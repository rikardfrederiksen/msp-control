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

        if led_channel != 2:
            raise NotImplementedError(
                "Only the 505 nm digital LED is implemented"
            )

        time_s, waveform = generate_bleach_waveform(
            config,
            control_mode="digital",
            command_voltage=command_voltage,
        )

        samples = len(waveform)
        timeout = len(waveform) / config.sample_rate_hz + 5.0

        try:
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

            return RawBleachAcquisition(
                time_s=time_s,
                monitor_voltage=np.asarray(
                    monitor_voltage,
                    dtype=float,
                ),
            )

        finally:
            try:
                self.daq.stop_bleach_digital_output()
            finally:
                self.daq.close()
