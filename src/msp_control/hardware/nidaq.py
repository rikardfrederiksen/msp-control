from dataclasses import dataclass

import nidaqmx
from nidaqmx.constants import (
    TerminalConfiguration,
    AcquisitionType,
    Edge,
)

import numpy as np


@dataclass(frozen=True)
class NIDaqConfig:
    """Hardware configuration for the NI DAQ used by the MSP."""

    device: str = "Dev1"

    pmt_channel: str = "ai0"

    pmt_min_v: float = 0.0
    pmt_max_v: float = 5.0

    sample_clock_terminal: str = "PFI1"
    start_trigger_terminal: str = "PFI2"

    shutter_line: str = "port0/line0"
    ir_line: str = "port0/line1"

    bleach_617_output: str = "ao1"
    bleach_617_monitor: str = "ai4"

    bleach_505_output: str = "port0/line2"
    bleach_505_monitor: str = "ai6"

    nominal_sample_rate: float = 10_000.0
    read_timeout: float = 10.0

    @property
    def pmt_physical_channel(self) -> str:
        return f"{self.device}/{self.pmt_channel}"

    @property
    def sample_clock_source(self) -> str:
        return f"/{self.device}/{self.sample_clock_terminal}"

    @property
    def start_trigger_source(self) -> str:
        return f"/{self.device}/{self.start_trigger_terminal}"

    @property
    def shutter_physical_channel(self) -> str:
        return f"{self.device}/{self.shutter_line}"

    @property
    def ir_physical_channel(self) -> str:
        return f"{self.device}/{self.ir_line}"

    @property
    def bleach_617_output_channel(self) -> str:
        return f"{self.device}/{self.bleach_617_output}"

    @property
    def bleach_617_monitor_channel(self) -> str:
        return f"{self.device}/{self.bleach_617_monitor}"

    @property
    def bleach_505_output_channel(self) -> str:
        return f"{self.device}/{self.bleach_505_output}"

    @property
    def bleach_505_monitor_channel(self) -> str:
        return f"{self.device}/{self.bleach_505_monitor}"


class NIDaq:
    """Interface to the NI DAQ used by the MSP."""

    def __init__(self, config: NIDaqConfig):
        self.config = config
        self._ai_task: nidaqmx.Task | None = None
        self._bleach_do_task: nidaqmx.Task | None = None
        self._bleach_ai_task: nidaqmx.Task | None = None

    def configure_ai(self) -> None:
        """Create and configure the PMT analog-input task."""

        self._ai_task = nidaqmx.Task()

        self._ai_task.ai_channels.add_ai_voltage_chan(
            self.config.pmt_physical_channel,
            terminal_config=TerminalConfiguration.DIFF,
            min_val=self.config.pmt_min_v,
            max_val=self.config.pmt_max_v,
        )

    def close(self) -> None:
        """Release DAQ resources."""

        if self._ai_task is not None:
            self._ai_task.close()
            self._ai_task = None

        if self._bleach_do_task is not None:
            self._bleach_do_task.close()
            self._bleach_do_task = None

        if self._bleach_ai_task is not None:
            self._bleach_ai_task.close()
            self._bleach_ai_task = None

    def configure_timing(self, total_samples: int) -> None:
        """Configure finite acquisition using the external Optoscan clock."""

        if self._ai_task is None:
            raise RuntimeError("Analog input must be configured first")

        if total_samples < 1:
            raise ValueError("total_samples must be at least one")

        self._ai_task.timing.cfg_samp_clk_timing(
            rate=self.config.nominal_sample_rate,
            source=self.config.sample_clock_source,
            active_edge=Edge.RISING,
            sample_mode=AcquisitionType.FINITE,
            samps_per_chan=total_samples,
        )

    def configure_start_trigger(self) -> None:
        """Configure the Optoscan trigger as the acquisition start trigger."""

        if self._ai_task is None:
            raise RuntimeError("Analog input must be configured first")

        self._ai_task.triggers.start_trigger.cfg_dig_edge_start_trig(
            trigger_source=self.config.start_trigger_source,
            trigger_edge=Edge.RISING,
        )

    def start(self) -> None:
        """Arm the analog-input task for acquisition."""

        if self._ai_task is None:
            raise RuntimeError("Analog input must be configured first")

        self._ai_task.start()


    def read_sweep(self, samples: int) -> list[float]:
        """Read one complete wavelength sweep from the analog-input task."""

        if self._ai_task is None:
            raise RuntimeError("Analog input must be configured first")

        if samples < 1:
            raise ValueError("samples must be at least one")

        return self._ai_task.read(
            number_of_samples_per_channel=samples,
            timeout=self.config.read_timeout,
        )

    def configure_bleach_digital_output(
        self,
        waveform: np.ndarray,
        sample_rate_hz: float,
    ) -> None:
        """Configure a finite digital output waveform for the 505 nm LED."""

        if self._bleach_do_task is not None:
            raise RuntimeError("Bleach digital output is already configured")

        waveform = np.asarray(waveform)

        if waveform.ndim != 1 or waveform.dtype != np.bool_:
            raise ValueError("Digital waveform must be a 1D boolean array")

        if len(waveform) < 2:
            raise ValueError("Digital waveform must contain at least two samples")

        if waveform[-1]:
            raise ValueError("Digital waveform must end with LED off")

        if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive")

        task = nidaqmx.Task()

        try:
            task.do_channels.add_do_chan(
                self.config.bleach_505_output_channel,
            )

            task.timing.cfg_samp_clk_timing(
                rate=sample_rate_hz,
                source=f"/{self.config.device}/ai/SampleClock",
                active_edge=Edge.RISING,
                sample_mode=AcquisitionType.FINITE,
                samps_per_chan=len(waveform),
            )

            task.write(
                waveform.tolist(),
                auto_start=False,
            )

        except Exception:
            task.close()
            raise

        self._bleach_do_task = task

    def configure_bleach_monitor(
        self,
        led_channel: int,
        samples: int,
        sample_rate_hz: float,
    ) -> None:
        """Configure finite analog acquisition of an LED monitor."""

        if self._bleach_ai_task is not None:
            raise RuntimeError("Bleach monitor is already configured")

        if led_channel == 1:
            physical_channel = self.config.bleach_617_monitor_channel
        elif led_channel == 2:
            physical_channel = self.config.bleach_505_monitor_channel
        else:
            raise ValueError("led_channel must be 1 or 2")

        if samples < 1:
            raise ValueError("samples must be at least one")

        if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive")

        task = nidaqmx.Task()

        try:
            task.ai_channels.add_ai_voltage_chan(
                physical_channel,
                terminal_config=TerminalConfiguration.DIFF,
            )

            task.timing.cfg_samp_clk_timing(
                rate=sample_rate_hz,
                sample_mode=AcquisitionType.FINITE,
                samps_per_chan=samples,
            )

        except Exception:
            task.close()
            raise

        self._bleach_ai_task = task

    def start_bleach_tasks(self) -> None:
        """Start synchronized bleach output and monitor acquisition."""

        if self._bleach_do_task is None:
            raise RuntimeError("Bleach digital output is not configured")

        if self._bleach_ai_task is None:
            raise RuntimeError("Bleach monitor is not configured")

        self._bleach_do_task.start()
        self._bleach_ai_task.start()

    def read_bleach_monitor(
        self,
        samples: int,
        timeout: float,
    ) -> np.ndarray:
        """Read the LED monitor waveform from the bleach AI task."""

        if self._bleach_ai_task is None:
            raise RuntimeError("Bleach monitor is not configured")

        if samples < 1:
            raise ValueError("samples must be at least one")

        if timeout <= 0:
            raise ValueError("timeout must be positive")

        data = self._bleach_ai_task.read(
            number_of_samples_per_channel=samples,
            timeout=timeout,
        )

        return np.asarray(data, dtype=float)

    def wait_for_bleach_output(self, timeout: float) -> None:
        """Wait for the digital bleach waveform to complete."""

        if self._bleach_do_task is None:
            raise RuntimeError("Bleach digital output is not configured")

        if timeout <= 0:
            raise ValueError("timeout must be positive")

        self._bleach_do_task.wait_until_done(timeout=timeout)

    def stop_bleach_digital_output(self) -> None:
        """Stop the bleach waveform and explicitly switch the LED off."""

        task = self._bleach_do_task

        try:
            if task is not None:
                try:
                    task.stop()
                finally:
                    task.close()
        finally:
            self._bleach_do_task = None

            self._write_digital_line(
                self.config.bleach_505_output_channel,
                False,
            )

    def open_shutter(self) -> None:
        """Open the PMT shutter."""

        self._write_digital_line(
            self.config.shutter_physical_channel,
            True,
        )

    def close_shutter(self) -> None:
        """Close the PMT shutter."""

        self._write_digital_line(
            self.config.shutter_physical_channel,
            False,
        )

    def ir_on(self) -> None:
        """Turn on the IR illumination."""

        self._write_digital_line(
            self.config.ir_physical_channel,
            True,
        )

    def ir_off(self) -> None:
        """Turn off the IR illumination."""

        self._write_digital_line(
            self.config.ir_physical_channel,
            False,
        )

    def _write_digital_line(
        self,
        physical_channel: str,
        value: bool,
    ) -> None:
        """Write a boolean value to a digital output line."""

        with nidaqmx.Task() as task:
            task.do_channels.add_do_chan(physical_channel)
            task.write(value)
