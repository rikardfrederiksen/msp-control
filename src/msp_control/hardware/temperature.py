from dataclasses import dataclass

import nidaqmx
import numpy as np

from nidaqmx.constants import (
    AcquisitionType,
    TerminalConfiguration,
)


@dataclass(frozen=True)
class TemperatureConfig:
    """Configuration for temperature measurement."""

    device: str = "Dev1"
    channel: str = "ai7"

    volts_per_C: float = 0.1

    min_v: float = 0.0
    max_v: float = 5.0

    sample_rate: float = 1000.0
    duration_s: float = 0.2

    @property
    def physical_channel(self) -> str:
        return f"{self.device}/{self.channel}"

    @property
    def sample_count(self) -> int:
        return round(self.sample_rate * self.duration_s)


@dataclass(frozen=True)
class TemperatureMeasurement:
    """Summary of one temperature measurement."""

    mean_C: float
    sd_C: float


class TemperatureMonitor:
    """Measure temperature from the TC324B monitor output."""

    def __init__(self, config: TemperatureConfig):
        self.config = config

    def measure(self) -> TemperatureMeasurement:
        """Acquire and summarize one temperature measurement."""

        with nidaqmx.Task() as task:
            task.ai_channels.add_ai_voltage_chan(
                self.config.physical_channel,
                terminal_config=TerminalConfiguration.DIFF,
                min_val=self.config.min_v,
                max_val=self.config.max_v,
            )

            task.timing.cfg_samp_clk_timing(
                rate=self.config.sample_rate,
                sample_mode=AcquisitionType.FINITE,
                samps_per_chan=self.config.sample_count,
            )

            voltage = np.asarray(
                task.read(
                    number_of_samples_per_channel=(
                        self.config.sample_count
                    ),
                ),
                dtype=float,
            )

        temperature_C = voltage / self.config.volts_per_C

        return TemperatureMeasurement(
            mean_C=float(np.mean(temperature_C)),
            sd_C=float(np.std(temperature_C)),
        )
