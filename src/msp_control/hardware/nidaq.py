from dataclasses import dataclass

import nidaqmx
from nidaqmx.constants import (
    TerminalConfiguration,
    AcquisitionType,
    Edge,
)

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

class NIDaq:
    """Interface to the NI DAQ used by the MSP."""

    def __init__(self, config: NIDaqConfig):
        self.config = config
        self._ai_task: nidaqmx.Task | None = None

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
