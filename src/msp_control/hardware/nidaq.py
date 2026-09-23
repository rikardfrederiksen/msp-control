from dataclasses import dataclass


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
