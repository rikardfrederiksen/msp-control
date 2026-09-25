from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ScanConfig:
    """Configuration for an Optoscan wavelength scan."""

    start_nm: float
    end_nm: float
    step_nm: float
    step_time_ms: float
    input_slit_nm: float
    output_slit_nm: float
    dark_scans: int
    data_scans: int

    def __post_init__(self):
        if self.end_nm <= self.start_nm:
            raise ValueError("end_nm must be greater than start_nm")

        if self.step_nm <= 0:
            raise ValueError("step_nm must be greater than zero")

        if self.step_time_ms <= 0:
            raise ValueError("step_time_ms must be greater than zero")

        if self.input_slit_nm <= 0 or self.output_slit_nm <= 0:
            raise ValueError("slit widths must be greater than zero")

        if self.dark_scans < 0:
            raise ValueError("dark_scans cannot be negative")

        if self.data_scans < 1:
            raise ValueError("data_scans must be at least one")

        n_steps = (self.end_nm - self.start_nm) / self.step_nm
        if not abs(n_steps - round(n_steps)) < 1e-9:
            raise ValueError(
                "wavelength range must be evenly divisible by step_nm"
            )

    @property
    def cycles(self) -> int:
        """Total number of wavelength sweeps performed by the Optoscan."""
        return self.dark_scans + 1 + self.data_scans

    @property
    def wavelength_points(self) -> int:
        """Number of wavelength points in one complete sweep."""
        return round((self.end_nm - self.start_nm) / self.step_nm) + 1

    @property
    def samples(self) -> int:
        """Total number of externally clocked DAQ samples."""
        return self.cycles * self.wavelength_points

    @property
    def wavelengths(self) -> np.ndarray:
        return np.linspace(
            self.start_nm,
            self.end_nm,
            self.wavelength_points,
        )

    def is_compatible_with(self, other: "ScanConfig") -> bool:
        """Return True if two configurations can share a baseline."""

        return (
            self.start_nm == other.start_nm
            and self.end_nm == other.end_nm
            and self.step_nm == other.step_nm
            and self.step_time_ms == other.step_time_ms
            and self.input_slit_nm == other.input_slit_nm
            and self.output_slit_nm == other.output_slit_nm
        )


def build_config_commands(config: ScanConfig) -> list[str]:
    """Build the Optoscan commands required to configure a wavelength scan."""

    return [
        "close_slits",
        f"{round(config.start_nm * 10)} scan_start !",
        f"{round(config.end_nm * 10)} scan_end !",
        f"{round(config.step_nm * 10)} scan_step_size !",
        f"{round(config.step_time_ms * 1000)} scan_step_time_lo !",
        "0 scan_step_time_hi !",
        "0 scan_data>ram",
        "0 scan_prog# !",
        "0 ram>scan_data",
        f"{config.cycles} cycles !",
        "compile_tables",
        "set_scantable",
        f"{round(config.input_slit_nm * 10)} scan_inslit_width !",
        f"{round(config.output_slit_nm * 10)} scan_exslit_width !",
    ]
