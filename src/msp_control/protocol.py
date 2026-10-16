from dataclasses import asdict, dataclass
import json
from pathlib import Path

from msp_control.config import ScanConfig


@dataclass(frozen=True)
class AcquisitionProtocol:
    start_nm: float
    end_nm: float
    step_nm: float
    step_time_ms: float
    input_slit_nm: float
    output_slit_nm: float
    dark_scans: int
    baseline_scans: int
    specimen_scans: int

    def scan_config(self, *, baseline: bool = False) -> ScanConfig:
        """Construct a ScanConfig for baseline or specimen acquisition."""
        return ScanConfig(
            start_nm=self.start_nm,
            end_nm=self.end_nm,
            step_nm=self.step_nm,
            step_time_ms=self.step_time_ms,
            input_slit_nm=self.input_slit_nm,
            output_slit_nm=self.output_slit_nm,
            dark_scans=self.dark_scans,
            data_scans=(
                self.baseline_scans if baseline else self.specimen_scans
            ),
        )

    def save(self, filename: str | Path) -> None:
        """Save the protocol as JSON."""
        Path(filename).write_text(
            json.dumps(asdict(self), indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, filename: str | Path) -> "AcquisitionProtocol":
        """Load and validate a protocol from JSON."""
        data = json.loads(Path(filename).read_text(encoding="utf-8"))
        protocol = cls(**data)

        if protocol.baseline_scans < 1 or protocol.specimen_scans < 1:
            raise ValueError("Scan counts must be at least one")

        protocol.scan_config(baseline=True)
        protocol.scan_config(baseline=False)

        return protocol
