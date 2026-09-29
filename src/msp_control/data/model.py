from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from msp_control.config import ScanConfig


class Polarization(Enum):
    TRANSVERSE = "T"
    LONGITUDINAL = "L"


@dataclass
class Baseline:
    """A baseline measurement used as the reference for specimen scans."""

    baseline_index: int
    polarization: Polarization
    config: ScanConfig
    wavelength: np.ndarray

    raw_dark: np.ndarray
    raw_transition: np.ndarray
    raw_baseline: np.ndarray

    dark_mean: np.ndarray
    baseline_mean: np.ndarray
    baseline_corrected: np.ndarray

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Scan:
    """A specimen measurement and its derived optical-density spectrum."""

    scan_index: int
    baseline_index: int
    polarization: Polarization
    config: ScanConfig
    wavelength: np.ndarray

    raw_dark: np.ndarray
    raw_transition: np.ndarray
    raw_specimen: np.ndarray

    dark_mean: np.ndarray
    specimen_mean: np.ndarray
    specimen_corrected: np.ndarray
    optical_density: np.ndarray

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScanGroup:
    scan_group_id: int
    label: str = ""
    scan_indices: list[int] = field(default_factory=list)

    def __post_init__(self):
        if not isinstance(self.scan_group_id, int):
            raise TypeError("scan_group_id must be an integer")

        if not isinstance(self.label, str):
            raise TypeError("label must be a string")

        if not all(
            isinstance(index, int)
            for index in self.scan_indices
        ):
            raise TypeError("scan_indices must contain integers")

        if len(self.scan_indices) != len(set(self.scan_indices)):
            raise ValueError(
                "scan_indices cannot contain duplicates"
            )


@dataclass
class Experiment:
    """A complete microspectrophotometry experiment."""

    shared_metadata: dict[str, Any] = field(default_factory=dict)
    baselines: list[Baseline] = field(default_factory=list)
    scans: list[Scan] = field(default_factory=list)
    scan_groups: list[ScanGroup] = field(default_factory=list)

    def add_baseline(self, baseline: Baseline) -> None:
        """Add a baseline to the experiment."""

        if any(
            existing.baseline_index == baseline.baseline_index
            for existing in self.baselines
        ):
            raise ValueError(
                f"Baseline index {baseline.baseline_index} already exists"
            )

        self.baselines.append(baseline)

    def get_baseline(self, baseline_index: int) -> Baseline:
        """Return a baseline by index."""

        for baseline in self.baselines:
            if baseline.baseline_index == baseline_index:
                return baseline

        raise KeyError(
            f"Baseline index {baseline_index} does not exist"
        )

    def get_baselines(
        self,
        polarization: Polarization | None = None,
    ) -> list[Baseline]:
        """Return baselines, optionally filtered by polarization."""

        if polarization is None:
            return list(self.baselines)

        return [
            baseline
            for baseline in self.baselines
            if baseline.polarization is polarization
        ]

    @property
    def next_baseline_index(self) -> int:
        """Return the next available baseline index."""

        if not self.baselines:
            return 0

        return max(
            baseline.baseline_index
            for baseline in self.baselines
        ) + 1

    def add_scan(
        self,
        scan: Scan,
    ) -> None:
        """Add a scan to the experiment."""

        if any(
            existing.scan_index == scan.scan_index
            for existing in self.scans
        ):
            raise ValueError(
                f"Scan index {scan.scan_index} already exists"
            )

        self.scans.append(scan)

    def get_scan(self, scan_index: int) -> Scan:
        """Return a scan by its experiment-wide index."""

        for scan in self.scans:
            if scan.scan_index == scan_index:
                return scan

        raise KeyError(
            f"Scan index {scan_index} does not exist"
        )

    @property
    def next_scan_index(self) -> int:
        """Return the next available experiment-wide scan index."""

        if not self.scans:
            return 0

        return max(
            scan.scan_index
            for scan in self.scans
        ) + 1

    def add_scan_group(
        self,
        scan_group: ScanGroup,
    ) -> None:
        """Add a scan group to the experiment."""

        if any(
            existing.scan_group_id == scan_group.scan_group_id
            for existing in self.scan_groups
        ):
            raise ValueError(
                f"Scan group {scan_group.scan_group_id} already exists"
            )

        existing_scan_indices = {
            scan.scan_index
            for scan in self.scans
        }

        for scan_index in scan_group.scan_indices:
            if scan_index not in existing_scan_indices:
                raise ValueError(
                    f"Scan {scan_index} does not exist"
                )

        self.scan_groups.append(scan_group)

    def get_scan_group(
        self,
        scan_group_id: int,
    ) -> ScanGroup:
        """Return a scan group by its experiment-wide ID."""

        for scan_group in self.scan_groups:
            if scan_group.scan_group_id == scan_group_id:
                return scan_group

        raise KeyError(
            f"Scan group {scan_group_id} does not exist"
        )

    def add_scan_to_group(
        self,
        scan_index: int,
        scan_group_id: int,
    ) -> None:
        """Add an existing scan to an existing scan group."""

        self.get_scan(scan_index)
        scan_group = self.get_scan_group(scan_group_id)

        if scan_index in scan_group.scan_indices:
            raise ValueError(
                f"Scan {scan_index} already belongs to "
                f"scan group {scan_group_id}"
            )

        scan_group.scan_indices.append(scan_index)

    def remove_scan_from_group(
        self,
        scan_index: int,
        scan_group_id: int,
    ) -> None:
        """Remove a scan from a scan group."""

        scan_group = self.get_scan_group(scan_group_id)

        if scan_index not in scan_group.scan_indices:
            raise ValueError(
                f"Scan {scan_index} does not belong to "
                f"scan group {scan_group_id}"
            )

        scan_group.scan_indices.remove(scan_index)

    def rename_scan_group(
        self,
        scan_group_id: int,
        label: str,
    ) -> None:
        """Change the label of a scan group."""

        if not isinstance(label, str):
            raise TypeError("label must be a string")

        scan_group = self.get_scan_group(scan_group_id)
        scan_group.label = label
