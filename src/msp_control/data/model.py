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
    raw_specimen: np.ndarray

    dark_mean: np.ndarray
    specimen_mean: np.ndarray
    specimen_corrected: np.ndarray
    optical_density: np.ndarray

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScanGroup:
    """A collection of related specimen scans."""

    scan_group_id: int | str
    scans: list[Scan] = field(default_factory=list)


@dataclass
class Experiment:
    """A complete microspectrophotometry experiment."""

    shared_metadata: dict[str, Any] = field(default_factory=dict)
    baselines: list[Baseline] = field(default_factory=list)
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
        scan_group_id: int | str,
    ) -> None:
        """Add a scan to a scan group."""

        if any(
            existing.scan_index == scan.scan_index
            for group in self.scan_groups
            for existing in group.scans
        ):
            raise ValueError(
                f"Scan index {scan.scan_index} already exists"
            )

        for group in self.scan_groups:
            if group.scan_group_id == scan_group_id:
                group.scans.append(scan)
                return

        group = ScanGroup(scan_group_id=scan_group_id)
        group.scans.append(scan)
        self.scan_groups.append(group)

    def get_scan(self, scan_index: int) -> Scan:
        """Return a scan by its experiment-wide index."""

        for group in self.scan_groups:
            for scan in group.scans:
                if scan.scan_index == scan_index:
                    return scan

        raise KeyError(
            f"Scan index {scan_index} does not exist"
        )

    @property
    def next_scan_index(self) -> int:
        """Return the next available experiment-wide scan index."""

        indices = [
            scan.scan_index
            for group in self.scan_groups
            for scan in group.scans
        ]

        if not indices:
            return 0

        return max(indices) + 1
