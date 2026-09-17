from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Baseline:
    """A baseline measurement used as the reference for specimen scans."""

    baseline_index: int
    wavelength: np.ndarray

    raw_dark: np.ndarray
    raw_baseline: np.ndarray

    dark: np.ndarray
    baseline: np.ndarray

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Scan:
    """A specimen measurement and its derived optical-density spectrum."""

    scan_index: int
    baseline_index: int
    wavelength: np.ndarray

    raw_dark: np.ndarray
    raw_specimen: np.ndarray

    dark: np.ndarray
    specimen: np.ndarray
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
    
