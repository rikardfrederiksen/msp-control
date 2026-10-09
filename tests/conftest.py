import h5py
import numpy as np
import pytest
from datetime import datetime, timezone

from dataclasses import replace

from msp_control.config import ScanConfig
from msp_control.data.model import Baseline, Polarization, Scan, ScanGroup, Experiment, Event
from msp_control.storage.hdf5 import (
    BaselineAlreadyExistsError,
    ScanAlreadyExistsError,
    ScanGroupAlreadyExistsError,
    read_baseline,
    write_baseline,
    write_scan,
    read_scan,
    write_experiment_metadata,
    read_experiment_metadata,
    write_scan_group,
    read_scan_group,
    write_experiment,
    read_experiment,
    _commit_pending_group,
    update_scan_group,
    create_experiment_file,
    read_event,
    write_event,
)


timestamp = datetime(
    2026, 10, 6,
    8, 31, 42, 381234,
    tzinfo=timezone.utc,
)


@pytest.fixture
def baseline():
    config = ScanConfig(
        start_nm=500,
        end_nm=520,
        step_nm=10,
        step_time_ms=2,
        input_slit_nm=4,
        output_slit_nm=4,
        dark_scans=2,
        data_scans=2,
    )

    return Baseline(
        baseline_index=7,
        timestamp=timestamp,
        polarization=Polarization.TRANSVERSE,
        config=config,
        wavelength=np.array([500.0, 510.0, 520.0]),
        raw_dark=np.array([
            [1.0, 2.0, 3.0],
            [3.0, 4.0, 5.0],
        ]),
        raw_transition=np.array([99.0, 99.0, 99.0]),
        raw_baseline=np.array([
            [11.0, 12.0, 13.0],
            [13.0, 14.0, 15.0],
        ]),
        dark_mean=np.array([2.0, 3.0, 4.0]),
        baseline_mean=np.array([12.0, 13.0, 14.0]),
        baseline_corrected=np.array([10.0, 10.0, 10.0]),
        metadata={
            "cell_type": "rod",
            "animal_number": 17,
            "temperature_C": 36.8,
            "accepted": True,
        },
    )


@pytest.fixture
def scan():
    config = ScanConfig(
        start_nm=500,
        end_nm=520,
        step_nm=10,
        step_time_ms=2,
        input_slit_nm=4,
        output_slit_nm=4,
        dark_scans=2,
        data_scans=2,
    )

    return Scan(
        scan_index=12,
        baseline_index=7,
        timestamp=timestamp,
        polarization=Polarization.TRANSVERSE,
        config=config,
        wavelength=np.array([500.0, 510.0, 520.0]),
        raw_dark=np.array([
            [1.0, 2.0, 3.0],
            [3.0, 4.0, 5.0],
        ]),
        raw_transition=np.array([99.0, 99.0, 99.0]),
        raw_specimen=np.array([
            [3.0, 4.0, 5.0],
            [3.0, 4.0, 5.0],
        ]),
        dark_mean=np.array([2.0, 3.0, 4.0]),
        specimen_mean=np.array([3.0, 4.0, 5.0]),
        specimen_corrected=np.array([1.0, 1.0, 1.0]),
        optical_density=np.array([1.0, 1.0, 1.0]),
        metadata={
            "cell_type": "rod",
            "temperature_C": 36.8,
            "accepted": True,
        },
    )
