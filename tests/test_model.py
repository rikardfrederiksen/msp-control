import numpy as np

import pytest
from datetime import datetime, timezone

from msp_control.config import ScanConfig
from msp_control.data.model import (
    Baseline,
    Experiment,
    Polarization,
    Scan,
    ScanGroup,
    Event,
)


config = ScanConfig(
    start_nm=350,
    end_nm=700,
    step_nm=1,
    step_time_ms=2,
    input_slit_nm=4,
    output_slit_nm=4,
    dark_scans=2,
    data_scans=5,
)

wavelength = config.wavelengths
timestamp = datetime(
    2026, 10, 6,
    8, 31, 42, 381234,
    tzinfo=timezone.utc,
)

# Fake baseline acquisition:
# 2 dark sweeps and 5 illuminated sweeps
raw_dark_baseline = np.random.normal(0.1, 0.005, (2, len(wavelength)))
raw_transition = np.array([99.0, 99.0, 99.0])
raw_baseline = np.random.normal(5.0, 0.05, (5, len(wavelength)))

dark_baseline = np.mean(raw_dark_baseline, axis=0)
baseline_mean = np.mean(raw_baseline, axis=0)
baseline_corrected = baseline_mean - dark_baseline

baseline = Baseline(
    baseline_index=0,
    timestamp=timestamp,
    polarization=Polarization.TRANSVERSE,
    config=config,
    wavelength=wavelength,
    raw_dark=raw_dark_baseline,
    raw_transition=raw_transition,
    raw_baseline=raw_baseline,
    dark_mean=dark_baseline,
    baseline_mean=baseline_mean,
    baseline_corrected=baseline_corrected,
    metadata={
        "number_of_dark_scans": 2,
        "number_of_scans": 5,
    },
)


# Fake specimen acquisition
raw_dark_specimen = np.random.normal(0.1, 0.005, (2, len(wavelength)))
raw_transition = np.array([99.0, 99.0, 99.0])
raw_specimen = np.random.normal(3.0, 0.05, (5, len(wavelength)))

dark_specimen = np.mean(raw_dark_specimen, axis=0)
specimen_mean = np.mean(raw_specimen, axis=0)
specimen_corrected = specimen_mean - dark_specimen

optical_density = -np.log10(
    specimen_corrected / baseline.baseline_corrected
)

scan = Scan(
    scan_index=0,
    baseline_index=baseline.baseline_index,
    timestamp=timestamp,
    polarization=Polarization.TRANSVERSE,
    config=config,
    wavelength=wavelength,
    raw_dark=raw_dark_specimen,
    raw_transition=raw_transition,
    raw_specimen=raw_specimen,
    dark_mean=dark_specimen,
    specimen_mean=specimen_mean,
    specimen_corrected=specimen_corrected,
    optical_density=optical_density,
    metadata={
        "number_of_dark_scans": 2,
        "number_of_scans": 5,
    },
)


scan_group = ScanGroup(
    scan_group_id=0,
    label="control",
    scan_indices=[0],
)


experiment = Experiment(
    shared_metadata={
        "species": "mouse",
        "animal": "test",
    },
    baselines=[baseline],
    scans=[scan],
    scan_groups=[scan_group],
)


def test_experiment():
    assert len(experiment.baselines) == 1
    assert len(experiment.scans) == 1
    assert len(experiment.scan_groups) == 1

    assert experiment.baselines[0].baseline_index == 0
    assert experiment.scans[0].baseline_index == 0

    assert experiment.baselines[0].baseline_corrected.shape == wavelength.shape
    assert experiment.scans[0].optical_density.shape == wavelength.shape

    assert experiment.baselines[0].config == config
    assert experiment.scans[0].config == config

    assert experiment.scan_groups[0].scan_indices == [0]

def test_add_baseline():
    experiment = Experiment()

    experiment.add_baseline(baseline)

    assert len(experiment.baselines) == 1
    assert experiment.baselines[0] is baseline


def test_get_baseline():
    experiment = Experiment()
    experiment.add_baseline(baseline)

    result = experiment.get_baseline(baseline.baseline_index)

    assert result is baseline


def test_add_baseline_rejects_duplicate_index():
    experiment = Experiment()
    experiment.add_baseline(baseline)

    with pytest.raises(
        ValueError,
        match=f"Baseline index {baseline.baseline_index} already exists",
    ):
        experiment.add_baseline(baseline)


def test_get_baselines_filters_by_polarization():
    baseline_t = baseline

    baseline_l = Baseline(
        baseline_index=1,
        timestamp=timestamp,
        polarization=Polarization.LONGITUDINAL,
        config=baseline.config,
        wavelength=baseline.wavelength,
        raw_dark=baseline.raw_dark,
        raw_transition=raw_transition,
        raw_baseline=baseline.raw_baseline,
        dark_mean=baseline.dark_mean,
        baseline_mean=baseline.baseline_mean,
        baseline_corrected=baseline.baseline_corrected,
    )

    experiment = Experiment()
    experiment.add_baseline(baseline_t)
    experiment.add_baseline(baseline_l)

    transverse = experiment.get_baselines(
        polarization=Polarization.TRANSVERSE
    )
    longitudinal = experiment.get_baselines(
        polarization=Polarization.LONGITUDINAL
    )

    assert transverse == [baseline_t]
    assert longitudinal == [baseline_l]
    all_baselines = experiment.get_baselines()

    assert len(all_baselines) == 2
    assert all_baselines[0] is baseline_t
    assert all_baselines[1] is baseline_l

def test_next_baseline_index():
    experiment = Experiment()

    assert experiment.next_baseline_index == 0

    baseline_0 = baseline

    baseline_2 = Baseline(
        baseline_index=2,
        polarization=Polarization.TRANSVERSE,
        timestamp=timestamp,
        config=baseline.config,
        wavelength=baseline.wavelength,
        raw_dark=baseline.raw_dark,
        raw_transition=raw_transition,
        raw_baseline=baseline.raw_baseline,
        dark_mean=baseline.dark_mean,
        baseline_mean=baseline.baseline_mean,
        baseline_corrected=baseline.baseline_corrected,
    )

    experiment.add_baseline(baseline_0)
    experiment.add_baseline(baseline_2)

    assert experiment.next_baseline_index == 3

def test_add_scan():
    experiment = Experiment()

    experiment.add_scan(scan)

    assert experiment.scans == [scan]


def test_get_scan():
    experiment = Experiment()

    experiment.add_scan(scan)

    result = experiment.get_scan(0)

    assert result is scan


def test_add_scan_rejects_duplicate_index():
    experiment = Experiment()

    experiment.add_scan(scan)

    with pytest.raises(
        ValueError,
        match=f"Scan index {scan.scan_index} already exists",
    ):
        experiment.add_scan(scan)

def test_next_scan_index():
    experiment = Experiment()

    assert experiment.next_scan_index == 0

    experiment.add_scan(scan)

    assert experiment.next_scan_index == 1


def test_scan_group_rejects_string_id():
    with pytest.raises(
        TypeError,
        match="scan_group_id must be an integer",
    ):
        ScanGroup(
            scan_group_id="control",
            label="control",
            scan_indices=[1,5],
        )


def test_scan_group():
    group = ScanGroup(
        scan_group_id=3,
        label="control",
        scan_indices=[1, 4, 7],
    )

    assert group.scan_group_id == 3
    assert group.label == "control"
    assert group.scan_indices == [1, 4, 7]


def test_scan_group_rejects_duplicate_scan_indices():
    with pytest.raises(
        ValueError,
        match="cannot contain duplicates",
    ):
        ScanGroup(
            scan_group_id=3,
            label="control",
            scan_indices=[1, 4, 4],
        )


def test_add_scan_group():
    experiment = Experiment(
        scans=[scan],
    )

    scan_group = ScanGroup(
        scan_group_id=3,
        label="control",
        scan_indices=[scan.scan_index],
    )

    experiment.add_scan_group(scan_group)

    assert len(experiment.scan_groups) == 1
    assert experiment.scan_groups[0] is scan_group


def test_add_scan_group_rejects_duplicate_id():
    experiment = Experiment(
        scans=[scan],
    )

    first = ScanGroup(
        scan_group_id=3,
        label="control",
        scan_indices=[scan.scan_index],
    )

    second = ScanGroup(
        scan_group_id=3,
        label="other",
        scan_indices=[],
    )

    experiment.add_scan_group(first)

    with pytest.raises(
        ValueError,
        match="Scan group 3 already exists",
    ):
        experiment.add_scan_group(second)


def test_add_scan_group_rejects_missing_scan():
    experiment = Experiment()

    scan_group = ScanGroup(
        scan_group_id=3,
        label="control",
        scan_indices=[99],
    )

    with pytest.raises(
        ValueError,
        match="Scan 99 does not exist",
    ):
        experiment.add_scan_group(scan_group)


def test_add_empty_scan_group():
    experiment = Experiment()

    scan_group = ScanGroup(
        scan_group_id=3,
        label="control",
        scan_indices=[],
    )

    experiment.add_scan_group(scan_group)

    assert experiment.scan_groups == [scan_group]


def test_get_scan_group():
    experiment = Experiment(
        scans=[scan],
    )

    scan_group = ScanGroup(
        scan_group_id=3,
        label="control",
        scan_indices=[scan.scan_index],
    )

    experiment.add_scan_group(scan_group)

    assert experiment.get_scan_group(3) is scan_group


def test_get_scan_group_rejects_missing_id():
    experiment = Experiment()

    with pytest.raises(
        KeyError,
        match="Scan group 3 does not exist",
    ):
        experiment.get_scan_group(3)

def test_add_scan_to_group():
    experiment = Experiment(
        scans=[scan],
    )

    scan_group = ScanGroup(
        scan_group_id=3,
        label="control",
    )
    experiment.add_scan_group(scan_group)

    experiment.add_scan_to_group(
        scan.scan_index,
        3,
    )

    assert scan_group.scan_indices == [scan.scan_index]


def test_add_scan_to_group_rejects_missing_scan():
    experiment = Experiment()

    scan_group = ScanGroup(
        scan_group_id=3,
        label="control",
    )
    experiment.add_scan_group(scan_group)

    with pytest.raises(
        KeyError,
        match="Scan index 99 does not exist",
    ):
        experiment.add_scan_to_group(99, 3)


def test_add_scan_to_group_rejects_duplicate():
    experiment = Experiment(
        scans=[scan],
    )

    scan_group = ScanGroup(
        scan_group_id=3,
        label="control",
        scan_indices=[scan.scan_index],
    )
    experiment.add_scan_group(scan_group)

    with pytest.raises(
        ValueError,
        match=f"Scan {scan.scan_index} already belongs",
    ):
        experiment.add_scan_to_group(
            scan.scan_index,
            3,
        )

def test_remove_scan_from_group():
    experiment = Experiment(
        scans=[scan],
    )

    scan_group = ScanGroup(
        scan_group_id=3,
        label="control",
        scan_indices=[scan.scan_index],
    )
    experiment.add_scan_group(scan_group)

    experiment.remove_scan_from_group(
        scan.scan_index,
        3,
    )

    assert scan_group.scan_indices == []


def test_remove_scan_from_group_rejects_missing_membership():
    experiment = Experiment(
        scans=[scan],
    )

    scan_group = ScanGroup(
        scan_group_id=3,
        label="control",
    )
    experiment.add_scan_group(scan_group)

    with pytest.raises(
        ValueError,
        match=f"Scan {scan.scan_index} does not belong",
    ):
        experiment.remove_scan_from_group(
            scan.scan_index,
            3,
        )

def test_rename_scan_group():
    experiment = Experiment()

    scan_group = ScanGroup(
        scan_group_id=3,
        label="control",
    )
    experiment.add_scan_group(scan_group)

    experiment.rename_scan_group(
        3,
        "dark adapted control",
    )

    assert scan_group.label == "dark adapted control"


def test_rename_scan_group_rejects_non_string_label():
    experiment = Experiment()

    scan_group = ScanGroup(
        scan_group_id=3,
        label="control",
    )
    experiment.add_scan_group(scan_group)

    with pytest.raises(
        TypeError,
        match="label must be a string",
    ):
        experiment.rename_scan_group(3, 42)

def test_add_event():
    event = Event(
        event_id=0,
        timestamp=timestamp,
        description="11-cis retinal added",
    )

    experiment = Experiment()
    experiment.add_event(event)

    assert experiment.events == [event]


def test_get_event():
    event = Event(
        event_id=0,
        timestamp=timestamp,
        description="11-cis retinal added",
    )

    experiment = Experiment(events=[event])

    assert experiment.get_event(0) is event


def test_add_event_rejects_duplicate_id():
    event = Event(
        event_id=0,
        timestamp=timestamp,
        description="11-cis retinal added",
    )

    experiment = Experiment(events=[event])

    with pytest.raises(ValueError):
        experiment.add_event(event)


def test_next_event_id():
    experiment = Experiment()

    assert experiment.next_event_id == 0

    experiment.add_event(
        Event(
            event_id=0,
            timestamp=timestamp,
            description="11-cis retinal added",
        )
    )

    assert experiment.next_event_id == 1
