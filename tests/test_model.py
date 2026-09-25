import numpy as np

import pytest

from msp_control.config import ScanConfig
from msp_control.data.model import (
    Baseline,
    Experiment,
    Polarization,
    Scan,
    ScanGroup,
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

# Fake baseline acquisition:
# 2 dark sweeps and 5 illuminated sweeps
raw_dark_baseline = np.random.normal(0.1, 0.005, (2, len(wavelength)))
raw_baseline = np.random.normal(5.0, 0.05, (5, len(wavelength)))

dark_baseline = np.mean(raw_dark_baseline, axis=0)
baseline_mean = np.mean(raw_baseline, axis=0)
baseline_corrected = baseline_mean - dark_baseline

baseline = Baseline(
    baseline_index=0,
    polarization=Polarization.TRANSVERSE,
    config=config,
    wavelength=wavelength,
    raw_dark=raw_dark_baseline,
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
    polarization=Polarization.TRANSVERSE,
    config=config,
    wavelength=wavelength,
    raw_dark=raw_dark_specimen,
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
    scans=[scan],
)


experiment = Experiment(
    shared_metadata={
        "species": "mouse",
        "animal": "test",
    },
    baselines=[baseline],
    scan_groups=[scan_group],
)


def test_experiment():
    assert len(experiment.baselines) == 1
    assert len(experiment.scan_groups) == 1
    assert len(experiment.scan_groups[0].scans) == 1

    assert experiment.baselines[0].baseline_index == 0
    assert experiment.scan_groups[0].scans[0].baseline_index == 0

    assert experiment.baselines[0].baseline_corrected.shape == wavelength.shape
    assert experiment.scan_groups[0].scans[0].optical_density.shape == wavelength.shape

    assert experiment.baselines[0].config == config
    assert experiment.scan_groups[0].scans[0].config == config

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
        polarization=Polarization.LONGITUDINAL,
        config=baseline.config,
        wavelength=baseline.wavelength,
        raw_dark=baseline.raw_dark,
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
        config=baseline.config,
        wavelength=baseline.wavelength,
        raw_dark=baseline.raw_dark,
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

    experiment.add_scan(scan, scan_group_id=0)

    assert len(experiment.scan_groups) == 1
    assert experiment.scan_groups[0].scan_group_id == 0
    assert len(experiment.scan_groups[0].scans) == 1
    assert experiment.scan_groups[0].scans[0] is scan


def test_get_scan():
    experiment = Experiment()
    experiment.add_scan(scan, scan_group_id="cell_1")

    result = experiment.get_scan(scan.scan_index)

    assert result is scan


def test_add_scan_rejects_duplicate_index_across_groups():
    experiment = Experiment()

    experiment.add_scan(scan, scan_group_id="cell_1")

    with pytest.raises(
        ValueError,
        match=f"Scan index {scan.scan_index} already exists",
    ):
        experiment.add_scan(scan, scan_group_id="cell_2")

def test_next_scan_index():
    experiment = Experiment()

    assert experiment.next_scan_index == 0

    scan_0 = scan

    scan_4 = Scan(
        scan_index=4,
        baseline_index=scan.baseline_index,
        polarization=scan.polarization,
        config=scan.config,
        wavelength=scan.wavelength,
        raw_dark=scan.raw_dark,
        raw_specimen=scan.raw_specimen,
        dark_mean=scan.dark_mean,
        specimen_mean=scan.specimen_mean,
        specimen_corrected=scan.specimen_corrected,
        optical_density=scan.optical_density,
    )

    experiment.add_scan(scan_0, scan_group_id="cell_1")
    experiment.add_scan(scan_4, scan_group_id="cell_2")

    assert experiment.next_scan_index == 5

def test_add_scan_to_existing_group():
    experiment = Experiment()

    scan_0 = scan

    scan_1 = Scan(
        scan_index=1,
        baseline_index=scan.baseline_index,
        polarization=scan.polarization,
        config=scan.config,
        wavelength=scan.wavelength,
        raw_dark=scan.raw_dark,
        raw_specimen=scan.raw_specimen,
        dark_mean=scan.dark_mean,
        specimen_mean=scan.specimen_mean,
        specimen_corrected=scan.specimen_corrected,
        optical_density=scan.optical_density,
    )

    experiment.add_scan(scan_0, scan_group_id="cell_1")
    experiment.add_scan(scan_1, scan_group_id="cell_1")

    assert len(experiment.scan_groups) == 1
    assert len(experiment.scan_groups[0].scans) == 2
    assert experiment.scan_groups[0].scans[0] is scan_0
    assert experiment.scan_groups[0].scans[1] is scan_1

