import numpy as np

from msp_control.data.model import Baseline, Experiment, Scan, ScanGroup


# Fake wavelength axis
wavelength = np.arange(350.0, 701.0, 1.0)

# Fake baseline acquisition:
# 2 dark sweeps and 5 illuminated sweeps
raw_dark_baseline = np.random.normal(0.1, 0.005, (2, len(wavelength)))
raw_baseline = np.random.normal(5.0, 0.05, (5, len(wavelength)))

dark_baseline = np.mean(raw_dark_baseline, axis=0)
baseline_signal = np.mean(raw_baseline, axis=0) - dark_baseline

baseline = Baseline(
    baseline_index=0,
    wavelength=wavelength,
    raw_dark=raw_dark_baseline,
    raw_baseline=raw_baseline,
    dark=dark_baseline,
    baseline=baseline_signal,
    metadata={
        "number_of_dark_scans": 2,
        "number_of_scans": 5,
    },
)


# Fake specimen acquisition
raw_dark_specimen = np.random.normal(0.1, 0.005, (2, len(wavelength)))
raw_specimen = np.random.normal(3.0, 0.05, (5, len(wavelength)))

dark_specimen = np.mean(raw_dark_specimen, axis=0)
specimen_signal = np.mean(raw_specimen, axis=0) - dark_specimen

optical_density = -np.log10(specimen_signal / baseline.baseline)

scan = Scan(
    scan_index=0,
    baseline_index=baseline.baseline_index,
    wavelength=wavelength,
    raw_dark=raw_dark_specimen,
    raw_specimen=raw_specimen,
    dark=dark_specimen,
    specimen=specimen_signal,
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

    assert experiment.baselines[0].baseline.shape == wavelength.shape
    assert experiment.scan_groups[0].scans[0].optical_density.shape == wavelength.shape
