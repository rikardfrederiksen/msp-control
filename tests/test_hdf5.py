import h5py
import numpy as np
import pytest

from dataclasses import replace

from msp_control.config import ScanConfig
from msp_control.data.model import Baseline, Polarization, Scan, ScanGroup, Experiment
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


def test_write_baseline(tmp_path):
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

    baseline = Baseline(
        baseline_index=7,
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
    )

    filename = tmp_path / "test.h5"

    write_baseline(filename, baseline)

    with h5py.File(filename, "r") as h5:
        assert h5.attrs["format"] == "msp-control"
        assert h5.attrs["format_version"] == 1

        group = h5["baselines/baseline_007"]

        assert group.attrs["baseline_index"] == 7
        assert group.attrs["polarization"] == "T"

        assert group["config"].attrs["start_nm"] == 500

        np.testing.assert_array_equal(
            group["data/raw_transition"][:],
            baseline.raw_transition,
        )

        np.testing.assert_array_equal(
            group["data/baseline_corrected"][:],
            baseline.baseline_corrected,
        )

def test_baseline_round_trip(tmp_path):
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

    baseline = Baseline(
        baseline_index=7,
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

    filename = tmp_path / "test.h5"

    write_baseline(filename, baseline)
    loaded = read_baseline(filename, baseline_index=7)

    assert loaded.baseline_index == baseline.baseline_index
    assert loaded.polarization is baseline.polarization
    assert loaded.config == baseline.config
    assert loaded.metadata == baseline.metadata
    assert isinstance(loaded.metadata["accepted"], bool)

    np.testing.assert_array_equal(
        loaded.wavelength,
        baseline.wavelength,
    )
    np.testing.assert_array_equal(
        loaded.raw_dark,
        baseline.raw_dark,
    )
    np.testing.assert_array_equal(
        loaded.raw_transition,
        baseline.raw_transition,
    )
    np.testing.assert_array_equal(
        loaded.raw_baseline,
        baseline.raw_baseline,
    )
    np.testing.assert_array_equal(
        loaded.dark_mean,
        baseline.dark_mean,
    )
    np.testing.assert_array_equal(
        loaded.baseline_mean,
        baseline.baseline_mean,
    )
    np.testing.assert_array_equal(
        loaded.baseline_corrected,
        baseline.baseline_corrected,
    )

def test_write_multiple_baselines(tmp_path):
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

    baseline_0 = Baseline(
        baseline_index=0,
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
    )

    baseline_1 = replace(
        baseline_0,
        baseline_index=1,
        polarization=Polarization.LONGITUDINAL,
    )

    filename = tmp_path / "test.h5"

    write_baseline(filename, baseline_0)
    write_baseline(filename, baseline_1)

    loaded_0 = read_baseline(filename, 0)
    loaded_1 = read_baseline(filename, 1)

    assert loaded_0.baseline_index == 0
    assert loaded_0.polarization is Polarization.TRANSVERSE

    assert loaded_1.baseline_index == 1
    assert loaded_1.polarization is Polarization.LONGITUDINAL


def test_write_baseline_rejects_duplicate_index(
    tmp_path,
    baseline,
):
    filename = tmp_path / "test.h5"

    write_baseline(filename, baseline)

    with pytest.raises(
        BaselineAlreadyExistsError,
        match=f"Baseline {baseline.baseline_index} already exists",
    ):
        write_baseline(filename, baseline)

    loaded = read_baseline(
        filename,
        baseline.baseline_index,
    )

    np.testing.assert_array_equal(
        loaded.raw_baseline,
        baseline.raw_baseline,
    )


def test_write_scan(tmp_path, baseline, scan):
    filename = tmp_path / "test.h5"

    write_baseline(filename, baseline)
    write_scan(filename, scan)

    with h5py.File(filename, "r") as h5:
        scan_group = h5["scans/scan_012"]

        assert scan_group.attrs["scan_index"] == 12
        assert scan_group.attrs["baseline_index"] == 7
        assert scan_group.attrs["polarization"] == "T"

        assert scan_group["config"].attrs["start_nm"] == 500

        np.testing.assert_array_equal(
            scan_group["data/raw_transition"][:],
            scan.raw_transition,
        )

        np.testing.assert_array_equal(
            scan_group["data/optical_density"][:],
            scan.optical_density,
        )

        assert scan_group["metadata"].attrs["cell_type"] == "rod"


def test_scan_round_trip(tmp_path, baseline, scan):
    filename = tmp_path / "test.h5"

    write_baseline(filename, baseline)
    write_scan(filename, scan)
    loaded = read_scan(filename, scan.scan_index)

    assert loaded.scan_index == scan.scan_index
    assert loaded.baseline_index == scan.baseline_index
    assert loaded.polarization is scan.polarization
    assert loaded.config == scan.config
    assert loaded.metadata == scan.metadata

    np.testing.assert_array_equal(
        loaded.wavelength,
        scan.wavelength,
    )
    np.testing.assert_array_equal(
        loaded.raw_dark,
        scan.raw_dark,
    )
    np.testing.assert_array_equal(
        loaded.raw_transition,
        scan.raw_transition,
    )
    np.testing.assert_array_equal(
        loaded.raw_specimen,
        scan.raw_specimen,
    )
    np.testing.assert_array_equal(
        loaded.dark_mean,
        scan.dark_mean,
    )
    np.testing.assert_array_equal(
        loaded.specimen_mean,
        scan.specimen_mean,
    )
    np.testing.assert_array_equal(
        loaded.specimen_corrected,
        scan.specimen_corrected,
    )
    np.testing.assert_array_equal(
        loaded.optical_density,
        scan.optical_density,
    )

def test_write_scan_rejects_duplicate_index(tmp_path, baseline, scan):
    filename = tmp_path / "test.h5"
    
    write_baseline(filename, baseline)
    write_scan(filename, scan)

    with pytest.raises(
        ScanAlreadyExistsError,
        match=f"Scan {scan.scan_index} already exists",
    ):
        write_scan(filename, scan)

    loaded = read_scan(
        filename,
        scan.scan_index,
    )

    np.testing.assert_array_equal(
        loaded.raw_specimen,
        scan.raw_specimen,
    )

def test_experiment_metadata_round_trip(tmp_path):
    filename = tmp_path / "test.h5"

    metadata = {
        "animal": "A17",
        "species": "mouse",
        "temperature_C": 36.8,
        "accepted": True,
    }

    write_experiment_metadata(filename, metadata)
    loaded = read_experiment_metadata(filename)

    assert loaded == metadata


def test_write_scan_group(tmp_path, baseline, scan):
    filename = tmp_path / "test.h5"

    scan_group = ScanGroup(
        scan_group_id=3,
        label="control",
        scans=[scan],
    )

    write_baseline(filename, baseline)
    write_scan(filename, scan)
    write_scan_group(filename, scan_group)

    with h5py.File(filename, "r") as h5:
        group = h5["scan_groups/group_003"]

        assert group.attrs["scan_group_id"] == 3
        assert group.attrs["label"] == "control"

        np.testing.assert_array_equal(
            group["scan_indices"][:],
            np.array([scan.scan_index]),
        )


def test_scan_group_round_trip(tmp_path, baseline, scan):
    filename = tmp_path / "test.h5"

    scan_group = ScanGroup(
        scan_group_id=3,
        label="control",
        scans=[scan],
    )

    # The scan itself must exist independently in /scans.
    write_baseline(filename, baseline)
    write_scan(filename, scan)
    write_scan_group(filename, scan_group)

    loaded = read_scan_group(
        filename,
        scan_group.scan_group_id,
    )

    assert loaded.scan_group_id == scan_group.scan_group_id
    assert loaded.label == scan_group.label
    assert len(loaded.scans) == 1

    loaded_scan = loaded.scans[0]

    assert loaded_scan.scan_index == scan.scan_index
    assert loaded_scan.baseline_index == scan.baseline_index

    np.testing.assert_array_equal(
        loaded_scan.optical_density,
        scan.optical_density,
    )

def test_write_scan_group_rejects_missing_scan(tmp_path, baseline, scan):
    filename = tmp_path / "test.h5"

    scan_group = ScanGroup(
        scan_group_id=3,
        label="control",
        scans=[scan],
    )

    with pytest.raises(
        ValueError,
        match=f"Scan {scan.scan_index} does not exist",
    ):
        write_scan_group(filename, scan_group)

    with h5py.File(filename, "r") as h5:
        assert "group_003" not in h5["scan_groups"]


def test_write_scan_group_rejects_duplicate_scan(tmp_path, baseline, scan):
    filename = tmp_path / "test.h5"

    write_baseline(filename, baseline)
    write_scan(filename, scan)

    scan_group = ScanGroup(
        scan_group_id=3,
        label="control",
        scans=[scan, scan],
    )

    with pytest.raises(
        ValueError,
        match="contains duplicate scan indices",
    ):
        write_scan_group(filename, scan_group)

    with h5py.File(filename, "r") as h5:
        assert "scan_groups/group_003" not in h5


def test_write_scan_group_rejects_duplicate_id(tmp_path, baseline, scan):
    filename = tmp_path / "test.h5"

    write_baseline(filename, baseline)
    write_scan(filename, scan)

    scan_group = ScanGroup(
        scan_group_id=3,
        label="control",
        scans=[scan],
    )

    write_scan_group(filename, scan_group)

    with pytest.raises(
        ScanGroupAlreadyExistsError,
        match="Scan group 3 already exists",
    ):
        write_scan_group(filename, scan_group)

    loaded = read_scan_group(filename, 3)

    assert loaded.scan_group_id == 3
    assert loaded.label == "control"
    assert len(loaded.scans) == 1
    assert loaded.scans[0].scan_index == scan.scan_index


def test_write_experiment(tmp_path, baseline, scan):
    filename = tmp_path / "test.h5"

    experiment = Experiment(
        shared_metadata={
            "animal": "A17",
            "species": "mouse",
        }
    )

    experiment.add_baseline(baseline)
    experiment.add_scan(
        scan,
        scan_group_id=3,
    )

    experiment.scan_groups[0].label = "control"

    write_experiment(filename, experiment)

    with h5py.File(filename, "r") as h5:
        assert "metadata" in h5
        assert "baselines/baseline_007" in h5
        assert "scans/scan_012" in h5
        assert "scan_groups/group_003" in h5

        assert h5["metadata"].attrs["animal"] == "A17"
        assert h5["metadata"].attrs["species"] == "mouse"

        assert (
            h5["scan_groups/group_003"].attrs["label"]
            == "control"
        )

        np.testing.assert_array_equal(
            h5["scan_groups/group_003/scan_indices"][:],
            np.array([12]),
        )

def test_experiment_round_trip(tmp_path, baseline, scan):
    filename = tmp_path / "test.h5"

    experiment = Experiment(
        shared_metadata={
            "animal": "A17",
            "species": "mouse",
        }
    )

    experiment.add_baseline(baseline)
    experiment.add_scan(
        scan,
        scan_group_id=3,
    )
    experiment.scan_groups[0].label = "control"

    write_experiment(filename, experiment)

    loaded = read_experiment(filename)

    assert loaded.shared_metadata == experiment.shared_metadata

    assert len(loaded.baselines) == 1
    loaded_baseline = loaded.baselines[0]

    assert loaded_baseline.baseline_index == baseline.baseline_index
    assert loaded_baseline.polarization is baseline.polarization

    np.testing.assert_array_equal(
        loaded_baseline.baseline_corrected,
        baseline.baseline_corrected,
    )

    assert len(loaded.scan_groups) == 1
    loaded_group = loaded.scan_groups[0]

    assert loaded_group.scan_group_id == 3
    assert loaded_group.label == "control"
    assert len(loaded_group.scans) == 1

    loaded_scan = loaded_group.scans[0]

    assert loaded_scan.scan_index == scan.scan_index
    assert loaded_scan.baseline_index == scan.baseline_index

    np.testing.assert_array_equal(
        loaded_scan.optical_density,
        scan.optical_density,
    )

def test_write_scan_rejects_missing_baseline(tmp_path, scan):
    filename = tmp_path / "test.h5"

    with pytest.raises(
        ValueError,
        match=f"Baseline {scan.baseline_index} does not exist",
    ):
        write_scan(filename, scan)

    with h5py.File(filename, "r") as h5:
        assert f"scans/scan_{scan.scan_index:03d}" not in h5
