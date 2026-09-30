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
    _commit_pending_group,
    update_scan_group,
    create_experiment_file,
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

    with h5py.File(filename, "r") as h5:
        assert "baselines/baseline_007" in h5
        assert "_pending/baseline_007" not in h5


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

    write_baseline(filename, baseline)
    write_scan(filename, scan)

    scan_group = ScanGroup(
        scan_group_id=0,
        label="control",
        scan_indices=[scan.scan_index],
    )

    write_scan_group(filename, scan_group)

    with h5py.File(filename, "r") as h5:
        group = h5[
            f"scan_groups/group_{scan_group.scan_group_id:03d}"
        ]

        assert (
            group.attrs["scan_group_id"]
            == scan_group.scan_group_id
        )
        assert group.attrs["label"] == scan_group.label

        np.testing.assert_array_equal(
            group["scan_indices"][:],
            np.array([scan.scan_index]),
        )


def test_scan_group_round_trip(tmp_path, baseline, scan):
    filename = tmp_path / "test.h5"

    # Referenced scans must already exist in /scans.
    write_baseline(filename, baseline)
    write_scan(filename, scan)

    scan_group = ScanGroup(
        scan_group_id=0,
        label="control",
        scan_indices=[scan.scan_index],
    )

    write_scan_group(filename, scan_group)

    loaded = read_scan_group(
        filename,
        scan_group.scan_group_id,
    )

    assert loaded.scan_group_id == scan_group.scan_group_id
    assert loaded.label == scan_group.label
    assert loaded.scan_indices == scan_group.scan_indices


def test_write_scan_group_rejects_missing_scan(tmp_path):
    filename = tmp_path / "test.h5"

    scan_group = ScanGroup(
        scan_group_id=3,
        label="control",
        scan_indices=[99],
    )

    with pytest.raises(
        ValueError,
        match="Scan 99 does not exist in the HDF5 file",
    ):
        write_scan_group(filename, scan_group)


def test_write_scan_group_rejects_duplicate_id(
    tmp_path,
    baseline,
    scan,
):
    filename = tmp_path / "test.h5"

    write_baseline(filename, baseline)
    write_scan(filename, scan)

    scan_group = ScanGroup(
        scan_group_id=3,
        label="control",
        scan_indices=[scan.scan_index],
    )

    write_scan_group(filename, scan_group)

    with pytest.raises(
        ScanGroupAlreadyExistsError,
        match="Scan group 3 already exists",
    ):
        write_scan_group(filename, scan_group)


def test_write_experiment(tmp_path, baseline, scan):
    filename = tmp_path / "test.h5"

    experiment = Experiment(
        shared_metadata={
            "animal": "A17",
            "species": "mouse",
        }
    )

    experiment.add_baseline(baseline)
    experiment.add_scan(scan)

    scan_group = ScanGroup(
        scan_group_id=3,
        label="control",
        scan_indices=[scan.scan_index],
    )
    experiment.scan_groups.append(scan_group)

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
    experiment.add_scan(scan)

    scan_group = ScanGroup(
        scan_group_id=3,
        label="control",
        scan_indices=[scan.scan_index],
    )
    experiment.scan_groups.append(scan_group)

    write_experiment(filename, experiment)

    loaded = read_experiment(filename)

    assert loaded.shared_metadata == experiment.shared_metadata

    # Baseline
    assert len(loaded.baselines) == 1
    loaded_baseline = loaded.baselines[0]

    assert loaded_baseline.baseline_index == baseline.baseline_index
    assert loaded_baseline.polarization is baseline.polarization

    np.testing.assert_array_equal(
        loaded_baseline.baseline_corrected,
        baseline.baseline_corrected,
    )

    # Scan
    assert len(loaded.scans) == 1
    loaded_scan = loaded.scans[0]

    assert loaded_scan.scan_index == scan.scan_index
    assert loaded_scan.baseline_index == scan.baseline_index

    np.testing.assert_array_equal(
        loaded_scan.optical_density,
        scan.optical_density,
    )

    # Scan group
    assert len(loaded.scan_groups) == 1
    loaded_group = loaded.scan_groups[0]

    assert loaded_group.scan_group_id == 3
    assert loaded_group.label == "control"
    assert loaded_group.scan_indices == [scan.scan_index]


def test_write_scan_rejects_missing_baseline(tmp_path, scan):
    filename = tmp_path / "test.h5"

    with pytest.raises(
        ValueError,
        match=f"Baseline {scan.baseline_index} does not exist",
    ):
        write_scan(filename, scan)

    with h5py.File(filename, "r") as h5:
        assert f"scans/scan_{scan.scan_index:03d}" not in h5


def test_write_rejects_non_msp_hdf5_file(
    tmp_path,
    baseline,
):
    filename = tmp_path / "other.h5"

    with h5py.File(filename, "w") as h5:
        h5.attrs["description"] = "Some unrelated HDF5 file"
        h5.create_dataset(
            "important_data",
            data=np.array([1, 2, 3]),
        )

    with pytest.raises(
        ValueError,
        match="not an MSP-control file",
    ):
        write_baseline(filename, baseline)

    # Make sure the rejected file was not modified.
    with h5py.File(filename, "r") as h5:
        assert h5.attrs["description"] == "Some unrelated HDF5 file"
        assert "important_data" in h5
        assert "format" not in h5.attrs
        assert "baselines" not in h5


def test_read_rejects_non_msp_hdf5_file(tmp_path):
    filename = tmp_path / "other.h5"

    with h5py.File(filename, "w") as h5:
        h5.create_group("baselines")

    with pytest.raises(
        ValueError,
        match="not an MSP-control file",
    ):
        read_baseline(filename, baseline_index=7)


def test_read_rejects_wrong_format(tmp_path):
    filename = tmp_path / "other.h5"

    with h5py.File(filename, "w") as h5:
        h5.attrs["format"] = "some-other-format"
        h5.attrs["format_version"] = 1

    with pytest.raises(
        ValueError,
        match="not an MSP-control file",
    ):
        read_baseline(filename, baseline_index=7)


def test_read_rejects_unsupported_format_version(tmp_path):
    filename = tmp_path / "future.h5"

    with h5py.File(filename, "w") as h5:
        h5.attrs["format"] = "msp-control"
        h5.attrs["format_version"] = 999
        h5.attrs["software_version"] = "99.0.0"

    with pytest.raises(
        ValueError,
        match="Unsupported MSP-control format version: 999",
    ):
        read_baseline(filename, baseline_index=7)


def test_read_rejects_missing_format_version(tmp_path):
    filename = tmp_path / "malformed.h5"

    with h5py.File(filename, "w") as h5:
        h5.attrs["format"] = "msp-control"

    with pytest.raises(
        ValueError,
        match="MSP-control file has no format version",
    ):
        read_baseline(filename, baseline_index=7)


def test_commit_pending_group(tmp_path):
    filename = tmp_path / "experiment.h5"

    with h5py.File(filename, "w") as h5:
        pending = h5.create_group("_pending/scan_007")
        pending.create_dataset(
            "data",
            data=np.array([1.0, 2.0, 3.0]),
        )

        _commit_pending_group(
            h5,
            "_pending/scan_007",
            "scans/scan_007",
        )

        assert "_pending/scan_007" not in h5
        assert "scans/scan_007" in h5

        np.testing.assert_array_equal(
            h5["scans/scan_007/data"][:],
            np.array([1.0, 2.0, 3.0]),
        )


def test_commit_pending_group_rejects_existing_destination(
    tmp_path,
):
    filename = tmp_path / "experiment.h5"

    with h5py.File(filename, "w") as h5:
        h5.create_group("_pending/scan_007")
        h5.create_group("scans/scan_007")

        with pytest.raises(
            ValueError,
            match="already exists",
        ):
            _commit_pending_group(
                h5,
                "_pending/scan_007",
                "scans/scan_007",
            )

        assert "_pending/scan_007" in h5
        assert "scans/scan_007" in h5


def test_write_baseline_cleans_up_after_write_failure(
    tmp_path,
    baseline,
    monkeypatch,
):
    filename = tmp_path / "experiment.h5"

    def fail_write(*args, **kwargs):
        raise RuntimeError("Simulated write failure")

    monkeypatch.setattr(
        "msp_control.storage.hdf5._write_scan_config",
        fail_write,
    )

    with pytest.raises(
        RuntimeError,
        match="Simulated write failure",
    ):
        write_baseline(filename, baseline)

    with h5py.File(filename, "r") as h5:
        group_name = f"baseline_{baseline.baseline_index:03d}"

        assert f"baselines/{group_name}" not in h5
        assert f"_pending/{group_name}" not in h5


def test_write_scan_cleans_up_after_write_failure(
    tmp_path,
    baseline,
    scan,
    monkeypatch,
):
    filename = tmp_path / "experiment.h5"

    write_baseline(filename, baseline)

    def fail_write(*args, **kwargs):
        raise RuntimeError("Simulated write failure")

    monkeypatch.setattr(
        "msp_control.storage.hdf5._write_scan_config",
        fail_write,
    )

    with pytest.raises(
        RuntimeError,
        match="Simulated write failure",
    ):
        write_scan(filename, scan)

    with h5py.File(filename, "r") as h5:
        group_name = f"scan_{scan.scan_index:03d}"

        assert f"scans/{group_name}" not in h5
        assert f"_pending/{group_name}" not in h5

        # The already committed baseline must be untouched.
        baseline_name = f"baseline_{baseline.baseline_index:03d}"
        assert f"baselines/{baseline_name}" in h5


def test_experiment_round_trip_with_ungrouped_scan(
    tmp_path,
    baseline,
    scan,
):
    filename = tmp_path / "test.h5"

    experiment = Experiment()
    experiment.add_baseline(baseline)
    experiment.add_scan(scan)

    write_experiment(filename, experiment)

    loaded = read_experiment(filename)

    assert len(loaded.scans) == 1
    assert loaded.scans[0].scan_index == scan.scan_index
    assert len(loaded.scan_groups) == 0


def test_update_scan_group(
    tmp_path,
    baseline,
    scan,
):
    filename = tmp_path / "test.h5"

    write_baseline(filename, baseline)
    write_scan(filename, scan)

    scan_group = ScanGroup(
        scan_group_id=3,
        label="control",
        scan_indices=[],
    )
    write_scan_group(filename, scan_group)

    scan_group.label = "dark adapted"
    scan_group.scan_indices.append(scan.scan_index)

    update_scan_group(filename, scan_group)

    loaded = read_scan_group(filename, 3)

    assert loaded.scan_group_id == 3
    assert loaded.label == "dark adapted"
    assert loaded.scan_indices == [scan.scan_index]


def test_update_scan_group_rejects_missing_group(tmp_path):
    filename = tmp_path / "test.h5"

    scan_group = ScanGroup(
        scan_group_id=3,
        label="control",
    )

    with pytest.raises(
        KeyError,
        match="Scan group 3 does not exist",
    ):
        update_scan_group(filename, scan_group)


def test_update_scan_group_rejects_missing_scan(
    tmp_path,
    baseline,
    scan,
):
    filename = tmp_path / "test.h5"

    write_baseline(filename, baseline)
    write_scan(filename, scan)

    original = ScanGroup(
        scan_group_id=3,
        label="control",
        scan_indices=[scan.scan_index],
    )
    write_scan_group(filename, original)

    updated = ScanGroup(
        scan_group_id=3,
        label="control",
        scan_indices=[99],
    )

    with pytest.raises(
        ValueError,
        match="Scan 99 does not exist in the HDF5 file",
    ):
        update_scan_group(filename, updated)


def test_update_scan_group_restores_old_group_after_commit_failure(
    tmp_path,
    baseline,
    scan,
    monkeypatch,
):
    filename = tmp_path / "test.h5"

    write_baseline(filename, baseline)
    write_scan(filename, scan)

    original = ScanGroup(
        scan_group_id=3,
        label="control",
        scan_indices=[scan.scan_index],
    )
    write_scan_group(filename, original)

    updated = ScanGroup(
        scan_group_id=3,
        label="dark adapted",
        scan_indices=[],
    )

    original_move = h5py.File.move
    move_count = 0

    def failing_move(self, source, dest):
        nonlocal move_count
        move_count += 1

        if move_count == 2:
            raise RuntimeError("simulated commit failure")

        return original_move(self, source, dest)

    monkeypatch.setattr(
        h5py.File,
        "move",
        failing_move,
    )

    with pytest.raises(
        RuntimeError,
        match="simulated commit failure",
    ):
        update_scan_group(filename, updated)

    loaded = read_scan_group(filename, 3)

    assert loaded.label == "control"
    assert loaded.scan_indices == [scan.scan_index]


def test_create_experiment_file(tmp_path):
    filename = tmp_path / "experiment.h5"

    create_experiment_file(filename)

    assert filename.exists()

    with h5py.File(filename, "r") as h5:
        assert h5.attrs["format"] == "msp-control"
        assert h5.attrs["format_version"] == 1


def test_create_experiment_file_rejects_existing_file(tmp_path):
    filename = tmp_path / "experiment.h5"

    filename.write_text("do not overwrite")

    with pytest.raises(
        FileExistsError,
        match="File already exists",
    ):
        create_experiment_file(filename)

    assert filename.read_text() == "do not overwrite"
