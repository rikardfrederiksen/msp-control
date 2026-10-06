from datetime import datetime

import numpy as np
import pytest

from msp_control.acquisition import RawAcquisition
from msp_control.config import ScanConfig
from msp_control.controller import MSPController
from msp_control.data.model import Experiment, Polarization, ScanGroup
from msp_control.storage.hdf5 import(
    read_baseline,
    read_scan,
    read_experiment_metadata,
    read_scan_group)


class FakeAcquisitionController:
    def __init__(self):
        self.acquire_count = 0

    def acquire(self, config):
        self.acquire_count += 1

        return RawAcquisition(
            dark=[
                np.array([1.0, 2.0, 3.0]),
                np.array([3.0, 4.0, 5.0]),
            ],
            transition=np.array([99.0, 99.0, 99.0]),
            data=[
                np.array([11.0, 12.0, 13.0]),
                np.array([13.0, 14.0, 15.0]),
            ],
        )


class FailingAcquisitionController:
    def acquire(self, config):
        raise RuntimeError("Acquisition failed")


class FakeClock:
    def __init__(self, timestamp: datetime):
        self.timestamp = timestamp

    def now(self) -> datetime:
        return self.timestamp


def test_acquire_baseline(tmp_path):
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

    acquisition = FakeAcquisitionController()

    controller = MSPController(
        acquisition=acquisition,
    )

    controller.create_new_experiment_file(
        tmp_path / "experiment.h5"
    )

    baseline = controller.acquire_baseline(
        config=config,
        polarization=Polarization.TRANSVERSE,
    )

    experiment = controller.experiment

    assert experiment is not None
    assert baseline.baseline_index == 0
    assert baseline.polarization is Polarization.TRANSVERSE
    assert baseline.config is config

    assert len(experiment.baselines) == 1
    assert experiment.baselines[0] is baseline

    np.testing.assert_allclose(
        baseline.baseline_corrected,
        [10.0, 10.0, 10.0],
    )

    stored_baseline = read_baseline(
        controller.filename,
        baseline.baseline_index,
    )

    np.testing.assert_allclose(
        stored_baseline.baseline_corrected,
        baseline.baseline_corrected,
    )



def test_acquire_baseline_does_not_modify_experiment_on_failure(
    tmp_path,
):
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

    controller = MSPController(
        acquisition=FailingAcquisitionController(),
    )

    controller.create_new_experiment_file(
        tmp_path / "experiment.h5"
    )

    experiment = controller.experiment
    assert experiment is not None

    with pytest.raises(RuntimeError, match="Acquisition failed"):
        controller.acquire_baseline(
            config=config,
            polarization=Polarization.TRANSVERSE,
        )

    assert experiment.baselines == []
    assert experiment.next_baseline_index == 0

def test_acquire_scan(tmp_path):
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

    acquisition = FakeAcquisitionController()

    controller = MSPController(
        acquisition=acquisition,
    )

    controller.create_new_experiment_file(
        tmp_path / "experiment.h5"
    )

    experiment = controller.experiment

    assert experiment is not None

    baseline = controller.acquire_baseline(
        config=config,
        polarization=Polarization.TRANSVERSE,
    )

    scan = controller.acquire_scan(
        config=config,
        polarization=Polarization.TRANSVERSE,
        baseline_index=baseline.baseline_index,
    )

    assert scan.scan_index == 0
    assert scan.baseline_index == baseline.baseline_index
    assert scan.polarization is Polarization.TRANSVERSE
    assert scan.config is config

    assert len(experiment.scans) == 1
    assert experiment.scans[0] is scan

    assert len(experiment.scan_groups) == 0

    np.testing.assert_allclose(
        scan.optical_density,
        [0.0, 0.0, 0.0],
    )

    stored_scan = read_scan(
        controller.filename,
        scan.scan_index,
    )

    np.testing.assert_allclose(
        stored_scan.optical_density,
        scan.optical_density,
    )
    
    assert stored_scan.baseline_index == baseline.baseline_index

def test_acquire_scan_rejects_missing_baseline_before_acquisition(
    tmp_path,
):
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

    acquisition = FakeAcquisitionController()

    controller = MSPController(
        acquisition=acquisition,
    )

    controller.create_new_experiment_file(
        tmp_path / "experiment.h5"
    )

    experiment = controller.experiment
    assert experiment is not None

    with pytest.raises(
        KeyError,
        match="Baseline index 7 does not exist",
    ):
        controller.acquire_scan(
            config=config,
            polarization=Polarization.TRANSVERSE,
            baseline_index=7,
        )

    assert acquisition.acquire_count == 0
    assert experiment.next_scan_index == 0
    assert experiment.scan_groups == []


def test_acquire_scan_rejects_wrong_polarization_before_acquisition(tmp_path):
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

    acquisition = FakeAcquisitionController()

    controller = MSPController(
        acquisition=acquisition,
    )

    controller.create_new_experiment_file(
        tmp_path / "experiment.h5"
    )

    experiment = controller.experiment
    assert experiment is not None
    
    baseline = controller.acquire_baseline(
        config=config,
        polarization=Polarization.TRANSVERSE,
    )

    acquisitions_before = acquisition.acquire_count

    with pytest.raises(
        ValueError,
        match="Baseline polarization does not match scan polarization",
    ):
        controller.acquire_scan(
            config=config,
            polarization=Polarization.LONGITUDINAL,
            baseline_index=baseline.baseline_index,
        )

    assert acquisition.acquire_count == acquisitions_before
    assert experiment.next_scan_index == 0
    assert experiment.scan_groups == []

def test_acquire_scan_rejects_incompatible_config_before_acquisition(tmp_path):
    baseline_config = ScanConfig(
        start_nm=500,
        end_nm=520,
        step_nm=10,
        step_time_ms=2,
        input_slit_nm=4,
        output_slit_nm=4,
        dark_scans=2,
        data_scans=2,
    )

    scan_config = ScanConfig(
        start_nm=500,
        end_nm=520,
        step_nm=5,
        step_time_ms=2,
        input_slit_nm=4,
        output_slit_nm=4,
        dark_scans=2,
        data_scans=2,
    )

    acquisition = FakeAcquisitionController()

    controller = MSPController(
        acquisition=acquisition,
    )

    controller.create_new_experiment_file(
        tmp_path / "experiment.h5"
    )

    experiment = controller.experiment
    assert experiment is not None

    baseline = controller.acquire_baseline(
        config=baseline_config,
        polarization=Polarization.TRANSVERSE,
    )

    acquisitions_before = acquisition.acquire_count

    with pytest.raises(
        ValueError,
        match="Baseline and specimen scan configurations are incompatible",
    ):
        controller.acquire_scan(
            config=scan_config,
            polarization=Polarization.TRANSVERSE,
            baseline_index=baseline.baseline_index,
        )

    assert acquisition.acquire_count == acquisitions_before
    assert experiment.next_scan_index == 0
    assert experiment.scan_groups == []


def test_create_new_experiment_file(tmp_path):
    filename = tmp_path / "experiment.h5"

    controller = MSPController(
        acquisition=FakeAcquisitionController(),
    )

    assert controller.experiment is None
    assert controller.filename is None

    controller.create_new_experiment_file(filename)

    assert filename.exists()
    assert controller.filename == filename
    assert isinstance(controller.experiment, Experiment)


def test_create_new_experiment_file_does_not_set_filename_on_failure(
    tmp_path,
):
    filename = tmp_path / "experiment.h5"
    filename.write_text("already exists")

    controller = MSPController(
        acquisition=FakeAcquisitionController(),
    )

    with pytest.raises(FileExistsError):
        controller.create_new_experiment_file(filename)

    assert controller.filename is None
    assert controller.experiment is None


def test_create_new_experiment_file_rejects_second_file(tmp_path):
    first_filename = tmp_path / "first.h5"
    second_filename = tmp_path / "second.h5"

    acquisition = FakeAcquisitionController()

    controller = MSPController(
        acquisition=acquisition,
    )

    controller.create_new_experiment_file(first_filename)

    first_experiment = controller.experiment

    with pytest.raises(
        RuntimeError,
        match="An experiment is already active",
    ):
        controller.create_new_experiment_file(second_filename)

    assert controller.experiment is first_experiment
    assert controller.filename == first_filename
    assert first_filename.exists()
    assert not second_filename.exists()


def test_close_experiment_allows_new_experiment(tmp_path):
    first_filename = tmp_path / "first.h5"
    second_filename = tmp_path / "second.h5"

    controller = MSPController(
        acquisition=FakeAcquisitionController(),
    )

    controller.create_new_experiment_file(first_filename)
    first_experiment = controller.experiment

    controller.close_experiment_file()

    assert controller.experiment is None
    assert controller.filename is None

    controller.create_new_experiment_file(second_filename)

    assert controller.experiment is not None
    assert controller.experiment is not first_experiment
    assert controller.filename == second_filename

    assert first_filename.exists()
    assert second_filename.exists()


def test_close_experiment_rejects_no_active_experiment():
    controller = MSPController(
        acquisition=FakeAcquisitionController(),
    )

    with pytest.raises(
        RuntimeError,
        match="No experiment is active",
    ):
        controller.close_experiment_file()


def test_acquire_baseline_rejects_no_active_experiment():
    acquisition = FakeAcquisitionController()

    controller = MSPController(
        acquisition=acquisition,
    )

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

    with pytest.raises(
        RuntimeError,
        match="No experiment is active",
    ):
        controller.acquire_baseline(
            config=config,
            polarization=Polarization.TRANSVERSE,
        )

    assert acquisition.acquire_count == 0
    

def test_acquire_baseline_does_not_modify_experiment_on_storage_failure(
    tmp_path,
    monkeypatch,
):
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

    acquisition = FakeAcquisitionController()

    controller = MSPController(
        acquisition=acquisition,
    )

    controller.create_new_experiment_file(
        tmp_path / "experiment.h5"
    )

    experiment = controller.experiment
    assert experiment is not None

    def failing_write_baseline(filename, baseline):
        raise OSError("Storage failed")

    monkeypatch.setattr(
        "msp_control.controller.write_baseline",
        failing_write_baseline,
    )

    with pytest.raises(
        OSError,
        match="Storage failed",
    ):
        controller.acquire_baseline(
            config=config,
            polarization=Polarization.TRANSVERSE,
        )

    assert experiment.baselines == []
    assert experiment.next_baseline_index == 0
    assert acquisition.acquire_count == 1


def test_acquire_scan_rejects_no_active_experiment():
    acquisition = FakeAcquisitionController()

    controller = MSPController(
        acquisition=acquisition,
    )

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

    with pytest.raises(
        RuntimeError,
        match="No experiment is active",
    ):
        controller.acquire_scan(
            config=config,
            polarization=Polarization.TRANSVERSE,
            baseline_index=0,
        )

    assert acquisition.acquire_count == 0


def test_acquire_scan_does_not_modify_experiment_on_storage_failure(
    tmp_path,
    monkeypatch,
):
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

    acquisition = FakeAcquisitionController()

    controller = MSPController(
        acquisition=acquisition,
    )

    controller.create_new_experiment_file(
        tmp_path / "experiment.h5"
    )

    experiment = controller.experiment
    assert experiment is not None

    baseline = controller.acquire_baseline(
        config=config,
        polarization=Polarization.TRANSVERSE,
    )

    acquisitions_before = acquisition.acquire_count

    def failing_write_scan(filename, scan):
        raise OSError("Storage failed")

    monkeypatch.setattr(
        "msp_control.controller.write_scan",
        failing_write_scan,
    )

    with pytest.raises(
        OSError,
        match="Storage failed",
    ):
        controller.acquire_scan(
            config=config,
            polarization=Polarization.TRANSVERSE,
            baseline_index=baseline.baseline_index,
        )

    assert experiment.scans == []
    assert experiment.next_scan_index == 0
    assert acquisition.acquire_count == acquisitions_before + 1

def test_update_experiment_metadata(tmp_path):
    filename = tmp_path / "experiment.h5"

    controller = MSPController(
        acquisition=FakeAcquisitionController(),
    )

    controller.create_new_experiment_file(filename)

    controller.update_experiment_metadata({
        "animal": "A123",
        "species": "mouse",
    })

    assert controller.experiment is not None
    assert controller.experiment.shared_metadata == {
        "animal": "A123",
        "species": "mouse",
    }

    controller.update_experiment_metadata({
        "animal": "A124",
    })

    assert controller.experiment.shared_metadata == {
        "animal": "A124",
        "species": "mouse",
    }

    stored_metadata = read_experiment_metadata(filename)

    assert stored_metadata == {
        "animal": "A124",
        "species": "mouse",
    }

def test_update_experiment_metadata_rejects_no_active_experiment():
    controller = MSPController(
        acquisition=FakeAcquisitionController(),
    )

    with pytest.raises(
        RuntimeError,
        match="No experiment is active",
    ):
        controller.update_experiment_metadata({
            "animal": "A123",
        })

def test_update_experiment_metadata_does_not_modify_experiment_on_storage_failure(
    tmp_path,
    monkeypatch,
):
    controller = MSPController(
        acquisition=FakeAcquisitionController(),
    )

    controller.create_new_experiment_file(
        tmp_path / "experiment.h5"
    )

    controller.update_experiment_metadata({
        "animal": "A123",
        "species": "mouse",
    })

    experiment = controller.experiment
    assert experiment is not None

    def failing_write_experiment_metadata(filename, metadata):
        raise OSError("Storage failed")

    monkeypatch.setattr(
        "msp_control.controller.write_experiment_metadata",
        failing_write_experiment_metadata,
    )

    with pytest.raises(
        OSError,
        match="Storage failed",
    ):
        controller.update_experiment_metadata({
            "animal": "A124",
        })

    assert experiment.shared_metadata == {
        "animal": "A123",
        "species": "mouse",
    }


def test_add_scan_group(tmp_path):
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

    controller = MSPController(
        acquisition=FakeAcquisitionController(),
    )

    controller.create_new_experiment_file(
        tmp_path / "experiment.h5"
    )

    baseline = controller.acquire_baseline(
        config=config,
        polarization=Polarization.TRANSVERSE,
    )

    scan = controller.acquire_scan(
        config=config,
        polarization=Polarization.TRANSVERSE,
        baseline_index=baseline.baseline_index,
    )

    scan_group = ScanGroup(
        scan_group_id=0,
        label="Control",
        scan_indices=[scan.scan_index],
    )

    controller.add_scan_group(scan_group)

    experiment = controller.experiment
    assert experiment is not None

    assert len(experiment.scan_groups) == 1
    assert experiment.scan_groups[0] is scan_group

    stored_group = read_scan_group(
        controller.filename,
        scan_group.scan_group_id,
    )

    assert stored_group.scan_group_id == 0
    assert stored_group.label == "Control"
    assert stored_group.scan_indices == [scan.scan_index]


def test_add_scan_group_rejects_no_active_experiment():
    controller = MSPController(
        acquisition=FakeAcquisitionController(),
    )

    scan_group = ScanGroup(
        scan_group_id=0,
        label="Control",
    )

    with pytest.raises(
        RuntimeError,
        match="No experiment is active",
    ):
        controller.add_scan_group(scan_group)


def test_add_scan_group_does_not_modify_experiment_on_storage_failure(
    tmp_path,
    monkeypatch,
):
    controller = MSPController(
        acquisition=FakeAcquisitionController(),
    )

    controller.create_new_experiment_file(
        tmp_path / "experiment.h5"
    )

    experiment = controller.experiment
    assert experiment is not None

    scan_group = ScanGroup(
        scan_group_id=0,
        label="Control",
    )

    def failing_write_scan_group(filename, scan_group):
        raise OSError("Storage failed")

    monkeypatch.setattr(
        "msp_control.controller.write_scan_group",
        failing_write_scan_group,
    )

    with pytest.raises(
        OSError,
        match="Storage failed",
    ):
        controller.add_scan_group(scan_group)

    assert experiment.scan_groups == []

def test_add_scan_to_group(tmp_path):
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

    controller = MSPController(
        acquisition=FakeAcquisitionController(),
    )

    controller.create_new_experiment_file(
        tmp_path / "experiment.h5"
    )

    baseline = controller.acquire_baseline(
        config=config,
        polarization=Polarization.TRANSVERSE,
    )

    scan = controller.acquire_scan(
        config=config,
        polarization=Polarization.TRANSVERSE,
        baseline_index=baseline.baseline_index,
    )

    scan_group = ScanGroup(
        scan_group_id=0,
        label="Control",
    )

    controller.add_scan_group(scan_group)

    # The group starts empty.
    assert scan_group.scan_indices == []

    controller.add_scan_to_group(
        scan_index=scan.scan_index,
        scan_group_id=scan_group.scan_group_id,
    )

    # The in-memory group was updated.
    assert scan_group.scan_indices == [
        scan.scan_index,
    ]

    # The persisted group was updated as well.
    stored_group = read_scan_group(
        controller.filename,
        scan_group.scan_group_id,
    )

    assert stored_group.scan_group_id == scan_group.scan_group_id
    assert stored_group.label == "Control"
    assert stored_group.scan_indices == [
        scan.scan_index,
    ]

def test_add_scan_to_group_does_not_modify_experiment_on_storage_failure(
    tmp_path,
    monkeypatch,
):
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

    controller = MSPController(
        acquisition=FakeAcquisitionController(),
    )

    controller.create_new_experiment_file(
        tmp_path / "experiment.h5"
    )

    baseline = controller.acquire_baseline(
        config=config,
        polarization=Polarization.TRANSVERSE,
    )

    scan = controller.acquire_scan(
        config=config,
        polarization=Polarization.TRANSVERSE,
        baseline_index=baseline.baseline_index,
    )

    scan_group = ScanGroup(
        scan_group_id=0,
        label="Control",
    )

    controller.add_scan_group(scan_group)

    def failing_update_scan_group(filename, scan_group):
        raise OSError("Storage failed")

    monkeypatch.setattr(
        "msp_control.controller.update_scan_group",
        failing_update_scan_group,
    )

    with pytest.raises(
        OSError,
        match="Storage failed",
    ):
        controller.add_scan_to_group(
            scan_index=scan.scan_index,
            scan_group_id=scan_group.scan_group_id,
        )

    assert scan_group.scan_indices == []

    stored_group = read_scan_group(
        controller.filename,
        scan_group.scan_group_id,
    )

    assert stored_group.scan_indices == []

def test_add_scan_to_group_rejects_duplicate_membership(tmp_path):
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

    controller = MSPController(
        acquisition=FakeAcquisitionController(),
    )

    controller.create_new_experiment_file(
        tmp_path / "experiment.h5"
    )

    baseline = controller.acquire_baseline(
        config=config,
        polarization=Polarization.TRANSVERSE,
    )

    scan = controller.acquire_scan(
        config=config,
        polarization=Polarization.TRANSVERSE,
        baseline_index=baseline.baseline_index,
    )

    scan_group = ScanGroup(
        scan_group_id=0,
        label="Control",
        scan_indices=[scan.scan_index],
    )

    controller.add_scan_group(scan_group)

    with pytest.raises(
        ValueError,
        match="already belongs",
    ):
        controller.add_scan_to_group(
            scan_index=scan.scan_index,
            scan_group_id=scan_group.scan_group_id,
        )

    assert scan_group.scan_indices == [scan.scan_index]

def test_add_scan_to_group_rejects_missing_scan(tmp_path):
    controller = MSPController(
        acquisition=FakeAcquisitionController(),
    )

    controller.create_new_experiment_file(
        tmp_path / "experiment.h5"
    )

    scan_group = ScanGroup(
        scan_group_id=0,
        label="Control",
    )

    controller.add_scan_group(scan_group)

    with pytest.raises(
        KeyError,
        match="Scan index 7 does not exist",
    ):
        controller.add_scan_to_group(
            scan_index=7,
            scan_group_id=0,
        )

    assert scan_group.scan_indices == []

def test_add_scan_to_group_rejects_missing_group(tmp_path):
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

    controller = MSPController(
        acquisition=FakeAcquisitionController(),
    )

    controller.create_new_experiment_file(
        tmp_path / "experiment.h5"
    )

    baseline = controller.acquire_baseline(
        config=config,
        polarization=Polarization.TRANSVERSE,
    )

    scan = controller.acquire_scan(
        config=config,
        polarization=Polarization.TRANSVERSE,
        baseline_index=baseline.baseline_index,
    )

    scan_group = ScanGroup(
        scan_group_id=0,
        label="Control",
    )

    controller.add_scan_group(scan_group)

    with pytest.raises(
        KeyError,
        match="Scan group 7 does not exist",
    ):
        controller.add_scan_to_group(
            scan_index=scan.scan_index,
            scan_group_id=7,
        )

    assert scan_group.scan_indices == []

def test_remove_scan_from_group(tmp_path):
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

    controller = MSPController(
        acquisition=FakeAcquisitionController(),
    )

    controller.create_new_experiment_file(
        tmp_path / "experiment.h5"
    )

    baseline = controller.acquire_baseline(
        config=config,
        polarization=Polarization.TRANSVERSE,
    )

    scan = controller.acquire_scan(
        config=config,
        polarization=Polarization.TRANSVERSE,
        baseline_index=baseline.baseline_index,
    )

    scan_group = ScanGroup(
        scan_group_id=0,
        label="Control",
        scan_indices=[scan.scan_index],
    )

    controller.add_scan_group(scan_group)

    controller.remove_scan_from_group(
        scan_index=scan.scan_index,
        scan_group_id=scan_group.scan_group_id,
    )

    assert scan_group.scan_indices == []

    stored_group = read_scan_group(
        controller.filename,
        scan_group.scan_group_id,
    )

    assert stored_group.scan_indices == []

def test_remove_scan_from_group_does_not_modify_experiment_on_storage_failure(
    tmp_path,
    monkeypatch,
):
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

    controller = MSPController(
        acquisition=FakeAcquisitionController(),
    )

    controller.create_new_experiment_file(
        tmp_path / "experiment.h5"
    )

    baseline = controller.acquire_baseline(
        config=config,
        polarization=Polarization.TRANSVERSE,
    )

    scan = controller.acquire_scan(
        config=config,
        polarization=Polarization.TRANSVERSE,
        baseline_index=baseline.baseline_index,
    )

    scan_group = ScanGroup(
        scan_group_id=0,
        label="Control",
        scan_indices=[scan.scan_index],
    )

    controller.add_scan_group(scan_group)

    def failing_update_scan_group(filename, scan_group):
        raise OSError("Storage failed")

    monkeypatch.setattr(
        "msp_control.controller.update_scan_group",
        failing_update_scan_group,
    )

    with pytest.raises(
        OSError,
        match="Storage failed",
    ):
        controller.remove_scan_from_group(
            scan_index=scan.scan_index,
            scan_group_id=scan_group.scan_group_id,
        )

    assert scan_group.scan_indices == [
        scan.scan_index,
    ]

    stored_group = read_scan_group(
        controller.filename,
        scan_group.scan_group_id,
    )

    assert stored_group.scan_indices == [
        scan.scan_index,
    ]

def test_remove_scan_from_group_rejects_nonmember(tmp_path):
    controller = MSPController(
        acquisition=FakeAcquisitionController(),
    )

    controller.create_new_experiment_file(
        tmp_path / "experiment.h5"
    )

    scan_group = ScanGroup(
        scan_group_id=0,
        label="Control",
    )

    controller.add_scan_group(scan_group)

    with pytest.raises(
        ValueError,
        match="Scan 7 does not belong to scan group 0",
    ):
        controller.remove_scan_from_group(
            scan_index=7,
            scan_group_id=0,
        )

    assert scan_group.scan_indices == []

    stored_group = read_scan_group(
        controller.filename,
        scan_group.scan_group_id,
    )

    assert stored_group.scan_indices == []

def test_remove_scan_from_group_rejects_missing_group(tmp_path):
    controller = MSPController(
        acquisition=FakeAcquisitionController(),
    )

    controller.create_new_experiment_file(
        tmp_path / "experiment.h5"
    )

    with pytest.raises(
        KeyError,
        match="Scan group 7 does not exist",
    ):
        controller.remove_scan_from_group(
            scan_index=0,
            scan_group_id=7,
        )

def test_rename_scan_group(tmp_path):
    controller = MSPController(
        acquisition=FakeAcquisitionController(),
    )

    controller.create_new_experiment_file(
        tmp_path / "experiment.h5"
    )

    scan_group = ScanGroup(
        scan_group_id=0,
        label="Control",
    )

    controller.add_scan_group(scan_group)

    controller.rename_scan_group(
        scan_group_id=0,
        label="Treatment",
    )

    assert scan_group.label == "Treatment"

    stored_group = read_scan_group(
        controller.filename,
        scan_group.scan_group_id,
    )

    assert stored_group.scan_group_id == 0
    assert stored_group.label == "Treatment"
    assert stored_group.scan_indices == []

def test_rename_scan_group_does_not_modify_experiment_on_storage_failure(
    tmp_path,
    monkeypatch,
):
    controller = MSPController(
        acquisition=FakeAcquisitionController(),
    )

    controller.create_new_experiment_file(
        tmp_path / "experiment.h5"
    )

    scan_group = ScanGroup(
        scan_group_id=0,
        label="Control",
    )

    controller.add_scan_group(scan_group)

    def failing_update_scan_group(filename, scan_group):
        raise OSError("Storage failed")

    monkeypatch.setattr(
        "msp_control.controller.update_scan_group",
        failing_update_scan_group,
    )

    with pytest.raises(
        OSError,
        match="Storage failed",
    ):
        controller.rename_scan_group(
            scan_group_id=0,
            label="Treatment",
        )

    # The live model must retain the old label.
    assert scan_group.label == "Control"

    # The persisted group must also retain the old label.
    stored_group = read_scan_group(
        controller.filename,
        scan_group.scan_group_id,
    )

    assert stored_group.scan_group_id == 0
    assert stored_group.label == "Control"
    assert stored_group.scan_indices == []

def test_rename_scan_group_requires_active_experiment():
    controller = MSPController(
        acquisition=FakeAcquisitionController(),
    )

    with pytest.raises(
        RuntimeError,
        match="No experiment is active",
    ):
        controller.rename_scan_group(
            scan_group_id=0,
            label="Treatment",
        )

def test_rename_scan_group_rejects_missing_group(tmp_path):
    controller = MSPController(
        acquisition=FakeAcquisitionController(),
    )

    controller.create_new_experiment_file(
        tmp_path / "experiment.h5"
    )

    with pytest.raises(
        KeyError,
        match="Scan group 7 does not exist",
    ):
        controller.rename_scan_group(
            scan_group_id=7,
            label="Treatment",
        )

def test_rename_scan_group_rejects_non_string_label(tmp_path):
    controller = MSPController(
        acquisition=FakeAcquisitionController(),
    )

    controller.create_new_experiment_file(
        tmp_path / "experiment.h5"
    )

    scan_group = ScanGroup(
        scan_group_id=0,
        label="Control",
    )

    controller.add_scan_group(scan_group)

    with pytest.raises(
        TypeError,
        match="label must be a string",
    ):
        controller.rename_scan_group(
            scan_group_id=0,
            label=123,
        )

    assert scan_group.label == "Control"

    stored_group = read_scan_group(
        controller.filename,
        scan_group.scan_group_id,
    )

    assert stored_group.label == "Control"

def test_acquire_baseline_uses_clock_timestamp(tmp_path):
    timestamp = datetime.fromisoformat(
        "2026-10-06T08:31:42.381234-07:00"
    )
    
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

    acquisition = FakeAcquisitionController()

    controller = MSPController(
        acquisition=acquisition,
        clock=FakeClock(timestamp),
    )

    controller.create_new_experiment_file(
        tmp_path / "experiment.h5"
    )

    baseline = controller.acquire_baseline(
        config=config,
        polarization=Polarization.TRANSVERSE,
    )

    assert baseline.timestamp == timestamp


def test_acquire_scan_uses_clock_timestamp(tmp_path):
    timestamp = datetime.fromisoformat(
        "2026-10-06T08:31:42.381234-07:00"
    )

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

    acquisition = FakeAcquisitionController()

    controller = MSPController(
        acquisition=acquisition,
        clock=FakeClock(timestamp),
    )

    controller.create_new_experiment_file(
        tmp_path / "experiment.h5"
    )

    baseline = controller.acquire_baseline(
        config=config,
        polarization=Polarization.TRANSVERSE,
    )

    scan = controller.acquire_scan(
        config=config,
        baseline_index=baseline.baseline_index,
        polarization=Polarization.TRANSVERSE,
    )

    assert scan.timestamp == timestamp
