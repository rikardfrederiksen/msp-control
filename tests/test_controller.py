import numpy as np
import pytest

from msp_control.acquisition import RawAcquisition
from msp_control.config import ScanConfig
from msp_control.controller import MSPController
from msp_control.data.model import Experiment, Polarization


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


def test_acquire_baseline():
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

    experiment = Experiment()
    acquisition = FakeAcquisitionController()

    controller = MSPController(
        experiment=experiment,
        acquisition=acquisition,
    )

    baseline = controller.acquire_baseline(
        config=config,
        polarization=Polarization.TRANSVERSE,
    )

    assert baseline.baseline_index == 0
    assert baseline.polarization is Polarization.TRANSVERSE
    assert baseline.config is config

    assert len(experiment.baselines) == 1
    assert experiment.baselines[0] is baseline

    np.testing.assert_allclose(
        baseline.baseline_corrected,
        [10.0, 10.0, 10.0],
    )


class FailingAcquisitionController:
    def acquire(self, config):
        raise RuntimeError("Acquisition failed")


def test_acquire_baseline_does_not_modify_experiment_on_failure():
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

    experiment = Experiment()

    controller = MSPController(
        experiment=experiment,
        acquisition=FailingAcquisitionController(),
    )

    with pytest.raises(RuntimeError, match="Acquisition failed"):
        controller.acquire_baseline(
            config=config,
            polarization=Polarization.TRANSVERSE,
        )

    assert experiment.baselines == []
    assert experiment.next_baseline_index == 0

def test_acquire_scan():
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

    experiment = Experiment()
    acquisition = FakeAcquisitionController()

    controller = MSPController(
        experiment=experiment,
        acquisition=acquisition,
    )

    baseline = controller.acquire_baseline(
        config=config,
        polarization=Polarization.TRANSVERSE,
    )

    scan = controller.acquire_scan(
        config=config,
        polarization=Polarization.TRANSVERSE,
        baseline_index=baseline.baseline_index,
        scan_group_id="cell_1",
    )

    assert scan.scan_index == 0
    assert scan.baseline_index == baseline.baseline_index
    assert scan.polarization is Polarization.TRANSVERSE
    assert scan.config is config

    assert len(experiment.scan_groups) == 1

    group = experiment.scan_groups[0]

    assert group.scan_group_id == "cell_1"
    assert len(group.scans) == 1
    assert group.scans[0] is scan

    np.testing.assert_allclose(
        scan.optical_density,
        [0.0, 0.0, 0.0],
    )

def test_acquire_scan_rejects_missing_baseline_before_acquisition():
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

    experiment = Experiment()
    acquisition = FakeAcquisitionController()

    controller = MSPController(
        experiment=experiment,
        acquisition=acquisition,
    )

    with pytest.raises(
        KeyError,
        match="Baseline index 7 does not exist",
    ):
        controller.acquire_scan(
            config=config,
            polarization=Polarization.TRANSVERSE,
            baseline_index=7,
            scan_group_id="cell_1",
        )

    assert acquisition.acquire_count == 0
    assert experiment.next_scan_index == 0
    assert experiment.scan_groups == []

def test_acquire_scan_rejects_wrong_polarization_before_acquisition():
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

    experiment = Experiment()
    acquisition = FakeAcquisitionController()

    controller = MSPController(
        experiment=experiment,
        acquisition=acquisition,
    )

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
            scan_group_id="cell_1",
        )

    assert acquisition.acquire_count == acquisitions_before
    assert experiment.next_scan_index == 0
    assert experiment.scan_groups == []

def test_acquire_scan_rejects_incompatible_config_before_acquisition():
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

    experiment = Experiment()
    acquisition = FakeAcquisitionController()

    controller = MSPController(
        experiment=experiment,
        acquisition=acquisition,
    )

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
            scan_group_id="cell_1",
        )

    assert acquisition.acquire_count == acquisitions_before
    assert experiment.next_scan_index == 0
    assert experiment.scan_groups == []

