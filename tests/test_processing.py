import numpy as np
import pytest

from msp_control.acquisition import RawAcquisition
from msp_control.data.model import Polarization
from msp_control.data.processing import process_baseline, process_scan
from msp_control.config import ScanConfig


def test_process_baseline():
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

    acquisition = RawAcquisition(
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

    baseline = process_baseline(
        acquisition=acquisition,
        config=config,
        baseline_index=7,
        polarization=Polarization.TRANSVERSE,
    )

    assert baseline.polarization is Polarization.TRANSVERSE

    np.testing.assert_array_equal(
        baseline.wavelength,
        [500.0, 510.0, 520.0],
    )

    np.testing.assert_array_equal(
        baseline.raw_dark,
        [
            [1.0, 2.0, 3.0],
            [3.0, 4.0, 5.0],
        ],
    )

    np.testing.assert_array_equal(
        baseline.raw_baseline,
        [
            [11.0, 12.0, 13.0],
            [13.0, 14.0, 15.0],
        ],
    )

    np.testing.assert_array_equal(
        baseline.dark_mean,
        [2.0, 3.0, 4.0],
    )

    np.testing.assert_array_equal(
        baseline.baseline_mean,
        [12.0, 13.0, 14.0],
    )

    np.testing.assert_array_equal(
        baseline.baseline_corrected,
        [10.0, 10.0, 10.0],
    )

    assert baseline.baseline_index == 7

def test_process_scan():
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

    baseline_acquisition = RawAcquisition(
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

    baseline = process_baseline(
        acquisition=baseline_acquisition,
        config=config,
        baseline_index=7,
        polarization=Polarization.TRANSVERSE,
    )

    specimen_acquisition = RawAcquisition(
        dark=[
            np.array([1.0, 2.0, 3.0]),
            np.array([3.0, 4.0, 5.0]),
        ],
        transition=np.array([99.0, 99.0, 99.0]),
        data=[
            np.array([2.0, 3.0, 4.0]),
            np.array([4.0, 5.0, 6.0]),
        ],
    )

    scan = process_scan(
        acquisition=specimen_acquisition,
        config=config,
        baseline=baseline,
        scan_index=12,
        polarization=Polarization.TRANSVERSE,
    )

    assert scan.scan_index == 12
    assert scan.baseline_index == 7
    assert scan.polarization is Polarization.TRANSVERSE

    np.testing.assert_array_equal(
        scan.wavelength,
        [500.0, 510.0, 520.0],
    )

    np.testing.assert_array_equal(
        scan.raw_dark,
        [
            [1.0, 2.0, 3.0],
            [3.0, 4.0, 5.0],
        ],
    )

    np.testing.assert_array_equal(
        scan.raw_specimen,
        [
            [2.0, 3.0, 4.0],
            [4.0, 5.0, 6.0],
        ],
    )

    np.testing.assert_array_equal(
        scan.dark_mean,
        [2.0, 3.0, 4.0],
    )

    np.testing.assert_array_equal(
        scan.specimen_mean,
        [3.0, 4.0, 5.0],
    )

    np.testing.assert_array_equal(
        scan.specimen_corrected,
        [1.0, 1.0, 1.0],
    )

    np.testing.assert_allclose(
        scan.optical_density,
        [1.0, 1.0, 1.0],
    )
def test_process_scan_rejects_mismatched_polarization():
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

    baseline_acquisition = RawAcquisition(
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

    baseline = process_baseline(
        acquisition=baseline_acquisition,
        config=config,
        baseline_index=7,
        polarization=Polarization.TRANSVERSE,
    )

    specimen_acquisition = RawAcquisition(
        dark=[
            np.array([1.0, 2.0, 3.0]),
            np.array([3.0, 4.0, 5.0]),
        ],
        transition=np.array([99.0, 99.0, 99.0]),
        data=[
            np.array([2.0, 3.0, 4.0]),
            np.array([4.0, 5.0, 6.0]),
        ],
    )

    with pytest.raises(
        ValueError,
        match="Baseline polarization does not match scan polarization",
    ):
        process_scan(
            acquisition=specimen_acquisition,
            config=config,
            baseline=baseline,
            scan_index=12,
            polarization=Polarization.LONGITUDINAL,
        )


def test_process_scan_rejects_incompatible_config():
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

    baseline_acquisition = RawAcquisition(
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

    baseline = process_baseline(
        acquisition=baseline_acquisition,
        config=baseline_config,
        baseline_index=7,
        polarization=Polarization.TRANSVERSE,
    )

    specimen_config = ScanConfig(
        start_nm=500,
        end_nm=520,
        step_nm=5,       # deliberately different
        step_time_ms=2,
        input_slit_nm=4,
        output_slit_nm=4,
        dark_scans=2,
        data_scans=2,
    )

    specimen_acquisition = RawAcquisition(
        dark=[
            np.ones(5),
            np.ones(5),
        ],
        transition=np.ones(5),
        data=[
            np.ones(5),
            np.ones(5),
        ],
    )

    with pytest.raises(
        ValueError,
        match="Baseline and specimen scan configurations are incompatible",
    ):
        process_scan(
            acquisition=specimen_acquisition,
            config=specimen_config,
            baseline=baseline,
            scan_index=12,
            polarization=Polarization.TRANSVERSE,
        )

def test_process_scan_allows_different_sweep_counts():
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

    baseline_acquisition = RawAcquisition(
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

    baseline = process_baseline(
        acquisition=baseline_acquisition,
        config=baseline_config,
        baseline_index=7,
        polarization=Polarization.TRANSVERSE,
    )

    specimen_config = ScanConfig(
        start_nm=500,
        end_nm=520,
        step_nm=10,
        step_time_ms=2,
        input_slit_nm=4,
        output_slit_nm=4,
        dark_scans=3,       # different, but allowed
        data_scans=4,       # different, but allowed
    )

    specimen_acquisition = RawAcquisition(
        dark=[
            np.array([1.0, 2.0, 3.0]),
            np.array([2.0, 3.0, 4.0]),
            np.array([3.0, 4.0, 5.0]),
        ],
        transition=np.array([99.0, 99.0, 99.0]),
        data=[
            np.array([2.0, 3.0, 4.0]),
            np.array([3.0, 4.0, 5.0]),
            np.array([4.0, 5.0, 6.0]),
            np.array([5.0, 6.0, 7.0]),
        ],
    )

    scan = process_scan(
        acquisition=specimen_acquisition,
        config=specimen_config,
        baseline=baseline,
        scan_index=12,
        polarization=Polarization.TRANSVERSE,
    )

    assert scan.config.dark_scans == 3
    assert scan.config.data_scans == 4
    assert scan.raw_dark.shape == (3, 3)
    assert scan.raw_specimen.shape == (4, 3)
