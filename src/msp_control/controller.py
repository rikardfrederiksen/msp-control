from msp_control.acquisition import AcquisitionController
from msp_control.config import ScanConfig
from msp_control.data.model import Baseline, Experiment, Polarization, Scan
from msp_control.data.processing import process_baseline, process_scan


class MSPController:
    """Coordinate high-level MSP experiment operations."""

    def __init__(
        self,
        experiment: Experiment,
        acquisition: AcquisitionController,
    ):
        self.experiment = experiment
        self.acquisition = acquisition

    def acquire_baseline(
        self,
        config: ScanConfig,
        polarization: Polarization,
    ) -> Baseline:
        """Acquire, process, and add a baseline to the experiment."""

        baseline_index = self.experiment.next_baseline_index

        raw = self.acquisition.acquire(config)

        baseline = process_baseline(
            acquisition=raw,
            config=config,
            baseline_index=baseline_index,
            polarization=polarization,
        )

        self.experiment.add_baseline(baseline)

        return baseline

    def acquire_scan(
        self,
        config: ScanConfig,
        polarization: Polarization,
        baseline_index: int,
        scan_group_id: int | str,
    ) -> Scan:
        """Acquire, process, and add a specimen scan to the experiment."""

        scan_index = self.experiment.next_scan_index
        baseline = self.experiment.get_baseline(baseline_index)

        if baseline.polarization is not polarization:
            raise ValueError(
                "Baseline polarization does not match scan polarization: "
                f"baseline={baseline.polarization.value}, "
                f"scan={polarization.value}"
            )

        if not baseline.config.is_compatible_with(config):
            raise ValueError(
                "Baseline and specimen scan configurations are incompatible"
            )

        raw = self.acquisition.acquire(config)

        scan = process_scan(
            acquisition=raw,
            config=config,
            baseline=baseline,
            scan_index=scan_index,
            polarization=polarization,
        )

        self.experiment.add_scan(
            scan,
            scan_group_id=scan_group_id,
        )

        return scan
