from pathlib import Path

import numpy as np

from msp_control.metadata import MetadataCollector
from msp_control.clock import Clock
from msp_control.acquisition import AcquisitionController
from msp_control.bleach import BleachController
from msp_control.config import ScanConfig
from msp_control.data.model import Baseline, Experiment, Polarization, Scan, ScanGroup, Event, Bleach
from msp_control.data.processing import process_baseline, process_scan
from msp_control.calibration.waveform import BleachWaveformConfig
from msp_control.storage.hdf5 import (
    create_experiment_file,
    write_baseline,
    write_scan,
    write_experiment_metadata,
    write_scan_group,
    update_scan_group,
    write_event,
    write_bleach,
    update_event,
)

class MSPController:
    def __init__(
        self,
        acquisition: AcquisitionController,
        clock: Clock | None = None,
        metadata_collector: MetadataCollector | None = None,
        bleach_controller: BleachController | None = None,
    ):
        self.acquisition = acquisition
        self.bleach_controller = bleach_controller
        self.clock = clock if clock is not None else Clock()
        self.metadata_collector = metadata_collector

        self.experiment: Experiment | None = None
        self.filename: Path | None = None

    def acquire_baseline(
        self,
        config: ScanConfig,
        polarization: Polarization,
    ) -> Baseline:
        """Acquire, process, persist, and add a baseline."""

        if self.experiment is None or self.filename is None:
            raise RuntimeError(
                "No experiment is active"
            )

        baseline_index = self.experiment.next_baseline_index
        timestamp = self.clock.now()

        # Collect metadata before acquisition, without committing it.
        pending_metadata = (
            self.metadata_collector.collect()
            if self.metadata_collector is not None
            else {}
        )

        raw = self.acquisition.acquire(config)

        baseline = process_baseline(
            acquisition=raw,
            config=config,
            baseline_index=baseline_index,
            polarization=polarization,
            timestamp=timestamp,
        )
        
        # Attach metadata only after successful acquisition and processing.
        baseline.metadata.update(pending_metadata)

        write_baseline(self.filename, baseline)

        self.experiment.add_baseline(baseline)

        return baseline

    def acquire_scan(
        self,
        config: ScanConfig,
        polarization: Polarization,
        baseline_index: int,
    ) -> Scan:
        """Acquire, process, persist, and add a specimen scan."""

        if self.experiment is None or self.filename is None:
            raise RuntimeError(
                "No experiment is active"
            )

        scan_index = self.experiment.next_scan_index
        timestamp = self.clock.now()
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

        # Collect metadata before acquisition, without committing it.
        pending_metadata = (
            self.metadata_collector.collect()
            if self.metadata_collector is not None
            else {}
        )
        
        raw = self.acquisition.acquire(config)

        scan = process_scan(
            acquisition=raw,
            config=config,
            baseline=baseline,
            scan_index=scan_index,
            polarization=polarization,
            timestamp=timestamp,
        )

        # Attach metadata only after successful acquisition and processing.
        scan.metadata.update(pending_metadata)

        # Persist the completed scan.
        write_scan(
            self.filename,
            scan,
        )
        
        self.experiment.add_scan(scan)

        return scan

    def acquire_bleach(
        self,
        led_channel: int,
        config: BleachWaveformConfig,
        nd_filter: float = 0.0,
        command_voltage: float | None = None,
    ) -> Bleach:
        """Acquire, persist, and register a bleaching exposure."""

        if self.experiment is None or self.filename is None:
            raise RuntimeError("No experiment is active")

        if self.bleach_controller is None:
            raise RuntimeError("No bleach controller configured")

        if led_channel not in (1, 2):
            raise ValueError("LED channel must be 1 or 2")

        if nd_filter < 0:
            raise ValueError("ND filter cannot be negative")

        if led_channel == 1:
            control_mode = "analog"

            if command_voltage is None:
                raise ValueError(
                    "command_voltage is required for analog bleaching"
                )

            if not np.isfinite(command_voltage) or not 0.0 <= command_voltage <= 10.0:
                raise ValueError(
                    "command_voltage must be between 0 and 10 V"
                )

        elif led_channel == 2:
            control_mode = "digital"

            if command_voltage is not None:
                raise ValueError(
                    "command_voltage must be None for digital bleaching"
                )

        bleach_index = self.experiment.next_bleach_index

        # Collect metadata before acquisition.
        pending_metadata = (
            self.metadata_collector.collect()
            if self.metadata_collector is not None
            else {}
        )

        # Persist the Event before activating the hardware.
        event = self.add_event("Bleach")

        try:
            raw = self.bleach_controller.acquire(
                led_channel=led_channel,
                config=config,
                command_voltage=command_voltage,
            )

        except Exception as exc:
            description = f"Failed bleach: {exc}"

            try:
                update_event(
                    self.filename,
                    event_id=event.event_id,
                    description=description,
                )
            except Exception as update_exc:
                print(f"Event update failed: {update_exc!r}")
            else:
                # Only update memory after persistence succeeds.
                event.description = description

            raise

        bleach = Bleach(
            bleach_index=bleach_index,
            event_id=event.event_id,
            timestamp=event.timestamp,
            led_channel=led_channel,
            command_voltage=command_voltage,
            requested_duration_s=config.duration_s,
            nd_filter=nd_filter,
            control_mode=control_mode,
            monitor_time_s=raw.time_s,
            monitor_voltage=raw.monitor_voltage,
            completed=True,
            metadata=pending_metadata,
        )

        # Persist before adding to the in-memory experiment.
        try:
            write_bleach(self.filename, bleach)

        except Exception as exc:
            description = f"Bleach completed; storage failed: {exc}"

            try:
                update_event(
                    self.filename,
                    event_id=event.event_id,
                    description=description,
                )
            except Exception:
                # Preserve the original storage exception.
                pass
            else:
                event.description = description

            raise

        self.experiment.add_bleach(bleach)

        return bleach

    def create_new_experiment_file(
        self,
        filename: str | Path,
    ) -> None:
        """Create and activate a new MSP experiment."""

        if self.experiment is not None:
            raise RuntimeError(
                "An experiment is already active"
            )

        filename = Path(filename)

        create_experiment_file(filename)

        self.experiment = Experiment()
        self.filename = filename

    def close_experiment_file(self) -> None:
        """Close the active MSP experiment."""

        if self.experiment is None:
            raise RuntimeError(
                "No experiment is active"
            )

        self.experiment = None
        self.filename = None

    def update_experiment_metadata(
        self,
        metadata: dict,
    ) -> None:
        """Add or update metadata for the active experiment."""

        if self.experiment is None or self.filename is None:
            raise RuntimeError(
                "No experiment is active"
            )

        write_experiment_metadata(
            self.filename,
            metadata,
        )

        self.experiment.shared_metadata.update(metadata)

    def add_scan_group(
        self,
        scan_group: ScanGroup,
    ) -> None:
        """Add and persist a scan group."""

        if self.experiment is None or self.filename is None:
            raise RuntimeError(
                "No experiment is active"
            )

        write_scan_group(
            self.filename,
            scan_group,
        )

        self.experiment.add_scan_group(scan_group)

    def add_scan_to_group(
        self,
        scan_index: int,
        scan_group_id: int,
    ) -> None:
        """Add a scan to an existing scan group."""

        if self.experiment is None or self.filename is None:
            raise RuntimeError(
                "No experiment is active"
            )

        # Validate the scan and group before touching storage.
        self.experiment.get_scan(scan_index)
        scan_group = self.experiment.get_scan_group(scan_group_id)

        if scan_index in scan_group.scan_indices:
            raise ValueError(
                f"Scan {scan_index} already belongs to "
                f"scan group {scan_group_id}"
            )

        updated_group = ScanGroup(
            scan_group_id=scan_group.scan_group_id,
            label=scan_group.label,
            scan_indices=[
                *scan_group.scan_indices,
                scan_index,
            ],
        )

        update_scan_group(
            self.filename,
            updated_group,
        )

        self.experiment.add_scan_to_group(
            scan_index,
            scan_group_id,
        )

    def remove_scan_from_group(
        self,
        scan_index: int,
        scan_group_id: int,
    ) -> None:
        """Remove a scan from an existing scan group."""

        if self.experiment is None or self.filename is None:
            raise RuntimeError(
                "No experiment is active"
            )

        scan_group = self.experiment.get_scan_group(scan_group_id)

        if scan_index not in scan_group.scan_indices:
            raise ValueError(
                f"Scan {scan_index} does not belong to "
                f"scan group {scan_group_id}"
            )

        updated_group = ScanGroup(
            scan_group_id=scan_group.scan_group_id,
            label=scan_group.label,
            scan_indices=[
                index
                for index in scan_group.scan_indices
                if index != scan_index
            ],
        )

        update_scan_group(
            self.filename,
            updated_group,
        )

        self.experiment.remove_scan_from_group(
            scan_index,
            scan_group_id,
        )

    def rename_scan_group(
        self,
        scan_group_id: int,
        label: str,
    ) -> None:
        """Rename an existing scan group."""

        if self.experiment is None or self.filename is None:
            raise RuntimeError(
                "No experiment is active"
            )

        scan_group = self.experiment.get_scan_group(
            scan_group_id
        )

        if not isinstance(label, str):
            raise TypeError("label must be a string")

        updated_group = ScanGroup(
            scan_group_id=scan_group.scan_group_id,
            label=label,
            scan_indices=list(scan_group.scan_indices),
        )

        update_scan_group(
            self.filename,
            updated_group,
        )

        self.experiment.rename_scan_group(
            scan_group_id,
            label,
        )


    def add_event(self, description: str) -> Event:
        """Record a timestamped event in the active experiment."""

        if self.experiment is None or self.filename is None:
            raise RuntimeError("No experiment is active")

        event = Event(
            event_id=self.experiment.next_event_id,
            timestamp=self.clock.now(),
            description=description,
        )

        write_event(self.filename, event)

        self.experiment.add_event(event)

        return event
