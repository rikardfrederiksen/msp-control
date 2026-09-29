from pathlib import Path

import h5py
import numpy as np

from msp_control.config import ScanConfig
from msp_control.data.model import Baseline, Polarization, Scan, ScanGroup, Experiment


FORMAT_NAME = "msp-control"
FORMAT_VERSION = 1
SOFTWARE_VERSION = "0.1.0"


class BaselineAlreadyExistsError(ValueError):
    pass


class ScanAlreadyExistsError(ValueError):
    pass


class ScanGroupAlreadyExistsError(ValueError):
    pass


def write_baseline(
    filename: str | Path,
    baseline: Baseline,
) -> None:
    """Write a baseline to an MSP HDF5 file."""

    _validate_metadata(baseline.metadata)

    with h5py.File(filename, "a") as h5:
        _initialize_file(h5)

        baselines_group = h5.require_group("baselines")
        pending_group = h5.require_group("_pending")

        group_name = f"baseline_{baseline.baseline_index:03d}"

        if group_name in baselines_group:
            raise BaselineAlreadyExistsError(
                f"Baseline {baseline.baseline_index} already exists"
            )

        if group_name in pending_group:
            del pending_group[group_name]

        baseline_group = pending_group.create_group(group_name)

        try:
            baseline_group.attrs["baseline_index"] = baseline.baseline_index
            baseline_group.attrs["polarization"] = baseline.polarization.value

            config_group = baseline_group.create_group("config")
            _write_scan_config(config_group, baseline.config)

            data_group = baseline_group.create_group("data")

            data_group.create_dataset(
                "wavelength",
                data=baseline.wavelength,
            )
            data_group.create_dataset(
                "raw_dark",
                data=baseline.raw_dark,
            )
            data_group.create_dataset(
                "raw_transition",
                data=baseline.raw_transition,
            )
            data_group.create_dataset(
                "raw_baseline",
                data=baseline.raw_baseline,
            )
            data_group.create_dataset(
                "dark_mean",
                data=baseline.dark_mean,
            )
            data_group.create_dataset(
                "baseline_mean",
                data=baseline.baseline_mean,
            )
            data_group.create_dataset(
                "baseline_corrected",
                data=baseline.baseline_corrected,
            )

            metadata_group = baseline_group.create_group("metadata")
            _write_metadata(metadata_group, baseline.metadata)

            _commit_pending_group(
                h5,
                f"_pending/{group_name}",
                f"baselines/{group_name}",
            )

        except Exception:
            if group_name in pending_group:
                del pending_group[group_name]
                h5.flush()

            raise


def write_scan(
    filename: str | Path,
    scan: Scan,
) -> None:
    """Write a specimen scan to an MSP HDF5 file."""

    _validate_metadata(scan.metadata)

    with h5py.File(filename, "a") as h5:
        _initialize_file(h5)

        baselines_group = h5.require_group("baselines")
        scans_group = h5.require_group("scans")
        pending_group = h5.require_group("_pending")

        baseline_name = f"baseline_{scan.baseline_index:03d}"

        # Validate provenance first.
        if baseline_name not in baselines_group:
            raise ValueError(
                f"Baseline {scan.baseline_index} does not exist "
                "in the HDF5 file"
            )

        group_name = f"scan_{scan.scan_index:03d}"

        # Protect completed measurement.
        if group_name in scans_group:
            raise ScanAlreadyExistsError(
                f"Scan {scan.scan_index} already exists"
            )

        # Discard an incomplete previous attempt.
        if group_name in pending_group:
            del pending_group[group_name]

        scan_group = pending_group.create_group(group_name)

        try:
            scan_group.attrs["scan_index"] = scan.scan_index
            scan_group.attrs["baseline_index"] = scan.baseline_index
            scan_group.attrs["polarization"] = scan.polarization.value

            config_group = scan_group.create_group("config")
            _write_scan_config(config_group, scan.config)

            data_group = scan_group.create_group("data")
            data_group.create_dataset(
                "wavelength",
                data=scan.wavelength,
            )
            data_group.create_dataset(
                "raw_dark",
                data=scan.raw_dark,
            )
            data_group.create_dataset(
                "raw_transition",
                data=scan.raw_transition,
            )
            data_group.create_dataset(
                "raw_specimen",
                data=scan.raw_specimen,
            )
            data_group.create_dataset(
                "dark_mean",
                data=scan.dark_mean,
            )
            data_group.create_dataset(
                "specimen_mean",
                data=scan.specimen_mean,
            )
            data_group.create_dataset(
                "specimen_corrected",
                data=scan.specimen_corrected,
            )
            data_group.create_dataset(
                "optical_density",
                data=scan.optical_density,
            )

            metadata_group = scan_group.create_group("metadata")
            _write_metadata(metadata_group, scan.metadata)

            _commit_pending_group(
                h5,
                f"_pending/{group_name}",
                f"scans/{group_name}",
            )

        except Exception:
            if group_name in pending_group:
                del pending_group[group_name]
                h5.flush()

            raise


def read_baseline(
    filename: str | Path,
    baseline_index: int,
) -> Baseline:
    """Read a baseline from an MSP HDF5 file."""

    with h5py.File(filename, "r") as h5:
        _validate_file(h5)
        
        baseline_group = h5[
            f"baselines/baseline_{baseline_index:03d}"
        ]

        config_group = baseline_group["config"]
        data_group = baseline_group["data"]
        metadata_group = baseline_group["metadata"]

        config = _read_scan_config(config_group)
        
        polarization = Polarization(
            baseline_group.attrs["polarization"]
        )

        return Baseline(
            baseline_index=int(
                baseline_group.attrs["baseline_index"]
            ),
            polarization=polarization,
            config=config,
            wavelength=data_group["wavelength"][:],
            raw_dark=data_group["raw_dark"][:],
            raw_transition=data_group["raw_transition"][:],
            raw_baseline=data_group["raw_baseline"][:],
            dark_mean=data_group["dark_mean"][:],
            baseline_mean=data_group["baseline_mean"][:],
            baseline_corrected=data_group["baseline_corrected"][:],
            metadata=_read_metadata(metadata_group),
        )


def read_scan(
    filename: str | Path,
    scan_index: int,
) -> Scan:
    """Read a specimen scan from an MSP HDF5 file."""

    with h5py.File(filename, "r") as h5:
        _validate_file(h5)
        
        scan_group = h5[
            f"scans/scan_{scan_index:03d}"
        ]

        config_group = scan_group["config"]
        data_group = scan_group["data"]
        metadata_group = scan_group["metadata"]

        config = _read_scan_config(config_group)

        polarization = Polarization(
            scan_group.attrs["polarization"]
        )

        return Scan(
            scan_index=int(
                scan_group.attrs["scan_index"]
            ),
            baseline_index=int(
                scan_group.attrs["baseline_index"]
            ),
            polarization=polarization,
            config=config,
            wavelength=data_group["wavelength"][:],
            raw_dark=data_group["raw_dark"][:],
            raw_transition=data_group["raw_transition"][:],
            raw_specimen=data_group["raw_specimen"][:],
            dark_mean=data_group["dark_mean"][:],
            specimen_mean=data_group["specimen_mean"][:],
            specimen_corrected=data_group[
                "specimen_corrected"
            ][:],
            optical_density=data_group["optical_density"][:],
            metadata=_read_metadata(metadata_group),
        )


def write_experiment_metadata(
    filename: str | Path,
    metadata: dict,
) -> None:
    """Write shared experiment metadata to an MSP HDF5 file."""

    _validate_metadata(metadata)

    with h5py.File(filename, "a") as h5:
        _initialize_file(h5)

        metadata_group = h5.require_group("metadata")
        _write_metadata(metadata_group, metadata)


def read_experiment_metadata(
    filename: str | Path,
) -> dict:
    """Read shared experiment metadata from an MSP HDF5 file."""

    with h5py.File(filename, "r") as h5:
        _validate_file(h5)
        
        metadata_group = h5["metadata"]
        return _read_metadata(metadata_group)


def write_scan_group(
    filename: str | Path,
    scan_group: ScanGroup,
) -> None:
    """Write scan-group membership to an MSP HDF5 file."""

    scan_indices = scan_group.scan_indices

    with h5py.File(filename, "a") as h5:
        _initialize_file(h5)

        scan_groups_group = h5.require_group("scan_groups")
        scans_group = h5.require_group("scans")

        group_name = f"group_{scan_group.scan_group_id:03d}"

        if group_name in scan_groups_group:
            raise ScanGroupAlreadyExistsError(
                f"Scan group {scan_group.scan_group_id} already exists"
            )

        for scan_index in scan_indices:
            scan_name = f"scan_{scan_index:03d}"

            if scan_name not in scans_group:
                raise ValueError(
                    f"Scan {scan_index} does not exist in the HDF5 file"
                )

        group = scan_groups_group.create_group(group_name)

        group.attrs["scan_group_id"] = scan_group.scan_group_id
        group.attrs["label"] = scan_group.label

        group.create_dataset(
            "scan_indices",
            data=scan_indices,
            dtype="i8",
        )


def read_scan_group(
    filename: str | Path,
    scan_group_id: int,
) -> ScanGroup:
    """Read a scan group from an MSP HDF5 file."""

    with h5py.File(filename, "r") as h5:
        _validate_file(h5)

        group = h5[
            f"scan_groups/group_{scan_group_id:03d}"
        ]

        stored_group_id = int(
            group.attrs["scan_group_id"]
        )
        label = str(group.attrs["label"])

        scan_indices = [
            int(index)
            for index in group["scan_indices"][:]
        ]

    return ScanGroup(
        scan_group_id=stored_group_id,
        label=label,
        scan_indices=scan_indices,
    )


def update_scan_group(
    filename: str | Path,
    scan_group: ScanGroup,
) -> None:
    """Update an existing scan group in an MSP HDF5 file."""

    with h5py.File(filename, "a") as h5:
        _initialize_file(h5)

        scan_groups_group = h5.require_group("scan_groups")
        scans_group = h5.require_group("scans")
        pending_group = h5.require_group("_pending")

        group_name = f"group_{scan_group.scan_group_id:03d}"
        final_path = f"/scan_groups/{group_name}"

        new_name = f"update_{group_name}"
        new_path = f"/_pending/{new_name}"

        old_name = f"old_{group_name}"
        old_path = f"/_pending/{old_name}"

        if group_name not in scan_groups_group:
            raise KeyError(
                f"Scan group {scan_group.scan_group_id} does not exist"
            )

        for scan_index in scan_group.scan_indices:
            scan_name = f"scan_{scan_index:03d}"

            if scan_name not in scans_group:
                raise ValueError(
                    f"Scan {scan_index} does not exist in the HDF5 file"
                )

        # Remove stale temporary state from an earlier interrupted update.
        if new_name in pending_group:
            del pending_group[new_name]

        if old_name in pending_group:
            del pending_group[old_name]

        try:
            pending = pending_group.create_group(new_name)

            pending.attrs["scan_group_id"] = (
                scan_group.scan_group_id
            )
            pending.attrs["label"] = scan_group.label

            pending.create_dataset(
                "scan_indices",
                data=scan_group.scan_indices,
                dtype="i8",
            )

            # Preserve the old version before committing the replacement.
            h5.move(final_path, old_path)

            # Commit the new version.
            h5.move(new_path, final_path)
            h5.flush()

            # The replacement succeeded, so the old version is no longer needed.
            del h5[old_path]
            h5.flush()

        except Exception:
            # Remove an uncommitted new version.
            if new_path in h5:
                del h5[new_path]

            # If the old version was moved aside but the new version was not
            # committed, restore the old version.
            if old_path in h5 and final_path not in h5:
                h5.move(old_path, final_path)

            h5.flush()
            raise


def read_experiment(
    filename: str | Path,
) -> Experiment:
    """Read an Experiment from an MSP HDF5 file."""

    shared_metadata = read_experiment_metadata(filename)

    experiment = Experiment(
        shared_metadata=shared_metadata,
    )

    with h5py.File(filename, "r") as h5:
        _validate_file(h5)

        baseline_indices = [
            int(group.attrs["baseline_index"])
            for group in h5["baselines"].values()
        ]

        scan_indices = [
            int(group.attrs["scan_index"])
            for group in h5["scans"].values()
        ]

        scan_group_ids = []

        if "scan_groups" in h5:
            scan_group_ids = [
                int(group.attrs["scan_group_id"])
                for group in h5["scan_groups"].values()
            ]

    for baseline_index in baseline_indices:
        baseline = read_baseline(
            filename,
            baseline_index,
        )
        experiment.add_baseline(baseline)

    for scan_index in scan_indices:
        scan = read_scan(
            filename,
            scan_index,
        )
        experiment.add_scan(scan)

    for scan_group_id in scan_group_ids:
        scan_group = read_scan_group(
            filename,
            scan_group_id,
        )
        experiment.scan_groups.append(scan_group)

    return experiment


def write_experiment(
    filename: str | Path,
    experiment: Experiment,
) -> None:
    """Write an Experiment to an MSP HDF5 file."""

    write_experiment_metadata(
        filename,
        experiment.shared_metadata,
    )

    for baseline in experiment.baselines:
        write_baseline(filename, baseline)

    for scan in experiment.scans:
        write_scan(filename, scan)

    for group in experiment.scan_groups:
        write_scan_group(filename, group)


def _write_metadata(
    group: h5py.Group,
    metadata: dict,
) -> None:
    """Write scalar metadata as HDF5 attributes."""

    for key, value in metadata.items():
        group.attrs[key] = value


def _read_metadata(
    group: h5py.Group,
) -> dict:
    """Read scalar metadata from HDF5 attributes."""

    metadata = {}

    for key, value in group.attrs.items():
        if isinstance(value, np.generic):
            value = value.item()

        metadata[key] = value

    return metadata


def _validate_metadata(metadata: dict) -> None:
    """Check that metadata can be represented as HDF5 attributes."""

    for key, value in metadata.items():
        if not isinstance(key, str):
            raise TypeError("Metadata keys must be strings")

        if not isinstance(value, (str, int, float, bool)):
            raise TypeError(
                f"Unsupported metadata type for {key!r}: "
                f"{type(value).__name__}"
            )


def _initialize_file(h5: h5py.File) -> None:
    """Initialize or validate an MSP-control HDF5 file."""

    if "format" in h5.attrs:
        _validate_file(h5)
        return

    if len(h5) > 0 or len(h5.attrs) > 0:
        raise ValueError(
            "HDF5 file is not an MSP-control file"
        )

    h5.attrs["format"] = FORMAT_NAME
    h5.attrs["format_version"] = FORMAT_VERSION
    h5.attrs["software_version"] = SOFTWARE_VERSION


def _validate_file(h5: h5py.File) -> None:
    """Validate an existing MSP-control HDF5 file."""

    if "format" not in h5.attrs:
        raise ValueError(
            "HDF5 file is not an MSP-control file"
        )

    if h5.attrs["format"] != FORMAT_NAME:
        raise ValueError(
            f"HDF5 file is not an MSP-control file: "
            f"format={h5.attrs['format']!r}"
        )

    if "format_version" not in h5.attrs:
        raise ValueError(
            "MSP-control file has no format version"
        )

    version = int(h5.attrs["format_version"])

    if version != FORMAT_VERSION:
        raise ValueError(
            f"Unsupported MSP-control format version: {version}"
        )

   
def _write_scan_config(
    group: h5py.Group,
    config: ScanConfig,
) -> None:
    """Write a ScanConfig as HDF5 attributes."""

    group.attrs["start_nm"] = config.start_nm
    group.attrs["end_nm"] = config.end_nm
    group.attrs["step_nm"] = config.step_nm
    group.attrs["step_time_ms"] = config.step_time_ms
    group.attrs["input_slit_nm"] = config.input_slit_nm
    group.attrs["output_slit_nm"] = config.output_slit_nm
    group.attrs["dark_scans"] = config.dark_scans
    group.attrs["data_scans"] = config.data_scans


def _read_scan_config(group: h5py.Group) -> ScanConfig:
    """Read a ScanConfig from HDF5 attributes."""

    return ScanConfig(
        start_nm=group.attrs["start_nm"],
        end_nm=group.attrs["end_nm"],
        step_nm=group.attrs["step_nm"],
        step_time_ms=group.attrs["step_time_ms"],
        input_slit_nm=group.attrs["input_slit_nm"],
        output_slit_nm=group.attrs["output_slit_nm"],
        dark_scans=group.attrs["dark_scans"],
        data_scans=group.attrs["data_scans"],
    )


def _commit_pending_group(
    h5: h5py.File,
    pending_path: str,
    final_path: str,
) -> None:
    """Commit a completed pending group to its final location."""

    if final_path in h5:
        raise ValueError(
            f"Cannot commit pending group: {final_path!r} already exists"
        )

    h5.move(pending_path, final_path)
    h5.flush()

