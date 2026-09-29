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

        group_name = f"baseline_{baseline.baseline_index:03d}"

        if group_name in baselines_group:
            raise BaselineAlreadyExistsError(
                f"Baseline {baseline.baseline_index} already exists"
            )

        baseline_group = baselines_group.create_group(group_name)

        baseline_group.attrs["baseline_index"] = baseline.baseline_index
        baseline_group.attrs["polarization"] = baseline.polarization.value

        config_group = baseline_group.create_group("config")
        _write_scan_config(config_group, baseline.config)

        data_group = baseline_group.create_group("data")

        data_group.create_dataset("wavelength", data=baseline.wavelength)
        data_group.create_dataset("raw_dark", data=baseline.raw_dark)
        data_group.create_dataset(
            "raw_transition",
            data=baseline.raw_transition,
        )
        data_group.create_dataset(
            "raw_baseline",
            data=baseline.raw_baseline,
        )
        data_group.create_dataset("dark_mean", data=baseline.dark_mean)
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

        baseline_name = f"baseline_{scan.baseline_index:03d}"

        if baseline_name not in baselines_group:
            raise ValueError(
                f"Baseline {scan.baseline_index} does not exist "
                "in the HDF5 file"
            )

        group_name = f"scan_{scan.scan_index:03d}"

        if group_name in scans_group:
            raise ScanAlreadyExistsError(
                f"Scan {scan.scan_index} already exists"
            )

        scan_group = scans_group.create_group(group_name)

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


def read_baseline(
    filename: str | Path,
    baseline_index: int,
) -> Baseline:
    """Read a baseline from an MSP HDF5 file."""

    with h5py.File(filename, "r") as h5:
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
        metadata_group = h5["metadata"]
        return _read_metadata(metadata_group)


def write_scan_group(
    filename: str | Path,
    scan_group: ScanGroup,
) -> None:
    """Write scan-group membership to an MSP HDF5 file."""

    scan_indices = [
        scan.scan_index
        for scan in scan_group.scans
    ]

    if len(scan_indices) != len(set(scan_indices)):
        raise ValueError(
            f"Scan group {scan_group.scan_group_id} "
            "contains duplicate scan indices"
        )

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
    """Read a scan group and its member scans from an MSP HDF5 file."""

    with h5py.File(filename, "r") as h5:
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

    scans = [
        read_scan(filename, scan_index)
        for scan_index in scan_indices
    ]

    return ScanGroup(
        scan_group_id=stored_group_id,
        label=label,
        scans=scans,
    )


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

    scans = {
        scan.scan_index: scan
        for group in experiment.scan_groups
        for scan in group.scans
    }

    for scan in scans.values():
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
    """Initialize a new MSP HDF5 file."""

    if "format" not in h5.attrs:
        h5.attrs["format"] = FORMAT_NAME
        h5.attrs["format_version"] = FORMAT_VERSION
        h5.attrs["software_version"] = SOFTWARE_VERSION

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


def read_experiment(
    filename: str | Path,
) -> Experiment:
    """Read an Experiment from an MSP HDF5 file."""

    shared_metadata = read_experiment_metadata(filename)

    experiment = Experiment(
        shared_metadata=shared_metadata,
    )

    with h5py.File(filename, "r") as h5:
        baseline_indices = [
            int(group.attrs["baseline_index"])
            for group in h5["baselines"].values()
        ]

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

    for scan_group_id in scan_group_ids:
        scan_group = read_scan_group(
            filename,
            scan_group_id,
        )
        experiment.scan_groups.append(scan_group)

    return experiment
