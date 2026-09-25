import numpy as np

from msp_control.acquisition import RawAcquisition
from msp_control.data.model import Baseline, Polarization, Scan
from msp_control.config import ScanConfig


def process_baseline(
    acquisition: RawAcquisition,
    config: ScanConfig,
    baseline_index: int,
    polarization: Polarization,
) -> Baseline:
    """Process a raw acquisition into an MSP baseline."""

    raw_dark = np.stack(acquisition.dark)
    raw_baseline = np.stack(acquisition.data)

    dark_mean = np.mean(raw_dark, axis=0)
    baseline_mean = np.mean(raw_baseline, axis=0)

    baseline_corrected = baseline_mean - dark_mean

    return Baseline(
        baseline_index=baseline_index,
        polarization=polarization,
        config=config,
        wavelength=config.wavelengths,
        raw_dark=raw_dark,
        raw_baseline=raw_baseline,
        dark_mean=dark_mean,
        baseline_mean=baseline_mean,
        baseline_corrected=baseline_corrected,
    )

def process_scan(
    acquisition: RawAcquisition,
    config: ScanConfig,
    baseline: Baseline,
    scan_index: int,
    polarization: Polarization,
) -> Scan:
    """Process a raw specimen acquisition into an MSP scan."""

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
   
    raw_dark = np.stack(acquisition.dark)
    raw_specimen = np.stack(acquisition.data)

    dark_mean = np.mean(raw_dark, axis=0)
    specimen_mean = np.mean(raw_specimen, axis=0)

    specimen_corrected = specimen_mean - dark_mean

    optical_density = -np.log10(
        specimen_corrected / baseline.baseline_corrected
    )

    return Scan(
        scan_index=scan_index,
        baseline_index=baseline.baseline_index,
        polarization=polarization,
        config=config,
        wavelength=config.wavelengths,
        raw_dark=raw_dark,
        raw_specimen=raw_specimen,
        dark_mean=dark_mean,
        specimen_mean=specimen_mean,
        specimen_corrected=specimen_corrected,
        optical_density=optical_density,
    )
