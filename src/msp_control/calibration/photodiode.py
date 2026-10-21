"""Photodiode measurements and PDA200C amplifier conversions."""

from importlib.resources import files

import numpy as np

PDA200C_GAINS = {
    "nA 1": 1e8,
    "nA 2": 1e7,
    "uA 1": 1e6,
    "uA 2": 1e5,
    "uA 3": 1e4,
    "mA 1": 1e3,
}


def voltage_to_current(
    voltage: float | np.ndarray,
    range_label: str,
    dark_voltage: float = 0.0,
) -> float | np.ndarray:
    """Convert dark-subtracted PDA200C output voltage to current (A)."""

    gain = PDA200C_GAINS[range_label]
    return (np.asarray(voltage) - dark_voltage) / gain

def responsivity_at_wavelength(wavelength_nm: float) -> float:
    """
    Return the photodiode responsivity at a specified wavelength.

    Parameters
    ----------
    wavelength_nm : float
        Wavelength in nanometers.

    Returns
    -------
    float
        Photodiode responsivity in A/W.

    Raises
    ------
    ValueError
        If the wavelength is outside the calibrated range.
    """
    calibration_file = (
        files("msp_control.calibration")
        .joinpath("data/PDGrasby.csv")
    )

    with calibration_file.open("r") as f:
        data = np.loadtxt(f, delimiter=",")

    wavelengths = data[:, 0]
    responsivities = data[:, 1]

    if not wavelengths[0] <= wavelength_nm <= wavelengths[-1]:
        raise ValueError(
            f"Wavelength {wavelength_nm} nm is outside the "
            f"calibrated range "
            f"{wavelengths[0]:g}–{wavelengths[-1]:g} nm"
        )

    return float(
        np.interp(wavelength_nm, wavelengths, responsivities)
    )


def current_to_power(
    current: float | np.ndarray,
    wavelength_nm: float,
) -> float | np.ndarray:
    """Convert photocurrent to optical power (W)."""

    responsivity = responsivity_at_wavelength(wavelength_nm)
    return np.asarray(current) / responsivity


def power_to_photon_flux_density(
    power_w: float | np.ndarray,
    wavelength_nm: float,
    area_um2: float,
) -> float | np.ndarray:
    """Convert optical power to photon flux density (photons/µm²/s)."""

    if not np.isfinite(wavelength_nm) or wavelength_nm <= 0:
        raise ValueError("Wavelength must be positive and finite")

    if not np.isfinite(area_um2) or area_um2 <= 0:
        raise ValueError("Illuminated area must be positive and finite")

    h = 6.62607015e-34
    c = 299792458

    photon_energy = h * c / (wavelength_nm * 1e-9)
    return np.asarray(power_w) / photon_energy / area_um2
