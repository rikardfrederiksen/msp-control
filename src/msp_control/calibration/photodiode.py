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
    voltage: float,
    range_label: str,
    dark_voltage: float = 0.0,
) -> float:
    """
    Convert PDA200C output voltage to photodiode current.

    Parameters
    ----------
    voltage : float
        Amplifier output voltage during illumination (V).
    range_label : str
        PDA200C front-panel range setting.
    dark_voltage : float
        Amplifier output voltage measured with the LED off (V).
        Defaults to zero.

    Returns
    -------
    float
        Dark-subtracted photodiode current (A).
    """
    gain = PDA200C_GAINS[range_label]
    return (voltage - dark_voltage) / gain


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
    current: float,
    wavelength_nm: float,
) -> float:
    """
    Convert photodiode photocurrent to optical power.

    Parameters
    ----------
    current : float
        Dark-subtracted photocurrent (A).
    wavelength_nm : float
        Wavelength of the incident light (nm).

    Returns
    -------
    float
        Optical power (W).
    """
    responsivity = responsivity_at_wavelength(wavelength_nm)

    return current / responsivity


def power_to_photon_flux_density(
    power_w: float,
    wavelength_nm: float,
    area_um2: float,
) -> float:
    """
    Convert optical power to photon flux density.

    Parameters
    ----------
    power_w : float
        Total optical power (W).
    wavelength_nm : float
        Wavelength of incident light (nm).
    area_um2 : float
        Illuminated area at the specimen plane (µm²).

    Returns
    -------
    float
        Photon flux density (photons/µm²/s).
    """
    if wavelength_nm <= 0:
        raise ValueError("Wavelength must be positive")

    if area_um2 <= 0:
        raise ValueError("Illuminated area must be positive")

    h = 6.62607015e-34  # Planck constant (J s)
    c = 299792458       # Speed of light (m/s)

    wavelength_m = wavelength_nm * 1e-9

    photon_energy = h * c / wavelength_m
    photons_per_second = power_w / photon_energy

    return photons_per_second / area_um2
