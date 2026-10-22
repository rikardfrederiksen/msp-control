
"""Create a synthetic MSP light calibration for GUI testing only."""

from datetime import datetime
from pathlib import Path

import numpy as np

from msp_control.calibration.model import (
    AnalogLEDCalibration,
    DigitalLEDCalibration,
    LightCalibration,
)
from msp_control.calibration.storage import (
    read_calibration,
    write_calibration,
)
from msp_control.calibration.settings import calibration_directory

def main():
    # Change this directory to the one used by find_calibration_files().
    output_dir = calibration_directory()
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = output_dir / "MOCK_TEST_ONLY_light_calibration.h5"

    # Analog LED: synthetic linear relationship from 0.02 to 10 V.
    voltage = np.linspace(0.02, 10.0, 101)
    analog_flux = 2.0e8 * voltage  # 1e9 photons/µm²/s at 5 V

    analog = AnalogLEDCalibration(
        led_channel=1,
        wavelength_nm=617.0,
        amplifier_range="MOCK",
        amplifier_gain=1.0,
        photodiode_responsivity=1.0,
        illuminated_area_um2=1.0,
        time_s=np.arange(len(voltage), dtype=float) * 0.01,
        command_voltage=voltage.copy(),
        monitor_voltage=voltage.copy(),
        photodiode_voltage=analog_flux / 1e9,
        calibration_voltage=voltage.copy(),
        calibration_monitor_voltage=voltage.copy(),
        photon_flux_density=analog_flux,
    )

    # Digital LED: synthetic on/off calibration.

    digital_state = np.array(
        [False, False, True, True, True, False, False],
        dtype=bool,
    )
    digital_flux = 2.0e9

    digital = DigitalLEDCalibration(
        led_channel=2,
        wavelength_nm=505.0,
        amplifier_range="MOCK",
        amplifier_gain=1.0,
        photodiode_responsivity=1.0,
        illuminated_area_um2=1.0,
        time_s=np.arange(len(digital_state), dtype=float) * 0.01,
        command_state=digital_state,
        monitor_voltage=digital_state.copy(),
        photodiode_voltage=digital_state.copy(),
        monitor_voltage_on=1.0,
        photon_flux_density_on=digital_flux,
    )

    calibration = LightCalibration(
        calibration_id="MOCK_TEST_ONLY",
        created_at=datetime.now(),
        analog_led=analog,
        digital_led=digital,
    )

    write_calibration(filename, calibration)

    # Verify that the saved file can be read by the production loader.
    loaded = read_calibration(filename)

    assert loaded.calibration_id == "MOCK_TEST_ONLY"
    assert loaded.analog_led.wavelength_nm == 617.0
    assert loaded.digital_led.wavelength_nm == 505.0

    print(f"Created and verified: {filename.resolve()}")


if __name__ == "__main__":
    main()
