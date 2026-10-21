from datetime import datetime

import h5py
import numpy as np
import pytest

from msp_control.calibration.model import (
    AnalogLEDCalibration,
    DigitalLEDCalibration,
    LightCalibration,
)
from msp_control.calibration.storage import write_calibration, read_calibration


def test_write_calibration(tmp_path):
    """Verify calibration HDF5 structure and overwrite protection."""

    analog = AnalogLEDCalibration(
        led_channel=1,
        wavelength_nm=617.0,
        amplifier_range="uA 1",
        amplifier_gain=1e6,
        photodiode_responsivity=0.4,
        illuminated_area_um2=10000.0,
        time_s=np.array([0.0, 0.1, 0.2, 0.3]),
        command_voltage=np.array([0.0, 1.0, 2.0, 0.0]),
        monitor_voltage=np.array([0.0, 0.2, 0.5, 0.0]),
        photodiode_voltage=np.array([0.0, 0.1, 0.3, 0.0]),
        calibration_voltage=np.array([1.0, 2.0]),
        calibration_monitor_voltage=np.array([0.2, 0.5]),
        photon_flux_density=np.array([1e6, 3e6]),
    )

    digital = DigitalLEDCalibration(
        led_channel=2,
        wavelength_nm=505.0,
        amplifier_range="uA 1",
        amplifier_gain=1e6,
        photodiode_responsivity=0.3,
        illuminated_area_um2=10000.0,
        time_s=np.array([0.0, 0.1, 0.2, 0.3]),
        command_state=np.array([False, True, True, False]),
        monitor_voltage=np.array([0.0, 0.8, 0.8, 0.0]),
        photodiode_voltage=np.array([0.0, 0.4, 0.4, 0.0]),
        monitor_voltage_on=0.8,
        photon_flux_density_on=4e6,
    )

    calibration = LightCalibration(
        calibration_id="test-calibration",
        created_at=datetime(2026, 10, 20, 12, 0),
        analog_led=analog,
        digital_led=digital,
    )

    filename = tmp_path / "test_calibration.h5"

    write_calibration(filename, calibration)

    with h5py.File(filename, "r") as h5:
        assert h5.attrs["format_name"] == "msp-light-calibration"
        assert h5.attrs["format_version"] == 1
        assert h5.attrs["calibration_id"] == "test-calibration"

        assert "analog_led" in h5
        assert "digital_led" in h5

        assert h5["analog_led"].attrs["wavelength_nm"] == 617.0
        assert h5["digital_led"].attrs["wavelength_nm"] == 505.0

        np.testing.assert_array_equal(
            h5["analog_led/raw/command_voltage"][:],
            analog.command_voltage,
        )

        np.testing.assert_array_equal(
            h5["analog_led/calibration/photon_flux_density"][:],
            analog.photon_flux_density,
        )

        np.testing.assert_array_equal(
            h5["digital_led/raw/command_state"][:],
            digital.command_state,
        )

        assert (
            h5["digital_led/calibration"].attrs["photon_flux_density_on"]
            == digital.photon_flux_density_on
        )

    # Existing calibration files must never be overwritten.
    with pytest.raises(FileExistsError):
        write_calibration(filename, calibration)

    # Verify complete write/read round trip.
    loaded = read_calibration(filename)

    assert loaded.calibration_id == calibration.calibration_id
    assert loaded.created_at == calibration.created_at

    for original, restored in (
        (calibration.analog_led, loaded.analog_led),
        (calibration.digital_led, loaded.digital_led),
    ):
        for field_name in original.__dataclass_fields__:
            expected = getattr(original, field_name)
            actual = getattr(restored, field_name)

            if isinstance(expected, np.ndarray):
                np.testing.assert_array_equal(actual, expected)
            else:
                assert actual == expected


@pytest.mark.parametrize(
    "field, value",
    [
        ("calibration_voltage", np.array([2.0, 1.0])),
        ("calibration_voltage", np.array([1.0, 1.0])),
        ("photon_flux_density", np.array([1e6, -3e6])),
        ("photon_flux_density", np.array([1e6, np.nan])),
        ("wavelength_nm", -617.0),
        ("illuminated_area_um2", 0.0),
        ("amplifier_gain", 0.0),
        ("photodiode_responsivity", -0.4),
        ("monitor_voltage", np.array([0.0, 0.2])),
        ("wavelength_nm", np.nan),
        ("illuminated_area_um2", np.inf),
        ("amplifier_gain", np.nan),
        ("photodiode_responsivity", np.inf),
    ],
)
def test_invalid_analog_calibration(field, value):
    """Invalid analog calibration data must be rejected."""

    data = dict(
        led_channel=1,
        wavelength_nm=617.0,
        amplifier_range="uA 1",
        amplifier_gain=1e6,
        photodiode_responsivity=0.4,
        illuminated_area_um2=10000.0,
        time_s=np.array([0.0, 0.1, 0.2, 0.3]),
        command_voltage=np.array([0.0, 1.0, 2.0, 0.0]),
        monitor_voltage=np.array([0.0, 0.2, 0.5, 0.0]),
        photodiode_voltage=np.array([0.0, 0.1, 0.3, 0.0]),
        calibration_voltage=np.array([1.0, 2.0]),
        calibration_monitor_voltage=np.array([0.2, 0.5]),
        photon_flux_density=np.array([1e6, 3e6]),
    )

    data[field] = value

    with pytest.raises(ValueError):
        AnalogLEDCalibration(**data)


@pytest.mark.parametrize(
    "field, value",
    [
        ("command_state", np.array([0, 1, 1, 0])),
        ("monitor_voltage", np.array([0.0, 0.8])),
        ("photodiode_voltage", np.array([0.0, np.nan, 0.4, 0.0])),
        ("monitor_voltage_on", np.nan),
        ("photon_flux_density_on", -1e6),
        ("photon_flux_density_on", np.inf),
        ("wavelength_nm", 0.0),
        ("illuminated_area_um2", np.nan),
        ("amplifier_gain", -1.0),
        ("photodiode_responsivity", 0.0),
    ],
)
def test_invalid_digital_calibration(field, value):
    """Invalid digital calibration data must be rejected."""

    data = dict(
        led_channel=2,
        wavelength_nm=505.0,
        amplifier_range="uA 1",
        amplifier_gain=1e6,
        photodiode_responsivity=0.3,
        illuminated_area_um2=10000.0,
        time_s=np.array([0.0, 0.1, 0.2, 0.3]),
        command_state=np.array([False, True, True, False]),
        monitor_voltage=np.array([0.0, 0.8, 0.8, 0.0]),
        photodiode_voltage=np.array([0.0, 0.4, 0.4, 0.0]),
        monitor_voltage_on=0.8,
        photon_flux_density_on=4e6,
    )

    data[field] = value

    with pytest.raises(ValueError):
        DigitalLEDCalibration(**data)
