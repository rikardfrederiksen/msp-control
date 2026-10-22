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


@pytest.fixture
def test_calibration():
    """Construct a valid light calibration for storage tests."""

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

    return LightCalibration(
        calibration_id="test-calibration",
        created_at=datetime(2026, 10, 20, 12, 0),
        analog_led=analog,
        digital_led=digital,
    )

def test_write_calibration(tmp_path, test_calibration):
    """Verify calibration HDF5 structure and overwrite protection."""

    filename = tmp_path / "test_calibration.h5"

    write_calibration(filename, test_calibration)

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
            test_calibration.analog_led.command_voltage,
        )

        np.testing.assert_array_equal(
            h5["analog_led/calibration/photon_flux_density"][:],
             test_calibration.analog_led.photon_flux_density,
        )

        np.testing.assert_array_equal(
            h5["digital_led/raw/command_state"][:],
             test_calibration.digital_led.command_state,
        )

        assert (
            h5["digital_led/calibration"].attrs["photon_flux_density_on"]
            ==  test_calibration.digital_led.photon_flux_density_on
        )

    # Existing calibration files must never be overwritten.
    with pytest.raises(FileExistsError):
        write_calibration(filename, test_calibration)

    # Verify complete write/read round trip.
    loaded = read_calibration(filename)

    assert loaded.calibration_id == test_calibration.calibration_id
    assert loaded.created_at == test_calibration.created_at

    for original, restored in (
        (test_calibration.analog_led, loaded.analog_led),
        (test_calibration.digital_led, loaded.digital_led),
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


def test_repeated_calibration_roundtrip(tmp_path, test_calibration):
    """Repeated-measurement metadata and SD arrays survive HDF5 storage."""

    from dataclasses import replace
    import numpy as np

    from msp_control.calibration.model import LightCalibration

    calibration = test_calibration

    analog = replace(
        calibration.analog_led,
        n_repeats=5,
        monitor_voltage_sd=np.full_like(
            calibration.analog_led.monitor_voltage, 0.02
        ),
        photodiode_voltage_sd=np.full_like(
            calibration.analog_led.photodiode_voltage, 0.01
        ),
    )

    digital = replace(
        calibration.digital_led,
        n_repeats=3,
        monitor_voltage_sd=np.full_like(
            calibration.digital_led.monitor_voltage, 0.03
        ),
        photodiode_voltage_sd=np.full_like(
            calibration.digital_led.photodiode_voltage, 0.015
        ),
    )

    repeated = replace(
        calibration,
        analog_led=analog,
        digital_led=digital,
    )

    filename = tmp_path / "repeated_calibration.h5"

    write_calibration(filename, repeated)
    loaded = read_calibration(filename)

    assert loaded.analog_led.n_repeats == 5
    assert loaded.digital_led.n_repeats == 3

    np.testing.assert_array_equal(
        loaded.analog_led.monitor_voltage_sd,
        analog.monitor_voltage_sd,
    )
    np.testing.assert_array_equal(
        loaded.analog_led.photodiode_voltage_sd,
        analog.photodiode_voltage_sd,
    )
    np.testing.assert_array_equal(
        loaded.digital_led.monitor_voltage_sd,
        digital.monitor_voltage_sd,
    )
    np.testing.assert_array_equal(
        loaded.digital_led.photodiode_voltage_sd,
        digital.photodiode_voltage_sd,
    )


def test_read_calibration_without_repeat_metadata(
    tmp_path, test_calibration
):
    """Older calibration files without repeat metadata remain readable."""

    filename = tmp_path / "legacy_calibration.h5"

    write_calibration(filename, test_calibration)

    # Simulate a calibration file created before repeat metadata existed.
    with h5py.File(filename, "r+") as h5:
        for led_name in ("analog_led", "digital_led"):
            group = h5[led_name]

            if "n_repeats" in group.attrs:
                del group.attrs["n_repeats"]

            raw = group["raw"]

            for dataset in (
                "monitor_voltage_sd",
                "photodiode_voltage_sd",
            ):
                if dataset in raw:
                    del raw[dataset]

    loaded = read_calibration(filename)

    for led in (loaded.analog_led, loaded.digital_led):
        assert led.n_repeats == 1
        assert led.monitor_voltage_sd is None
        assert led.photodiode_voltage_sd is None
