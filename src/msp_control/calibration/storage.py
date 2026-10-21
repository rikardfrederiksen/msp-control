"""HDF5 storage for MSP light calibrations."""

from pathlib import Path
from datetime import datetime

import h5py

from msp_control.calibration.model import (
    AnalogLEDCalibration,
    DigitalLEDCalibration,
    LightCalibration,
)


FORMAT_NAME = "msp-light-calibration"
FORMAT_VERSION = 1


def write_calibration(
    filename: str | Path,
    calibration: LightCalibration,
) -> None:
    """Write a complete light calibration to a new HDF5 file."""

    filename = Path(filename)

    # Exclusive creation prevents overwriting existing calibrations.
    with h5py.File(filename, "x") as h5:
        h5.attrs["format_name"] = FORMAT_NAME
        h5.attrs["format_version"] = FORMAT_VERSION
        h5.attrs["calibration_id"] = calibration.calibration_id
        h5.attrs["created_at"] = calibration.created_at.isoformat()

        # Analog LED
        analog = calibration.analog_led
        group = h5.create_group("analog_led")

        _write_led_attributes(group, analog)

        raw = group.create_group("raw")
        raw.create_dataset("time_s", data=analog.time_s)
        raw.create_dataset(
            "command_voltage", data=analog.command_voltage
        )
        raw.create_dataset(
            "monitor_voltage", data=analog.monitor_voltage
        )
        raw.create_dataset(
            "photodiode_voltage", data=analog.photodiode_voltage
        )

        processed = group.create_group("calibration")
        processed.create_dataset(
            "command_voltage", data=analog.calibration_voltage
        )
        processed.create_dataset(
            "monitor_voltage",
            data=analog.calibration_monitor_voltage,
        )
        processed.create_dataset(
            "photon_flux_density",
            data=analog.photon_flux_density,
        )

        # Digital LED
        digital = calibration.digital_led
        group = h5.create_group("digital_led")

        _write_led_attributes(group, digital)

        raw = group.create_group("raw")
        raw.create_dataset("time_s", data=digital.time_s)
        raw.create_dataset(
            "command_state", data=digital.command_state
        )
        raw.create_dataset(
            "monitor_voltage", data=digital.monitor_voltage
        )
        raw.create_dataset(
            "photodiode_voltage", data=digital.photodiode_voltage
        )

        processed = group.create_group("calibration")
        processed.attrs["monitor_voltage_on"] = (
            digital.monitor_voltage_on
        )
        processed.attrs["photon_flux_density_on"] = (
            digital.photon_flux_density_on
        )


def _write_led_attributes(group, led) -> None:
    """Write attributes shared by analog and digital LEDs."""

    group.attrs["led_channel"] = led.led_channel
    group.attrs["wavelength_nm"] = led.wavelength_nm
    group.attrs["amplifier_range"] = led.amplifier_range
    group.attrs["amplifier_gain"] = led.amplifier_gain
    group.attrs["photodiode_responsivity"] = (
        led.photodiode_responsivity
    )
    group.attrs["illuminated_area_um2"] = (
        led.illuminated_area_um2
    )


def read_calibration(
    filename: str | Path,
) -> LightCalibration:
    """Read and validate an MSP light calibration HDF5 file."""

    with h5py.File(filename, "r") as h5:

        if h5.attrs.get("format_name") != FORMAT_NAME:
            raise ValueError("Not an MSP light calibration file")

        if h5.attrs.get("format_version") != FORMAT_VERSION:
            raise ValueError("Unsupported calibration format version")

        analog_group = h5["analog_led"]
        analog_raw = analog_group["raw"]
        analog_processed = analog_group["calibration"]

        analog = AnalogLEDCalibration(
            led_channel=int(analog_group.attrs["led_channel"]),
            wavelength_nm=float(analog_group.attrs["wavelength_nm"]),
            amplifier_range=str(analog_group.attrs["amplifier_range"]),
            amplifier_gain=float(analog_group.attrs["amplifier_gain"]),
            photodiode_responsivity=float(
                analog_group.attrs["photodiode_responsivity"]
            ),
            illuminated_area_um2=float(
                analog_group.attrs["illuminated_area_um2"]
            ),
            time_s=analog_raw["time_s"][:],
            command_voltage=analog_raw["command_voltage"][:],
            monitor_voltage=analog_raw["monitor_voltage"][:],
            photodiode_voltage=analog_raw["photodiode_voltage"][:],
            calibration_voltage=analog_processed["command_voltage"][:],
            calibration_monitor_voltage=analog_processed["monitor_voltage"][:],
            photon_flux_density=analog_processed["photon_flux_density"][:],
        )

        digital_group = h5["digital_led"]
        digital_raw = digital_group["raw"]
        digital_processed = digital_group["calibration"]

        digital = DigitalLEDCalibration(
            led_channel=int(digital_group.attrs["led_channel"]),
            wavelength_nm=float(digital_group.attrs["wavelength_nm"]),
            amplifier_range=str(digital_group.attrs["amplifier_range"]),
            amplifier_gain=float(digital_group.attrs["amplifier_gain"]),
            photodiode_responsivity=float(
                digital_group.attrs["photodiode_responsivity"]
            ),
            illuminated_area_um2=float(
                digital_group.attrs["illuminated_area_um2"]
            ),
            time_s=digital_raw["time_s"][:],
            command_state=digital_raw["command_state"][:],
            monitor_voltage=digital_raw["monitor_voltage"][:],
            photodiode_voltage=digital_raw["photodiode_voltage"][:],
            monitor_voltage_on=float(
                digital_processed.attrs["monitor_voltage_on"]
            ),
            photon_flux_density_on=float(
                digital_processed.attrs["photon_flux_density_on"]
            ),
        )

        return LightCalibration(
            calibration_id=str(h5.attrs["calibration_id"]),
            created_at=datetime.fromisoformat(
                h5.attrs["created_at"]
            ),
            analog_led=analog,
            digital_led=digital,
        )
