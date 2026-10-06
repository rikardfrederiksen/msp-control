import pytest

from msp_control.hardware.temperature import (
    TemperatureConfig,
    TemperatureMeasurement,
)


def test_temperature_config_physical_channel():
    config = TemperatureConfig(
        device="SimDev1",
        channel="ai7",
    )

    assert config.physical_channel == "SimDev1/ai7"


def test_temperature_config_sample_count():
    config = TemperatureConfig(
        sample_rate=1000.0,
        duration_s=0.2,
    )

    assert config.sample_count == 200


def test_temperature_measurement():
    measurement = TemperatureMeasurement(
        mean_C=36.8,
        sd_C=0.12,
    )

    assert measurement.mean_C == pytest.approx(36.8)
    assert measurement.sd_C == pytest.approx(0.12)
