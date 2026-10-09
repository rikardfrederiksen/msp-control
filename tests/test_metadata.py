import serial

from nidaqmx.errors import DaqError
from unittest.mock import MagicMock
import pytest

from msp_control.hardware.temperature import TemperatureMeasurement
from msp_control.hardware.stage import StagePosition
from msp_control.hardware.field_stop import FieldStopPosition, FieldStopConfig
from msp_control.metadata import MetadataCollector


def test_collect_field_stop_metadata():
    field_stop = MagicMock()
    field_stop.config = FieldStopConfig(
        horizontal_reference_deg=40.0
    )
    field_stop.read_position.return_value = FieldStopPosition(
        stage_angle_deg=130.0623,
        angle_deg=90.0623,
    )

    collector = MetadataCollector(field_stop=field_stop)

    metadata = collector.collect()

    assert metadata == {
        "field_stop_stage_angle_deg": pytest.approx(130.0623),
        "field_stop_angle_deg": pytest.approx(90.0623),
        "field_stop_horizontal_reference_deg": pytest.approx(40.0),
    }

    field_stop.read_position.assert_called_once_with()


def test_collect_unhomed_field_stop():
    field_stop = MagicMock()
    field_stop.read_position.side_effect = RuntimeError(
        "Field-stop rotation stage is not homed"
    )

    collector = MetadataCollector(field_stop=field_stop)

    metadata = collector.collect()

    assert metadata == {}
    field_stop.read_position.assert_called_once_with()


def test_collect_without_field_stop():
    collector = MetadataCollector(field_stop=None)

    metadata = collector.collect()

    assert metadata == {}


def test_collect_stage_metadata():
    stage = MagicMock()
    stage.read_position.return_value = StagePosition(
        x_um=125.0,
        y_um=250.0,
        z_um=375.0,
    )

    collector = MetadataCollector(stage=stage)

    metadata = collector.collect()

    assert metadata == {
        "stage_x_um": pytest.approx(125.0),
        "stage_y_um": pytest.approx(250.0),
        "stage_z_um": pytest.approx(375.0),
    }

    stage.read_position.assert_called_once_with()


def test_collect_stage_connection_failure():
    stage = MagicMock()
    stage.read_position.side_effect = serial.SerialException(
        "Could not open COM5"
    )

    collector = MetadataCollector(stage=stage)

    metadata = collector.collect()

    assert metadata == {}
    stage.read_position.assert_called_once_with()


def test_stage_failure_preserves_field_stop_metadata():
    field_stop = MagicMock()
    field_stop.config = FieldStopConfig(
        horizontal_reference_deg=40.0
    )
    field_stop.read_position.return_value = FieldStopPosition(
        stage_angle_deg=130.0,
        angle_deg=90.0,
    )

    stage = MagicMock()
    stage.read_position.side_effect = serial.SerialException(
        "Could not open COM5"
    )

    collector = MetadataCollector(
        field_stop=field_stop,
        stage=stage,
    )

    metadata = collector.collect()

    assert metadata == {
        "field_stop_stage_angle_deg": 130.0,
        "field_stop_angle_deg": 90.0,
        "field_stop_horizontal_reference_deg": 40.0,
    }

    field_stop.read_position.assert_called_once_with()
    stage.read_position.assert_called_once_with()


def test_collect_temperature_metadata():
    temperature = MagicMock()

    temperature.measure.return_value = TemperatureMeasurement(
        mean_C=21.35,
        sd_C=0.025,
    )

    collector = MetadataCollector(temperature=temperature)

    metadata = collector.collect()

    assert metadata == {
        "temperature_mean_C": pytest.approx(21.35),
        "temperature_sd_C": pytest.approx(0.025),
    }

    temperature.measure.assert_called_once_with()


def test_collect_temperature_failure():
    temperature = MagicMock()

    temperature.measure.side_effect = DaqError(
        "NI DAQ device not available",
        error_code=-200220,
    )

    collector = MetadataCollector(temperature=temperature)

    metadata = collector.collect()

    assert metadata == {}
    temperature.measure.assert_called_once_with()


def test_temperature_failure_preserves_other_metadata():
    field_stop = MagicMock()
    field_stop.config = FieldStopConfig(
        horizontal_reference_deg=40.0
    )
    field_stop.read_position.return_value = FieldStopPosition(
        stage_angle_deg=130.0,
        angle_deg=90.0,
    )

    stage = MagicMock()
    stage.read_position.return_value = StagePosition(
        x_um=125.0,
        y_um=250.0,
        z_um=375.0,
    )

    temperature = MagicMock()
    temperature.measure.side_effect = DaqError(
        "NI DAQ device not available",
        error_code=-200220,
    )

    collector = MetadataCollector(
        field_stop=field_stop,
        stage=stage,
        temperature=temperature,
    )

    metadata = collector.collect()

    assert metadata == {
        "field_stop_stage_angle_deg": 130.0,
        "field_stop_angle_deg": 90.0,
        "field_stop_horizontal_reference_deg": 40.0,
        "stage_x_um": 125.0,
        "stage_y_um": 250.0,
        "stage_z_um": 375.0,
    }

    field_stop.read_position.assert_called_once_with()
    stage.read_position.assert_called_once_with()
    temperature.measure.assert_called_once_with()
