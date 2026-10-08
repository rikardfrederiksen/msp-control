from unittest.mock import MagicMock

import pytest

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
