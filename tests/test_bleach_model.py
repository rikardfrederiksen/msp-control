from datetime import datetime

import numpy as np
import pytest

from msp_control.data.model import Bleach, Experiment, Event


def make_bleach(**overrides):
    values = dict(
        bleach_index=0,
        event_id=0,
        timestamp=datetime(2026, 10, 9, 12, 0),
        led_channel=1,
        command_voltage=2.5,
        requested_duration_s=10.0,
        nd_filter=2.0,
        monitor_time_s=np.array([0.0, 0.1, 0.2]),
        monitor_voltage=np.array([0.0, 0.5, 0.5]),
        completed=True,
        calibration_id="msp_bleach_20261009",
    )
    values.update(overrides)
    return Bleach(**values)


def test_bleach_creation():
    bleach = make_bleach()

    assert bleach.bleach_index == 0
    assert bleach.event_id == 0
    assert bleach.led_channel == 1
    assert bleach.requested_duration_s == 10.0
    assert bleach.completed is True

    np.testing.assert_array_equal(
        bleach.monitor_voltage,
        [0.0, 0.5, 0.5],
    )


def test_bleach_rejects_invalid_timestamp():
    with pytest.raises(TypeError, match="timestamp"):
        make_bleach(timestamp="2026-10-09")


def test_bleach_rejects_negative_duration():
    with pytest.raises(ValueError, match="duration"):
        make_bleach(requested_duration_s=-1.0)


def test_bleach_rejects_invalid_voltage():
    with pytest.raises(ValueError, match="voltage"):
        make_bleach(command_voltage=12.0)


def test_bleach_rejects_mismatched_monitor_arrays():
    with pytest.raises(ValueError, match="monitor"):
        make_bleach(
            monitor_time_s=np.array([0.0, 0.1]),
            monitor_voltage=np.array([0.0, 0.5, 0.5]),
        )


def test_bleach_allows_interrupted_exposure():
    bleach = make_bleach(
        completed=False,
        monitor_time_s=np.array([0.0, 0.1]),
        monitor_voltage=np.array([0.0, 0.5]),
    )

    assert bleach.completed is False
    assert len(bleach.monitor_voltage) == 2


def test_bleach_rejects_negative_nd():
    with pytest.raises(ValueError, match="ND"):
        make_bleach(nd_filter=-1.0)


def test_experiment_add_and_get_bleach():
    experiment = Experiment()
    experiment.add_event(
        Event(
            event_id=0,
            timestamp=datetime(2026, 10, 9, 12, 0),
            description="Bleach started",
        )
    )

    bleach = make_bleach()
    experiment.add_bleach(bleach)

    assert experiment.get_bleach(0) is bleach
    assert experiment.next_bleach_index == 1


def test_experiment_next_bleach_index():
    experiment = Experiment()
    assert experiment.next_bleach_index == 0

    experiment.add_event(
        Event(0, datetime(2026, 10, 9, 12, 0), "Bleach started")
    )
    experiment.add_event(
        Event(1, datetime(2026, 10, 9, 12, 5), "Bleach started")
    )

    experiment.add_bleach(make_bleach(bleach_index=2, event_id=0))
    experiment.add_bleach(make_bleach(bleach_index=5, event_id=1))

    assert experiment.next_bleach_index == 6


def test_experiment_rejects_duplicate_bleach():
    experiment = Experiment()
    experiment.add_event(
        Event(0, datetime(2026, 10, 9, 12, 0), "Bleach started")
    )

    experiment.add_bleach(make_bleach())

    with pytest.raises(ValueError, match="already exists"):
        experiment.add_bleach(make_bleach())


def test_experiment_rejects_bleach_without_event():
    experiment = Experiment()

    with pytest.raises(ValueError, match="Event"):
        experiment.add_bleach(make_bleach())

    assert experiment.bleaches == []
