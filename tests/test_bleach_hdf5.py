from datetime import datetime

import pytest
import numpy as np

from msp_control.data.model import Bleach, Event, Experiment
from msp_control.storage.hdf5 import (
    create_experiment_file,
    write_event,
    write_bleach,
    read_bleach,
    read_experiment,
    write_experiment,
)


def test_bleach_hdf5_round_trip(tmp_path):
    filename = tmp_path / "experiment.h5"
    create_experiment_file(filename)

    timestamp = datetime(2026, 10, 9, 12, 0)

    event = Event(
        event_id=0,
        timestamp=timestamp,
        description="Bleach started",
    )
    write_event(filename, event)

    bleach = Bleach(
        bleach_index=0,
        event_id=0,
        timestamp=timestamp,
        led_channel=1,
        command_voltage=2.5,
        requested_duration_s=10.0,
        nd_filter=2.0,
        monitor_time_s=np.array([0.0, 0.1, 0.2]),
        monitor_voltage=np.array([0.0, 0.5, 0.5]),
        completed=True,
        calibration_id="msp_bleach_20261009",
        metadata={"notes": "Test exposure"},
    )

    write_bleach(filename, bleach)
    restored = read_bleach(filename, 0)

    assert restored.bleach_index == bleach.bleach_index
    assert restored.event_id == bleach.event_id
    assert restored.timestamp == bleach.timestamp
    assert restored.led_channel == bleach.led_channel
    assert restored.command_voltage == bleach.command_voltage
    assert restored.requested_duration_s == bleach.requested_duration_s
    assert restored.nd_filter == bleach.nd_filter
    assert restored.completed is True
    assert restored.calibration_id == bleach.calibration_id
    assert restored.metadata == bleach.metadata

    np.testing.assert_array_equal(
        restored.monitor_time_s,
        bleach.monitor_time_s,
    )
    np.testing.assert_array_equal(
        restored.monitor_voltage,
        bleach.monitor_voltage,
    )




def test_interrupted_bleach_round_trip(tmp_path):
    filename = tmp_path / "experiment.h5"
    create_experiment_file(filename)

    timestamp = datetime(2026, 10, 9, 12, 0)
    write_event(filename, Event(0, timestamp, "Bleach started"))

    bleach = Bleach(
        bleach_index=0,
        event_id=0,
        timestamp=timestamp,
        led_channel=1,
        command_voltage=2.5,
        requested_duration_s=10.0,
        nd_filter=2.0,
        monitor_time_s=np.array([0.0, 0.1]),
        monitor_voltage=np.array([0.0, 0.5]),
        completed=False,
        calibration_id=None,
    )

    write_bleach(filename, bleach)
    restored = read_bleach(filename, 0)

    assert restored.completed is False
    assert restored.calibration_id is None

    np.testing.assert_array_equal(
        restored.monitor_time_s,
        bleach.monitor_time_s,
    )
    np.testing.assert_array_equal(
        restored.monitor_voltage,
        bleach.monitor_voltage,
    )


def test_duplicate_bleach_does_not_overwrite(tmp_path):
    filename = tmp_path / "experiment.h5"
    create_experiment_file(filename)

    timestamp = datetime(2026, 10, 9, 12, 0)
    write_event(filename, Event(0, timestamp, "Bleach started"))

    bleach = Bleach(
        bleach_index=0,
        event_id=0,
        timestamp=timestamp,
        led_channel=1,
        command_voltage=2.5,
        requested_duration_s=10.0,
        nd_filter=2.0,
        monitor_time_s=np.array([0.0, 0.1]),
        monitor_voltage=np.array([0.0, 0.5]),
        completed=True,
    )

    write_bleach(filename, bleach)

    with pytest.raises(ValueError, match="already exists"):
        write_bleach(filename, bleach)

    restored = read_bleach(filename, 0)
    assert restored.command_voltage == 2.5
    assert restored.completed is True


def test_bleach_requires_existing_event(tmp_path):
    filename = tmp_path / "experiment.h5"
    create_experiment_file(filename)

    bleach = Bleach(
        bleach_index=0,
        event_id=99,
        timestamp=datetime(2026, 10, 9, 12, 0),
        led_channel=1,
        command_voltage=2.5,
        requested_duration_s=10.0,
        nd_filter=2.0,
        monitor_time_s=np.array([0.0, 0.1]),
        monitor_voltage=np.array([0.0, 0.5]),
        completed=True,
    )

    with pytest.raises(ValueError, match="Event"):
        write_bleach(filename, bleach)


def test_experiment_with_bleach_round_trip(
    tmp_path,
    baseline,
    scan,
):
    filename = tmp_path / "experiment.h5"

    timestamp = datetime(2026, 10, 9, 12, 0)

    experiment = Experiment()

    # A valid experiment includes a baseline and a scan.
    experiment.add_baseline(baseline)
    experiment.add_scan(scan)

    event = Event(
        event_id=0,
        timestamp=timestamp,
        description="Bleach started",
    )
    experiment.add_event(event)

    bleach = Bleach(
        bleach_index=0,
        event_id=0,
        timestamp=timestamp,
        led_channel=1,
        command_voltage=2.5,
        requested_duration_s=10.0,
        nd_filter=2.0,
        monitor_time_s=np.array([0.0, 0.1, 0.2]),
        monitor_voltage=np.array([0.0, 0.5, 0.5]),
        completed=True,
        calibration_id="msp_bleach_20261009",
    )
    experiment.add_bleach(bleach)

    write_experiment(filename, experiment)
    restored = read_experiment(filename)

    assert len(restored.baselines) == 1
    assert len(restored.scans) == 1
    assert len(restored.events) == 1
    assert len(restored.bleaches) == 1

    restored_bleach = restored.get_bleach(0)

    assert restored_bleach.event_id == 0
    assert restored_bleach.completed is True
    assert restored_bleach.nd_filter == 2.0
    assert restored_bleach.calibration_id == bleach.calibration_id

    np.testing.assert_array_equal(
        restored_bleach.monitor_voltage,
        bleach.monitor_voltage,
    )

    assert restored.next_bleach_index == 1
    assert restored.next_event_id == 1
