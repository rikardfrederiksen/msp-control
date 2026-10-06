from datetime import datetime

from msp_control.clock import Clock


def test_clock_returns_timezone_aware_datetime():
    clock = Clock()

    timestamp = clock.now()

    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo is not None
    assert timestamp.utcoffset() is not None
