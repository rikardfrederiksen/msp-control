from unittest.mock import Mock
import pytest

from msp_control.hardware.manager import HardwareManager


def test_reinitialize_optoscan():
    optoscan = Mock()
    field_stop = Mock()

    manager = HardwareManager(
        optoscan=optoscan,
        field_stop=field_stop,
    )

    manager.reinitialize_optoscan()

    optoscan.initialize.assert_called_once_with()
    field_stop.connect.assert_not_called()
    field_stop.close.assert_not_called()

def test_initialize_hardware():
    optoscan = Mock()
    optoscan.is_open = False

    field_stop = Mock()
    field_stop.is_connected = False

    manager = HardwareManager(optoscan, field_stop)
    manager.initialize()

    optoscan.initialize.assert_called_once_with()
    field_stop.connect.assert_called_once_with()

def test_initialize_skips_connected_devices():
    optoscan = Mock()
    optoscan.is_open = True

    field_stop = Mock()
    field_stop.is_connected = True

    manager = HardwareManager(optoscan, field_stop)
    manager.initialize()

    optoscan.initialize.assert_not_called()
    field_stop.connect.assert_not_called()


def test_initialize_only_optoscan():
    optoscan = Mock()
    optoscan.is_open = False

    field_stop = Mock()
    field_stop.is_connected = True

    manager = HardwareManager(optoscan, field_stop)
    manager.initialize()

    optoscan.initialize.assert_called_once_with()
    field_stop.connect.assert_not_called()


def test_initialize_only_field_stop():
    optoscan = Mock()
    optoscan.is_open = True

    field_stop = Mock()
    field_stop.is_connected = False

    manager = HardwareManager(optoscan, field_stop)
    manager.initialize()

    optoscan.initialize.assert_not_called()
    field_stop.connect.assert_called_once_with()


def test_field_stop_failure_preserves_optoscan():
    optoscan = Mock()
    optoscan.is_open = False

    field_stop = Mock()
    field_stop.is_connected = False
    field_stop.connect.side_effect = RuntimeError(
        "Field stop connection failed"
    )

    manager = HardwareManager(optoscan, field_stop)

    with pytest.raises(RuntimeError, match="Field stop connection failed"):
        manager.initialize()

    optoscan.initialize.assert_called_once_with()
    optoscan.close.assert_not_called()
    field_stop.connect.assert_called_once_with()


def test_disconnect():
    optoscan = Mock()
    field_stop = Mock()

    manager = HardwareManager(optoscan, field_stop)
    manager.disconnect()

    field_stop.close.assert_called_once_with()
    optoscan.close.assert_called_once_with()


def test_disconnect_continues_after_field_stop_failure():
    optoscan = Mock()
    field_stop = Mock()

    field_stop.close.side_effect = RuntimeError(
        "Field stop disconnection failed"
    )

    manager = HardwareManager(optoscan, field_stop)

    with pytest.raises(
        RuntimeError,
        match="Field stop disconnection failed",
    ):
        manager.disconnect()

    optoscan.close.assert_called_once_with()
