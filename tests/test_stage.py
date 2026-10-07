from unittest.mock import MagicMock, patch
import pytest

from msp_control.hardware.stage import (
    StageConfig,
    StagePosition,
    StagePositionMonitor,
)


def test_stage_config():
    config = StageConfig()

    assert config.port == "COM5"
    assert config.baudrate == 128000
    assert config.timeout == pytest.approx(1.0)
    assert config.um_per_microstep == pytest.approx(0.0625)
    assert config.drive == 1


def test_stage_position():
    position = StagePosition(
        x_um=16000.0,
        y_um=16104.0,
        z_um=25000.0,
    )

    assert position.x_um == pytest.approx(16000.0)
    assert position.y_um == pytest.approx(16104.0)
    assert position.z_um == pytest.approx(25000.0)


def test_stage_microsteps_to_um():
    config = StageConfig()

    assert 256000 * config.um_per_microstep == pytest.approx(16000.0)


def test_decode_position_response():
    response = (
        b"\x01"
        b"\x6f\x80\x03\x00"
        b"\x94\xee\x03\x00"
        b"\x80\x1a\x06\x00"
        b"\x0d"
    )

    position = StagePositionMonitor.decode_position_response(response)

    assert position.x_um == pytest.approx(229487 * 0.0625)
    assert position.y_um == pytest.approx(257684 * 0.0625)
    assert position.z_um == pytest.approx(400000 * 0.0625)


def test_decode_position_response_rejects_wrong_length():
    response = b"\x01\x00\r"

    with pytest.raises(ValueError, match="14 bytes"):
        StagePositionMonitor.decode_position_response(response)


def test_decode_position_response_rejects_missing_terminator():
    response = (
        b"\x01"
        b"\x6f\x80\x03\x00"
        b"\x94\xee\x03\x00"
        b"\x80\x1a\x06\x00"
        b"\x00"
    )

    with pytest.raises(ValueError, match="terminator"):
        StagePositionMonitor.decode_position_response(response)


def test_decode_position_response_rejects_wrong_drive():
    response = (
        b"\x02"
        b"\x6f\x80\x03\x00"
        b"\x94\xee\x03\x00"
        b"\x80\x1a\x06\x00"
        b"\x0d"
    )

    with pytest.raises(ValueError, match="drive"):
        StagePositionMonitor.decode_position_response(response)


def test_read_position():
    response = (
        b"\x01"
        b"\x6f\x80\x03\x00"
        b"\x94\xee\x03\x00"
        b"\x80\x1a\x06\x00"
        b"\x0d"
    )

    serial_port = MagicMock()
    serial_port.read_until.return_value = response

    with patch(
        "msp_control.hardware.stage.serial.Serial",
        return_value=serial_port,
    ) as serial_class:
        monitor = StagePositionMonitor()
        position = monitor.read_position()

    serial_class.assert_called_once_with(
        port="COM5",
        baudrate=128000,
        bytesize=8,
        parity="N",
        stopbits=1,
        timeout=1.0,
    )

    serial_port.write.assert_called_once_with(b"C")
    serial_port.read_until.assert_called_once_with(b"\r")
    serial_port.close.assert_called_once()

    assert position.x_um == pytest.approx(229487 * 0.0625)
    assert position.y_um == pytest.approx(257684 * 0.0625)
    assert position.z_um == pytest.approx(400000 * 0.0625)


def test_read_position_closes_serial_port_on_read_failure():
    serial_port = MagicMock()
    serial_port.read_until.side_effect = RuntimeError("Serial read failed")

    with patch(
        "msp_control.hardware.stage.serial.Serial",
        return_value=serial_port,
    ):
        monitor = StagePositionMonitor()

        with pytest.raises(RuntimeError, match="Serial read failed"):
            monitor.read_position()

    serial_port.close.assert_called_once()
