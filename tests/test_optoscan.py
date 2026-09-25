import pytest
from msp_control.hardware.optoscan import OptoscanSerial
from msp_control.config import ScanConfig, build_config_commands

import numpy as np

class FakeSerial:
    def __init__(self, responses=None):
        self.is_open = True
        self.responses = list(responses or [b"ok\n"])
        self.written = b""
        self.flushed = False
        self.input_buffer_reset = False

    def write(self, data):
        self.written += data

    def flush(self):
        self.flushed = True

    def reset_input_buffer(self):
        self.input_buffer_reset = True

    @property
    def in_waiting(self):
        if not self.responses:
            return 0
        return len(self.responses[0])

    def read(self, size):
        response = self.responses.pop(0)
        return response[:size]

    def readline(self):
        return self.responses.pop(0)

def test_scan_config():
    config = ScanConfig(
        start_nm=360,
        end_nm=720,
        step_nm=2,
        step_time_ms=2,
        input_slit_nm=4,
        output_slit_nm=4,
        dark_scans=2,
        data_scans=5,
    )

    assert config.cycles == 8
    assert config.wavelength_points == 181
    assert config.samples == 1448


def test_scan_config_rejects_zero_step():
    with pytest.raises(ValueError):
        ScanConfig(
            start_nm=360,
            end_nm=720,
            step_nm=0,
            step_time_ms=2,
            input_slit_nm=4,
            output_slit_nm=4,
            dark_scans=2,
            data_scans=5,
        )


def test_scan_config_rejects_invalid_range():
    with pytest.raises(ValueError):
        ScanConfig(
            start_nm=360,
            end_nm=721,
            step_nm=2,
            step_time_ms=2,
            input_slit_nm=4,
            output_slit_nm=4,
            dark_scans=2,
            data_scans=5,
        )


def test_build_config_commands():
    config = ScanConfig(
        start_nm=360,
        end_nm=720,
        step_nm=2,
        step_time_ms=2,
        input_slit_nm=4,
        output_slit_nm=4,
        dark_scans=2,
        data_scans=5,
    )

    commands = build_config_commands(config)

    assert commands == [
        "close_slits",
        "3600 scan_start !",
        "7200 scan_end !",
        "20 scan_step_size !",
        "2000 scan_step_time_lo !",
        "0 scan_step_time_hi !",
        "0 scan_data>ram",
        "0 scan_prog# !",
        "0 ram>scan_data",
        "8 cycles !",
        "compile_tables",
        "set_scantable",
        "40 scan_inslit_width !",
        "40 scan_exslit_width !",
    ]

def test_send_command():
    optoscan = OptoscanSerial(port="/dev/null")
    fake_serial = FakeSerial(
        responses=[b"close_slits  ok\r\n"]
    )
    optoscan._serial = fake_serial

    response = optoscan.send_command("close_slits")

    assert fake_serial.written == b"close_slits\n"
    assert fake_serial.flushed
    assert response == "close_slits  ok"

def test_enter_diagnostic_mode():
    optoscan = OptoscanSerial(port="/dev/null")
    fake_serial = FakeSerial(
        responses=[
            b"\x1b",
            b"\x1b\x1f\r\n",
        ]
    )
    optoscan._serial = fake_serial

    optoscan.enter_diagnostic_mode()

    assert fake_serial.input_buffer_reset
    assert fake_serial.written == b"\n9"

def test_exit_diagnostic_mode():
    optoscan = OptoscanSerial(port="/dev/null")
    fake_serial = FakeSerial(
        responses=[
            b"menu \x1b\x1b",
        ]
    )
    optoscan._serial = fake_serial

    optoscan.exit_diagnostic_mode()

    assert fake_serial.written == b"menu\n"
    assert fake_serial.flushed

def test_read_until_accumulates_response():
    optoscan = OptoscanSerial(port="/dev/null")
    fake_serial = FakeSerial(
        responses=[
            b"9. Enter diagn",
            b"ostic mode\r\n",
        ]
    )
    optoscan._serial = fake_serial

    response = optoscan._read_until(b"9. Enter diagnostic mode")

    assert response == b"9. Enter diagnostic mode\r\n"

def test_configure_scan():
    config = ScanConfig(
        start_nm=360,
        end_nm=720,
        step_nm=2,
        step_time_ms=2,
        input_slit_nm=4,
        output_slit_nm=4,
        dark_scans=2,
        data_scans=5,
    )

    commands = build_config_commands(config)

    fake_serial = FakeSerial(
        responses=[
            f"{command}  ok\r\n".encode("ascii")
            for command in commands
        ]
    )

    optoscan = OptoscanSerial("COM1")
    optoscan._serial = fake_serial

    responses = optoscan.configure_scan(config)

    expected = b"".join(
        (command + "\n").encode("ascii")
        for command in commands
    )

    assert fake_serial.written == expected
    assert len(responses) == len(commands)

def test_run_scan():
    fake_serial = FakeSerial(
        responses=[
            b"run_scan_prog Scanning. Press any key to stop now \r\n",
            b" ok\r\n",
        ]
    )

    optoscan = OptoscanSerial("COM1")
    optoscan._serial = fake_serial

    start_response, completion_response = optoscan.run_scan()

    assert fake_serial.written == b"run_scan_prog\n"
    assert fake_serial.flushed
    assert "Scanning." in start_response
    assert completion_response == " ok\r\n"

def test_run_scan_rejects_unexpected_start_response():
    fake_serial = FakeSerial(
        responses=[
            b"unexpected response\r\n",
        ]
    )

    optoscan = OptoscanSerial("COM1")
    optoscan._serial = fake_serial

    with pytest.raises(RuntimeError, match="scan-start"):
        optoscan.run_scan()

def test_start_scan():
    fake_serial = FakeSerial(
        responses=[
            b"run_scan_prog Scanning. Press any key to stop now \r\n",
            b" ok\r\n",
        ]
    )

    optoscan = OptoscanSerial("COM1")
    optoscan._serial = fake_serial

    response = optoscan.start_scan()

    assert fake_serial.written == b"run_scan_prog\n"
    assert fake_serial.flushed
    assert "Scanning." in response

    # The completion response must still be waiting.
    assert fake_serial.responses == [b" ok\r\n"]


def test_wait_for_scan_complete():
    fake_serial = FakeSerial(
        responses=[
            b" ok\r\n",
        ]
    )

    optoscan = OptoscanSerial("COM1")
    optoscan._serial = fake_serial

    response = optoscan.wait_for_scan_complete()

    assert response == " ok\r\n"

def test_wait_for_scan_complete_rejects_unexpected_response():
    fake_serial = FakeSerial(
        responses=[
            b"unexpected response\r\n",
        ]
    )

    optoscan = OptoscanSerial("COM1")
    optoscan._serial = fake_serial

    with pytest.raises(RuntimeError, match="scan-completion"):
        optoscan.wait_for_scan_complete()

def test_scan_config_wavelengths():
    config = ScanConfig(
        start_nm=360,
        end_nm=720,
        step_nm=2,
        step_time_ms=2,
        input_slit_nm=4,
        output_slit_nm=4,
        dark_scans=2,
        data_scans=5,
    )

    expected = np.arange(360, 722, 2, dtype=float)

    np.testing.assert_array_equal(
        config.wavelengths,
        expected,
    )

