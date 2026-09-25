from dataclasses import dataclass

from msp_control.config import ScanConfig, build_config_commands

import serial
import time
import numpy as np

class OptoscanSerial:
    """Serial connection to a Cairn Optoscan controller."""

    def __init__(
        self,
        port: str,
        baudrate: int = 9600,
        timeout: float = 10.0,
    ):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self._serial = None

    def open(self) -> None:
        """Open the serial connection."""
        self._serial = serial.Serial(
            port=self.port,
            baudrate=self.baudrate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=self.timeout,
            xonxoff=False,
            rtscts=False,
            dsrdtr=False,
        )

    def close(self) -> None:
        """Close the serial connection."""
        if self._serial is not None:
            self._serial.close()
            self._serial = None

    @property
    def is_open(self) -> bool:
        return self._serial is not None and self._serial.is_open

    def _read_until(self, expected: bytes, timeout: float = 1.0) -> bytes:
        """Read serial data until the expected byte sequence is received."""

        deadline = time.monotonic() + timeout
        response = bytearray()

        while time.monotonic() < deadline:
            waiting = self._serial.in_waiting

            if waiting:
                response.extend(self._serial.read(waiting))

                if expected in response:
                    return bytes(response)

            time.sleep(0.01)

        raise TimeoutError(
            f"Timed out waiting for {expected!r}; received {bytes(response)!r}"
        )

    def enter_diagnostic_mode(self) -> None:
        """Enter the Optoscan diagnostic/Forth command interface."""

        if not self.is_open:
            raise RuntimeError("Optoscan serial port is not open")

        self._serial.reset_input_buffer()

        # Wake the controller/menu interface.
        self._serial.write(b"\n")
        self._serial.flush()
        self._read_until(b"\x1b")

        # Menu option 9 enters diagnostic mode.
        # Consume the complete response: b"\x1b\x1f\r\n".
        self._serial.write(b"9")
        self._serial.flush()
        self._read_until(b"\r\n")

    def exit_diagnostic_mode(self) -> None:
        """Exit the diagnostic/Forth interface and return to the main menu."""

        if not self.is_open:
            raise RuntimeError("Optoscan serial port is not open")

        self._serial.write(b"menu\n")
        self._serial.flush()

        self._read_until(b"\x1b\x1b")

    def send_command(self, command: str) -> str:
        """Send one command to the Optoscan and return its response."""

        if not self.is_open:
            raise RuntimeError("Optoscan serial port is not open")

        message = (command + "\n").encode("ascii")
        self._serial.write(message)
        self._serial.flush()

        response = self._serial.readline()

        if not response:
            raise TimeoutError(
                f"No response from Optoscan after command: {command!r}"
            )

        response_text = response.decode("ascii").strip()

        if "error" in response_text.lower():
            raise RuntimeError(
                f"Optoscan error after command {command!r}: {response_text!r}"
            )

        if "ok" not in response_text.lower():
            raise RuntimeError(
                f"Unexpected Optoscan response after command "
                f"{command!r}: {response_text!r}"
            )

        return response_text

    def configure_scan(self, config: ScanConfig) -> list[str]:
        """Configure the Optoscan for a wavelength scan.

        Sends the complete scan-configuration command sequence to the
        controller and returns the controller responses.
        """

        responses = []

        for command in build_config_commands(config):
            responses.append(self.send_command(command))

        return responses

    def start_scan(self) -> str:
        """Start the configured scan and return after scanning begins."""

        if self._serial is None:
            raise RuntimeError("Serial port is not open")

        self._serial.write(b"run_scan_prog\n")
        self._serial.flush()

        response = self._serial.readline().decode("ascii")

        if "Scanning." not in response:
            raise RuntimeError(
                f"Unexpected scan-start response: {response!r}"
            )

        return response


    def wait_for_scan_complete(self) -> str:
        """Wait for the Optoscan to report that the scan has completed."""

        if self._serial is None:
            raise RuntimeError("Serial port is not open")

        response = self._serial.readline().decode("ascii")

        if "ok" not in response:
            raise RuntimeError(
                f"Unexpected scan-completion response: {response!r}"
            )

        return response


    def run_scan(self) -> tuple[str, str]:
        """Run a complete scan and wait for it to finish."""

        start_response = self.start_scan()
        completion_response = self.wait_for_scan_complete()

        return start_response, completion_response
