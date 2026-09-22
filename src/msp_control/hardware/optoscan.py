from dataclasses import dataclass
import serial
import time

@dataclass(frozen=True)
class ScanConfig:
    """Configuration for an Optoscan wavelength scan."""

    start_nm: float
    end_nm: float
    step_nm: float
    step_time_ms: float
    input_slit_nm: float
    output_slit_nm: float
    dark_scans: int
    data_scans: int

    def __post_init__(self):
        if self.end_nm <= self.start_nm:
            raise ValueError("end_nm must be greater than start_nm")

        if self.step_nm <= 0:
            raise ValueError("step_nm must be greater than zero")

        if self.step_time_ms <= 0:
            raise ValueError("step_time_ms must be greater than zero")

        if self.input_slit_nm <= 0 or self.output_slit_nm <= 0:
            raise ValueError("slit widths must be greater than zero")

        if self.dark_scans < 0:
            raise ValueError("dark_scans cannot be negative")

        if self.data_scans < 1:
            raise ValueError("data_scans must be at least one")

        n_steps = (self.end_nm - self.start_nm) / self.step_nm
        if not abs(n_steps - round(n_steps)) < 1e-9:
            raise ValueError(
                "wavelength range must be evenly divisible by step_nm"
            )

    @property
    def cycles(self) -> int:
        """Total number of wavelength sweeps performed by the Optoscan."""
        return self.dark_scans + 1 + self.data_scans

    @property
    def wavelength_points(self) -> int:
        """Number of wavelength points in one complete sweep."""
        return round((self.end_nm - self.start_nm) / self.step_nm) + 1

    @property
    def samples(self) -> int:
        """Total number of externally clocked DAQ samples."""
        return self.cycles * self.wavelength_points

def build_config_commands(config: ScanConfig) -> list[str]:
    """Build the Optoscan commands required to configure a wavelength scan."""

    return [
        "close_slits",
        f"{round(config.start_nm * 10)} scan_start !",
        f"{round(config.end_nm * 10)} scan_end !",
        f"{round(config.step_nm * 10)} scan_step_size !",
        f"{round(config.step_time_ms * 1000)} scan_step_time_lo !",
        "0 scan_step_time_hi !",
        "0 scan_data>ram",
        "0 scan_prog# !",
        "0 ram>scan_data",
        f"{config.cycles} cycles !",
        "compile_tables",
        "set_scantable",
        f"{round(config.input_slit_nm * 10)} scan_inslit_width !",
        f"{round(config.output_slit_nm * 10)} scan_exslit_width !",
    ]

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
        self._serial.write(b"9")
        self._serial.flush()
        self._read_until(b"\x1b")

    def exit_diagnostic_mode(self) -> None:
        """Exit the diagnostic/Forth interface and return to the main menu."""

        if not self.is_open:
            raise RuntimeError("Optoscan serial port is not open")

        self._serial.write(b"menu\n")
        self._serial.flush()

        self._read_until(b"9. Enter diagnostic mode")

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
