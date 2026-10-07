import serial
import struct
from dataclasses import dataclass

@dataclass(frozen=True)
class StageConfig:
    port: str = "COM5"
    baudrate: int = 128000
    timeout: float = 1.0
    um_per_microstep: float = 0.0625
    drive: int = 1

@dataclass(frozen=True)
class StagePosition:
    x_um: float
    y_um: float
    z_um: float


class StagePositionMonitor:
    def __init__(self, config: StageConfig = StageConfig()):
        self.config = config

    def read_position(self) -> StagePosition:
        ser = serial.Serial(
            port=self.config.port,
            baudrate=self.config.baudrate,
            bytesize=8,
            parity="N",
            stopbits=1,
            timeout=self.config.timeout,
        )

        try:
            ser.write(b"C")
            response = ser.read_until(b"\r")
        finally:
            ser.close()

        return self.decode_position_response(
            response,
            config=self.config,
        )

    @staticmethod
    def decode_position_response(
        response: bytes,
        config: StageConfig = StageConfig(),
    ) -> StagePosition:
        if len(response) != 14:
            raise ValueError(
                f"Position response must be 14 bytes, got {len(response)}"
            )

        if response[-1] != 0x0D:
            raise ValueError("Position response has invalid terminator")

        drive, x, y, z = struct.unpack("<BIII", response[:-1])

        if drive != config.drive:
            raise ValueError(
                f"Position response is for drive {drive}, "
                f"expected drive {config.drive}"
            )

        return StagePosition(
            x_um=x * config.um_per_microstep,
            y_um=y * config.um_per_microstep,
            z_um=z * config.um_per_microstep,
        )

