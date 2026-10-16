from msp_control.hardware.optoscan import OptoscanSerial
from msp_control.hardware.field_stop import FieldStopMonitor


class HardwareManager:
    """Manage persistent connections to MSP hardware devices."""
    def __init__(
        self,
        optoscan: OptoscanSerial,
        field_stop: FieldStopMonitor,
    ):
        self.optoscan = optoscan
        self.field_stop = field_stop

    def reinitialize_optoscan(self) -> None:
        """Reestablish communication with the Optoscan."""
        self.optoscan.initialize()

    def initialize(self) -> None:
        """Initialize persistent hardware connections."""

        if not self.optoscan.is_open:
            self.optoscan.initialize()

        if not self.field_stop.is_connected:
            self.field_stop.connect()

    def disconnect(self) -> None:
        """Disconnect both persistent hardware devices."""
        try:
            self.field_stop.close()
        finally:
            self.optoscan.close()
