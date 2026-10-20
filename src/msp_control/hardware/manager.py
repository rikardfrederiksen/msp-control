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

        optoscan_was_open = self.optoscan.is_open
        field_stop_was_connected = self.field_stop.is_connected

        try:
            if not optoscan_was_open:
                self.optoscan.initialize()

            if not field_stop_was_connected:
                self.field_stop.connect()

        except Exception:
            # Only close devices that were not connected before
            # this initialization attempt.
            if not field_stop_was_connected:
                try:
                    self.field_stop.close()
                except Exception:
                    pass

            if not optoscan_was_open:
                try:
                    self.optoscan.close()
                except Exception:
                    pass

            raise

    def disconnect(self) -> None:
        """Disconnect both persistent hardware devices."""
        try:
            self.field_stop.close()
        finally:
            self.optoscan.close()
