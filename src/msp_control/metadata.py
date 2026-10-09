import serial

from nidaqmx.errors import DaqError

from msp_control.hardware.temperature import TemperatureMonitor
from msp_control.hardware.field_stop import FieldStopMonitor
from msp_control.hardware.stage import StagePositionMonitor

class MetadataCollector:
    """Collect instrument metadata before an acquisition."""

    def __init__(
        self,
        field_stop: FieldStopMonitor | None = None,
        stage: StagePositionMonitor | None = None,
        temperature: TemperatureMonitor | None = None,
    ):
        self.field_stop = field_stop
        self.stage = stage
        self.temperature = temperature

    def collect(self) -> dict:
        """Return a snapshot of the current instrument metadata."""

        metadata = {}

        if self.field_stop is not None:
            try:
                position = self.field_stop.read_position()
            except RuntimeError as exc:
                if "not homed" not in str(exc):
                    raise
            else:
                metadata.update({
                    "field_stop_stage_angle_deg": position.stage_angle_deg,
                    "field_stop_angle_deg": position.angle_deg,
                    "field_stop_horizontal_reference_deg": (
                        self.field_stop.config.horizontal_reference_deg
                    ),
                })

        if self.stage is not None:
            try:
                position = self.stage.read_position()
            except (serial.SerialException, ValueError):
                pass
            else:
                metadata.update({
                    "stage_x_um": position.x_um,
                    "stage_y_um": position.y_um,
                    "stage_z_um": position.z_um,
                })

        if self.temperature is not None:
            try:
                measurement = self.temperature.measure()
            except DaqError:
                pass
            else:
                metadata.update({
                    "temperature_mean_C": measurement.mean_C,
                    "temperature_sd_C": measurement.sd_C,
                })

        return metadata
