from msp_control.hardware.field_stop import FieldStopMonitor


class MetadataCollector:
    """Collect instrument metadata before an acquisition."""

    def __init__(
        self,
        field_stop: FieldStopMonitor | None = None,
    ):
        self.field_stop = field_stop

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

        return metadata
