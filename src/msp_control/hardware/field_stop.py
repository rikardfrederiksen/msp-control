from dataclasses import dataclass


@dataclass(frozen=True)
class FieldStopConfig:
    serial_number: str = "27503323"
    poll_interval_ms: int = 250
    settings_timeout_ms: int = 5000
    kinesis_path: str = r"C:\Program Files\Thorlabs\Kinesis"


@dataclass(frozen=True)
class FieldStopPosition:
    stage_angle_deg: float


class KinesisFieldStopDevice:
    def __init__(
        self,
        config: FieldStopConfig,
        device_manager,
        kcube_class,
    ):
        self.config = config
        self.device_manager = device_manager
        self.kcube_class = kcube_class
        self.device = None

    def connect(self):
        self.device_manager.BuildDeviceList()

        self.device = self.kcube_class.CreateKCubeDCServo(
            self.config.serial_number
        )

        try:
            self.device.Connect(self.config.serial_number)
            self.device.WaitForSettingsInitialized(
                self.config.settings_timeout_ms
            )
            self.device.LoadMotorConfiguration(
                self.config.serial_number
            )
            self.device.StartPolling(
                self.config.poll_interval_ms
            )

        except Exception:
            try:
                self.device.Disconnect()
            finally:
                self.device = None
            raise

    def is_homed(self) -> bool:
        return bool(self.device.Status.IsHomed)

    def get_position(self) -> float:
        return float(self.device.Position)

    def close(self):
        if self.device is not None:
            self.device.StopPolling()
            self.device.Disconnect()
            self.device = None

    @classmethod
    def from_kinesis(cls, config: FieldStopConfig = FieldStopConfig()):
        device_manager, kcube_class = load_kinesis(config)

        return cls(
            config=config,
            device_manager=device_manager,
            kcube_class=kcube_class,
        )

class FieldStopMonitor:
    def __init__(self, device):
        self.device = device

    def connect(self):
        self.device.connect()

    def read_position(self) -> FieldStopPosition:
        if not self.device.is_homed():
            raise RuntimeError("Field-stop rotation stage is not homed")

        return FieldStopPosition(
            stage_angle_deg=float(self.device.get_position())
        )

    def close(self):
        self.device.close()


def load_kinesis(config: FieldStopConfig):
    import sys

    if config.kinesis_path not in sys.path:
        sys.path.append(config.kinesis_path)

    try:
        import clr
    except ImportError as exc:
        raise RuntimeError(
            "pythonnet is required to use the Thorlabs field-stop stage"
        ) from exc

    clr.AddReference("Thorlabs.MotionControl.DeviceManagerCLI")
    clr.AddReference("Thorlabs.MotionControl.KCube.DCServoCLI")

    from Thorlabs.MotionControl.DeviceManagerCLI import DeviceManagerCLI
    from Thorlabs.MotionControl.KCube.DCServoCLI import KCubeDCServo

    return DeviceManagerCLI, KCubeDCServo
