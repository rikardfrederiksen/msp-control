import pytest
from unittest.mock import MagicMock, patch


from msp_control.hardware.field_stop import (
    FieldStopConfig,
    FieldStopMonitor,
    FieldStopPosition,
    KinesisFieldStopDevice,
)


def test_field_stop_config():
    config = FieldStopConfig()

    assert config.serial_number == "27503323"
    assert config.poll_interval_ms == 250
    assert config.settings_timeout_ms == 5000


def test_field_stop_position():
    position = FieldStopPosition(
        stage_angle_deg=130.0623,
        angle_deg=90.0623,
    )

    assert position.stage_angle_deg == pytest.approx(130.0623)
    assert position.angle_deg == pytest.approx(90.0623)


def test_field_stop_monitor_connects_device():
    device = MagicMock()
    config = FieldStopConfig()
    monitor = FieldStopMonitor(device=device, config=config)

    monitor.connect()

    device.connect.assert_called_once()


def test_field_stop_monitor_reads_position():
    device = MagicMock()
    device.is_homed.return_value = True
    device.get_position.return_value = 130.0

    config = FieldStopConfig()
    monitor = FieldStopMonitor(device=device, config=config)

    position = monitor.read_position()

    device.get_position.assert_called_once()
    assert position.stage_angle_deg == pytest.approx(130.0)
    assert position.angle_deg == pytest.approx(90.0)


@pytest.mark.parametrize(
    "stage_angle, expected_angle",
    [
        (40.0, 0.0),      # Horizontal
        (130.0, 90.0),    # Vertical
        (220.0, 0.0),     # Horizontal + 180°
        (310.0, 90.0),    # Vertical + 180°
        (20.0, 160.0),    # Below reference
        (85.0, 45.0),     # Intermediate orientation
    ],
)


def test_field_stop_calibration(stage_angle, expected_angle):
    device = MagicMock()
    device.is_homed.return_value = True
    device.get_position.return_value = stage_angle

    config = FieldStopConfig()
    monitor = FieldStopMonitor(device=device, config=config)

    position = monitor.read_position()

    assert position.stage_angle_deg == pytest.approx(stage_angle)
    assert position.angle_deg == pytest.approx(expected_angle)


def test_field_stop_custom_calibration_reference():
    device = MagicMock()
    device.is_homed.return_value = True
    device.get_position.return_value = 135.0

    config = FieldStopConfig(horizontal_reference_deg=45.0)
    monitor = FieldStopMonitor(device=device, config=config)

    position = monitor.read_position()

    assert position.stage_angle_deg == pytest.approx(135.0)
    assert position.angle_deg == pytest.approx(90.0)


def test_field_stop_monitor_rejects_unhomed_device():
    device = MagicMock()
    device.is_homed.return_value = False

    config = FieldStopConfig()
    monitor = FieldStopMonitor(device=device, config=config)

    with pytest.raises(RuntimeError, match="not homed"):
        monitor.read_position()


def test_field_stop_monitor_closes_device():
    device = MagicMock()
    config = FieldStopConfig()
    monitor = FieldStopMonitor(device=device, config=config)

    monitor.close()

    device.close.assert_called_once()


def test_kinesis_device_connect():
    device_manager = MagicMock()
    kcube_class = MagicMock()
    kcube = MagicMock()
    settings_option = MagicMock()

    kcube_class.CreateKCubeDCServo.return_value = kcube

    device = KinesisFieldStopDevice(
        config=FieldStopConfig(),
        device_manager=device_manager,
        kcube_class=kcube_class,
        settings_option=settings_option,
    )

    device.connect()

    device_manager.BuildDeviceList.assert_called_once()
    kcube_class.CreateKCubeDCServo.assert_called_once_with("27503323")
    kcube.Connect.assert_called_once_with("27503323")
    kcube.WaitForSettingsInitialized.assert_called_once_with(5000)
    kcube.LoadMotorConfiguration.assert_called_once_with(
        "27503323",
        settings_option,
    )
    kcube.StartPolling.assert_called_once_with(250)

def test_kinesis_device_reports_homed_status():
    kcube = MagicMock()
    kcube.Status.IsHomed = True

    device = KinesisFieldStopDevice(
        config=FieldStopConfig(),
        device_manager=MagicMock(),
        kcube_class=MagicMock(),
    )
    device.device = kcube

    assert device.is_homed() is True


def test_kinesis_device_reads_position():
    kcube = MagicMock()
    kcube.Position = 130.0623

    device = KinesisFieldStopDevice(
        config=FieldStopConfig(),
        device_manager=MagicMock(),
        kcube_class=MagicMock(),
    )
    device.device = kcube

    assert device.get_position() == pytest.approx(130.0623)


def test_kinesis_device_reads_dotnet_decimal_position():
    class FakeDotNetDecimal:
        def ToString(self):
            return "130.0623"

        def __float__(self):
            raise TypeError("Cannot convert .NET Decimal directly")

    kcube = MagicMock()
    kcube.Position = FakeDotNetDecimal()

    device = KinesisFieldStopDevice(
        config=FieldStopConfig(),
        device_manager=MagicMock(),
        kcube_class=MagicMock(),
    )
    device.device = kcube

    assert device.get_position() == pytest.approx(130.0623)


def test_kinesis_device_closes():
    kcube = MagicMock()

    device = KinesisFieldStopDevice(
        config=FieldStopConfig(),
        device_manager=MagicMock(),
        kcube_class=MagicMock(),
    )
    device.device = kcube

    device.close()

    kcube.StopPolling.assert_called_once()
    kcube.Disconnect.assert_called_once()
    assert device.device is None


def test_kinesis_device_disconnects_if_initialization_fails():
    device_manager = MagicMock()
    kcube_class = MagicMock()
    kcube = MagicMock()

    kcube_class.CreateKCubeDCServo.return_value = kcube
    kcube.WaitForSettingsInitialized.side_effect = RuntimeError(
        "Initialization failed"
    )

    device = KinesisFieldStopDevice(
        config=FieldStopConfig(),
        device_manager=device_manager,
        kcube_class=kcube_class,
        settings_option=MagicMock(),
    )

    with pytest.raises(RuntimeError, match="Initialization failed"):
        device.connect()

    kcube.Disconnect.assert_called_once()
    assert device.device is None


def test_kinesis_device_from_kinesis():
    device_manager = MagicMock()
    kcube_class = MagicMock()
    settings_option = MagicMock()

    with patch(
        "msp_control.hardware.field_stop.load_kinesis",
        return_value=(device_manager, kcube_class, settings_option),
    ):
        device = KinesisFieldStopDevice.from_kinesis()

    assert device.device_manager is device_manager
    assert device.kcube_class is kcube_class
    assert device.settings_option is settings_option


def test_kinesis_device_rejects_missing_settings_option():
    device_manager = MagicMock()
    kcube_class = MagicMock()

    device = KinesisFieldStopDevice(
        config=FieldStopConfig(),
        device_manager=device_manager,
        kcube_class=kcube_class,
    )

    with pytest.raises(RuntimeError, match="settings option"):
        device.connect()

    device_manager.BuildDeviceList.assert_not_called()
    kcube_class.CreateKCubeDCServo.assert_not_called()
