from unittest.mock import patch

from msp_control.gui.app import create_hardware_manager


def test_hardware_factory_shares_device_instances():
    with patch(
        "msp_control.gui.app.KinesisFieldStopDevice.from_kinesis"
    ):
        hardware, metadata, acquisition, bleach = create_hardware_manager()

    assert acquisition.optoscan is hardware.optoscan
    assert metadata.field_stop is hardware.field_stop

    assert metadata.stage is not None
    assert metadata.temperature is not None
    assert acquisition.daq is bleach.daq
