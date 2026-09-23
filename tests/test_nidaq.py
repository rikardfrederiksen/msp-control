from msp_control.hardware.nidaq import NIDaqConfig


def test_nidaq_config():
    config = NIDaqConfig()

    assert config.pmt_physical_channel == "Dev1/ai0"
    assert config.sample_clock_source == "/Dev1/PFI1"
    assert config.start_trigger_source == "/Dev1/PFI2"
    assert config.shutter_physical_channel == "Dev1/port0/line0"
    assert config.ir_physical_channel == "Dev1/port0/line1"

    assert config.pmt_min_v == 0.0
    assert config.pmt_max_v == 5.0
    assert config.nominal_sample_rate == 10_000.0
    assert config.read_timeout == 10.0
