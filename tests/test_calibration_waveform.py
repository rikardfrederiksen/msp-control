import pytest
import numpy as np

from msp_control.calibration.waveform import CalibrationWaveformConfig, generate_calibration_waveform


def test_default_calibration_waveform_config():
    config = CalibrationWaveformConfig()

    assert config.dark_before_s == 1.0
    assert config.ramp_up_s == 10.0
    assert config.hold_s == 1.0
    assert config.ramp_down_s == 10.0
    assert config.dark_after_s == 1.0
    assert config.max_voltage == 10.0
    assert config.sample_rate_hz == 1000.0


def test_custom_max_voltage():
    config = CalibrationWaveformConfig(max_voltage=8.0)

    assert config.max_voltage == 8.0


def test_default_waveform():
    config = CalibrationWaveformConfig()

    time_s, voltage = generate_calibration_waveform(config)

    assert len(time_s) == 23000
    assert len(voltage) == 23000

    assert time_s[0] == pytest.approx(0.0)
    assert time_s[-1] == pytest.approx(22.999)

    # Initial darkness
    assert np.all(voltage[:1000] == 0)

    # Upward ramp
    assert voltage[1000] == pytest.approx(0.0)
    assert voltage[6000] == pytest.approx(5.0)
    assert voltage[10999] == pytest.approx(9.999)

    # Hold at maximum voltage
    assert np.all(voltage[11000:12000] == 10.0)

    # Downward ramp
    assert voltage[12000] == pytest.approx(10.0)
    assert voltage[17000] == pytest.approx(5.0)
    assert voltage[21999] == pytest.approx(0.001)

    # Final darkness
    assert np.all(voltage[22000:] == 0)


def test_waveform_custom_max_voltage():
    config = CalibrationWaveformConfig(max_voltage=8.0)

    _, voltage = generate_calibration_waveform(config)

    assert voltage.max() == pytest.approx(8.0)
    assert np.all(voltage[11000:12000] == 8.0)


def test_waveform_custom_sample_rate():
    config = CalibrationWaveformConfig(sample_rate_hz=100.0)

    time_s, voltage = generate_calibration_waveform(config)

    assert len(time_s) == 2300
    assert len(voltage) == 2300
    assert time_s[-1] == pytest.approx(22.99)
