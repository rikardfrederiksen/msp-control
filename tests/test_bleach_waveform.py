import numpy as np
import pytest

from msp_control.calibration.waveform import (
    BleachWaveformConfig,
    generate_bleach_waveform,
)


def test_digital_bleach_waveform():
    config = BleachWaveformConfig(
        duration_s=10.0,
        sample_rate_hz=1000.0,
        pre_dark_s=0.1,
        post_dark_s=0.05,
    )

    time, output = generate_bleach_waveform(
        config,
        control_mode="digital",
        command_voltage=None,
    )

    assert len(time) == 10150
    assert len(output) == 10150
    assert output.dtype == np.bool_

    np.testing.assert_array_equal(output[:100], False)
    np.testing.assert_array_equal(output[100:10100], True)
    np.testing.assert_array_equal(output[10100:], False)

    assert time[0] == 0.0
    assert time[-1] == pytest.approx(10.149)


def test_analog_bleach_waveform():
    config = BleachWaveformConfig(
        duration_s=2.0,
        sample_rate_hz=1000.0,
        pre_dark_s=0.1,
        post_dark_s=0.05,
    )

    time, output = generate_bleach_waveform(
        config,
        control_mode="analog",
        command_voltage=2.5,
    )

    assert len(output) == 2150
    assert output.dtype == np.float64

    np.testing.assert_array_equal(output[:100], 0.0)
    np.testing.assert_array_equal(output[100:2100], 2.5)
    np.testing.assert_array_equal(output[2100:], 0.0)


def test_analog_bleach_requires_voltage():
    config = BleachWaveformConfig(duration_s=1.0)

    with pytest.raises(ValueError, match="command_voltage"):
        generate_bleach_waveform(
            config,
            control_mode="analog",
            command_voltage=None,
        )


def test_digital_bleach_rejects_voltage():
    config = BleachWaveformConfig(duration_s=1.0)

    with pytest.raises(ValueError, match="command_voltage"):
        generate_bleach_waveform(
            config,
            control_mode="digital",
            command_voltage=2.5,
        )


def test_bleach_waveform_rejects_invalid_mode():
    config = BleachWaveformConfig(duration_s=1.0)

    with pytest.raises(ValueError, match="control_mode"):
        generate_bleach_waveform(
            config,
            control_mode="invalid",
            command_voltage=None,
        )


def test_bleach_waveform_rejects_invalid_duration():
    with pytest.raises(ValueError, match="duration"):
        BleachWaveformConfig(duration_s=0.0)


def test_bleach_waveform_rejects_invalid_sample_rate():
    with pytest.raises(ValueError, match="sample_rate"):
        BleachWaveformConfig(
            duration_s=1.0,
            sample_rate_hz=0.0,
        )
