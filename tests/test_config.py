import pytest

from msp_control.config import ScanConfig


@pytest.mark.parametrize(
    "field,value",
    [
        ("start_nm", 490),
        ("end_nm", 530),
        ("step_nm", 5),
        ("step_time_ms", 3),
        ("input_slit_nm", 5),
        ("output_slit_nm", 5),
    ],
)
def test_scan_config_incompatible_spectral_settings(field, value):
    settings = {
        "start_nm": 500,
        "end_nm": 520,
        "step_nm": 10,
        "step_time_ms": 2,
        "input_slit_nm": 4,
        "output_slit_nm": 4,
        "dark_scans": 2,
        "data_scans": 5,
    }

    config1 = ScanConfig(**settings)

    settings[field] = value
    config2 = ScanConfig(**settings)

    assert not config1.is_compatible_with(config2)

def test_scan_config_compatible_with_different_sweep_counts():
    config1 = ScanConfig(
        start_nm=500,
        end_nm=520,
        step_nm=10,
        step_time_ms=2,
        input_slit_nm=4,
        output_slit_nm=4,
        dark_scans=2,
        data_scans=5,
    )

    config2 = ScanConfig(
        start_nm=500,
        end_nm=520,
        step_nm=10,
        step_time_ms=2,
        input_slit_nm=4,
        output_slit_nm=4,
        dark_scans=3,
        data_scans=10,
    )

    assert config1.is_compatible_with(config2)
