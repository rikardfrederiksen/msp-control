import pytest
from msp_control.hardware.optoscan import ScanConfig, build_config_commands


def test_scan_config():
    config = ScanConfig(
        start_nm=360,
        end_nm=720,
        step_nm=2,
        step_time_ms=2,
        input_slit_nm=4,
        output_slit_nm=4,
        dark_scans=2,
        data_scans=5,
    )

    assert config.cycles == 8
    assert config.wavelength_points == 181
    assert config.samples == 1448


def test_scan_config_rejects_zero_step():
    with pytest.raises(ValueError):
        ScanConfig(
            start_nm=360,
            end_nm=720,
            step_nm=0,
            step_time_ms=2,
            input_slit_nm=4,
            output_slit_nm=4,
            dark_scans=2,
            data_scans=5,
        )


def test_scan_config_rejects_invalid_range():
    with pytest.raises(ValueError):
        ScanConfig(
            start_nm=360,
            end_nm=721,
            step_nm=2,
            step_time_ms=2,
            input_slit_nm=4,
            output_slit_nm=4,
            dark_scans=2,
            data_scans=5,
        )


def test_build_config_commands():
    config = ScanConfig(
        start_nm=360,
        end_nm=720,
        step_nm=2,
        step_time_ms=2,
        input_slit_nm=4,
        output_slit_nm=4,
        dark_scans=2,
        data_scans=5,
    )

    commands = build_config_commands(config)

    assert commands == [
        "close_slits",
        "3600 scan_start !",
        "7200 scan_end !",
        "20 scan_step_size !",
        "2000 scan_step_time_lo !",
        "0 scan_step_time_hi !",
        "0 scan_data>ram",
        "0 scan_prog# !",
        "0 ram>scan_data",
        "8 cycles !",
        "compile_tables",
        "set_scantable",
        "40 scan_inslit_width !",
        "40 scan_exslit_width !",
    ]
