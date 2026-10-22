"""Calibration file discovery and default calibration settings."""

import json
from pathlib import Path


def calibration_directory() -> Path:
    """Return the directory containing saved LED calibrations."""
    return Path(__file__).resolve().parents[3] / "config" / "calibrations"


def find_calibration_files() -> list[Path]:
    """Return available calibration files, newest first."""

    calibration_dir = calibration_directory()

    if not calibration_dir.is_dir():
        return []

    return sorted(
        calibration_dir.glob("*.h5"),
        key=lambda path: path.name,
        reverse=True,
    )


def calibration_settings_path() -> Path:
    """Return the path to the calibration settings file."""
    return (
        Path(__file__).resolve().parents[3]
        / "config"
        / "calibration_settings.json"
    )


def load_default_calibration() -> str | None:
    """Return the configured default calibration filename."""

    path = calibration_settings_path()

    if not path.is_file():
        return None

    with path.open("r", encoding="utf-8") as f:
        settings = json.load(f)

    return settings.get("default_calibration")


def save_default_calibration(filename: str) -> None:
    """Save the default calibration filename."""

    path = calibration_settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(
            {"default_calibration": filename},
            f,
            indent=2,
        )
