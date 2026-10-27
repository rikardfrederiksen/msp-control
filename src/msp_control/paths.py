"""Paths to external MSP configuration files."""

import sys
from pathlib import Path


def config_directory() -> Path:
    """Return the external configuration directory."""

    if getattr(sys, "frozen", False):
        # Running as a PyInstaller executable.
        root = Path(sys.executable).resolve().parent
    else:
        # Running from source.
        root = Path(__file__).resolve().parents[2]

    return root / "config"
