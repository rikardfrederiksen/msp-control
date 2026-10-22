from pathlib import Path
from unittest.mock import patch

from PySide6.QtWidgets import QApplication

from msp_control.gui.app import MSPMainWindow


def test_calibration_selection_after_dialog():
    """Closing without saving preserves the manual selection."""

    app = QApplication.instance() or QApplication([])

    calibration_files = [
        Path("/test/old_calibration.h5"),
        Path("/test/new_calibration.h5"),
    ]

    with (
        patch(
            "msp_control.gui.app.find_calibration_files",
            return_value=calibration_files,
        ),
        patch(
            "msp_control.gui.app.load_default_calibration",
            return_value="new_calibration.h5",
        ),
    ):
        window = MSPMainWindow()

        # Manually select the older calibration.
        index = window.calibration_combo.findText(
            "old_calibration.h5"
        )
        window.calibration_combo.setCurrentIndex(index)

        # Simulate opening and closing without saving.
        with patch(
            "msp_control.gui.app.CalibrationDialog"
        ) as MockDialog:
            MockDialog.return_value.calibration_saved = False
            window.on_new_calibration()

        assert (
            window.calibration_combo.currentText()
            == "old_calibration.h5"
        )

        # Simulate saving a new calibration.
        with patch(
            "msp_control.gui.app.CalibrationDialog"
        ) as MockDialog:
            MockDialog.return_value.calibration_saved = True
            window.on_new_calibration()

        assert (
            window.calibration_combo.currentText()
            == "new_calibration.h5"
        )

    window.close()
