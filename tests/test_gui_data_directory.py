from unittest.mock import patch

from PySide6.QtWidgets import QApplication

from msp_control.gui.app import MSPMainWindow


def test_open_experiment_starts_at_data_root(tmp_path):
    """Open Experiment always starts at the data root."""

    app = QApplication.instance() or QApplication([])

    window = MSPMainWindow()
    window.data_root = tmp_path

    try:
        with patch(
            "msp_control.gui.app.QFileDialog.getOpenFileName",
            return_value=("", ""),
        ) as file_dialog:
            window.on_open_experiment()

        file_dialog.assert_called_once_with(
            window,
            "Open Experiment",
            str(tmp_path),
            "MSP Experiment (*.h5)",
        )

    finally:
        window.close()


def test_new_experiment_uses_experimenter_directory(tmp_path):
    """New Experiment starts in the selected experimenter's folder."""

    app = QApplication.instance() or QApplication([])

    user_directory = tmp_path / "Rikard"
    user_directory.mkdir()

    window = MSPMainWindow()
    window.data_root = tmp_path

    try:
        with (
            patch(
                "msp_control.gui.app.ExperimentMetadataDialog"
            ) as metadata_dialog,
            patch(
                "msp_control.gui.app.QFileDialog.getSaveFileName",
                return_value=("", ""),
            ) as file_dialog,
        ):
            metadata_dialog.return_value.exec.return_value = True
            metadata_dialog.return_value.get_metadata.return_value = {
                "experimenter": "RF"
            }

            window.on_new_experiment()

        file_dialog.assert_called_once_with(
            window,
            "Create New Experiment",
            str(user_directory),
            "MSP Experiment (*.h5)",
        )

    finally:
        window.close()


def test_new_experiment_falls_back_to_data_root(tmp_path):
    """Use the data root when the experimenter's folder is missing."""

    app = QApplication.instance() or QApplication([])

    window = MSPMainWindow()
    window.data_root = tmp_path

    # Deliberately do not create tmp_path / "Chris".

    try:
        with (
            patch(
                "msp_control.gui.app.ExperimentMetadataDialog"
            ) as metadata_dialog,
            patch(
                "msp_control.gui.app.QFileDialog.getSaveFileName",
                return_value=("", ""),
            ) as file_dialog,
        ):
            metadata_dialog.return_value.exec.return_value = True
            metadata_dialog.return_value.get_metadata.return_value = {
                "experimenter": "CM"
            }

            window.on_new_experiment()

        file_dialog.assert_called_once_with(
            window,
            "Create New Experiment",
            str(tmp_path),
            "MSP Experiment (*.h5)",
        )

        assert not (tmp_path / "Chris").exists()

    finally:
        window.close()
