from dataclasses import replace

import pytest

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from msp_control.gui.app import MSPMainWindow
from msp_control.storage.hdf5 import (
    create_experiment_file,
    write_baseline,
    write_scan,
    read_scan,
)


@pytest.fixture
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_save_metadata_updates_highlighted_scan(
    tmp_path, baseline, scan, qapp
):
    """Saving metadata updates the highlighted scan, not the checked scan."""

    filename = tmp_path / "test.h5"

    create_experiment_file(filename)
    write_baseline(filename, baseline)
    write_scan(filename, scan)

    # Create a second scan with a different index.
    second_scan = replace(
        scan,
        scan_index=scan.scan_index + 1,
        metadata=scan.metadata.copy(),
    )
    write_scan(filename, second_scan)

    window = MSPMainWindow()

    try:
        window.load_experiment(filename)

        assert window.scan_list.count() == 2

        first_item = window.scan_list.item(0)
        second_item = window.scan_list.item(1)

        # Check the first scan for plotting.
        first_item.setCheckState(Qt.CheckState.Checked)
        second_item.setCheckState(Qt.CheckState.Unchecked)

        # Highlight the second scan for metadata editing.
        window.scan_list.setCurrentItem(second_item)

        window.cell_type_combo.setCurrentIndex(
            window.cell_type_combo.findData("Cone")
        )
        window.scan_comment_edit.setPlainText("Test comment")

        assert window.save_scan_metadata_button.isEnabled()

        window.save_scan_metadata_button.click()

        assert not window.save_scan_metadata_button.isEnabled()

        # The highlighted scan should have been updated.
        saved_second = read_scan(filename, second_scan.scan_index)
        assert saved_second.metadata["cell_type"] == "Cone"
        assert saved_second.metadata["comment"] == "Test comment"

        # The checked scan should remain unchanged.
        saved_first = read_scan(filename, scan.scan_index)
        assert saved_first.metadata == scan.metadata

    finally:
        window.close()
