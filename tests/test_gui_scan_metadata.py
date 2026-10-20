from dataclasses import replace

import pytest
import numpy as np

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


def test_new_scan_replaces_displayed_scan(
    tmp_path, baseline, scan, qapp
):
    """A newly acquired scan replaces the displayed spectrum."""

    filename = tmp_path / "test.h5"

    create_experiment_file(filename)
    write_baseline(filename, baseline)
    write_scan(filename, scan)

    window = MSPMainWindow()

    try:
        window.load_experiment(filename)

        # Initially display the existing scan.
        first_item = window.scan_list.item(0)
        first_item.setCheckState(Qt.CheckState.Checked)

        assert window.scan_curves[scan.scan_index].isVisible()

        # Simulate a newly acquired scan with different OD data.
        new_scan = replace(
            scan,
            scan_index=scan.scan_index + 1,
            optical_density=scan.optical_density + 0.5,
            metadata=scan.metadata.copy(),
        )

        # The controller normally adds the scan before notifying the GUI.
        window.experiment.add_scan(new_scan)

        # Change plot settings before acquiring the new scan.
        window.plot_mode_combo.setCurrentIndex(
            window.plot_mode_combo.findData("scan")
        )
        window.zero_correction_checkbox.setChecked(True)
        window.show_legend_checkbox.setChecked(True)

        assert window.plot_mode_combo.currentData() == "scan"
        assert window.zero_correction_checkbox.isChecked()
        assert window.show_legend_checkbox.isChecked()
        
        window.on_scan_acquired(new_scan)

        # New acquisitions restore the default plot settings.
        assert window.plot_mode_combo.currentData() == "od"
        assert not window.zero_correction_checkbox.isChecked()
        assert not window.show_legend_checkbox.isChecked()

        # Zero-correction cursors should be hidden.
        assert not window.zero_start_line.isVisible()
        assert not window.zero_end_line.isVisible()

        # The legend should be hidden.
        assert not window.spectrum_legend.isVisible()

        new_item = window.scan_list.item(1)

        # Only the new scan should be checked.
        assert first_item.checkState() == Qt.CheckState.Unchecked
        assert new_item.checkState() == Qt.CheckState.Checked

        # The new scan should also be highlighted.
        assert window.scan_list.currentItem() is new_item

        # The old curve should be hidden and the new curve visible.
        assert not window.scan_curves[scan.scan_index].isVisible()
        assert window.scan_curves[new_scan.scan_index].isVisible()

        # Verify that the plotted data are the new scan's OD values.
        x, y = window.scan_curves[new_scan.scan_index].getData()

        np.testing.assert_allclose(x, new_scan.wavelength)
        np.testing.assert_allclose(y, new_scan.optical_density)

    finally:
        window.close()


def test_plot_display_modes_use_correct_data(
    tmp_path, baseline, scan, qapp
):
    """Display modes use the correct data without modifying the model."""

    filename = tmp_path / "test.h5"

    create_experiment_file(filename)
    write_baseline(filename, baseline)
    write_scan(filename, scan)

    window = MSPMainWindow()

    try:
        window.load_experiment(filename)

        # Select the scan for plotting.
        item = window.scan_list.item(0)
        item.setCheckState(Qt.CheckState.Checked)

        curve = window.scan_curves[scan.scan_index]

        # Keep a copy of the original OD values.
        original_od = scan.optical_density.copy()

        # Optical density mode.
        window.plot_mode_combo.setCurrentIndex(
            window.plot_mode_combo.findData("od")
        )

        x, y = curve.getData()

        np.testing.assert_allclose(x, scan.wavelength)
        np.testing.assert_allclose(y, scan.optical_density)

        # Dark-corrected specimen signal.
        window.plot_mode_combo.setCurrentIndex(
            window.plot_mode_combo.findData("scan")
        )

        x, y = curve.getData()

        np.testing.assert_allclose(x, scan.wavelength)
        np.testing.assert_allclose(y, scan.specimen_corrected)

        # Dark-corrected associated baseline.
        window.plot_mode_combo.setCurrentIndex(
            window.plot_mode_combo.findData("baseline")
        )

        x, y = curve.getData()

        np.testing.assert_allclose(x, baseline.wavelength)
        np.testing.assert_allclose(y, baseline.baseline_corrected)

        # Switching display modes must not modify stored OD.
        np.testing.assert_array_equal(
            window.experiment.get_scan(scan.scan_index).optical_density,
            original_od,
        )

    finally:
        window.close()


def test_shared_baseline_plotted_only_once(
    tmp_path, baseline, scan, qapp
):
    """Scans sharing a baseline display only one baseline curve."""

    filename = tmp_path / "test.h5"

    create_experiment_file(filename)
    write_baseline(filename, baseline)
    write_scan(filename, scan)

    second_scan = replace(
        scan,
        scan_index=scan.scan_index + 1,
        metadata=scan.metadata.copy(),
    )
    write_scan(filename, second_scan)

    window = MSPMainWindow()

    try:
        window.load_experiment(filename)

        first_item = window.scan_list.item(0)
        second_item = window.scan_list.item(1)

        first_item.setCheckState(Qt.CheckState.Checked)
        second_item.setCheckState(Qt.CheckState.Checked)

        first_curve = window.scan_curves[scan.scan_index]
        second_curve = window.scan_curves[second_scan.scan_index]

        # Both specimen spectra should be visible in OD mode.
        assert first_curve.isVisible()
        assert second_curve.isVisible()

        # In Baseline mode, only one shared baseline is displayed.
        window.plot_mode_combo.setCurrentIndex(
            window.plot_mode_combo.findData("baseline")
        )

        assert first_curve.isVisible()
        assert not second_curve.isVisible()

        # Unchecking the first scan should transfer baseline
        # visibility to the second scan.
        first_item.setCheckState(Qt.CheckState.Unchecked)

        assert not first_curve.isVisible()
        assert second_curve.isVisible()

        # Returning to Scan mode should display the checked
        # specimen spectrum normally.
        window.plot_mode_combo.setCurrentIndex(
            window.plot_mode_combo.findData("scan")
        )

        assert not first_curve.isVisible()
        assert second_curve.isVisible()

    finally:
        window.close()


def test_zero_correction_does_not_modify_scan(
    tmp_path, baseline, scan, qapp
):
    """Zero correction changes displayed OD, not stored data."""

    filename = tmp_path / "test.h5"

    create_experiment_file(filename)
    write_baseline(filename, baseline)
    write_scan(filename, scan)

    window = MSPMainWindow()

    try:
        window.load_experiment(filename)

        item = window.scan_list.item(0)
        item.setCheckState(Qt.CheckState.Checked)

        curve = window.scan_curves[scan.scan_index]
        stored_scan = window.experiment.get_scan(scan.scan_index)
        original_od = stored_scan.optical_density.copy()

        # Choose an interval containing several wavelength samples.
        wavelength = stored_scan.wavelength
        start = float(wavelength[0])
        end = float(wavelength[-1])

        window.zero_start_spin.setValue(start)
        window.zero_end_spin.setValue(end)
        window.zero_correction_checkbox.setChecked(True)

        x, y = curve.getData()

        mask = (x >= start) & (x <= end) & np.isfinite(y)

        assert np.any(mask)
        np.testing.assert_allclose(
            np.mean(y[mask]),
            0.0,
            atol=1e-12,
        )

        # The stored OD array must remain unchanged.
        np.testing.assert_array_equal(
            stored_scan.optical_density,
            original_od,
        )

        # Disabling correction restores the original spectrum.
        window.zero_correction_checkbox.setChecked(False)

        x, y = curve.getData()

        np.testing.assert_allclose(x, wavelength)
        np.testing.assert_array_equal(y, original_od)

    finally:
        window.close()


def test_plot_zoom_preserved_when_toggling_legend(
    tmp_path, baseline, scan, qapp
):
    """Legend changes preserve zoom; scan selection resets it."""

    filename = tmp_path / "test.h5"

    create_experiment_file(filename)
    write_baseline(filename, baseline)
    write_scan(filename, scan)

    window = MSPMainWindow()

    try:
        window.load_experiment(filename)

        item = window.scan_list.item(0)
        item.setCheckState(Qt.CheckState.Checked)

        view_box = window.spectrum_plot.getViewBox()

        # Set a deliberately narrow viewing range.
        view_box.setRange(
            xRange=(400, 500),
            yRange=(-0.5, 0.5),
            padding=0,
        )

        original_range = view_box.viewRange()

        # Toggling the legend should preserve the viewing range.
        window.show_legend_checkbox.setChecked(True)

        np.testing.assert_allclose(
            view_box.viewRange(),
            original_range,
        )

        # Changing scan visibility should restore automatic scaling.
        item.setCheckState(Qt.CheckState.Unchecked)
        item.setCheckState(Qt.CheckState.Checked)

        assert not np.allclose(
            view_box.viewRange(),
            original_range,
        )

    finally:
        window.close()
