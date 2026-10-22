from datetime import datetime
from unittest.mock import Mock, patch

import numpy as np

from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtWidgets import QApplication

from msp_control.calibration.settings import load_default_calibration
from tests.test_calibration_storage import test_calibration
from msp_control.calibration.storage import read_calibration
from msp_control.gui.calibration_dialog import CalibrationDialog
from msp_control.calibration.model import (
    AnalogLEDCalibration,
    DigitalLEDCalibration,
    LightCalibration,
)


def test_calibration_worker_thread():
    app = QApplication.instance() or QApplication([])

    dialog = CalibrationDialog()

    expected_calibration = object()

    workflow = Mock()
    workflow.acquire.return_value = expected_calibration

    config = dialog.get_calibration_config()

    loop = QEventLoop()
    timeout = QTimer()
    timeout.setSingleShot(True)
    timeout.setInterval(5000)

    timeout.timeout.connect(loop.quit)

    dialog.acquisition_thread_finished_for_test = False

    original_cleanup = dialog.on_acquisition_thread_finished

    def on_thread_finished():
        original_cleanup()
        dialog.acquisition_thread_finished_for_test = True
        loop.quit()

    # Replace the cleanup slot for this test.
    dialog.on_acquisition_thread_finished = on_thread_finished

    dialog.start_acquisition(
        workflow=workflow,
        config=config,
        calibration_id="test_thread",
        created_at=datetime(2026, 10, 21, 12, 0),
    )

    timeout.start()
    loop.exec()
    timeout.stop()

    assert dialog.acquisition_thread_finished_for_test
    assert dialog.calibration is expected_calibration
    assert dialog.save_button.isEnabled()
    assert dialog.acquisition_thread is None

    workflow.acquire.assert_called_once()


def test_calibration_dialog_cannot_close_during_acquisition():
    app = QApplication.instance() or QApplication([])

    dialog = CalibrationDialog()

    # Simulate an active acquisition.
    dialog.acquisition_thread = object()

    dialog.show()
    app.processEvents()

    dialog.close()
    app.processEvents()

    assert dialog.isVisible()

    # Simulate acquisition completion.
    dialog.acquisition_thread = None

    dialog.close()
    app.processEvents()

    assert not dialog.isVisible()


def test_calibration_plot_selector():
    """Verify that each plot mode displays the correct calibration data."""

    app = QApplication.instance() or QApplication([])
    dialog = CalibrationDialog()

    analog = AnalogLEDCalibration(
        led_channel=1,
        wavelength_nm=617.0,
        amplifier_range="uA 1",
        amplifier_gain=1e6,
        photodiode_responsivity=0.4,
        illuminated_area_um2=10000.0,
        time_s=np.array([0.0, 1.0, 2.0, 3.0]),
        command_voltage=np.array([0.0, 2.0, 4.0, 0.0]),
        monitor_voltage=np.array([0.0, 0.4, 0.8, 0.0]),
        photodiode_voltage=np.array([0.05, 0.25, 0.45, 0.05]),
        calibration_voltage=np.array([0.0, 2.0, 4.0]),
        calibration_monitor_voltage=np.array([0.0, 0.4, 0.8]),
        photon_flux_density=np.array([0.0, 1e6, 2e6]),
    )

    digital = DigitalLEDCalibration(
        led_channel=2,
        wavelength_nm=505.0,
        amplifier_range="uA 1",
        amplifier_gain=1e6,
        photodiode_responsivity=0.3,
        illuminated_area_um2=10000.0,
        time_s=np.array([0.0, 1.0, 2.0]),
        command_state=np.array([False, True, False]),
        monitor_voltage=np.array([0.0, 2.0, 0.0]),
        photodiode_voltage=np.array([0.05, 0.5, 0.05]),
        monitor_voltage_on=2.0,
        photon_flux_density_on=1e7,
    )

    dialog.calibration = LightCalibration(
        calibration_id="test_plot",
        created_at=datetime(2026, 10, 21, 12, 0),
        analog_led=analog,
        digital_led=digital,
    )

    expected = {
        "Waveform vs time": (
            analog.time_s,
            analog.command_voltage,
        ),
        "Photodiode response": (
            analog.command_voltage,
            analog.photodiode_voltage,
        ),
        "LED monitor response": (
            analog.command_voltage,
            analog.monitor_voltage,
        ),
        "Photon flux density": (
            analog.calibration_voltage,
            analog.photon_flux_density,
        ),
    }

    for selection, (expected_x, expected_y) in expected.items():
        dialog.plot_selector.setCurrentText(selection)
        dialog.update_calibration_plot()

        curves = dialog.plot_widget.listDataItems()

        if selection in ("Photodiode response", "LED monitor response"):
            legend = dialog.plot_widget.plotItem.legend

            assert legend is not None
            assert len(legend.items) == 2

        if selection in ("Photodiode response", "LED monitor response"):
            assert len(curves) == 2

            peak_indices = np.flatnonzero(
                np.isclose(expected_x, np.max(expected_x))
            )
            first_peak = peak_indices[0]
            last_peak = peak_indices[-1]

            up_x, up_y = curves[0].getData()
            down_x, down_y = curves[1].getData()

            np.testing.assert_allclose(
                up_x, expected_x[:first_peak + 1]
            )
            np.testing.assert_allclose(
                up_y, expected_y[:first_peak + 1]
            )
            np.testing.assert_allclose(
                down_x, expected_x[last_peak:]
            )
            np.testing.assert_allclose(
                down_y, expected_y[last_peak:]
            )

        else:
            assert len(curves) == 1

            x, y = curves[0].getData()

            np.testing.assert_allclose(x, expected_x)
            np.testing.assert_allclose(y, expected_y)


def test_save_calibration_from_dialog(
    tmp_path, monkeypatch, test_calibration
):
    """Saving from the dialog creates a readable calibration file."""

    app = QApplication.instance() or QApplication([])
    dialog = CalibrationDialog()

    # Use a temporary working directory so no real calibration is saved.
    monkeypatch.setattr(
        "msp_control.gui.calibration_dialog.calibration_directory",
        lambda: tmp_path / "config" / "calibrations",
    )
    monkeypatch.setattr(
        "msp_control.calibration.settings.calibration_settings_path",
        lambda: tmp_path / "config" / "calibration_settings.json",
    )

    dialog.calibration = test_calibration
    dialog.save_button.setEnabled(True)

    dialog.save_button.click()
    assert dialog.calibration_saved is True

    filename = (
        tmp_path
        / "config"
        / "calibrations"
        / "test-calibration.h5"
    )

    assert load_default_calibration() == "test-calibration.h5"
    assert filename.exists()

    loaded = read_calibration(filename)

    assert loaded.calibration_id == test_calibration.calibration_id
    assert loaded.analog_led.wavelength_nm == 617.0
    assert loaded.digital_led.wavelength_nm == 505.0

    assert not dialog.save_button.isEnabled()


def test_calibration_dialog_not_saved_initially():
    """Opening a calibration dialog does not mark it as saved."""

    app = QApplication.instance() or QApplication([])
    dialog = CalibrationDialog()

    assert dialog.calibration_saved is False

    dialog.reject()
    assert dialog.calibration_saved is False


def test_calibration_repetition_controls():
    app = QApplication.instance() or QApplication([])

    dialog = CalibrationDialog()

    assert dialog.analog_repeats.value() == 5
    assert dialog.digital_repeats.value() == 5

    dialog.analog_repeats.setValue(3)
    dialog.digital_repeats.setValue(7)

    assert dialog.analog_repeats.value() == 3
    assert dialog.digital_repeats.value() == 7

    dialog.close()


def test_calibration_config_repetition_counts():
    """GUI repetition settings propagate into LightCalibrationConfig."""

    app = QApplication.instance() or QApplication([])

    dialog = CalibrationDialog()

    dialog.analog_repeats.setValue(3)
    dialog.digital_repeats.setValue(7)

    config = dialog.get_calibration_config()

    assert config.analog_repeats == 3
    assert config.digital_repeats == 7

    # Existing waveform configuration remains intact.
    assert config.analog_waveform.sample_rate_hz == 1000
    assert config.digital_waveform.sample_rate_hz == 1000

    dialog.close()


def test_calibration_acquire_button_connected():
    """Acquire button invokes the calibration acquisition entry point."""

    app = QApplication.instance() or QApplication([])

    with patch.object(
        CalibrationDialog,
        "on_acquire_calibration",
    ) as acquire:
        dialog = CalibrationDialog()

        assert dialog.acquire_button.isEnabled()

        dialog.acquire_button.click()

        acquire.assert_called_once_with()

        dialog.close()


def test_calibration_progress_display():
    """Progress updates display the LED and repetition number."""

    app = QApplication.instance() or QApplication([])

    dialog = CalibrationDialog()

    dialog.on_acquisition_progress("analog", 2, 5)

    assert dialog.status_label.text() == (
        "Analog LED: acquiring 2/5"
    )

    dialog.on_acquisition_progress("digital", 1, 3)

    assert dialog.status_label.text() == (
        "Digital LED: acquiring 1/3"
    )

    dialog.close()
