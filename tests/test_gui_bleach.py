from unittest.mock import Mock

from PySide6.QtCore import QEventLoop, QTimer, QEvent, QCoreApplication
from PySide6.QtWidgets import QApplication

from msp_control.gui.app import MSPMainWindow
from msp_control.storage.hdf5 import create_experiment_file


def test_analog_bleach_worker(tmp_path):
    """Bleach button starts a worker with the correct parameters."""

    app = QApplication.instance() or QApplication([])

    filename = tmp_path / "bleach_test.h5"
    create_experiment_file(filename)

    window = MSPMainWindow()

    try:
        window.load_experiment(filename)

        # Simulate initialized hardware without opening devices.
        window.hardware_initialized = True
        window.controller.bleach_controller = Mock()
        window.update_acquisition_controls()

        assert window.bleach_button.isEnabled()

        # Configure an analog bleaching exposure.
        window.bleach_led_combo.setCurrentIndex(
            window.bleach_led_combo.findData(1)
        )
        window.bleach_duration_spin.setValue(0.25)
        window.bleach_voltage_spin.setValue(0.5)

        nd_filter = window.bleach_nd_combo.currentData()

        # Mock the high-level controller method, not the hardware.
        expected_bleach = Mock()
        expected_bleach.bleach_index = 0

        window.controller.acquire_bleach = Mock(
            return_value=expected_bleach
        )

        loop = QEventLoop()
        timeout = QTimer()
        timeout.setSingleShot(True)
        timeout.setInterval(5000)
        timeout.timeout.connect(loop.quit)

        window.bleach_button.click()

        assert window.acquisition_thread is not None
        assert not window.bleach_button.isEnabled()
        assert not window.acquire_baseline_button.isEnabled()

        # Wait for thread cleanup.
        thread = window.acquisition_thread
        thread.finished.connect(loop.quit)

        timeout.start()
        loop.exec()
        timeout.stop()

        # Process any remaining queued GUI signals.
        app.processEvents()

        assert window.acquisition_thread is None
        assert window.bleach_button.isEnabled()

        # Verify the controller received the GUI settings.
        window.controller.acquire_bleach.assert_called_once()

        kwargs = window.controller.acquire_bleach.call_args.kwargs

        assert kwargs["led_channel"] == 1
        assert kwargs["config"].duration_s == 0.25
        assert kwargs["nd_filter"] == nd_filter
        assert kwargs["command_voltage"] == 0.5

    finally:
        window.close()
        app.processEvents()
        QCoreApplication.sendPostedEvents(
            None, QEvent.Type.DeferredDelete
        )
        app.processEvents()


def test_digital_bleach_worker(tmp_path):
    """Digital bleaching must not pass an analog command voltage."""

    app = QApplication.instance() or QApplication([])

    filename = tmp_path / "digital_bleach_test.h5"
    create_experiment_file(filename)

    window = MSPMainWindow()

    try:
        window.load_experiment(filename)

        window.hardware_initialized = True
        window.controller.bleach_controller = Mock()
        window.update_acquisition_controls()

        # Select the digital LED.
        window.bleach_led_combo.setCurrentIndex(
            window.bleach_led_combo.findData(2)
        )
        window.bleach_duration_spin.setValue(0.5)

        assert not window.bleach_voltage_spin.isEnabled()

        expected_bleach = Mock()
        expected_bleach.bleach_index = 0

        window.controller.acquire_bleach = Mock(
            return_value=expected_bleach
        )

        loop = QEventLoop()
        timeout = QTimer()
        timeout.setSingleShot(True)
        timeout.setInterval(5000)
        timeout.timeout.connect(loop.quit)

        window.bleach_button.click()

        thread = window.acquisition_thread
        assert thread is not None

        thread.finished.connect(loop.quit)

        timeout.start()
        if thread.isRunning():
            loop.exec()
        timeout.stop()

        app.processEvents()

        assert window.acquisition_thread is None
        assert window.bleach_button.isEnabled()

        window.controller.acquire_bleach.assert_called_once()

        kwargs = window.controller.acquire_bleach.call_args.kwargs

        assert kwargs["led_channel"] == 2
        assert kwargs["config"].duration_s == 0.5
        assert kwargs["command_voltage"] is None

    finally:
        window.close()
        app.processEvents()
        QCoreApplication.sendPostedEvents(
            None, QEvent.Type.DeferredDelete
        )
        app.processEvents()


from unittest.mock import patch


def test_bleach_worker_failure(tmp_path):
    """A failed bleach releases the acquisition thread and restores controls."""

    app = QApplication.instance() or QApplication([])

    filename = tmp_path / "failed_bleach_test.h5"
    create_experiment_file(filename)

    window = MSPMainWindow()

    try:
        window.load_experiment(filename)

        window.hardware_initialized = True
        window.controller.bleach_controller = Mock()
        window.update_acquisition_controls()

        # Simulate a hardware acquisition failure.
        window.controller.acquire_bleach = Mock(
            side_effect=RuntimeError("Simulated DAQ failure")
        )

        loop = QEventLoop()
        timeout = QTimer()
        timeout.setSingleShot(True)
        timeout.setInterval(5000)
        timeout.timeout.connect(loop.quit)

        # Prevent the error dialog from blocking the test.
        with patch(
            "msp_control.gui.app.QMessageBox.critical"
        ) as error_dialog:

            window.bleach_button.click()

            thread = window.acquisition_thread
            assert thread is not None

            assert not window.bleach_button.isEnabled()

            thread.finished.connect(loop.quit)

            timeout.start()
            if thread.isRunning():
                loop.exec()
            timeout.stop()

            app.processEvents()

            assert window.acquisition_thread is None
            assert window.bleach_button.isEnabled()
            assert window.acquire_baseline_button.isEnabled()

            # Verify that the error was reported.
            error_dialog.assert_called_once()

            assert "Simulated DAQ failure" in (
                error_dialog.call_args.args[2]
            )

            # Verify the failure was logged.
            assert "Bleach failed" in (
                window.message_log.toPlainText()
            )

    finally:
        window.close()
        app.processEvents()
        QCoreApplication.sendPostedEvents(
            None, QEvent.Type.DeferredDelete
        )
        app.processEvents()
