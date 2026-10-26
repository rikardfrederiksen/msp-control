from unittest.mock import Mock, patch

from PySide6.QtWidgets import QApplication

from msp_control.gui.app import MSPMainWindow


def test_close_disconnects_hardware():
    """Closing the main window disconnects persistent hardware."""

    app = QApplication.instance() or QApplication([])

    hardware_manager = Mock()
    window = MSPMainWindow(hardware_manager=hardware_manager)

    window.close()

    hardware_manager.disconnect.assert_called_once()
    assert not window.isVisible()


def test_close_during_acquisition_does_not_disconnect():
    """An active acquisition prevents window and hardware shutdown."""

    app = QApplication.instance() or QApplication([])

    hardware_manager = Mock()
    window = MSPMainWindow(hardware_manager=hardware_manager)

    # Simulate an active acquisition.
    window.acquisition_thread = Mock()

    try:
        # Prevent the warning dialog from blocking the test.
        with patch(
            "msp_control.gui.app.QMessageBox.warning"
        ) as warning:
            result = window.close()

        assert result is False
        warning.assert_called_once()
        hardware_manager.disconnect.assert_not_called()

    finally:
        # Restore normal shutdown behavior.
        window.acquisition_thread = None
        window.close()


def test_close_handles_hardware_disconnect_error():
    """The window closes even if hardware disconnection fails."""

    app = QApplication.instance() or QApplication([])

    hardware_manager = Mock()
    hardware_manager.disconnect.side_effect = RuntimeError(
        "Serial connection failed"
    )

    window = MSPMainWindow(hardware_manager=hardware_manager)

    result = window.close()

    assert result is True
    hardware_manager.disconnect.assert_called_once()
    assert not window.isVisible()

    # The failure should be recorded in the GUI log.
    assert "Serial connection failed" in window.message_log.toPlainText()
