import sys
from types import SimpleNamespace
from unittest.mock import Mock

from PySide6.QtWidgets import QApplication

from msp_control.gui.app import MSPMainWindow


app = QApplication(sys.argv)
window = MSPMainWindow()

# Simulate an initialized Optoscan.
fake_optoscan = SimpleNamespace(
    is_open=True,
    open_slits=Mock(),
    close_slits=Mock(),
)

window.hardware_manager = SimpleNamespace(
    optoscan=fake_optoscan,
)

window.hardware_initialized = True
window.update_acquisition_controls()

# Use known manual settings.
window.manual_wavelength_spin.setValue(700)
window.manual_input_slit_spin.setValue(2)
window.manual_output_slit_spin.setValue(4)

# Exercise the GUI buttons.
window.open_slits_button.click()
window.close_slits_button.click()

# Verify the calls.
fake_optoscan.open_slits.assert_called_once_with(
    wavelength_nm=700,
    input_slit_nm=2,
    output_slit_nm=4,
)

fake_optoscan.close_slits.assert_called_once_with()

print("Open slits command: OK")
print("Close slits command: OK")

window.close()
app.quit()
