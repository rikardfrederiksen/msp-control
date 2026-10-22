import sys
import time
from types import SimpleNamespace
from datetime import datetime

import numpy as np

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from msp_control.gui.app import MSPMainWindow


class FakeController:
    """Simulate baseline acquisition without hardware or disk writes."""

    def __init__(self, experiment):
        self.experiment = experiment

    def acquire_baseline(self, config, polarization):
        time.sleep(2)

        wavelength = config.wavelengths

        baseline = SimpleNamespace(
            baseline_index=self.experiment.next_baseline_index,
            polarization=polarization,
            config=config,
            wavelength=wavelength,
            baseline_corrected=np.ones_like(wavelength),
        )

        self.experiment.add_baseline(baseline)

        return baseline

    def acquire_scan(self, config, polarization, baseline_index):
        """Simulate specimen acquisition without hardware or disk writes."""
        time.sleep(5)
        #raise RuntimeError("Simulated monochromator communication failure")

        baseline = self.experiment.get_baseline(baseline_index)

        if baseline.polarization is not polarization:
            raise ValueError("Baseline polarization does not match")

        if not baseline.config.is_compatible_with(config):
            raise ValueError("Incompatible scan configuration")

        wavelength = config.wavelengths

        # Simulate a Gaussian absorption spectrum.
        optical_density = 0.5 * np.exp(
            -0.5 * ((wavelength - 500) / 40) ** 2
        )

        scan = SimpleNamespace(
            scan_index=self.experiment.next_scan_index,
            baseline_index=baseline_index,
            polarization=polarization,
            config=config,
            wavelength=wavelength,
            optical_density=optical_density,
            timestamp=window.controller.clock.now(),
            metadata={},
        )

        self.experiment.add_scan(scan)

        return scan

app = QApplication(sys.argv)
window = MSPMainWindow()

# Open the existing experiment.
window.load_experiment("data/msp_hardware_test_002.h5")

# Replace only the acquisition methods.
fake = FakeController(window.controller.experiment)
window.controller.acquire_baseline = fake.acquire_baseline
window.controller.acquire_scan = fake.acquire_scan

# Simulate initialized hardware.
window.hardware_initialized = True
window.update_acquisition_controls()

stage = "baseline"


def check_completion():
    global stage

    if window.acquisition_thread is not None:
        QTimer.singleShot(100, check_completion)
        return

    if stage == "baseline":
        print("Baseline acquisition completed")
        print("Baseline count:", window.baseline_list.count())
        print("Selected baseline:", window.active_baseline_index)

        stage = "scan"

        # Check whether the GUI permits specimen acquisition.
        window.update_acquisition_controls()
        print("Acquire Scan enabled:", window.acquire_scan_button.isEnabled())

        if not window.acquire_scan_button.isEnabled():
            window.close()
            app.quit()
            return

        window.on_acquire_scan()
        QTimer.singleShot(100, check_completion)

    elif stage == "scan":
        print("Specimen acquisition finished")
        print("Scan count:", window.scan_list.count())
        print("OD plot curves:", len(window.spectrum_plot.listDataItems()))
        print("Acquisition thread:", window.acquisition_thread)
        print("Acquire Baseline enabled:", window.acquire_baseline_button.isEnabled())
        print("Acquire Scan enabled:", window.acquire_scan_button.isEnabled())
        print("Start wavelength enabled:", window.start_nm_spin.isEnabled())
        print("Load Protocol enabled:", window.load_protocol_action.isEnabled())

        stage = "done"


# Start the baseline acquisition after the event loop begins.
QTimer.singleShot(100, window.on_acquire_baseline)
QTimer.singleShot(200, check_completion)

window.show()
sys.exit(app.exec())
