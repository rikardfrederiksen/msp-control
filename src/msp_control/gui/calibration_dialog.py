"""GUI for LED light calibration."""

from datetime import datetime
from pathlib import Path

import numpy as np
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QFormLayout,
    QLabel,
    QPushButton,
    QDoubleSpinBox,
    QComboBox,
    QSpinBox,
    QMessageBox,
)
import pyqtgraph as pg
from PySide6.QtCore import QObject, QThread, Signal, Slot

from msp_control.calibration.settings import (
    calibration_directory,
    save_default_calibration,
)
from msp_control.hardware.nidaq import NIDaq, NIDaqConfig
from msp_control.calibration.acquisition import CalibrationController
from msp_control.gui.spot_size_widget import SpotSizeWidget
from msp_control.calibration.waveform import (
    CalibrationWaveformConfig,
    BleachWaveformConfig,
)
from msp_control.calibration.storage import write_calibration
from msp_control.calibration.workflow import LightCalibrationConfig, LightCalibrationWorkflow


class CalibrationAcquisitionWorker(QObject):
    """Run LED calibration acquisition outside the GUI thread."""

    finished = Signal(object)
    failed = Signal(str)
    progress = Signal(str, int, int)

    def __init__(
        self,
        workflow: LightCalibrationWorkflow,
        config: LightCalibrationConfig,
        calibration_id: str,
        created_at: datetime,
    ):
        super().__init__()

        self.workflow = workflow
        self.config = config
        self.calibration_id = calibration_id
        self.created_at = created_at

    @Slot()
    def run(self):
        try:
            calibration = self.workflow.acquire(
                config=self.config,
                calibration_id=self.calibration_id,
                created_at=self.created_at,
                progress_callback=self.progress.emit,
            )
        except Exception as exc:
            self.failed.emit(str(exc))
        else:
            self.finished.emit(calibration)


class CalibrationDialog(QDialog):
    """Configure, acquire, and inspect LED calibrations."""

    def __init__(self, parent=None):
        super().__init__(parent)

        self.calibration = None
        self.acquisition_thread = None
        self.acquisition_worker = None
        self.calibration_saved = False
        self.setWindowTitle("Light Calibration")
        self.resize(1000, 1200)

        layout = QVBoxLayout(self)

        # -----------------------------------------------
        # Calibration configuration
        # -----------------------------------------------

        configuration_group = QGroupBox("Calibration Configuration")
        configuration_layout = QHBoxLayout(configuration_group)

        analog_group = QGroupBox("Analog LED")
        analog_form = QFormLayout(analog_group)

        self.analog_wavelength = QDoubleSpinBox()
        self.analog_wavelength.setRange(200, 1000)
        self.analog_wavelength.setValue(617)
        self.analog_wavelength.setSuffix(" nm")
        analog_form.addRow("Wavelength:", self.analog_wavelength)

        self.analog_gain = QComboBox()
        self.analog_gain.addItems(
            ["nA 1", "nA 2", "uA 1", "uA 2", "uA 3", "mA 1"]
        )
        self.analog_gain.setCurrentText("uA 1")
        analog_form.addRow("Photodiode range:", self.analog_gain)

        self.analog_repeats = QSpinBox()
        self.analog_repeats.setRange(1, 100)
        self.analog_repeats.setValue(5)
        analog_form.addRow("Repetitions:", self.analog_repeats)
        
        digital_group = QGroupBox("Digital LED")
        digital_form = QFormLayout(digital_group)

        self.digital_wavelength = QDoubleSpinBox()
        self.digital_wavelength.setRange(200, 1000)
        self.digital_wavelength.setValue(505)
        self.digital_wavelength.setSuffix(" nm")
        digital_form.addRow("Wavelength:", self.digital_wavelength)

        self.digital_gain = QComboBox()
        self.digital_gain.addItems(
            ["nA 1", "nA 2", "uA 1", "uA 2", "uA 3", "mA 1"]
        )
        self.digital_gain.setCurrentText("uA 1")
        digital_form.addRow("Photodiode range:", self.digital_gain)

        self.digital_repeats = QSpinBox()
        self.digital_repeats.setRange(1, 100)
        self.digital_repeats.setValue(5)
        digital_form.addRow("Repetitions:", self.digital_repeats)

        configuration_layout.addWidget(analog_group)
        configuration_layout.addWidget(digital_group)

        layout.addWidget(configuration_group)

        # -----------------------------------------------
        # Area settings
        # -----------------------------------------------

        area_group = QGroupBox("Illuminated Area")
        area_layout = QFormLayout(area_group)

        self.spot_size = SpotSizeWidget()
        area_layout.addRow("Aperture size:", self.spot_size)

        layout.addWidget(area_group)

        # -----------------------------------------------
        # Acquisition settings
        # -----------------------------------------------

        timing_group = QGroupBox("Acquisition Settings")
        timing_form = QFormLayout(timing_group)

        self.ramp_duration = QDoubleSpinBox()
        self.ramp_duration.setRange(0.5, 120.0)
        self.ramp_duration.setDecimals(1)
        self.ramp_duration.setValue(10.0)
        self.ramp_duration.setSuffix(" s")
        timing_form.addRow("Analog ramp duration (each direction):", self.ramp_duration)

        self.max_voltage = QDoubleSpinBox()
        self.max_voltage.setRange(0.1, 10.0)
        self.max_voltage.setDecimals(1)
        self.max_voltage.setValue(10.0)
        self.max_voltage.setSuffix(" V")
        timing_form.addRow("Maximum analog voltage:", self.max_voltage)

        self.digital_duration = QDoubleSpinBox()
        self.digital_duration.setRange(0.1, 30.0)
        self.digital_duration.setDecimals(1)
        self.digital_duration.setValue(1.0)
        self.digital_duration.setSuffix(" s")
        timing_form.addRow("Digital LED ON duration:", self.digital_duration)

        self.sample_rate = QSpinBox()
        self.sample_rate.setRange(100, 10000)
        self.sample_rate.setSingleStep(100)
        self.sample_rate.setValue(1000)
        self.sample_rate.setSuffix(" Hz")
        timing_form.addRow("Sampling rate:", self.sample_rate)

        layout.addWidget(timing_group)

        # -----------------------------------------------
        # Acquisition
        # -----------------------------------------------

        acquisition_layout = QHBoxLayout()

        self.acquire_button = QPushButton("Acquire Calibration")
        self.acquire_button.clicked.connect(self.on_acquire_calibration)

        self.status_label = QLabel("Ready")

        acquisition_layout.addWidget(self.acquire_button)
        acquisition_layout.addWidget(self.status_label)
        acquisition_layout.addStretch()

        layout.addLayout(acquisition_layout)

        # -----------------------------------------------
        # Calibration plots
        # -----------------------------------------------

        plot_controls = QHBoxLayout()

        plot_controls.addWidget(QLabel("Display:"))

        self.plot_selector = QComboBox()
        self.plot_selector.addItems([
            "Waveform vs time",
            "Photodiode response",
            "LED monitor response",
            "Photon flux density",
        ])

        plot_controls.addWidget(self.plot_selector)
        plot_controls.addStretch()

        layout.addLayout(plot_controls)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)

        layout.addWidget(self.plot_widget, stretch=1)

        self.plot_selector.currentIndexChanged.connect(
            self.update_calibration_plot
        )

        self.update_calibration_plot()

        # -----------------------------------------------
        # Acceptance
        # -----------------------------------------------

        button_layout = QHBoxLayout()
        button_layout.addStretch()

        self.save_button = QPushButton("Save Calibration")
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self.on_save_calibration)

        self.cancel_button = QPushButton("Close")
        self.cancel_button.clicked.connect(self.reject)

        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.cancel_button)

        layout.addLayout(button_layout)

    @Slot(str)
    def on_acquisition_failed(self, error_message):
        """Handle an acquisition failure."""

        self.calibration = None
        self.status_label.setText(f"Acquisition failed: {error_message}")

    @Slot(object)
    def on_acquisition_finished(self, calibration):
        """Handle a successfully completed calibration."""

        self.calibration = calibration
        self.status_label.setText("Calibration acquired successfully")
        self.save_button.setEnabled(True)

        self.update_calibration_plot()

    @Slot()
    def on_save_calibration(self):
        """Save the acquired calibration to an HDF5 file."""

        if self.calibration is None:
            return

        calibration_dir = calibration_directory()
        calibration_dir.mkdir(parents=True, exist_ok=True)

        filename = calibration_dir / f"{self.calibration.calibration_id}.h5"

        try:
            write_calibration(filename, self.calibration)
            save_default_calibration(filename.name)
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Calibration Save Failed",
                str(exc),
            )
            return

        self.status_label.setText(f"Saved: {filename.name}")
        self.save_button.setEnabled(False)
        self.calibration_saved = True

    @Slot()
    def on_acquisition_thread_finished(self):
        """Release references after the worker thread stops."""

        self.acquisition_thread = None
        self.acquisition_worker = None
        self.acquire_button.setEnabled(True)

    @Slot(str, int, int)
    def on_acquisition_progress(self, led, current, total):
        """Display the acquisition currently in progress."""

        led_name = "Analog" if led == "analog" else "Digital"

        self.status_label.setText(
            f"{led_name} LED: acquiring {current}/{total}"
        )

    def on_acquire_calibration(self):
        """Start a complete LED calibration."""

        config = self.get_calibration_config()

        created_at = datetime.now()
        calibration_id = created_at.strftime("led_cal_%Y%m%d_%H%M%S")

        daq = NIDaq(NIDaqConfig())
        controller = CalibrationController(daq)
        workflow = LightCalibrationWorkflow(controller)

        self.start_acquisition(
            workflow=workflow,
            config=config,
            calibration_id=calibration_id,
            created_at=created_at,
        )

    def get_calibration_config(self) -> LightCalibrationConfig:
        """Build a calibration configuration from the GUI controls."""

        sample_rate = float(self.sample_rate.value())
        area_um2 = self.spot_size.area_um2

        analog_waveform = CalibrationWaveformConfig(
            ramp_up_s=self.ramp_duration.value(),
            ramp_down_s=self.ramp_duration.value(),
            max_voltage=self.max_voltage.value(),
            sample_rate_hz=sample_rate,
        )

        digital_waveform = BleachWaveformConfig(
            duration_s=self.digital_duration.value(),
            sample_rate_hz=sample_rate,
        )

        return LightCalibrationConfig(
            analog_waveform=analog_waveform,
            digital_waveform=digital_waveform,
            analog_wavelength_nm=self.analog_wavelength.value(),
            digital_wavelength_nm=self.digital_wavelength.value(),
            analog_amplifier_range=self.analog_gain.currentText(),
            digital_amplifier_range=self.digital_gain.currentText(),
            analog_area_um2=area_um2,
            digital_area_um2=area_um2,
            analog_repeats=self.analog_repeats.value(),
            digital_repeats=self.digital_repeats.value(),
        )

    def start_acquisition(self, workflow, config, calibration_id, created_at):
        """Start calibration acquisition in a worker thread."""

        if self.acquisition_thread is not None:
            raise RuntimeError("Calibration acquisition already running")

        self.calibration = None
        self.acquire_button.setEnabled(False)
        self.save_button.setEnabled(False)
        self.status_label.setText("Acquiring calibration...")

        self.acquisition_thread = QThread(self)

        self.acquisition_worker = CalibrationAcquisitionWorker(
            workflow=workflow,
            config=config,
            calibration_id=calibration_id,
            created_at=created_at,
        )

        self.acquisition_worker.moveToThread(self.acquisition_thread)

        self.acquisition_thread.started.connect(
            self.acquisition_worker.run
        )

        self.acquisition_worker.finished.connect(
            self.on_acquisition_finished
        )
        self.acquisition_worker.failed.connect(
            self.on_acquisition_failed
        )

        self.acquisition_worker.progress.connect(
            self.on_acquisition_progress
        )
        
        self.acquisition_worker.finished.connect(
            self.acquisition_thread.quit
        )
        self.acquisition_worker.failed.connect(
            self.acquisition_thread.quit
        )

        self.acquisition_worker.finished.connect(
            self.acquisition_worker.deleteLater
        )
        self.acquisition_worker.failed.connect(
            self.acquisition_worker.deleteLater
        )

        self.acquisition_thread.finished.connect(
            self.on_acquisition_thread_finished
        )
        self.acquisition_thread.finished.connect(
            self.acquisition_thread.deleteLater
        )


        self.acquisition_thread.start()

    def update_calibration_plot(self):
        """Display the selected analog LED calibration measurement."""

        selection = self.plot_selector.currentText()

        self.plot_widget.clear()

        # Remove the previous legend before creating a new one.
        if self.plot_widget.plotItem.legend is not None:
            self.plot_widget.plotItem.legend.scene().removeItem(
                self.plot_widget.plotItem.legend
            )
            self.plot_widget.plotItem.legend = None

        if selection in ("Photodiode response", "LED monitor response"):
            self.plot_widget.addLegend()

        if selection == "Waveform vs time":
            self.plot_widget.setLabel("bottom", "Time", units="s")
            self.plot_widget.setLabel("left", "Command voltage", units="V")

        elif selection == "Photodiode response":
            self.plot_widget.setLabel("bottom", "Command voltage", units="V")
            self.plot_widget.setLabel("left", "Photodiode voltage", units="V")

        elif selection == "LED monitor response":
            self.plot_widget.setLabel("bottom", "Command voltage", units="V")
            self.plot_widget.setLabel("left", "LED monitor voltage", units="V")

        elif selection == "Photon flux density":
            self.plot_widget.setLabel("bottom", "Command voltage", units="V")
            self.plot_widget.setLabel(
                "left", "Photon flux density", units="photons/µm²/s"
            )

        if self.calibration is None:
            return

        analog = self.calibration.analog_led

        if selection == "Waveform vs time":
            x = analog.time_s
            y = analog.command_voltage

        elif selection == "Photodiode response":
            x = analog.command_voltage
            y = analog.photodiode_voltage

        elif selection == "LED monitor response":
            x = analog.command_voltage
            y = analog.monitor_voltage

        elif selection == "Photon flux density":
            x = analog.calibration_voltage
            y = analog.photon_flux_density

        else:
            return

        if selection in ("Photodiode response", "LED monitor response"):

            # Find the first and last occurrence of maximum command voltage.
            peak_indices = np.flatnonzero(
                np.isclose(x, np.max(x))
            )

            first_peak = peak_indices[0]
            last_peak = peak_indices[-1]

            # Ascending ramp
            self.plot_widget.plot(
                x[:first_peak + 1],
                y[:first_peak + 1],
                pen=pg.mkPen("#0072B2", width=2),
                name="Ascending",
            )

            # Descending ramp
            self.plot_widget.plot(
                x[last_peak:],
                y[last_peak:],
                pen=pg.mkPen("#D55E00", width=2),
                name="Descending",
            )

        else:
            self.plot_widget.plot(
                x,
                y,
                pen=pg.mkPen("#56B4E9", width=2),
            )
        
    def reject(self):
        """Prevent closing the dialog during acquisition."""

        if self.acquisition_thread is not None:
            self.status_label.setText(
                "Please wait for calibration acquisition to finish"
            )
            return

        super().reject()

    def closeEvent(self, event):
        """Prevent closing the dialog during acquisition."""

        if self.acquisition_thread is not None:
            self.status_label.setText(
                "Please wait for calibration acquisition to finish"
            )
            event.ignore()
            return

        super().closeEvent(event)
