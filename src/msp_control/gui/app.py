import sys
from pathlib import Path
from datetime import datetime
import json

import numpy as np

from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QSplitter,
    QListWidget,
    QListWidgetItem,
    QWidget,
    QVBoxLayout,
    QLabel,
    QFrame,
    QTreeWidget,
    QTreeWidgetItem,
    QMessageBox,
    QFileDialog,
    QGroupBox,
    QPlainTextEdit,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QAbstractItemView,
    QLineEdit,
    QPushButton,
    QHBoxLayout,
    QScrollArea,
    QFormLayout,
    QDoubleSpinBox,
    QSpinBox,
    QRadioButton,
    QComboBox,
    QCheckBox,
    QSizePolicy,
)

from PySide6.QtGui import QFont
from PySide6.QtCore import Qt, QTimer, QObject, QThread, Signal, Slot
import pyqtgraph as pg

from msp_control.data.model import Polarization
from msp_control.protocol import AcquisitionProtocol
from msp_control.config import ScanConfig
from msp_control.acquisition import AcquisitionController
from msp_control.hardware.nidaq import NIDaq, NIDaqConfig
from msp_control.metadata import MetadataCollector
from msp_control.hardware.stage import StagePositionMonitor, StageConfig
from msp_control.hardware.temperature import TemperatureMonitor, TemperatureConfig
from msp_control.hardware.manager import HardwareManager
from msp_control.hardware.optoscan import OptoscanSerial
from msp_control.controller import MSPController
from msp_control.metadata_registry import load_metadata_registry
from msp_control.gui.experiment_metadata_dialog import (
    ExperimentMetadataDialog,
)
from msp_control.gui.calibration_dialog import CalibrationDialog
from msp_control.hardware.field_stop import (
    FieldStopConfig,
    FieldStopMonitor,
    KinesisFieldStopDevice,
)
from msp_control.calibration.settings import (
    find_calibration_files,
    calibration_settings_path,
    load_default_calibration,
    save_default_calibration,
)
from msp_control.calibration.storage import read_calibration
from msp_control.calibration.bleach import estimate_bleach
from msp_control.bleach import BleachController
from msp_control.calibration.waveform import BleachWaveformConfig


SCAN_COLORS = [
    "#56B4E9",  # Sky blue
    "#E69F00",  # Orange
    "#009E73",  # Bluish green
    "#F0E442",  # Yellow
    "#CC79A7",  # Reddish purple
    "#D55E00",  # Vermilion
    "#0072B2",  # Blue
    "#FFFFFF",  # White
    "#999999",  # Gray
    "#AA77FF",  # Violet
]


def load_cell_types() -> list[str]:
    """Load the available cell types from the project configuration."""

    config_path = (
        Path(__file__).resolve().parents[3]
        / "config"
        / "cell_types.json"
    )

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    return config["cell_types"]


class BaselineAcquisitionWorker(QObject):
    """Run baseline acquisition outside the GUI thread."""

    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, controller, config, polarization, metadata=None):
        super().__init__()
        self.controller = controller
        self.config = config
        self.polarization = polarization
        self.metadata = metadata

    @Slot()
    def run(self):
        try:
           baseline = self.controller.acquire_baseline(
               config=self.config,
               polarization=self.polarization,
               metadata=self.metadata,
           )
        except Exception as exc:
            self.failed.emit(str(exc))
        else:
            self.finished.emit(baseline)


class ScanAcquisitionWorker(QObject):
    """Run specimen acquisition outside the GUI thread."""

    finished = Signal(object)
    failed = Signal(str)

    def __init__(
        self, controller, config, polarization, baseline_index, metadata=None
    ):
        super().__init__()
        self.controller = controller
        self.config = config
        self.polarization = polarization
        self.baseline_index = baseline_index
        self.metadata = metadata

    @Slot()
    def run(self):
        try:
            scan = self.controller.acquire_scan(
                config=self.config,
                polarization=self.polarization,
                baseline_index=self.baseline_index,
                metadata=self.metadata,
            )
        except Exception as exc:
            self.failed.emit(str(exc))
        else:
            self.finished.emit(scan)


class BleachAcquisitionWorker(QObject):
    """Run bleaching outside the GUI thread."""

    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, controller, led_channel, config,
                 nd_filter, command_voltage):
        super().__init__()
        self.controller = controller
        self.led_channel = led_channel
        self.config = config
        self.nd_filter = nd_filter
        self.command_voltage = command_voltage

    @Slot()
    def run(self):
        try:
            bleach = self.controller.acquire_bleach(
                led_channel=self.led_channel,
                config=self.config,
                nd_filter=self.nd_filter,
                command_voltage=self.command_voltage,
            )
        except Exception as exc:
            self.failed.emit(str(exc))
        else:
            self.finished.emit(bleach)


class MSPMainWindow(QMainWindow):
    """Main window for the MSP control application."""

    def __init__(self, hardware_manager=None):
        super().__init__()

        self.hardware_manager = hardware_manager
        self.hardware_initialized = False

        self.acquisition_thread = None
        self.acquisition_worker = None

        self.setWindowTitle("MSP Control")
        
        screen = QApplication.primaryScreen()
        if screen is not None:
            available = screen.availableGeometry()
            self.resize(
                min(1600, int(available.width() * 0.95)),
                min(900, int(available.height() * 0.95)),
            )
        else:
            self.resize(1600, 900)

        self.controller = MSPController()
        self.experiment = None

        self.message_log = QPlainTextEdit()
        self.message_log.setReadOnly(True)
        self.message_log.setMaximumBlockCount(1000)

        # Load the controlled metadata vocabulary
        self.metadata_registry = load_metadata_registry(
            "config/experiment_metadata.json"
        )

        # File menu
        file_menu = self.menuBar().addMenu("&File")

        new_experiment_action = file_menu.addAction(
            "&New Experiment..."
        )
        new_experiment_action.triggered.connect(
            self.on_new_experiment
        )

        open_experiment_action = file_menu.addAction(
            "&Open Experiment..."
        )
        open_experiment_action.triggered.connect(
            self.on_open_experiment
        )

        close_experiment_action = file_menu.addAction(
            "&Close Experiment"
        )
        close_experiment_action.triggered.connect(
            self.on_close_experiment
        )

        file_menu.addSeparator()

        self.load_protocol_action = file_menu.addAction(
            "&Load Protocol..."
        )
        
        self.load_protocol_action.triggered.connect(
            self.on_load_protocol
        )

        save_protocol_action = file_menu.addAction(
            "&Save Protocol As..."
        )
        save_protocol_action.triggered.connect(
            self.on_save_protocol
        )

        # Edit menu
        edit_menu = self.menuBar().addMenu("&Edit")

        edit_metadata_action = edit_menu.addAction(
            "Experiment &Metadata..."
        )
        edit_metadata_action.triggered.connect(
            self.on_edit_experiment_metadata
        )

        # Hardware menu
        hardware_menu = self.menuBar().addMenu("&Hardware")

        initialize_hardware_action = hardware_menu.addAction(
            "&Initialize Hardware"
        )
        initialize_hardware_action.triggered.connect(
            self.on_initialize_hardware
        )
        
        # --------------------------------------------------
        # Main OD spectrum plot
        # --------------------------------------------------

        self.spectrum_plot = pg.PlotWidget()
        self.spectrum_plot.getAxis("left").enableAutoSIPrefix(False)
        self.spectrum_legend = self.spectrum_plot.addLegend(
            offset=(-10, 10)
        )
        self.spectrum_legend.setLabelTextSize("16pt")
        self.spectrum_legend.setLabelTextColor("#B0B0B0")
        self.spectrum_legend.hide()
        self.spectrum_plot.setBackground("#171B22")

        # Axis tick labels
        tick_font = QFont()
        tick_font.setPointSize(16)
        tick_font.setBold(True)

        # Axis titles
        self.axis_label_style = {
            "color": "#B0B0B0",
            "font-size": "18pt",
            "font-weight": "bold",
        }

        for axis_name in ("left", "bottom"):
            axis = self.spectrum_plot.getAxis(axis_name)

            # Thicker axis lines and tick marks
            axis.setPen(pg.mkPen(self.axis_label_style["color"], width=2.5))

            # Tick label appearance
            axis.setTextPen(pg.mkPen(self.axis_label_style["color"]))
            axis.setTickFont(tick_font)

        # Axis titles
        self.spectrum_plot.setLabel(
            "bottom", "Wavelength", units="nm", **self.axis_label_style
        )
        self.spectrum_plot.setLabel(
            "left", "Optical density", **self.axis_label_style
        )

        self.spectrum_plot.showGrid(x=True, y=True, alpha=0.2)

        # --------------------------------------------------
        # Specimen scan list
        # --------------------------------------------------

        self.scan_list = QListWidget()
        self.scan_list.setMinimumWidth(120)
        self.scan_list.setMinimumHeight(200)

        self.scan_curves = {}

        self.scan_list.itemChanged.connect(
            self.on_scan_visibility_changed
        )

        self.scan_list.currentItemChanged.connect(
            self.on_scan_selection_changed
        )

        # --------------------------------------------------
        # Baseline selection list
        # --------------------------------------------------

        self.baseline_list = QListWidget()
        self.baseline_list.setMaximumHeight(450)

        self.baseline_list.currentItemChanged.connect(
            self.on_active_baseline_changed
        )

        self.active_baseline_index = None

        # --------------------------------------------------
        # Column 1: Baselines and experiment metadata
        # --------------------------------------------------

        baseline_panel = QWidget()
        baseline_layout = QVBoxLayout(baseline_panel)

        baseline_layout.addWidget(QLabel("Baseline spectrum"))

        self.baseline_plot = pg.PlotWidget()
        self.baseline_plot.setBackground("#171B22")
        self.baseline_plot.setMinimumHeight(150)
        self.baseline_plot.setMaximumHeight(220)

        # Baseline plot axis styling
        baseline_tick_font = QFont()
        baseline_tick_font.setPointSize(11)
        baseline_tick_font.setBold(True)

        baseline_label_style = {
            "color": "#B0B0B0",
            "font-size": "12pt",
            "font-weight": "bold",
        }

        for axis_name in ("left", "bottom"):
            axis = self.baseline_plot.getAxis(axis_name)

            axis.setPen(pg.mkPen("#B0B0B0", width=2))
            axis.setTextPen(pg.mkPen("#B0B0B0"))
            axis.setTickFont(baseline_tick_font)

        self.baseline_plot.setLabel(
            "bottom", "Wavelength", units="nm", **baseline_label_style,
        )
        self.baseline_plot.setLabel(
            "left", "Signal", units="V", **baseline_label_style,
        )

        self.baseline_plot.showGrid(x=True, y=True, alpha=0.2)

        baseline_layout.addWidget(self.baseline_plot)

        baseline_layout.addWidget(QLabel("Baselines"))
        baseline_layout.addWidget(self.baseline_list)

        baseline_layout.addWidget(QLabel("Experiment metadata"))

        self.experiment_metadata_label = QLabel("No experiment metadata")
        self.experiment_metadata_label.setWordWrap(True)
        self.experiment_metadata_label.setAlignment(
            Qt.AlignmentFlag.AlignTop
        )

        baseline_layout.addWidget(self.experiment_metadata_label)
        baseline_layout.addStretch()

        # --------------------------------------------------
        # Column 2: Specimen scans and scan metadata
        # --------------------------------------------------

        scan_panel = QWidget()
        scan_layout = QVBoxLayout(scan_panel)

        scan_layout.addWidget(QLabel("Specimen scans"))
        scan_layout.addWidget(self.scan_list)

        scan_layout.addWidget(QLabel("Scan metadata"))

        self.scan_metadata_view = QTreeWidget()
        self.scan_metadata_view.setColumnCount(2)
        self.scan_metadata_view.setHeaderLabels(["Parameter", "Value"])
        self.scan_metadata_view.setRootIsDecorated(False)
        self.scan_metadata_view.setAlternatingRowColors(True)

        scan_layout.addWidget(self.scan_metadata_view)

        # --------------------------------------------------
        # Column 3: Main plot and plotting controls
        # --------------------------------------------------

        plot_panel = QWidget()
        plot_layout = QVBoxLayout(plot_panel)

        plot_layout.addWidget(
            self.spectrum_plot,
            stretch=1
        )

        plot_controls_layout = QHBoxLayout()

        plot_controls_layout.addWidget(QLabel("Display:"))

        self.plot_mode_combo = QComboBox()
        self.plot_mode_combo.addItem("Optical density", userData="od")
        self.plot_mode_combo.addItem("Scan", userData="scan")
        self.plot_mode_combo.addItem("Baseline", userData="baseline")

        plot_controls_layout.addWidget(self.plot_mode_combo)
        self.zero_correction_checkbox = QCheckBox("Zero correction")
        plot_controls_layout.addWidget(self.zero_correction_checkbox)

        self.zero_start_spin = QDoubleSpinBox()
        self.zero_start_spin.setRange(200, 1000)
        self.zero_start_spin.setValue(700)
        self.zero_start_spin.setSuffix(" nm")
        plot_controls_layout.addWidget(self.zero_start_spin)

        self.zero_end_spin = QDoubleSpinBox()
        self.zero_end_spin.setRange(200, 1000)
        self.zero_end_spin.setValue(720)
        self.zero_end_spin.setSuffix(" nm")
        plot_controls_layout.addWidget(self.zero_end_spin)

        self.show_legend_checkbox = QCheckBox("Show legend")
        plot_controls_layout.addWidget(self.show_legend_checkbox)

        self.plot_home_button = QPushButton("Home")
        self.plot_home_button.setToolTip(
            "Automatically scale both axes to the displayed spectra"
        )
        plot_controls_layout.addWidget(self.plot_home_button)

        plot_controls_layout.addWidget(QLabel("Mouse:"))

        self.zoom_mode_combo = QComboBox()
        self.zoom_mode_combo.addItem("X + Y", userData="xy")
        self.zoom_mode_combo.addItem("X only", userData="x")
        self.zoom_mode_combo.addItem("Y only", userData="y")

        self.zoom_mode_combo.setToolTip(
            "Select which axes respond to mouse-wheel zoom and dragging"
        )

        plot_controls_layout.addWidget(self.zoom_mode_combo)
        
        plot_controls_layout.addStretch()

        self.zero_start_line = pg.InfiniteLine(
            pos=self.zero_start_spin.value(),
            angle=90,
            movable=True,
            pen=pg.mkPen("#D55E00", width=1.5, style=Qt.PenStyle.DashLine),
            hoverPen=pg.mkPen("#D55E00", width=2.5),
        )

        self.zero_end_line = pg.InfiniteLine(
            pos=self.zero_end_spin.value(),
            angle=90,
            movable=True,
            pen=pg.mkPen("#0072B2", width=1.5, style=Qt.PenStyle.DashLine),
            hoverPen=pg.mkPen("#0072B2", width=2.5),
        )

        self.spectrum_plot.addItem(self.zero_start_line, ignoreBounds=True)
        self.spectrum_plot.addItem(self.zero_end_line, ignoreBounds=True)

        self.zero_start_line.hide()
        self.zero_end_line.hide()

        plot_layout.addLayout(plot_controls_layout)

        self.plot_mode_combo.currentIndexChanged.connect(
            self.on_plot_mode_changed
        )

        self.zero_correction_checkbox.toggled.connect(
            lambda checked: self.refresh_spectrum_plot(auto_range=False)
        )
        self.zero_start_spin.valueChanged.connect(
            lambda value: self.refresh_spectrum_plot(auto_range=False)
        )

        self.zero_end_spin.valueChanged.connect(
            lambda value: self.refresh_spectrum_plot(auto_range=False)
        )

        self.zero_start_spin.valueChanged.connect(
            self.zero_start_line.setValue
        )
        self.zero_end_spin.valueChanged.connect(
            self.zero_end_line.setValue
        )

        self.zero_start_line.sigPositionChanged.connect(
            lambda: self.on_zero_cursor_moved(
                self.zero_start_line, self.zero_start_spin
            )
        )
        self.zero_end_line.sigPositionChanged.connect(
            lambda: self.on_zero_cursor_moved(
                self.zero_end_line, self.zero_end_spin
            )
        )
        self.zero_start_line.sigPositionChangeFinished.connect(
            lambda: self.refresh_spectrum_plot(auto_range=False)
        )

        self.zero_end_line.sigPositionChangeFinished.connect(
            lambda: self.refresh_spectrum_plot(auto_range=False)
        )
        
        self.plot_home_button.clicked.connect(
            self.spectrum_plot.autoRange
        )

        self.zoom_mode_combo.currentIndexChanged.connect(
            self.on_zoom_mode_changed
        )

        self.show_legend_checkbox.toggled.connect(
            lambda checked: self.refresh_spectrum_plot(auto_range=False)
        )
        
        # --------------------------------------------------
        # Column 4: MSP and bleaching controls
        # --------------------------------------------------

        # Scrollable MSP control panel
        hardware_panel = QScrollArea()
        hardware_panel.setWidgetResizable(True)
        hardware_panel.setFrameShape(QScrollArea.Shape.NoFrame)

        self.hardware_controls = QWidget()
        self.hardware_layout = QVBoxLayout(self.hardware_controls)

        # --------------------------------------------------
        # Scan configuration
        # --------------------------------------------------

        scan_config_group = QGroupBox("Scan Configuration")
        scan_config_layout = QFormLayout(scan_config_group)

        self.start_nm_spin = QDoubleSpinBox()
        self.start_nm_spin.setRange(300, 1000)
        self.start_nm_spin.setValue(360)
        self.start_nm_spin.setSuffix(" nm")

        self.end_nm_spin = QDoubleSpinBox()
        self.end_nm_spin.setRange(300, 1000)
        self.end_nm_spin.setValue(720)
        self.end_nm_spin.setSuffix(" nm")

        self.step_nm_spin = QDoubleSpinBox()
        self.step_nm_spin.setRange(0.1, 100)
        self.step_nm_spin.setValue(2)
        self.step_nm_spin.setSuffix(" nm")

        self.step_time_spin = QDoubleSpinBox()
        self.step_time_spin.setRange(1, 1000)
        self.step_time_spin.setValue(2)
        self.step_time_spin.setSuffix(" ms")

        self.input_slit_spin = QDoubleSpinBox()
        self.input_slit_spin.setRange(0.1, 20)
        self.input_slit_spin.setValue(4)
        self.input_slit_spin.setSuffix(" nm")

        self.output_slit_spin = QDoubleSpinBox()
        self.output_slit_spin.setRange(0.1, 20)
        self.output_slit_spin.setValue(4)
        self.output_slit_spin.setSuffix(" nm")

        self.dark_scans_spin = QSpinBox()
        self.dark_scans_spin.setRange(0, 100)
        self.dark_scans_spin.setValue(2)

        self.baseline_scans_spin = QSpinBox()
        self.baseline_scans_spin.setRange(1, 100)
        self.baseline_scans_spin.setValue(20)

        self.specimen_scans_spin = QSpinBox()
        self.specimen_scans_spin.setRange(1, 100)
        self.specimen_scans_spin.setValue(3)

        scan_config_layout.addRow("Start wavelength:", self.start_nm_spin)
        scan_config_layout.addRow("End wavelength:", self.end_nm_spin)
        scan_config_layout.addRow("Step size:", self.step_nm_spin)
        scan_config_layout.addRow("Step time:", self.step_time_spin)
        scan_config_layout.addRow("Input slit:", self.input_slit_spin)
        scan_config_layout.addRow("Output slit:", self.output_slit_spin)
        scan_config_layout.addRow("Dark scans:", self.dark_scans_spin)
        scan_config_layout.addRow("Baseline scans:", self.baseline_scans_spin)
        scan_config_layout.addRow("Specimen scans:", self.specimen_scans_spin)

        self.hardware_layout.addWidget(scan_config_group)

        # Field-stop dimensions used for subsequent acquisitions
        field_stop_group = QGroupBox("Field Stop Dimensions")
        field_stop_layout = QFormLayout(field_stop_group)

        self.field_stop_height_spin = QDoubleSpinBox()
        self.field_stop_height_spin.setRange(0.1, 1000.0)
        self.field_stop_height_spin.setDecimals(1)
        self.field_stop_height_spin.setSingleStep(0.5)
        self.field_stop_height_spin.setValue(6.0)
        self.field_stop_height_spin.setSuffix(" µm")

        self.field_stop_width_spin = QDoubleSpinBox()
        self.field_stop_width_spin.setRange(0.1, 1000.0)
        self.field_stop_width_spin.setDecimals(1)
        self.field_stop_width_spin.setSingleStep(0.5)
        self.field_stop_width_spin.setValue(3.0)
        self.field_stop_width_spin.setSuffix(" µm")

        field_stop_layout.addRow("Height:", self.field_stop_height_spin)
        field_stop_layout.addRow("Width:", self.field_stop_width_spin)

        self.hardware_layout.addWidget(field_stop_group)

        for widget in (
                self.start_nm_spin,
                self.end_nm_spin,
                self.step_nm_spin,
                self.step_time_spin,
                self.input_slit_spin,
                self.output_slit_spin,
        ):
            widget.valueChanged.connect(
                self.update_acquisition_controls
            )

        # --------------------------------------------------
        # Acquisition controls
        # --------------------------------------------------

        acquisition_group = QGroupBox("Acquisition")
        acquisition_layout = QVBoxLayout(acquisition_group)

        # Manual monochromator controls
        monochromator_group = QGroupBox("Monochromator")
        monochromator_layout = QVBoxLayout(monochromator_group)

        # Open / close buttons
        slit_button_layout = QHBoxLayout()

        self.open_slits_button = QPushButton("Open slits")
        self.close_slits_button = QPushButton("Close slits")

        self.open_slits_button.clicked.connect(self.on_open_slits)
        self.close_slits_button.clicked.connect(self.on_close_slits)

        slit_button_layout.addWidget(self.open_slits_button)
        slit_button_layout.addWidget(self.close_slits_button)

        monochromator_layout.addLayout(slit_button_layout)

        # Manual wavelength and slit widths
        manual_settings_layout = QHBoxLayout()

        self.manual_wavelength_spin = QDoubleSpinBox()
        self.manual_wavelength_spin.setRange(200, 1000)
        self.manual_wavelength_spin.setValue(700)
        self.manual_wavelength_spin.setSuffix(" nm")

        self.manual_input_slit_spin = QDoubleSpinBox()
        self.manual_input_slit_spin.setRange(0.1, 50)
        self.manual_input_slit_spin.setValue(4)
        self.manual_input_slit_spin.setSuffix(" nm")

        self.manual_output_slit_spin = QDoubleSpinBox()
        self.manual_output_slit_spin.setRange(0.1, 50)
        self.manual_output_slit_spin.setValue(4)
        self.manual_output_slit_spin.setSuffix(" nm")

        for label, spin in (
            ("Wavelength", self.manual_wavelength_spin),
            ("Input slit", self.manual_input_slit_spin),
            ("Output slit", self.manual_output_slit_spin),
        ):
            column = QVBoxLayout()
            column.addWidget(QLabel(label))
            column.addWidget(spin)
            manual_settings_layout.addLayout(column)

        monochromator_layout.addLayout(manual_settings_layout)

        self.hardware_layout.addWidget(monochromator_group)

        # Polarization selection
        polarization_layout = QHBoxLayout()

        self.transverse_radio = QRadioButton("Transverse (T)")
        self.longitudinal_radio = QRadioButton("Longitudinal (L)")

        self.transverse_radio.setChecked(True)

        polarization_layout.addWidget(self.transverse_radio)
        polarization_layout.addWidget(self.longitudinal_radio)

        acquisition_layout.addLayout(polarization_layout)

        self.transverse_radio.toggled.connect(
            self.update_acquisition_controls
        )

        # Acquisition buttons
        self.acquire_baseline_button = QPushButton("Acquire Baseline")
        self.acquire_scan_button = QPushButton("Acquire Scan")

        # Disabled until acquisition is connected
        self.acquire_baseline_button.setEnabled(False)
        self.acquire_scan_button.setEnabled(False)

        acquisition_layout.addWidget(self.acquire_baseline_button)
        acquisition_layout.addWidget(self.acquire_scan_button)

        self.hardware_layout.addWidget(acquisition_group)

        # Controls that must not change during acquisition.
        self.acquisition_settings = (
            self.start_nm_spin,
            self.end_nm_spin,
            self.step_nm_spin,
            self.step_time_spin,
            self.input_slit_spin,
            self.output_slit_spin,
            self.dark_scans_spin,
            self.baseline_scans_spin,
            self.specimen_scans_spin,
            self.transverse_radio,
            self.longitudinal_radio,
        )

        # Manual monochromator controls will be added here later.
        self.manual_hardware_controls = (
            self.open_slits_button,
            self.close_slits_button,
            self.manual_wavelength_spin,
            self.manual_input_slit_spin,
            self.manual_output_slit_spin,
        )

        # Action of acquistion buttons
        self.acquire_baseline_button.clicked.connect(
            self.on_acquire_baseline
        )

        self.acquire_scan_button.clicked.connect(
            self.on_acquire_scan
        )

        # --------------------------------------------------
        # Light calibration
        # --------------------------------------------------

        calibration_group = QGroupBox("Light Calibration")
        calibration_layout = QVBoxLayout(calibration_group)

        # Calibration file selection
        self.calibration_combo = QComboBox()
        self.refresh_calibration_list()

        calibration_layout.addWidget(
            QLabel("Calibration file:")
        )
        calibration_layout.addWidget(self.calibration_combo)

        # New calibration button
        self.new_calibration_button = QPushButton("New Calibration...")
        self.new_calibration_button.clicked.connect(
            self.on_new_calibration
        )

        calibration_layout.addWidget(self.new_calibration_button)

        self.hardware_layout.addWidget(calibration_group)

        # --------------------------------------------------
        # Bleaching controls
        # --------------------------------------------------

        self.bleaching_group = QGroupBox("Bleaching")
        bleaching_layout = QVBoxLayout(self.bleaching_group)
        bleaching_layout.setSpacing(8)

        def add_bleach_field(row, label, widget):
            """Add a compact control with its label above it."""
            field = QVBoxLayout()
            field.setSpacing(2)
            field.addWidget(QLabel(label))
            field.addWidget(widget)
            row.addLayout(field)

        # --------------------------------------------------
        # First row: LED, duration, ND filter
        # --------------------------------------------------

        bleach_row1 = QHBoxLayout()
        bleach_row1.setSpacing(8)

        self.bleach_led_combo = QComboBox()
        add_bleach_field(bleach_row1, "LED", self.bleach_led_combo)

        self.bleach_duration_spin = QDoubleSpinBox()
        self.bleach_duration_spin.setRange(0.01, 3600.0)
        self.bleach_duration_spin.setDecimals(2)
        self.bleach_duration_spin.setValue(1.0)
        self.bleach_duration_spin.setSuffix(" s")
        add_bleach_field(
            bleach_row1, "Duration", self.bleach_duration_spin
        )

        self.bleach_nd_combo = QComboBox()

        nd_config_path = (
            Path(__file__).resolve().parents[3]
            / "config"
            / "nd_filters.json"
        )

        with nd_config_path.open(encoding="utf-8") as f:
            nd_values = json.load(f)["nd_filters"]

        for nd in nd_values:
            self.bleach_nd_combo.addItem(f"{nd:.2f}", float(nd))

        add_bleach_field(bleach_row1, "ND", self.bleach_nd_combo)

        bleach_row1.addStretch()
        bleaching_layout.addLayout(bleach_row1)

        # --------------------------------------------------
        # Second row: voltage, photon flux, pigment peak,
        # and estimated bleach fraction
        # --------------------------------------------------

        bleach_row2 = QHBoxLayout()
        bleach_row2.setSpacing(8)

        self.bleach_voltage_spin = QDoubleSpinBox()
        self.bleach_voltage_spin.setRange(0.02, 10.0)
        self.bleach_voltage_spin.setDecimals(2)
        self.bleach_voltage_spin.setSingleStep(0.1)
        self.bleach_voltage_spin.setValue(1.0)
        self.bleach_voltage_spin.setSuffix(" V")
        add_bleach_field(
            bleach_row2, "Voltage", self.bleach_voltage_spin
        )

        self.bleach_flux_label = QLabel("—")
        self.bleach_flux_label.setMinimumWidth(75)
        self.bleach_flux_label.setToolTip(
            "Photon flux density at the specimen "
            "(photons/µm²/s), after ND attenuation"
        )
        add_bleach_field(
            bleach_row2, "Photon flux", self.bleach_flux_label
        )

        self.bleach_lambda_max_spin = QDoubleSpinBox()
        self.bleach_lambda_max_spin.setRange(350.0, 700.0)
        self.bleach_lambda_max_spin.setDecimals(0)
        self.bleach_lambda_max_spin.setSingleStep(1.0)
        self.bleach_lambda_max_spin.setValue(500.0)
        self.bleach_lambda_max_spin.setSuffix(" nm")
        add_bleach_field(
            bleach_row2, "A1 λmax", self.bleach_lambda_max_spin
        )

        self.bleach_fraction_label = QLabel("—")
        self.bleach_fraction_label.setMinimumWidth(50)
        add_bleach_field(
            bleach_row2, "Bleach (%)", self.bleach_fraction_label
        )

        bleach_row2.addStretch()
        bleaching_layout.addLayout(bleach_row2)

        # --------------------------------------------------
        # Third row: Bleach execution
        # --------------------------------------------------

        bleach_row3 = QHBoxLayout()

        self.bleach_button = QPushButton("Bleach")
        self.bleach_button.setEnabled(False)

        bleach_row3.addWidget(self.bleach_button)
        self.bleach_button.clicked.connect(self.on_acquire_bleach)
        bleaching_layout.addLayout(bleach_row3)
        self.refresh_bleach_leds()

        # --------------------------------------------------
        # Prevent unnecessary horizontal expansion
        # --------------------------------------------------

        for widget in (
            self.bleach_led_combo,
            self.bleach_duration_spin,
            self.bleach_nd_combo,
            self.bleach_voltage_spin,
            self.bleach_lambda_max_spin,
        ):
            widget.setSizePolicy(
                QSizePolicy.Policy.Preferred,
                QSizePolicy.Policy.Fixed,
            )

        self.hardware_layout.addWidget(self.bleaching_group)

        # Update LED wavelengths when the calibration changes
        self.calibration_combo.currentIndexChanged.connect(
            self.refresh_bleach_leds
        )
        self.bleach_led_combo.currentIndexChanged.connect(
            self.update_bleach_estimate
        )
        self.bleach_duration_spin.valueChanged.connect(
            self.update_bleach_estimate
        )
        self.bleach_nd_combo.currentIndexChanged.connect(
            self.update_bleach_estimate
        )
        self.bleach_voltage_spin.valueChanged.connect(
            self.update_bleach_estimate
        )
        self.bleach_lambda_max_spin.valueChanged.connect(
            self.update_bleach_estimate
        )

        self.calibration_combo.currentIndexChanged.connect(
            self.update_bleach_estimate
        )

        self.calibration_combo.currentIndexChanged.connect(
            self.update_bleach_voltage_range
        )

        # Initialize controls from the currently selected calibration.
        self.update_bleach_voltage_range()
        self.update_bleach_estimate()

        hardware_panel.setWidget(self.hardware_controls)

        # --------------------------------------------------
        # Main horizontal splitter
        # --------------------------------------------------

        self.splitter = QSplitter(Qt.Orientation.Horizontal)

        self.splitter.addWidget(baseline_panel)
        self.splitter.addWidget(scan_panel)
        self.splitter.addWidget(plot_panel)
        self.splitter.addWidget(hardware_panel)

        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 0)
        self.splitter.setStretchFactor(2, 1)
        self.splitter.setStretchFactor(3, 0)

        self.splitter.setSizes([300, 350, 700, 350])

        # Lower panel: messages, events, and clocks
        self.lower_splitter = QSplitter(Qt.Orientation.Horizontal)

        messages_panel = QGroupBox("Messages / Log")

        scan_editor_panel = QGroupBox("Scan Metadata Editor")
        events_panel = QGroupBox("Experiment Events")
        clocks_panel = QGroupBox("Clocks")

        # Messages panel
        messages_layout = QVBoxLayout(messages_panel)
        messages_layout.addWidget(self.message_log)

        # Scan metadata editor
        scan_editor_layout = QVBoxLayout(scan_editor_panel)

        self.editing_scan_label = QLabel("No scan selected")
        scan_editor_layout.addWidget(self.editing_scan_label)

        scan_editor_form = QFormLayout()

        self.cell_type_combo = QComboBox()
        self.cell_type_combo.addItem("Not assigned", userData=None)

        for cell_type in load_cell_types():
            self.cell_type_combo.addItem(cell_type, userData=cell_type)

        self.cell_type_combo.setEnabled(False)

        scan_editor_form.addRow("Cell type:", self.cell_type_combo)

        self.scan_comment_edit = QPlainTextEdit()
        self.scan_comment_edit.setPlaceholderText(
            "Comments about this recording..."
        )
        self.scan_comment_edit.setEnabled(False)

        scan_editor_form.addRow("Comment:", self.scan_comment_edit)

        scan_editor_layout.addLayout(scan_editor_form)

        self.save_scan_metadata_button = QPushButton("Save Metadata")
        self.save_scan_metadata_button.setEnabled(False)

        self.save_scan_metadata_button.clicked.connect(
            self.on_save_scan_metadata
        )

        self.cell_type_combo.currentIndexChanged.connect(
            self.on_scan_metadata_edited
        )
        self.scan_comment_edit.textChanged.connect(
            self.on_scan_metadata_edited
        )

        scan_editor_layout.addWidget(self.save_scan_metadata_button)

        scan_editor_layout.addStretch()

        # Experiment events panel
        events_layout = QVBoxLayout(events_panel)

        self.events_table = QTableWidget(0, 2)
        self.events_table.setHorizontalHeaderLabels(["Time", "Event"])

        self.events_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.events_table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.events_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )

        self.events_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents
        )
        self.events_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch
        )

        events_layout.addWidget(self.events_table)

        # Manual event entry
        event_entry_layout = QHBoxLayout()

        self.event_description = QLineEdit()
        self.event_description.setPlaceholderText("Event description")

        self.record_event_button = QPushButton("Record Event")

        event_entry_layout.addWidget(self.event_description)
        event_entry_layout.addWidget(self.record_event_button)

        events_layout.addLayout(event_entry_layout)

        self.record_event_button.clicked.connect(self.on_record_event)
        self.event_description.returnPressed.connect(self.on_record_event)

        self.events_table.itemSelectionChanged.connect(
            self.update_elapsed_time
        )

        # Clocks panel
        clocks_layout = QVBoxLayout(clocks_panel)

        clocks_layout.addWidget(QLabel("Current time"))

        self.current_time_label = QLabel("--:--:--")
        self.current_time_label.setStyleSheet(
            "font-size: 24px; font-weight: bold;"
        )
        clocks_layout.addWidget(self.current_time_label)

        self.selected_event_label = QLabel("Time since: No event selected")
        self.selected_event_label.setWordWrap(True)
        clocks_layout.addWidget(self.selected_event_label)

        self.elapsed_event_label = QLabel("--:--:--")
        self.elapsed_event_label.setStyleSheet(
            "font-size: 24px; font-weight: bold;"
        )
        clocks_layout.addWidget(self.elapsed_event_label)

        clocks_layout.addStretch()

        for panel in (
            messages_panel,
            scan_editor_panel,
            events_panel,
            clocks_panel,
        ):
            self.lower_splitter.addWidget(panel)

        self.lower_splitter.setSizes([300, 300, 350, 200])

        self.update_event_controls()
        self.update_acquisition_controls()
        
        # Main window: upper workspace and lower information panel
        self.main_splitter = QSplitter(Qt.Orientation.Vertical)
        self.main_splitter.addWidget(self.splitter)
        self.main_splitter.addWidget(self.lower_splitter)

        self.main_splitter.setStretchFactor(0, 3)
        self.main_splitter.setStretchFactor(1, 1)
        self.main_splitter.setSizes([750, 250])

        self.clock_timer = QTimer(self)
        self.clock_timer.timeout.connect(self.update_clocks)
        self.clock_timer.start(1000)

        self.update_clocks()

        self.setCentralWidget(self.main_splitter)

        self.log_message("MSP Control started")

    def closeEvent(self, event):
        """Safely disconnect hardware when closing the application."""

        if self.acquisition_thread is not None:
            QMessageBox.warning(
                self,
                "Acquisition in Progress",
                "Wait for the current acquisition to finish before exiting.",
            )
            event.ignore()
            return

        if self.hardware_manager is not None:
            try:
                self.hardware_manager.disconnect()
            except Exception as exc:
                self.log_message(f"Hardware shutdown error: {exc}")

        super().closeEvent(event)

    def load_experiment(self, filename: str | Path):
        """Load an experiment and display its OD spectra."""

        if self.controller.experiment is not None:
            self.controller.close_experiment_file()

        self.controller.open_experiment_file(filename)

        self.experiment = self.controller.experiment

        self.update_experiment_metadata_display(
            self.experiment.shared_metadata
        )

        self.baseline_list.clear()
        self.active_baseline_index = None
        self.baseline_plot.clear()

        for baseline in self.experiment.baselines:
            item = QListWidgetItem(
                f"Baseline {baseline.baseline_index:03d} "
                f"({baseline.polarization.value})"
            )

            item.setData(
                Qt.ItemDataRole.UserRole,
                baseline.baseline_index,
            )

            self.baseline_list.addItem(item)

        if self.baseline_list.count() > 0:
            self.baseline_list.setCurrentRow(
                self.baseline_list.count() - 1
            )

        self.scan_list.clear()
        self.scan_metadata_view.clear()
        self.spectrum_plot.clear()
        self.scan_curves.clear()

        for scan in self.experiment.scans:
            color = SCAN_COLORS[scan.scan_index % len(SCAN_COLORS)]

            curve = self.spectrum_plot.plot(
                scan.wavelength,
                scan.optical_density,
                pen=pg.mkPen(color=color, width=3.5),
            )

            curve.setVisible(False)

            self.scan_curves[scan.scan_index] = curve

            item = QListWidgetItem(
                f"Scan {scan.scan_index:03d} ({scan.polarization.value})"
            )

            item.setData(Qt.ItemDataRole.UserRole, scan.scan_index)
            item.setFlags(
                item.flags() | Qt.ItemFlag.ItemIsUserCheckable
            )
            item.setCheckState(Qt.CheckState.Unchecked)

            self.scan_list.addItem(item)

        self.setWindowTitle(f"MSP Control - {Path(filename).name}")

        self.refresh_events_table()
        self.update_event_controls()
        self.update_acquisition_controls()

    def update_experiment_metadata_display(self, metadata):
        """Display experiment metadata using readable labels."""

        if not metadata:
            self.experiment_metadata_label.setText(
                "No experiment metadata"
            )
            return

        registry = self.metadata_registry

        species_id = metadata.get("species")
        species = registry["species"].get(species_id, {})

        genotype_id = metadata.get("genotype")
        genotype_label = species.get(
            "genotypes", {}
        ).get(genotype_id, genotype_id)

        def lookup(section, key):
            value = metadata.get(key)
            return registry[section].get(value, value)

        fields = [
            ("Species", species.get("label", species_id)),
            ("Genotype", genotype_label),
            ("Genotype notes", metadata.get("genotype_notes")),
            ("DOB", metadata.get("dob")),
            ("Age", (
                f"{metadata['age_days']} days"
                if "age_days" in metadata else None
            )),
            ("Sex", lookup("sex", "sex")),
            ("Animal ID", metadata.get("animal_id")),
            ("Cage card", metadata.get("cage_card")),
            ("Animal identifier", metadata.get("animal_identifier")),
            ("Preparation", lookup("preparations", "preparation")),
            ("Experimenter", lookup("experimenters", "experimenter")),
            ("Notes", metadata.get("notes")),
        ]

        lines = [
            f"{label}: {value}"
            for label, value in fields
            if value is not None and value != ""
        ]

        self.experiment_metadata_label.setText(
            "\n".join(lines) if lines else "No experiment metadata"
        )

    def log_message(self, message: str) -> None:
        """Append a timestamped message to the instrument log."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.message_log.appendPlainText(f"[{timestamp}] {message}")

    def refresh_events_table(self) -> None:
        """Display events from the active experiment."""

        self.events_table.setRowCount(0)

        experiment = self.controller.experiment
        if experiment is None:
            return

        events = sorted(
            experiment.events,
            key=lambda event: event.timestamp,
        )

        for event in events:
            row = self.events_table.rowCount()
            self.events_table.insertRow(row)

            time_item = QTableWidgetItem(
                event.timestamp.strftime("%H:%M:%S")
            )
            description_item = QTableWidgetItem(event.description)

            # Retain the event ID without displaying it.
            time_item.setData(
                Qt.ItemDataRole.UserRole,
                event.event_id,
            )

            self.events_table.setItem(row, 0, time_item)
            self.events_table.setItem(row, 1, description_item)

        # Select the most recent event by default.
        if events:
            self.events_table.selectRow(len(events) - 1)

    def update_event_controls(self) -> None:
        """Enable event recording only when an experiment is active."""
        self.record_event_button.setEnabled(
            self.controller.experiment is not None
        )

    def update_acquisition_controls(self) -> None:
        """Enable acquisition controls when their requirements are met."""

        acquiring = self.acquisition_thread is not None

        # Disable scan settings during acquisition.
        for widget in self.acquisition_settings:
            widget.setEnabled(not acquiring)

        # Disable bleach settings during acquisition.
        for widget in (
            self.calibration_combo,
            self.bleach_led_combo,
            self.bleach_duration_spin,
            self.bleach_nd_combo,
            self.bleach_lambda_max_spin,
        ):
            widget.setEnabled(not acquiring)

        # Analog voltage is only applicable to the analog LED.
        self.bleach_voltage_spin.setEnabled(
            not acquiring
            and self.bleach_led_combo.currentData() == 1
        )

        # Manual hardware controls.
        manual_ready = (
            self.hardware_initialized
            and self.hardware_manager is not None
            and self.hardware_manager.optoscan.is_open
            and not acquiring
        )

        for widget in self.manual_hardware_controls:
            widget.setEnabled(manual_ready)

        # Baseline and bleach acquisition require an active experiment.
        ready = (
            self.controller.experiment is not None
            and self.hardware_initialized
            and not acquiring
        )

        self.acquire_baseline_button.setEnabled(ready)
        self.bleach_button.setEnabled(ready)

        # Scan acquisition additionally requires a compatible baseline.
        scan_ready = False

        if ready and self.active_baseline_index is not None:
            baseline = self.controller.experiment.get_baseline(
                self.active_baseline_index
            )

            try:
                config = self.get_scan_config()

                scan_ready = bool(
                    baseline.polarization is self.get_polarization()
                    and baseline.config.is_compatible_with(config)
                )

            except ValueError:
                pass

        self.load_protocol_action.setEnabled(not acquiring)
        self.acquire_scan_button.setEnabled(scan_ready)

    def update_clocks(self) -> None:
        """Update the live clock and elapsed event time."""

        now = datetime.now()
        self.current_time_label.setText(now.strftime("%H:%M:%S"))

        self.update_elapsed_time()

    def update_elapsed_time(self) -> None:
        """Display elapsed time since the selected experiment event."""

        experiment = self.controller.experiment
        row = self.events_table.currentRow()

        if experiment is None or row < 0:
            self.elapsed_event_label.setText("--:--:--")
            self.selected_event_label.setText("Time since: No event selected")
            return

        item = self.events_table.item(row, 0)

        if item is None:
            self.elapsed_event_label.setText("--:--:--")
            self.selected_event_label.setText("Time since: No event selected")
            return

        event_id = item.data(Qt.ItemDataRole.UserRole)
        event = experiment.get_event(event_id)

        self.selected_event_label.setText(
            f"Time since: {event.description}"
        )

        elapsed = datetime.now().astimezone() - event.timestamp
        total_seconds = max(0, int(elapsed.total_seconds()))

        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)

        self.elapsed_event_label.setText(
            f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        )

    def get_scan_config(self, *, baseline: bool = False) -> ScanConfig:
        """Build a scan configuration from the GUI controls."""

        data_scans = (
            self.baseline_scans_spin.value()
            if baseline
            else self.specimen_scans_spin.value()
        )

        return ScanConfig(
            start_nm=self.start_nm_spin.value(),
            end_nm=self.end_nm_spin.value(),
            step_nm=self.step_nm_spin.value(),
            step_time_ms=self.step_time_spin.value(),
            input_slit_nm=self.input_slit_spin.value(),
            output_slit_nm=self.output_slit_spin.value(),
            dark_scans=self.dark_scans_spin.value(),
            data_scans=data_scans,
        )

    def get_acquisition_protocol(self) -> AcquisitionProtocol:
        """Construct an acquisition protocol from the GUI settings."""

        return AcquisitionProtocol(
            start_nm=self.start_nm_spin.value(),
            end_nm=self.end_nm_spin.value(),
            step_nm=self.step_nm_spin.value(),
            step_time_ms=self.step_time_spin.value(),
            input_slit_nm=self.input_slit_spin.value(),
            output_slit_nm=self.output_slit_spin.value(),
            dark_scans=self.dark_scans_spin.value(),
            baseline_scans=self.baseline_scans_spin.value(),
            specimen_scans=self.specimen_scans_spin.value(),
        )

    def set_acquisition_protocol(self, protocol: AcquisitionProtocol) -> None:
        """Populate the GUI controls from an acquisition protocol."""

        self.start_nm_spin.setValue(protocol.start_nm)
        self.end_nm_spin.setValue(protocol.end_nm)
        self.step_nm_spin.setValue(protocol.step_nm)
        self.step_time_spin.setValue(protocol.step_time_ms)
        self.input_slit_spin.setValue(protocol.input_slit_nm)
        self.output_slit_spin.setValue(protocol.output_slit_nm)
        self.dark_scans_spin.setValue(protocol.dark_scans)
        self.baseline_scans_spin.setValue(protocol.baseline_scans)
        self.specimen_scans_spin.setValue(protocol.specimen_scans)

        self.update_acquisition_controls()

    def get_polarization(self) -> Polarization:
        """Return the polarization selected in the GUI."""

        if self.transverse_radio.isChecked():
            return Polarization.TRANSVERSE

        return Polarization.LONGITUDINAL

    def get_field_stop_metadata(self) -> dict:
        """Return the current field-stop dimensions."""
        return {
            "field_stop_height_um": self.field_stop_height_spin.value(),
            "field_stop_width_um": self.field_stop_width_spin.value(),
        }

    def refresh_calibration_list(self):
        """Refresh available calibrations and select the configured default."""

        self.calibration_combo.clear()
        self.calibration_combo.addItem("No calibration selected", None)

        for calibration_file in find_calibration_files():
            self.calibration_combo.addItem(
                calibration_file.name,
                calibration_file,
            )

        default_filename = load_default_calibration()

        if default_filename is not None:
            for index in range(1, self.calibration_combo.count()):
                calibration_file = self.calibration_combo.itemData(index)

                if calibration_file.name == default_filename:
                    self.calibration_combo.setCurrentIndex(index)
                    break

    def on_acquire_bleach(self) -> None:
        """Start a bleaching exposure in a worker thread."""

        if (
            not self.hardware_initialized
            or self.controller.experiment is None
            or self.controller.bleach_controller is None
            or self.acquisition_thread is not None
        ):
            return

        led_channel = self.bleach_led_combo.currentData()

        try:
            config = BleachWaveformConfig(
                duration_s=self.bleach_duration_spin.value()
            )

            nd_filter = float(self.bleach_nd_combo.currentData())

            command_voltage = (
                self.bleach_voltage_spin.value()
                if led_channel == 1
                else None
            )

        except (TypeError, ValueError) as exc:
            QMessageBox.warning(self, "Invalid bleach settings", str(exc))
            return

        self.acquisition_thread = QThread(self)

        self.acquisition_worker = BleachAcquisitionWorker(
            controller=self.controller,
            led_channel=led_channel,
            config=config,
            nd_filter=nd_filter,
            command_voltage=command_voltage,
        )

        self.acquisition_worker.moveToThread(self.acquisition_thread)

        self.acquisition_thread.started.connect(self.acquisition_worker.run)

        self.acquisition_worker.finished.connect(self.on_bleach_finished)
        self.acquisition_worker.failed.connect(self.on_bleach_failed)

        self.acquisition_worker.finished.connect(self.acquisition_thread.quit)
        self.acquisition_worker.failed.connect(self.acquisition_thread.quit)

        self.acquisition_worker.finished.connect(self.acquisition_worker.deleteLater)
        self.acquisition_worker.failed.connect(self.acquisition_worker.deleteLater)

        self.acquisition_thread.finished.connect(
            self.on_acquisition_thread_finished
        )
        self.acquisition_thread.finished.connect(
            self.acquisition_thread.deleteLater
        )

        self.update_acquisition_controls()

        self.log_message(
            f"Starting bleach: LED {led_channel}, "
            f"{config.duration_s:g} s, ND {nd_filter:g}"
        )

        self.acquisition_thread.start()

    @Slot(object)
    def on_bleach_finished(self, bleach) -> None:
        """Handle a completed bleaching exposure."""

        self.log_message(
            f"Bleach {bleach.bleach_index} completed successfully."
        )

        self.refresh_events_table()

    @Slot(str)
    def on_bleach_failed(self, message: str) -> None:
        """Handle a failed bleaching exposure."""

        self.log_message(f"Bleach failed: {message}")

        self.refresh_events_table()

        QMessageBox.critical(
            self,
            "Bleach failed",
            message,
        )

    def on_acquire_baseline(self) -> None:
        """Start baseline acquisition in a worker thread."""

        if self.controller.experiment is None or not self.hardware_initialized:
            return

        if self.acquisition_thread is not None:
            return

        try:
            config = self.get_scan_config(baseline=True)
            polarization = self.get_polarization()
        except ValueError as exc:
            QMessageBox.warning(
                self,
                "Invalid Scan Configuration",
                str(exc),
            )
            return

        self.log_message(
            f"Acquiring {polarization.value} baseline "
            f"({config.data_scans} scans)"
        )

        self.acquisition_thread = QThread(self)
        self.acquisition_worker = BaselineAcquisitionWorker(
            self.controller,
            config,
            polarization,
            metadata=self.get_field_stop_metadata(),
        )

        self.acquisition_worker.moveToThread(self.acquisition_thread)

        self.acquisition_thread.started.connect(self.acquisition_worker.run)

        self.acquisition_worker.finished.connect(self.on_baseline_acquired)
        self.acquisition_worker.failed.connect(self.on_baseline_failed)

        self.acquisition_worker.finished.connect(self.acquisition_thread.quit)
        self.acquisition_worker.failed.connect(self.acquisition_thread.quit)

        self.acquisition_thread.finished.connect(
            self.acquisition_worker.deleteLater
        )
        self.acquisition_thread.finished.connect(
            self.acquisition_thread.deleteLater
        )
        self.acquisition_thread.finished.connect(
            self.on_acquisition_thread_finished
        )

        self.update_acquisition_controls()
        self.acquisition_thread.start()

    @Slot(object)
    def on_baseline_acquired(self, baseline) -> None:
        """Display a successfully acquired baseline."""

        self.log_message(
            f"Baseline {baseline.baseline_index} acquired successfully"
        )

        item = QListWidgetItem(
            f"Baseline {baseline.baseline_index:03d} "
            f"({baseline.polarization.value})"
        )

        item.setData(
            Qt.ItemDataRole.UserRole,
            baseline.baseline_index,
        )

        self.baseline_list.addItem(item)
        self.baseline_list.setCurrentItem(item)

    @Slot(str)
    def on_baseline_failed(self, message: str) -> None:
        """Report an acquisition error."""

        self.log_message(f"Baseline acquisition failed: {message}")

        QMessageBox.critical(
            self,
            "Baseline Acquisition Failed",
            message,
        )

    def on_acquisition_thread_finished(self) -> None:
        """Clean up after an acquisition."""

        self.acquisition_thread = None
        self.acquisition_worker = None
        self.update_acquisition_controls()

    def on_acquire_scan(self) -> None:
        """Start specimen acquisition in a worker thread."""

        # Recheck the acquisition conditions.
        self.update_acquisition_controls()

        if not self.acquire_scan_button.isEnabled():
            return

        try:
            config = self.get_scan_config()
            polarization = self.get_polarization()
            baseline_index = self.active_baseline_index

        except ValueError as exc:
            QMessageBox.warning(
                self,
                "Invalid Scan Configuration",
                str(exc),
            )
            return

        self.log_message(
            f"Acquiring {polarization.value} specimen scan "
            f"using baseline {baseline_index:03d}"
        )

        self.acquisition_thread = QThread(self)

        self.acquisition_worker = ScanAcquisitionWorker(
            controller=self.controller,
            config=config,
            polarization=polarization,
            baseline_index=baseline_index,
            metadata=self.get_field_stop_metadata(),
        )

        self.acquisition_worker.moveToThread(self.acquisition_thread)

        self.acquisition_thread.started.connect(
            self.acquisition_worker.run
        )

        self.acquisition_worker.finished.connect(
            self.on_scan_acquired
        )
        self.acquisition_worker.failed.connect(
            self.on_scan_failed
        )

        self.acquisition_worker.finished.connect(
            self.acquisition_thread.quit
        )
        self.acquisition_worker.failed.connect(
            self.acquisition_thread.quit
        )

        self.acquisition_thread.finished.connect(
            self.acquisition_worker.deleteLater
        )
        self.acquisition_thread.finished.connect(
            self.acquisition_thread.deleteLater
        )
        self.acquisition_thread.finished.connect(
            self.on_acquisition_thread_finished
        )

        self.update_acquisition_controls()
        self.acquisition_thread.start()

    @Slot(object)
    def on_scan_acquired(self, scan) -> None:
        """Display a successfully acquired specimen scan."""

        self.log_message(
            f"Scan {scan.scan_index} acquired successfully"
        )

        color = SCAN_COLORS[scan.scan_index % len(SCAN_COLORS)]

        curve = self.spectrum_plot.plot(
            scan.wavelength,
            scan.optical_density,
            pen=pg.mkPen(color=color, width=3.5),
        )

        self.scan_curves[scan.scan_index] = curve

        item = QListWidgetItem(
            f"Scan {scan.scan_index:03d} ({scan.polarization.value})"
        )

        item.setData(
            Qt.ItemDataRole.UserRole,
            scan.scan_index,
        )

        item.setFlags(
            item.flags() | Qt.ItemFlag.ItemIsUserCheckable
        )
        # Hide and uncheck all previously displayed scans.
        for i in range(self.scan_list.count()):
            existing_item = self.scan_list.item(i)
            existing_item.setCheckState(Qt.CheckState.Unchecked)
        item.setCheckState(Qt.CheckState.Checked)

        self.scan_list.addItem(item)

        # Select the new scan to display its metadata.
        self.scan_list.setCurrentItem(item)

        self.plot_mode_combo.setCurrentIndex(0)
        self.zero_correction_checkbox.setChecked(False)
        self.show_legend_checkbox.setChecked(False)
        self.refresh_spectrum_plot()

    @Slot(str)
    def on_scan_failed(self, message: str) -> None:
        """Report a specimen acquisition error."""

        self.log_message(f"Scan acquisition failed: {message}")

        QMessageBox.critical(
            self,
            "Scan Acquisition Failed",
            message,
        )

    def on_scan_visibility_changed(self, item):
        """Update the plot when a scan checkbox changes."""
        self.refresh_spectrum_plot()

    def on_active_baseline_changed(self, current, previous):
        """Update the active baseline and its preview."""

        if current is None:
            self.active_baseline_index = None
            self.baseline_plot.clear()
            self.update_acquisition_controls()
            return

        self.active_baseline_index = current.data(
            Qt.ItemDataRole.UserRole
        )

        baseline = next(
            b for b in self.experiment.baselines
            if b.baseline_index == self.active_baseline_index
        )

        self.baseline_plot.clear()

        self.baseline_plot.plot(
            baseline.wavelength,
            baseline.baseline_corrected,
            pen=pg.mkPen(color="#56B4E9", width=2.5),
        )

        self.update_acquisition_controls()

    def on_scan_selection_changed(self, current, previous):
        """Display metadata for the highlighted specimen scan."""

        self.scan_metadata_view.clear()

        if current is None or self.experiment is None:
            self.editing_scan_label.setText("No scan selected")

            self.cell_type_combo.blockSignals(True)
            self.scan_comment_edit.blockSignals(True)

            self.cell_type_combo.setCurrentIndex(0)
            self.scan_comment_edit.clear()

            self.cell_type_combo.blockSignals(False)
            self.scan_comment_edit.blockSignals(False)

            self.cell_type_combo.setEnabled(False)
            self.scan_comment_edit.setEnabled(False)
            self.save_scan_metadata_button.setEnabled(False)
            return

        scan_index = current.data(Qt.ItemDataRole.UserRole)

        scan = next(
            s for s in self.experiment.scans
            if s.scan_index == scan_index
        )

        self.editing_scan_label.setText(f"Editing Scan {scan_index:03d}")

        # Prevent programmatic updates from triggering change detection.
        self.cell_type_combo.blockSignals(True)
        self.scan_comment_edit.blockSignals(True)

        cell_type = scan.metadata.get("cell_type")
        index = self.cell_type_combo.findData(cell_type)
        self.cell_type_combo.setCurrentIndex(index if index >= 0 else 0)

        self.scan_comment_edit.setPlainText(
            scan.metadata.get("comment", "")
        )

        self.cell_type_combo.blockSignals(False)
        self.scan_comment_edit.blockSignals(False)

        self.cell_type_combo.setEnabled(True)
        self.scan_comment_edit.setEnabled(True)
        self.save_scan_metadata_button.setEnabled(False)

        config = scan.config

        rows = [
            ("Time", scan.timestamp.strftime("%H:%M:%S")),
            ("Polarization", scan.polarization.value),
            ("Baseline", f"{scan.baseline_index:03d}"),
            ("Wavelength range",
             f"{config.start_nm}–{config.end_nm} nm"),
            ("Wavelength step", f"{config.step_nm} nm"),
            ("Step time", f"{config.step_time_ms} ms"),
            ("Slit widths",
             f"{config.input_slit_nm} / {config.output_slit_nm} nm"),
            ("Dark scans", str(config.dark_scans)),
            ("Specimen scans", str(config.data_scans)),
        ]

        labels = {
            "temperature_mean_C": ("Temperature", "°C"),
            "temperature_sd_C": ("Temperature SD", "°C"),
            "field_stop_angle_deg": ("Field stop angle", "°"),
            "field_stop_stage_angle_deg": ("Field stop stage angle", "°"),
            "field_stop_horizontal_reference_deg": (
                "Field stop reference", "°"
            ),
            "stage_x_um": ("Stage X", "µm"),
            "stage_y_um": ("Stage Y", "µm"),
            "stage_z_um": ("Stage Z", "µm"),
        }

        for key, value in scan.metadata.items():
            label, unit = labels.get(key, (key.replace("_", " ").title(), ""))

            if isinstance(value, (int, float)):
                display_value = f"{value:.2f}"
            else:
                display_value = str(value)

            if unit:
                display_value += f" {unit}"

            rows.append((label, display_value))

        for parameter, value in rows:
            self.scan_metadata_view.addTopLevelItem(
                QTreeWidgetItem([parameter, value])
            )

        self.scan_metadata_view.resizeColumnToContents(0)

    def on_scan_metadata_edited(self):
        """Enable Save Metadata when the editor differs from saved values."""
        if self.experiment is None:
            return

        current = self.scan_list.currentItem()
        if current is None:
            return

        scan_index = current.data(Qt.ItemDataRole.UserRole)
        scan = next(
            (s for s in self.experiment.scans
             if s.scan_index == scan_index),
            None,
        )
        if scan is None:
            return

        cell_type = self.cell_type_combo.currentData()
        comment = self.scan_comment_edit.toPlainText()

        modified = (
            cell_type != scan.metadata.get("cell_type")
            or comment != scan.metadata.get("comment", "")
        )

        self.save_scan_metadata_button.setEnabled(modified)

    def on_save_scan_metadata(self):
        """Save editable metadata for the highlighted specimen scan."""

        if self.experiment is None:
            return

        current = self.scan_list.currentItem()
        if current is None:
            return

        scan_index = current.data(Qt.ItemDataRole.UserRole)

        metadata = {
            "cell_type": self.cell_type_combo.currentData(),
            "comment": self.scan_comment_edit.toPlainText(),
        }

        try:
            self.controller.update_scan_metadata(
                scan_index,
                metadata,
            )
        except Exception as exc:
            self.log_message(
                f"Failed to save metadata for scan {scan_index:03d}: {exc}"
            )
            return

        self.save_scan_metadata_button.setEnabled(False)

        # Refresh the read-only metadata table.
        self.on_scan_selection_changed(current, None)

        self.log_message(
            f"Metadata saved for scan {scan_index:03d}"
        )

    def on_new_experiment(self):
        """Create a new experiment with its metadata."""

        if self.controller.experiment is not None:
            QMessageBox.information(
                self,
                "Experiment Already Open",
                "Close the current experiment before creating a new one.",
            )
            return

        dialog = ExperimentMetadataDialog(
            self.metadata_registry,
            parent=self,
        )

        if not dialog.exec():
            return

        metadata = dialog.get_metadata()

        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Create New Experiment",
            "data/",
            "MSP Experiment (*.h5)",
        )

        if not filename:
            return

        filename = Path(filename)

        if filename.suffix.lower() != ".h5":
            filename = filename.with_suffix(".h5")

        if filename.exists():
            QMessageBox.warning(
                self,
                "File Already Exists",
                f"The file already exists:\n\n{filename}\n\n"
                "Choose a different filename.",
            )
            return

        try:
            self.controller.create_new_experiment_file(filename)
            self.controller.update_experiment_metadata(metadata)

        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            QMessageBox.critical(
                self,
                "Experiment Creation Failed",
                f"Could not create the experiment:\n\n{exc}",
            )
            return

        self.experiment = self.controller.experiment

        self.update_experiment_metadata_display(
            self.experiment.shared_metadata
        )

        self.baseline_list.clear()
        self.active_baseline_index = None
        self.baseline_plot.clear()

        self.scan_list.clear()
        self.scan_metadata_view.clear()
        self.spectrum_plot.clear()
        self.scan_curves.clear()

        self.setWindowTitle(f"MSP Control - {filename.name}")
        self.refresh_events_table()
        self.update_event_controls()
        self.update_acquisition_controls()

    def on_edit_experiment_metadata(self):
        """Open the metadata dialog for the loaded experiment."""

        if self.controller.experiment is None:
            return

        dialog = ExperimentMetadataDialog(
            self.metadata_registry,
            metadata=self.experiment.shared_metadata,
            parent=self,
        )

        if dialog.exec():
            metadata = dialog.get_metadata()

            try:
                self.controller.update_experiment_metadata(metadata)

            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                QMessageBox.critical(
                    self,
                    "Metadata Save Failed",
                    f"Could not save experiment metadata:\n\n{exc}",
                )
                return

            self.update_experiment_metadata_display(
                self.controller.experiment.shared_metadata
            )

    def on_close_experiment(self):
        """Close the active experiment and clear its display."""

        if self.acquisition_thread is not None:
            QMessageBox.warning(
                self,
                "Acquisition in Progress",
                "Wait for the current acquisition to finish before closing the experiment.",
            )
            return

        if self.controller.experiment is None:
            return

        self.controller.close_experiment_file()
        self.experiment = None

        # Clear experiment metadata.
        self.update_experiment_metadata_display({})

        # Clear baselines.
        self.baseline_list.clear()
        self.active_baseline_index = None
        self.baseline_plot.clear()

        # Clear scans and spectra.
        self.scan_list.clear()
        self.scan_curves.clear()
        self.spectrum_plot.clear()

        # Clear scan metadata.
        self.scan_metadata_view.clear()

        # Clear experiment events and reset the elapsed-time clock.
        self.refresh_events_table()
        self.update_elapsed_time()

        self.update_event_controls()
        self.setWindowTitle("MSP Control")
        self.update_acquisition_controls()
        
    def on_open_experiment(self):
        """Open an existing experiment from an HDF5 file."""

        if self.controller.experiment is not None:
            QMessageBox.information(
                self,
                "Experiment Already Open",
                "Close the current experiment before opening another one.",
            )
            return

        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Open Experiment",
            "data/",
            "MSP Experiment (*.h5)",
        )

        if not filename:
            return

        try:
            self.load_experiment(filename)

        except (OSError, RuntimeError, TypeError, ValueError, KeyError) as exc:
            QMessageBox.critical(
                self,
                "Experiment Open Failed",
                f"Could not open the experiment:\n\n{exc}",
            )

    def on_load_protocol(self) -> None:
        """Load acquisition settings from a JSON protocol."""

        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Load Acquisition Protocol",
            "",
            "JSON files (*.json)",
        )

        if not filename:
            return

        try:
            protocol = AcquisitionProtocol.load(filename)

            # Check that all values fit the GUI controls.
            fields = [
                (self.start_nm_spin, protocol.start_nm),
                (self.end_nm_spin, protocol.end_nm),
                (self.step_nm_spin, protocol.step_nm),
                (self.step_time_spin, protocol.step_time_ms),
                (self.input_slit_spin, protocol.input_slit_nm),
                (self.output_slit_spin, protocol.output_slit_nm),
                (self.dark_scans_spin, protocol.dark_scans),
                (self.baseline_scans_spin, protocol.baseline_scans),
                (self.specimen_scans_spin, protocol.specimen_scans),
            ]

            for widget, value in fields:
                if not widget.minimum() <= value <= widget.maximum():
                    raise ValueError(
                        f"Protocol value {value} is outside the "
                        f"allowed range for this control"
                    )

            self.set_acquisition_protocol(protocol)
            self.log_message(f"Loaded protocol: {filename}")

        except (OSError, ValueError, TypeError, KeyError) as exc:
            QMessageBox.warning(
                self,
                "Load Protocol Failed",
                str(exc),
            )

    def on_save_protocol(self) -> None:
        """Save the current acquisition settings to JSON."""

        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save Acquisition Protocol",
            "",
            "JSON files (*.json)",
        )

        if not filename:
            return

        try:
            protocol = self.get_acquisition_protocol()

            # Validate both acquisition configurations.
            protocol.scan_config(baseline=True)
            protocol.scan_config(baseline=False)

            protocol.save(filename)
            self.log_message(f"Saved protocol: {filename}")

        except (OSError, ValueError, TypeError) as exc:
            QMessageBox.warning(
                self,
                "Save Protocol Failed",
                str(exc),
            )

    def on_initialize_hardware(self) -> None:
        if self.acquisition_thread is not None:
            QMessageBox.warning(
                self,
                "Acquisition in Progress",
                "Wait for the current acquisition to finish before reinitializing hardware.",
            )
            return

        self.log_message("Initializing hardware...")

        try:
            if self.hardware_manager is None:
                (
                    hardware_manager,
                    metadata_collector,
                    acquisition_controller,
                    bleach_controller,
                ) = create_hardware_manager()

                hardware_manager.initialize()

                # Attach hardware only after successful initialization.
                self.hardware_manager = hardware_manager
                self.controller.metadata_collector = metadata_collector
                self.controller.acquisition = acquisition_controller
                self.controller.bleach_controller = bleach_controller

            else:
                self.hardware_manager.initialize()

            self.hardware_initialized = True
            self.update_acquisition_controls()

            self.log_message("Hardware initialized successfully.")

            QMessageBox.information(
                self,
                "Hardware",
                "Hardware initialized successfully.",
            )

        except Exception as exc:
            self.hardware_initialized = False
            self.update_acquisition_controls()
            self.log_message(f"Hardware initialization failed: {exc}")

            QMessageBox.critical(
                self,
                "Hardware initialization failed",
                str(exc),
            )

    def on_open_slits(self):
        """Position the monochromator and open its slits."""
        if (
            not self.hardware_initialized
            or self.hardware_manager is None
            or self.acquisition_thread is not None
        ):
            return

        try:
            self.hardware_manager.optoscan.open_slits(
                wavelength_nm=self.manual_wavelength_spin.value(),
                input_slit_nm=self.manual_input_slit_spin.value(),
                output_slit_nm=self.manual_output_slit_spin.value(),
            )

        except Exception as exc:
            QMessageBox.critical(
                self,
                "Optoscan Error",
                f"Could not open the slits:\n{exc}",
            )

    def on_close_slits(self):
        """Close the monochromator slits."""
        if (
            not self.hardware_initialized
            or self.hardware_manager is None
            or self.acquisition_thread is not None
        ):
            return

        try:
            self.hardware_manager.optoscan.close_slits()

        except Exception as exc:
            QMessageBox.critical(
                self,
                "Optoscan Error",
                f"Could not close the slits:\n{exc}",
            )

    def on_record_event(self) -> None:
        """Record a manually entered experiment event."""

        if self.controller.experiment is None:
            return

        description = self.event_description.text().strip()

        if not description:
            return

        if self.controller.experiment is None:
            QMessageBox.warning(
                self,
                "No Active Experiment",
                "Open or create an experiment before recording an event.",
            )
            return

        try:
            event = self.controller.add_event(description)

        except Exception as exc:
            self.log_message(f"Failed to record event: {exc}")
            QMessageBox.critical(
                self,
                "Event Recording Failed",
                str(exc),
            )
            return

        self.refresh_events_table()
        self.event_description.clear()

        self.log_message(
            f"Event recorded: {event.description}"
        )

    def on_plot_mode_changed(self, index):
        """Update the displayed quantity without modifying stored data."""

        self.refresh_spectrum_plot()

    def on_zero_cursor_moved(self, line, spinbox):
        """Synchronize a wavelength control with its plot cursor."""

        spinbox.blockSignals(True)
        spinbox.setValue(line.value())
        spinbox.blockSignals(False)

    def on_zoom_mode_changed(self):
        """Control which plot axes respond to mouse interactions."""

        mode = self.zoom_mode_combo.currentData()

        view_box = self.spectrum_plot.getViewBox()

        view_box.setMouseEnabled(
            x=mode in ("xy", "x"),
            y=mode in ("xy", "y"),
        )

    def on_new_calibration(self):
        """Open the calibration dialog and refresh after a successful save."""

        dialog = CalibrationDialog(self)
        dialog.exec()

        if dialog.calibration_saved:
            self.refresh_calibration_list()

    def refresh_bleach_leds(self):
        """Populate the bleach LED selector from the selected calibration."""

        selected_channel = self.bleach_led_combo.currentData()

        self.bleach_led_combo.clear()

        calibration_file = self.calibration_combo.currentData()

        if calibration_file is None:
            self.bleach_led_combo.addItem("Analog", 1)
            self.bleach_led_combo.addItem("Digital", 2)
            return

        try:
            calibration = read_calibration(calibration_file)
        except (OSError, KeyError, ValueError) as exc:
            self.log_message(f"Unable to load light calibration: {exc}")
            self.bleach_led_combo.addItem("Analog", 1)
            self.bleach_led_combo.addItem("Digital", 2)
            return

        self.bleach_led_combo.addItem(
            f"Analog ({calibration.analog_led.wavelength_nm:g} nm)",
            calibration.analog_led.led_channel,
        )

        self.bleach_led_combo.addItem(
            f"Digital ({calibration.digital_led.wavelength_nm:g} nm)",
            calibration.digital_led.led_channel,
        )

        for index in range(self.bleach_led_combo.count()):
            if self.bleach_led_combo.itemData(index) == selected_channel:
                self.bleach_led_combo.setCurrentIndex(index)
                break

    def update_bleach_voltage_range(self):
        """Set analog bleach voltage limits from the selected calibration."""

        calibration_file = self.calibration_combo.currentData()

        if calibration_file is None:
            self.bleach_voltage_spin.setRange(0.02, 10.0)
            return

        try:
            calibration = read_calibration(calibration_file)

            voltages = calibration.analog_led.calibration_voltage

            minimum = float(voltages.min())
            maximum = float(voltages.max())

            self.bleach_voltage_spin.setRange(minimum, maximum)

        except (OSError, KeyError, ValueError) as exc:
            self.log_message(
                f"Unable to determine bleach voltage range: {exc}"
            )
            self.bleach_voltage_spin.setRange(0.02, 10.0)

            
    def update_bleach_estimate(self):
        """Update photon flux density and estimated bleach fraction."""

        self.bleach_flux_label.setText("—")
        self.bleach_fraction_label.setText("—")

        led_channel = self.bleach_led_combo.currentData()

        # Analog LED requires a voltage; digital LED is on/off.
        is_analog = led_channel == 1
        self.bleach_voltage_spin.setEnabled(
            is_analog and self.acquisition_thread is None
        )

        calibration_file = self.calibration_combo.currentData()
        if calibration_file is None:
            return

        try:
            calibration = read_calibration(calibration_file)

            if is_analog:
                led_calibration = calibration.analog_led
                command_voltage = self.bleach_voltage_spin.value()
            elif led_channel == 2:
                led_calibration = calibration.digital_led
                command_voltage = None
            else:
                return

            estimate = estimate_bleach(
                calibration=led_calibration,
                duration_s=self.bleach_duration_spin.value(),
                nd_filter=self.bleach_nd_combo.currentData(),
                lambda_max_nm=self.bleach_lambda_max_spin.value(),
                command_voltage=command_voltage,
            )

        except (OSError, KeyError, ValueError) as exc:
            self.log_message(f"Unable to calculate bleach estimate: {exc}")
            return

        self.bleach_flux_label.setText(
            f"{estimate.photon_flux_density:.2e}"
        )
        self.bleach_fraction_label.setText(
            f"{100.0 * estimate.fraction_bleached:.1f}%"
        )

    def refresh_spectrum_plot(self, auto_range=True):
        """Update displayed spectra and optionally reset the plot range."""

        if self.experiment is None:
            return

        mode = self.plot_mode_combo.currentData()

        plot_item = self.spectrum_plot.getPlotItem()

        for line in (self.zero_start_line, self.zero_end_line):
            if line not in plot_item.items:
                self.spectrum_plot.addItem(line, ignoreBounds=True)

        show_zero_cursors = (
            mode == "od"
            and self.zero_correction_checkbox.isChecked()
        )

        self.zero_start_line.setVisible(show_zero_cursors)
        self.zero_end_line.setVisible(show_zero_cursors)

        displayed_baselines = set()
        self.spectrum_legend.clear()

        for i in range(self.scan_list.count()):
            item = self.scan_list.item(i)
            scan_index = item.data(Qt.ItemDataRole.UserRole)
            scan = self.experiment.get_scan(scan_index)

            curve = self.scan_curves.get(scan_index)

            if curve is None:
                continue

            if mode == "od":
                wavelength = scan.wavelength
                values = scan.optical_density

                if self.zero_correction_checkbox.isChecked():
                    start = self.zero_start_spin.value()
                    end = self.zero_end_spin.value()

                    mask = (
                        (wavelength >= min(start, end))
                        & (wavelength <= max(start, end))
                        & np.isfinite(values)
                    )

                    if np.any(mask):
                        values = values - np.mean(values[mask])

            elif mode == "scan":
                wavelength = scan.wavelength
                values = scan.specimen_corrected

            elif mode == "baseline":
                baseline = self.experiment.get_baseline(
                    scan.baseline_index
                )
                wavelength = baseline.wavelength
                values = baseline.baseline_corrected

            else:
                return

            curve.setData(wavelength, values)

            visible = item.checkState() == Qt.CheckState.Checked

            if mode == "baseline" and visible:
                if scan.baseline_index in displayed_baselines:
                    visible = False
                else:
                    displayed_baselines.add(scan.baseline_index)

            curve.setVisible(visible)
            if visible and self.show_legend_checkbox.isChecked():
                if mode == "baseline":
                    label = (
                        f"Baseline {baseline.baseline_index:03d} "
                        f"({baseline.polarization.value})"
                    )
                else:
                    label = (
                        f"Scan {scan.scan_index:03d} "
                        f"({scan.polarization.value})"
                    )

                self.spectrum_legend.addItem(curve, label)
        self.spectrum_legend.setVisible(
            self.show_legend_checkbox.isChecked()
        )

        if mode == "od":
            self.spectrum_plot.setLabel(
                "left", "Optical density", **self.axis_label_style
            )
        else:
            self.spectrum_plot.setLabel(
                "left", "Signal", units="V", **self.axis_label_style
            )
        if auto_range:
            self.spectrum_plot.autoRange()

def create_hardware_manager() -> tuple[
    HardwareManager,
    MetadataCollector,
    AcquisitionController,
]:
    """Construct hardware objects without connecting to devices."""

    optoscan = OptoscanSerial("COM2")

    field_stop_config = FieldStopConfig()
    field_stop_device = KinesisFieldStopDevice.from_kinesis(
        field_stop_config
    )
    field_stop = FieldStopMonitor(
        field_stop_device,
        field_stop_config,
    )

    hardware_manager = HardwareManager(
        optoscan=optoscan,
        field_stop=field_stop,
    )

    stage = StagePositionMonitor(StageConfig())
    temperature = TemperatureMonitor(TemperatureConfig())

    metadata_collector = MetadataCollector(
        field_stop=field_stop,
        stage=stage,
        temperature=temperature,
    )

    daq = NIDaq(NIDaqConfig())

    acquisition_controller = AcquisitionController(
        daq=daq,
        optoscan=optoscan,
    )

    bleach_controller = BleachController(daq=daq)

    return (
        hardware_manager,
        metadata_collector,
        acquisition_controller,
        bleach_controller,
    )


def main():
    app = QApplication(sys.argv)

    window = MSPMainWindow()
    window.show()

    return app.exec()

if __name__ == "__main__":
    sys.exit(main())
