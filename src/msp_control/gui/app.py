import sys
from pathlib import Path

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
)

from PySide6.QtCore import Qt
import pyqtgraph as pg

from msp_control.controller import MSPController
from msp_control.metadata_registry import load_metadata_registry
from msp_control.gui.experiment_metadata_dialog import (
    ExperimentMetadataDialog,
)


class MSPMainWindow(QMainWindow):
    """Main window for the MSP control application."""

    def __init__(self):
        super().__init__()

        self.setWindowTitle("MSP Control")
        self.resize(1400, 800)

        self.controller = MSPController()
        self.experiment = None

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

        # Edit menu
        edit_menu = self.menuBar().addMenu("&Edit")

        edit_metadata_action = edit_menu.addAction(
            "Experiment &Metadata..."
        )
        edit_metadata_action.triggered.connect(
            self.on_edit_experiment_metadata
        )

        # --------------------------------------------------
        # Main OD spectrum plot
        # --------------------------------------------------

        self.spectrum_plot = pg.PlotWidget()
        self.spectrum_plot.setBackground("w")

        self.spectrum_plot.setLabel(
            "bottom", "Wavelength", units="nm"
        )
        self.spectrum_plot.setLabel(
            "left", "Optical density"
        )

        self.spectrum_plot.showGrid(x=True, y=True, alpha=0.2)

        # --------------------------------------------------
        # Specimen scan list
        # --------------------------------------------------

        self.scan_list = QListWidget()
        self.scan_list.setMinimumWidth(120)

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
        self.baseline_list.setMaximumHeight(150)

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
        self.baseline_plot.setBackground("w")
        self.baseline_plot.setMinimumHeight(150)
        self.baseline_plot.setMaximumHeight(220)

        self.baseline_plot.setLabel(
            "bottom", "Wavelength", units="nm"
        )
        self.baseline_plot.setLabel(
            "left", "Signal", units="V"
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

        plot_layout.addWidget(QLabel("Plot controls"))

        # --------------------------------------------------
        # Column 4: MSP and bleaching controls
        # --------------------------------------------------

        hardware_panel = QWidget()
        hardware_layout = QVBoxLayout(hardware_panel)

        hardware_layout.addWidget(QLabel("MSP controls"))
        hardware_layout.addWidget(QLabel("Bleaching controls"))
        hardware_layout.addStretch()

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

        self.splitter.setSizes([280, 190, 700, 230])

        self.setCentralWidget(self.splitter)

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
        self.spectrum_plot.clear()
        self.scan_curves.clear()

        colors = [
            "#0072B2",
            "#D55E00",
            "#009E73",
            "#CC79A7",
            "#E69F00",
            "#56B4E9",
        ]

        for scan in self.experiment.scans:
            color = colors[scan.scan_index % len(colors)]

            curve = self.spectrum_plot.plot(
                scan.wavelength,
                scan.optical_density,
                pen=pg.mkPen(color=color, width=1.5),
            )

            self.scan_curves[scan.scan_index] = curve

            item = QListWidgetItem(
                f"Scan {scan.scan_index:03d} ({scan.polarization.value})"
            )

            item.setData(Qt.ItemDataRole.UserRole, scan.scan_index)
            item.setFlags(
                item.flags() | Qt.ItemFlag.ItemIsUserCheckable
            )
            item.setCheckState(Qt.CheckState.Checked)

            self.scan_list.addItem(item)

        self.setWindowTitle(f"MSP Control - {Path(filename).name}")

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

    def on_scan_visibility_changed(self, item):
        """Show or hide a scan when its checkbox changes."""

        scan_index = item.data(Qt.ItemDataRole.UserRole)
        curve = self.scan_curves.get(scan_index)

        if curve is not None:
            visible = item.checkState() == Qt.CheckState.Checked
            curve.setVisible(visible)

    def on_active_baseline_changed(self, current, previous):
        """Update the active baseline and its preview."""

        if current is None:
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
            pen=pg.mkPen(color="#0072B2", width=1.5),
        )

    def on_scan_selection_changed(self, current, previous):
        """Display metadata for the highlighted specimen scan."""

        self.scan_metadata_view.clear()

        if current is None:
            return

        scan_index = current.data(Qt.ItemDataRole.UserRole)

        scan = next(
            s for s in self.experiment.scans
            if s.scan_index == scan_index
        )

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
        self.scan_list.clear()
        self.spectrum_plot.clear()
        self.scan_curves.clear()
        self.active_baseline_index = None

        self.setWindowTitle(f"MSP Control - {filename.name}")

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

        self.setWindowTitle("MSP Control")

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

def main():
    app = QApplication(sys.argv)

    window = MSPMainWindow()
    #window.load_experiment("data/msp_hardware_test_002.h5")
    window.show()

    return app.exec()

if __name__ == "__main__":
    sys.exit(main())
