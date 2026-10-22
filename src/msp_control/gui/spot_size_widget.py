"""Widget for specifying the illuminated area."""

from PySide6.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QLabel,
    QDoubleSpinBox,
    QComboBox,
)

from msp_control.calibration.geometry import Spotsize


class SpotSizeWidget(QWidget):
    """Specify a circular spot by diameter or radius."""

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.size_spin = QDoubleSpinBox()
        self.size_spin.setRange(0.1, 100000.0)
        self.size_spin.setDecimals(1)
        self.size_spin.setValue(100.0)
        self.size_spin.setSuffix(" µm")

        self.measurement_combo = QComboBox()
        self.measurement_combo.addItems(["Diameter", "Radius"])

        self.area_label = QLabel()

        layout.addWidget(self.size_spin)
        layout.addWidget(self.measurement_combo)
        layout.addWidget(self.area_label)

        self.size_spin.valueChanged.connect(self.update_area)
        self.measurement_combo.currentIndexChanged.connect(
            self.update_area
        )

        self.update_area()

    @property
    def area_um2(self) -> float:
        """Return the calculated illuminated area in µm²."""

        size = self.size_spin.value()

        if self.measurement_combo.currentText() == "Diameter":
            return Spotsize.circle_area(diameter=size)

        return Spotsize.circle_area(radius=size)

    def update_area(self):
        """Update the displayed area."""

        self.area_label.setText(
            f"Area: {self.area_um2:,.0f} µm²"
        )
