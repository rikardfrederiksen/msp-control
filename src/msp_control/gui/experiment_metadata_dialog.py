from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QVBoxLayout,
    QDateEdit,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
)

from PySide6.QtCore import QDate


class ExperimentMetadataDialog(QDialog):
    """Dialog for creating or editing experiment metadata."""

    def __init__(
        self,
        registry: dict,
        metadata: dict | None = None,
        parent=None,
    ):
        super().__init__(parent)

        self.registry = registry

        self.setWindowTitle("Experiment metadata")
        self.resize(500, 550)

        layout = QVBoxLayout(self)
        form = QFormLayout()

        # Species selection
        self.species_combo = QComboBox()

        for species_id, species in registry["species"].items():
            self.species_combo.addItem(
                species["label"],
                userData=species_id,
            )

        form.addRow("Species:", self.species_combo)

        # Genotype selection
        self.genotype_combo = QComboBox()
        form.addRow("Genotype:", self.genotype_combo)

        # Experiment date is fixed when the dialog is created.
        # Later, editing an existing experiment will use its stored date.
        self.experiment_date = QDate.currentDate()

        # Date of birth
        self.dob_edit = QDateEdit()
        self.dob_edit.setCalendarPopup(True)
        self.dob_edit.setDisplayFormat("yyyy-MM-dd")
        self.dob_edit.setMaximumDate(self.experiment_date)
        self.dob_edit.setDate(self.experiment_date)

        form.addRow("Date of birth:", self.dob_edit)

        # Calculated age (read-only)
        self.age_label = QLabel()
        form.addRow("Age:", self.age_label)

        self.dob_edit.dateChanged.connect(self.update_age)
        self.update_age()

        # Sex
        self.sex_combo = QComboBox()

        for identifier, label in registry["sex"].items():
            self.sex_combo.addItem(label, userData=identifier)

        form.addRow("Sex:", self.sex_combo)

        # Animal identification
        self.animal_id_edit = QLineEdit()
        form.addRow("Animal ID:", self.animal_id_edit)

        self.cage_card_edit = QLineEdit()
        form.addRow("Cage card:", self.cage_card_edit)

        self.animal_identifier_edit = QLineEdit()
        form.addRow("Animal identifier:", self.animal_identifier_edit)

        # Additional genotype information
        self.genotype_notes_edit = QLineEdit()
        form.addRow("Genotype notes:", self.genotype_notes_edit)

        # Preparation
        self.preparation_combo = QComboBox()

        for identifier, label in registry["preparations"].items():
            self.preparation_combo.addItem(label, userData=identifier)

        form.addRow("Preparation:", self.preparation_combo)

        # Experimenter
        self.experimenter_combo = QComboBox()

        for identifier, label in registry["experimenters"].items():
            self.experimenter_combo.addItem(label, userData=identifier)

        form.addRow("Experimenter:", self.experimenter_combo)

        # General notes
        self.notes_edit = QPlainTextEdit()
        self.notes_edit.setPlaceholderText("Experiment notes")
        self.notes_edit.setMaximumHeight(90)

        form.addRow("Notes:", self.notes_edit)

        layout.addLayout(form)

        # OK / Cancel
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )

        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout.addWidget(buttons)

        # Update genotype options when species changes
        self.species_combo.currentIndexChanged.connect(
            self.update_genotypes
        )

        self.species_combo.currentIndexChanged.connect(
            self.update_age
        )

        self.update_genotypes()

        if metadata is not None:
            self.set_metadata(metadata)

    def update_genotypes(self, index=None):
        """Populate genotypes for the selected species."""

        species_id = self.species_combo.currentData()

        self.genotype_combo.clear()

        if species_id is None:
            return

        genotypes = self.registry["species"][species_id]["genotypes"]

        for genotype_id, label in genotypes.items():
            self.genotype_combo.addItem(
                label,
                userData=genotype_id,
            )

    def update_age(self, date=None):
        """Update age from DOB and the experiment date."""

        age_days = self.dob_edit.date().daysTo(
            self.experiment_date
        )

        species_id = self.species_combo.currentData()

        if species_id == "mouse":
            age_text = f"P{age_days} ({age_days} days)"
        else:
            age_text = f"{age_days} days"

        self.age_label.setText(age_text)
   
    def get_metadata(self) -> dict:
        """Return standardized experiment metadata."""

        age_days = self.dob_edit.date().daysTo(
            self.experiment_date
        )

        return {
            "species": self.species_combo.currentData(),
            "genotype": self.genotype_combo.currentData(),
            "genotype_notes": self.genotype_notes_edit.text().strip(),
            "dob": self.dob_edit.date().toString("yyyy-MM-dd"),
            "experiment_date": self.experiment_date.toString("yyyy-MM-dd"),
            "age_days": age_days,
            "sex": self.sex_combo.currentData(),
            "animal_id": self.animal_id_edit.text().strip(),
            "cage_card": self.cage_card_edit.text().strip(),
            "animal_identifier": self.animal_identifier_edit.text().strip(),
            "preparation": self.preparation_combo.currentData(),
            "experimenter": self.experimenter_combo.currentData(),
            "notes": self.notes_edit.toPlainText().strip(),
        }

    def set_metadata(self, metadata: dict):
        """Populate the dialog from existing experiment metadata."""

        # Preserve the original experiment date.
        date_string = metadata.get("experiment_date")
        if date_string:
            date = QDate.fromString(date_string, "yyyy-MM-dd")
            if not date.isValid():
                raise ValueError("Invalid experiment date")
            self.experiment_date = date
            self.dob_edit.setMaximumDate(date)

        # Species must be set before genotype because
        # changing species repopulates the genotype dropdown.
        def select(combo, value, field):
            index = combo.findData(value)
            if index < 0:
                raise ValueError(
                    f"Unknown {field} identifier: {value}"
                )
            combo.setCurrentIndex(index)

        if "species" in metadata:
            select(
                self.species_combo,
                metadata["species"],
                "species",
            )

        if "genotype" in metadata:
            select(
                self.genotype_combo,
                metadata["genotype"],
                "genotype",
            )

        if "dob" in metadata:
            dob = QDate.fromString(metadata["dob"], "yyyy-MM-dd")
            if not dob.isValid():
                raise ValueError("Invalid date of birth")
            self.dob_edit.setDate(dob)

        for key, combo in (
            ("sex", self.sex_combo),
            ("preparation", self.preparation_combo),
            ("experimenter", self.experimenter_combo),
        ):
            if key in metadata:
                select(combo, metadata[key], key)

        for key, widget in (
            ("animal_id", self.animal_id_edit),
            ("cage_card", self.cage_card_edit),
            ("animal_identifier", self.animal_identifier_edit),
            ("genotype_notes", self.genotype_notes_edit),
        ):
            widget.setText(metadata.get(key, ""))

        self.notes_edit.setPlainText(metadata.get("notes", ""))

        self.update_age()
