import json
from pathlib import Path


class MetadataRegistryError(ValueError):
    """Invalid experiment metadata registry."""


def reject_duplicate_keys(pairs):
    """Construct a dictionary while rejecting duplicate JSON keys."""
    result = {}

    for key, value in pairs:
        if key in result:
            raise MetadataRegistryError(
                f"Duplicate registry identifier: {key}"
            )
        result[key] = value

    return result


def load_metadata_registry(filename: str | Path) -> dict:
    """Load and validate a metadata registry from JSON."""

    filename = Path(filename)

    with filename.open("r", encoding="utf-8") as f:
        registry = json.load(
            f,
            object_pairs_hook=reject_duplicate_keys,
        )

    required_sections = (
        "species",
        "preparations",
        "sex",
        "experimenters",
    )

    for section in required_sections:
        values = registry.get(section)

        if not isinstance(values, dict) or not values:
            raise MetadataRegistryError(
                f"Registry section '{section}' must be a nonempty dictionary"
            )

    for species_id, species in registry["species"].items():
        if not isinstance(species, dict):
            raise MetadataRegistryError(
                f"Invalid species entry: {species_id}"
            )

        label = species.get("label")
        genotypes = species.get("genotypes")

        if not isinstance(label, str) or not label.strip():
            raise MetadataRegistryError(
                f"Species '{species_id}' has no valid label"
            )

        if not isinstance(genotypes, dict) or not genotypes:
            raise MetadataRegistryError(
                f"Species '{species_id}' has no valid genotypes"
            )

        for genotype_id, genotype_label in genotypes.items():
            if not isinstance(genotype_label, str) or not genotype_label.strip():
                raise MetadataRegistryError(
                    f"Invalid genotype '{genotype_id}' for '{species_id}'"
                )

    for section in ("preparations", "sex", "experimenters"):
        for identifier, label in registry[section].items():
            if not isinstance(label, str) or not label.strip():
                raise MetadataRegistryError(
                    f"Invalid entry '{identifier}' in '{section}'"
                )

    return registry
