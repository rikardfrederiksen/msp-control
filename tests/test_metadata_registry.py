import json

import pytest

from msp_control.metadata_registry import (
    MetadataRegistryError,
    load_metadata_registry,
)


@pytest.fixture
def valid_registry():
    return {
        "species": {
            "mouse": {
                "label": "Mouse (Mus musculus)",
                "genotypes": {
                    "WT": "Wild type",
                    "Arr1_KO": "Arr1−/−",
                },
            }
        },
        "preparations": {
            "isolated_retina": "Isolated retina",
        },
        "sex": {
            "female": "Female",
            "male": "Male",
        },
        "experimenters": {
            "RF": "Rikard Frederiksen",
        },
    }


def write_registry(tmp_path, registry):
    filename = tmp_path / "experiment_metadata.json"
    filename.write_text(
        json.dumps(registry),
        encoding="utf-8",
    )
    return filename


def test_load_valid_registry(tmp_path, valid_registry):
    filename = write_registry(tmp_path, valid_registry)

    result = load_metadata_registry(filename)

    assert result == valid_registry


def test_missing_required_section(tmp_path, valid_registry):
    del valid_registry["species"]

    filename = write_registry(tmp_path, valid_registry)

    with pytest.raises(MetadataRegistryError):
        load_metadata_registry(filename)


def test_empty_genotype_list(tmp_path, valid_registry):
    valid_registry["species"]["mouse"]["genotypes"] = {}

    filename = write_registry(tmp_path, valid_registry)

    with pytest.raises(MetadataRegistryError):
        load_metadata_registry(filename)


def test_invalid_genotype_label(tmp_path, valid_registry):
    valid_registry["species"]["mouse"]["genotypes"]["WT"] = ""

    filename = write_registry(tmp_path, valid_registry)

    with pytest.raises(MetadataRegistryError):
        load_metadata_registry(filename)


def test_empty_experimenter_list(tmp_path, valid_registry):
    valid_registry["experimenters"] = {}

    filename = write_registry(tmp_path, valid_registry)

    with pytest.raises(MetadataRegistryError):
        load_metadata_registry(filename)


def test_duplicate_genotype_identifier(tmp_path):
    filename = tmp_path / "experiment_metadata.json"

    filename.write_text(
        """
        {
            "species": {
                "mouse": {
                    "label": "Mouse",
                    "genotypes": {
                        "WT": "Wild type",
                        "WT": "Another wild type"
                    }
                }
            },
            "preparations": {"retina": "Retina"},
            "sex": {"unknown": "Unknown"},
            "experimenters": {"RF": "Rikard Frederiksen"}
        }
        """,
        encoding="utf-8",
    )

    with pytest.raises(
        MetadataRegistryError,
        match="Duplicate registry identifier: WT",
    ):
        load_metadata_registry(filename)
