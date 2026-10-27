from msp_control.gui.data_directory import experiment_directory


def test_existing_experimenter_directory(tmp_path):
    user_directory = tmp_path / "Rikard"
    user_directory.mkdir()

    result = experiment_directory(
        tmp_path,
        "RF",
        {"RF": "Rikard", "CM": "Chris"},
    )

    assert result == user_directory


def test_missing_experimenter_directory(tmp_path):
    result = experiment_directory(
        tmp_path,
        "CM",
        {"RF": "Rikard", "CM": "Chris"},
    )

    assert result == tmp_path


def test_unknown_experimenter(tmp_path):
    result = experiment_directory(
        tmp_path,
        "XX",
        {"RF": "Rikard"},
    )

    assert result == tmp_path
