from pathlib import Path


def experiment_directory(
    data_root: Path,
    experimenter: str,
    experimenters: dict[str, str],
) -> Path:
    """Return the starting directory for a new experiment."""

    data_root = Path(data_root)

    name = experimenters.get(experimenter)

    if name:
        user_directory = data_root / name

        if user_directory.is_dir():
            return user_directory

    return data_root
