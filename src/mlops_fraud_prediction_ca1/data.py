"""Load the base fraud-prediction datasets without cleaning source values."""

import logging
from pathlib import Path

import pandas as pd

logger: logging.Logger = logging.getLogger(__name__)


def load_base_data(data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read the three base CSVs from an explicitly selected directory.

    Args:
        data_dir: Directory containing customers_base.csv, orders_base.csv,
            and labels_base.csv. Relative paths use the current working
            directory; a leading tilde is expanded.

    Returns:
        DataFrames in the order (customers, orders, labels). Pandas infers
        types and missing values using its read_csv defaults. No cleaning,
        date parsing, joining, or source-file modification is performed.

    Raises:
        FileNotFoundError: One or more required files are absent. All paths
            are checked before any CSV is read.
        OSError: An input cannot be read.
        pandas.errors.EmptyDataError: An input has no columns to parse.
        pandas.errors.ParserError: An input cannot be parsed as CSV.

    Logs each filename and its loaded dimensions at INFO level. Logging
    handlers and verbosity are configured by the caller.
    """
    directory: Path = Path(data_dir).expanduser()
    paths: tuple[Path, Path, Path] = (
        directory / "customers_base.csv",
        directory / "orders_base.csv",
        directory / "labels_base.csv",
    )
    missing: list[str] = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required CSV files not found: " + ", ".join(missing))

    frames: list[pd.DataFrame] = []
    for path in paths:
        logger.info("Loading %s", path)
        frame: pd.DataFrame = pd.read_csv(path)
        logger.info("Loaded %s: %d rows, %d columns", path.name, *frame.shape)
        frames.append(frame)

    return frames[0], frames[1], frames[2]
