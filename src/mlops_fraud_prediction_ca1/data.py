"""Load the base fraud-prediction datasets without cleaning source values."""

import logging
from pathlib import Path

import pandas as pd


def load_base_data(
    data_dir: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read the three base CSVs from an explicitly selected directory.

    Args:
        data_dir: Directory containing customers_base.csv, orders_base.csv,
            and labels_base.csv. Relative paths use the current working
            directory. A leading `~` refers to the user's home directory,
            so `~/datasets/fraud` can be used instead of a full absolute path.

    Returns:
        DataFrames in the order (customers, orders, labels). IDs, categories,
        flags and dates use explicit string dtypes; numeric columns and missing
        values follow pandas read_csv defaults. Leading zeros and flag spelling
        are preserved across batches. No cleaning,
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
    logger: logging.Logger = logging.getLogger(__name__)

    # Path() does not expand "~" automatically. Expand it so callers can use
    # home-relative directories such as "~/datasets/fraud".
    directory: Path = Path(data_dir).expanduser()
    paths: tuple[Path, Path, Path] = (
        directory / "customers_base.csv",
        directory / "orders_base.csv",
        directory / "labels_base.csv",
    )
    missing_files: list[str] = [str(path) for path in paths if not path.is_file()]
    if len(missing_files) > 0:
        raise FileNotFoundError(
            "Required CSV files not found: " + ", ".join(missing_files)
        )

    logger.info("Loading base datasets from %s", directory)
    frames: list[pd.DataFrame] = []
    for path in paths:
        # Pandas' default returns a DataFrame; PyCharm misreads its Unpack overloads.
        # noinspection PyTypeChecker,PyArgumentList
        frame: pd.DataFrame = pd.read_csv(
            path,
            dtype={
                column: "string"
                for column in (
                    "order_id",
                    "customer_id",
                    "country_code",
                    "mobile_verified",
                    "is_fraud",
                    "order_date",
                    "first_order_datetime",
                    "collect_type",
                    "payment_method",
                )
            },
        )
        logger.info(
            "Loaded %-18s | %9s rows | %2d columns",
            path.name,
            f"{len(frame):,}",
            len(frame.columns),
        )
        frames.append(frame)

    return frames[0], frames[1], frames[2]
