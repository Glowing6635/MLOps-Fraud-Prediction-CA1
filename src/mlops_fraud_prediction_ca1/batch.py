"""Run the batch pipeline from three CSV URLs."""

import argparse
import sys
from collections.abc import Sequence
from urllib.parse import urlsplit

from .config import SettingError, load_settings


def _csv_url(value: str) -> str:
    try:
        url = urlsplit(value)
        if url.scheme in {"http", "https"} and url.hostname:
            return value
    except ValueError:
        pass
    raise argparse.ArgumentTypeError("expected an absolute HTTP or HTTPS URL")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for table in ("customers", "orders", "labels"):
        parser.add_argument(
            f"{table}_url", type=_csv_url, help=f"URL of the {table} CSV"
        )
    _, extra = parser.parse_known_args(argv)
    if extra:
        parser.error("expected exactly three CSV URLs")

    try:
        settings = load_settings()
    except SettingError as exc:
        sys.stderr.write(f"Configuration error: {exc}\n")
        return 1

    sys.stdout.write(
        f"Database host: {settings.db_host}\n"
        f"Database name: {settings.db_name}\n"
        f"MLflow experiment: {settings.mlflow_experiment_name}\n"
    )
    sys.stdout.flush()
    sys.stderr.write("Batch stopped: fetch and validate stage (#19) is unavailable.\n")
    return 1


if __name__ == "__main__":
    sys.exit(main())
