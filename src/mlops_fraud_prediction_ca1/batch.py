"""The one-command batch pipeline (issue #27): load a batch, retrain and report, within 3 minutes.

`uv run run-batch <customers URL> <orders URL> <labels URL>` needs no other input.
Before any work it prints the database and MLflow experiment it will use, and a new
`etl_run_id` that tags every MLflow run it logs. It then runs five stages in order,
printing each one's wall-clock time against its budget in D27-01 of
docs/decision-log.md:

1. Fetch and validate: `batch_files.read_batch`.
2. Clean and load: count each table's rows, then `loading.load_batch`.
3. Read training data: the labelled training and validation windows, through
   `rds_reads.read_labelled_orders`.
4. Train and log: the baseline model, through the shared harness (`training.train`).
5. Report: each table's rows before and after, the MLflow runs logged with this
   `etl_run_id`, and a link that lists them on the MLflow server.

A failed stage stops the run with exit status 1. The load is one transaction, so a
failure before it ends leaves the database as it was. After it, the batch stays
loaded, and running the command again is safe: it adds no rows and logs new runs.
"""

import argparse
import logging
import sys
import time
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from urllib.parse import quote, urlsplit

import mlflow
import pandas as pd
from mlflow.exceptions import MlflowException
from sqlalchemy import Engine
from sqlalchemy.exc import SQLAlchemyError

from .batch_files import BatchError, read_batch
from .config import SettingError, Settings, create_db_engine, load_settings
from .loading import load_batch
from .models import baseline_model
from .rds_reads import read_labelled_orders
from .top_k import K
from .training import train

# Each stage's wall-clock budget in seconds, from D27-01 of docs/decision-log.md.
# Change them only together with that entry.
_BUDGETS = {
    "Fetch and validate": 25,
    "Clean and load": 30,
    "Read training data": 15,
    "Train and log": 85,
    "Report": 15,
}
# The whole command's limit (issue #27). The 10 seconds the stages leave are for
# start-up and shutdown.
_COMMAND_LIMIT = 180
# A dead link fails after about 4 x 10 s plus 17 s of retry waits, not the
# downloader's default of about 4 x 30 s plus 17 s, so the run stops well within its limit.
_DOWNLOAD_TIMEOUT = 10
_TABLES = ("batches", "customers", "orders", "labels")


def _csv_url(value: str) -> str:
    try:
        url = urlsplit(value)
        if url.scheme in {"http", "https"} and url.hostname:
            return value
    except ValueError:
        pass
    raise argparse.ArgumentTypeError("expected an absolute HTTP or HTTPS URL")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for table in ("customers", "orders", "labels"):
        parser.add_argument(
            f"{table}_url", type=_csv_url, help=f"URL of the {table} CSV"
        )
    # parse_args would echo extra URLs and expose their signed query strings.
    args, extra = parser.parse_known_args(argv)
    if extra:
        parser.error("expected exactly three CSV URLs")

    try:
        settings = load_settings()
    except SettingError as exc:
        sys.stderr.write(f"Configuration error: {exc}\n")
        return 1

    etl_run_id = uuid.uuid4().hex
    sys.stdout.write(
        f"Database host: {settings.db_host}\n"
        f"Database name: {settings.db_name}\n"
        f"MLflow experiment: {settings.mlflow_experiment_name}\n"
        f"ETL run ID: {etl_run_id}\n"
    )
    # MLflow prints an emoji when a run ends, which Windows cannot encode when piped.
    sys.stdout.reconfigure(errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    started = time.perf_counter()
    engine = create_db_engine(settings)
    try:
        _run(
            engine,
            settings,
            (args.customers_url, args.orders_url, args.labels_url),
            etl_run_id,
        )
    except (BatchError, SQLAlchemyError, MlflowException, ValueError) as error:
        sys.stderr.write(f"Batch stopped: {error}\n")
        return 1
    finally:
        engine.dispose()
    seconds = time.perf_counter() - started
    over = "  OVER BUDGET" if seconds > _COMMAND_LIMIT else ""
    sys.stdout.write(f"Total: {seconds:.1f} s of {_COMMAND_LIMIT} s{over}\n")
    return 0


def _run(
    engine: Engine, settings: Settings, urls: tuple[str, str, str], etl_run_id: str
) -> None:
    with _stage("Fetch and validate"):
        batch = read_batch(*urls, timeout=_DOWNLOAD_TIMEOUT)
    sys.stdout.write(
        f"Batch {batch.batch_id}: {len(batch.customers)} customers,"
        f" {len(batch.orders)} orders, {len(batch.labels)} labels\n"
    )
    with _stage("Clean and load"):
        before = _count_rows(engine)
        load_batch(engine, batch)
    with _stage("Read training data"):
        windows = tuple(
            _labelled_orders(engine, purpose) for purpose in ("train", "validation")
        )
    with _stage("Train and log"):
        # TODO(#25): run the per-batch retraining of every candidate instead, once it
        # reaches this branch; until then the baseline stands in for the candidates.
        # Keep the etl_run_id tag on every run: the report counts and links runs by it.
        train(
            baseline_model(),
            k=K,
            seed=0,
            labelled_windows=windows,
            tags={
                "mlflow.runName": f"baseline-{batch.batch_id}-{etl_run_id}",
                "etl_run_id": etl_run_id,
                "batch_id": batch.batch_id,
                # A per-batch run is an evaluation, never a champion candidate (#25).
                "champion_eligible": "False",
            },
        )
    with _stage("Report"):
        after = _count_rows(engine)
        experiment = mlflow.get_experiment_by_name(settings.mlflow_experiment_name)
        run_filter = f"tags.etl_run_id = '{etl_run_id}'"
        runs = mlflow.search_runs(
            [experiment.experiment_id], run_filter, output_format="list"
        )
        sys.stdout.write("Rows per table, before and after:\n")
        for table in _TABLES:
            sys.stdout.write(f"  {table}: {before[table]} -> {after[table]}\n")
        sys.stdout.write(
            f"MLflow runs logged: {len(runs)}\n"
            f"MLflow runs: {settings.mlflow_tracking_uri.rstrip('/')}/#/experiments/"
            f"{experiment.experiment_id}?searchFilter={quote(run_filter)}\n"
        )


@contextmanager
def _stage(name: str) -> Iterator[None]:
    """Time one stage and print its time against its budget, or that it failed."""
    start = time.perf_counter()
    try:
        yield
    except Exception:
        sys.stdout.write(f"{name}: failed after {time.perf_counter() - start:.1f} s\n")
        raise
    seconds = time.perf_counter() - start
    over = "  OVER BUDGET" if seconds > _BUDGETS[name] else ""
    sys.stdout.write(f"{name}: {seconds:.1f} s of {_BUDGETS[name]} s{over}\n")


def _count_rows(engine: Engine) -> dict[str, int]:
    with engine.connect() as connection:
        return {
            table: connection.exec_driver_sql(f"SELECT COUNT(*) FROM {table}").scalar()
            for table in _TABLES
        }


def _labelled_orders(engine: Engine, purpose: str) -> pd.DataFrame:
    rows = read_labelled_orders(engine, purpose=purpose)
    # The harness needs a 0 or 1 label; None means the source label was neither.
    return rows[rows["is_fraud"].notna()]


if __name__ == "__main__":
    sys.exit(main())
