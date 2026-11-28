"""Run the batch pipeline from three CSV URLs.

Five stages run in order, each timed against its D27-01 target. Real stages
for #19, #20 and #25 are not wired yet, so the command stops at the first stage.
"""

import argparse
import sys
import time
import uuid
from collections.abc import Callable, Sequence
from urllib.parse import urlsplit

from .batch_files import BatchError
from .config import SettingError, load_settings
from .stages import (
    COMMAND_LIMIT_SECONDS,
    STAGE_TARGETS,
    TABLES,
    StageFailed,
    Stages,
    StageUnavailable,
    mlflow_run_url,
    unwired_stages,
)


def _csv_url(value: str) -> str:
    try:
        url = urlsplit(value)
        if url.scheme in {"http", "https"} and url.hostname:
            return value
    except ValueError:
        pass
    raise argparse.ArgumentTypeError("expected an absolute HTTP or HTTPS URL")


def _write(text: str) -> None:
    sys.stdout.write(text + "\n")
    sys.stdout.flush()


def _stop(message: str) -> int:
    sys.stderr.write(f"Batch stopped: {message}\n")
    return 1


def _describe(error: Exception) -> str:
    # Only our own error types carry messages written to be printed; any other
    # message could echo a signed URL or a database credential.
    if isinstance(error, BatchError | StageFailed):
        return f"{type(error).__name__}: {error}"
    return type(error).__name__


def _stage_line(
    number: int, name: str, seconds: float, target: int, note: str = ""
) -> str:
    over = " OVER TARGET" if seconds > target else ""
    return f"Stage {number}/5 {name}: {seconds:.2f} s (target {target} s){over}{note}"


def main(
    argv: Sequence[str] | None = None,
    *,
    stages: Stages | None = None,
    clock: Callable[[], float] = time.perf_counter,
    new_etl_run_id: Callable[[], str] = lambda: uuid.uuid4().hex,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for table in TABLES:
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

    etl_run_id = new_etl_run_id()
    _write(
        f"Database host: {settings.db_host}\n"
        f"Database name: {settings.db_name}\n"
        f"MLflow experiment: {settings.mlflow_experiment_name}\n"
        f"etl_run_id: {etl_run_id}"
    )

    stages = unwired_stages() if stages is None else stages
    started = clock()
    elapsed: list[float] = []
    state: dict = {}

    def fetch():
        batch = stages.fetch.run(args.customers_url, args.orders_url, args.labels_url)
        state["batch"] = batch
        sizes = ", ".join(f"{t} {len(getattr(batch, t))} rows" for t in TABLES)
        _write(f"Batch {batch.batch_id} input: {sizes}")

    def load():
        state["before"] = stages.load.run(state["batch"])

    def read():
        state["rows"] = stages.read.run()

    def train():
        outcome = stages.train.run(state["rows"], state["batch"].batch_id, etl_run_id)
        state["outcome"] = outcome
        if outcome.failures:
            raise StageFailed(
                "candidates failed: " + ", ".join(sorted(outcome.failures))
            )

    def report():
        after = stages.count.run()
        outcome = state["outcome"]
        for table in TABLES:
            _write(
                f"{table}: {state['before'][table]} rows before, {after[table]} after"
            )
        _write(f"MLflow runs logged: {outcome.runs_logged}")
        _write(
            "MLflow URL: "
            + mlflow_run_url(
                settings.mlflow_tracking_uri, outcome.experiment_id, etl_run_id
            )
        )

    steps = (fetch, load, read, train, report)
    for number, ((name, target), step) in enumerate(
        zip(STAGE_TARGETS, steps, strict=True), start=1
    ):
        stage_started = clock()
        try:
            step()
        except StageUnavailable as error:
            return _stop(str(error))
        except Exception as error:  # noqa: BLE001 - any stage failure must exit 1
            _write(
                _stage_line(number, name, clock() - stage_started, target, " FAILED")
            )
            return _stop(f"{name} stage failed ({_describe(error)}).")
        elapsed.append(clock() - stage_started)
        _write(_stage_line(number, name, elapsed[-1], target))

    total = clock() - started
    target_total = sum(target for _, target in STAGE_TARGETS)
    _write(
        f"Stages total: {sum(elapsed):.2f} s (target {target_total} s)\n"
        f"Elapsed since start: {total:.2f} s "
        f"(command limit {COMMAND_LIMIT_SECONDS} s, excluding process startup)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
