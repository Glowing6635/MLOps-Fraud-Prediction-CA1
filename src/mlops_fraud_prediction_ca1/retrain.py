"""Per-batch evaluation-mode retraining (PBI #25)."""

import argparse
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from importlib import import_module

import pandas as pd

from .rds_reads import read_training_snapshot
from .retrain_adapter import make_fit_and_log, make_mlflow_logger

TRAIN_AND_LOG_SECONDS = 85


@dataclass(frozen=True)
class ModelSpec:
    name: str
    owner: str
    build: Callable[[], object]
    params: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationInputs:
    models: Sequence[ModelSpec]
    engineers: Sequence[str]
    validation_rows: pd.DataFrame
    evaluate_top_k: Callable
    k: int
    champion_run_id: str
    champion_model: object


@dataclass
class RetrainReport:
    etl_run_id: str
    batch_id: str
    counts: dict[str, int]
    results: dict[str, dict] = field(default_factory=dict)
    failures: dict[str, str] = field(default_factory=dict)

    @property
    def exit_code(self) -> int:
        return int(bool(self.failures))


def validate_manifest(models: Sequence[ModelSpec], engineers: Sequence[str]) -> None:
    names = [model.name for model in models]
    if not names or len(set(names)) != len(names) or not all(names):
        raise ValueError("The manifest needs nonempty, unique model names.")
    roster = set(engineers)
    if not roster or "" in roster:
        raise ValueError("Supply the expected engineer roster.")
    owners = {model.owner for model in models}
    if roster != owners:
        raise ValueError(
            f"Manifest owners differ from roster: missing={sorted(roster - owners)}, "
            f"unexpected={sorted(owners - roster)}."
        )


def retrain(
    inputs: EvaluationInputs,
    *,
    batch_id: str,
    etl_run_id: str,
    read_rows: Callable[[], tuple[pd.DataFrame, dict[str, int]]],
    log_run: Callable,
    log: Callable[[str], None],
) -> RetrainReport:
    validate_manifest(inputs.models, inputs.engineers)
    rows, exclusions = read_rows()
    train_rows = rows.loc[rows["is_fraud"].isin([0, 1])]
    new_rows = int(train_rows["batch_id"].eq(batch_id).sum())
    counts = {
        **exclusions,
        "train_old_rows": len(train_rows) - new_rows,
        "train_new_rows": new_rows,
        "excluded_invalid_label_rows": len(rows) - len(train_rows),
        "train_orders": len(train_rows),
        "excluded_rows": exclusions["excluded_date_rows"] + len(rows) - len(train_rows),
    }
    report = RetrainReport(etl_run_id, batch_id, counts)
    log(f"etl_run_id={etl_run_id} batch_id={batch_id} {counts}")
    started = time.perf_counter()
    fit_and_log = make_fit_and_log(inputs, log_run)
    for spec in inputs.models:
        name = f"{spec.name}-{batch_id}-{etl_run_id}"
        try:
            report.results[spec.name] = fit_and_log(
                spec, train_rows, name, counts, batch_id, etl_run_id
            )
        except Exception as error:  # noqa: BLE001 - isolate each candidate failure
            report.failures[spec.name] = f"{type(error).__name__}: {error}"
            log(f"FAILED {name}: {report.failures[spec.name]}")
    log(
        f"Train and log: {time.perf_counter() - started:.2f}s / {TRAIN_AND_LOG_SECONDS}s"
    )
    return report


def run_evaluation_mode(
    engine, inputs, *, batch_id, log_run, etl_run_id=None, log=print
) -> RetrainReport:
    """Read the #154 training snapshot once and share it across all candidates."""
    return retrain(
        inputs,
        batch_id=batch_id,
        etl_run_id=uuid.uuid4().hex if etl_run_id is None else etl_run_id,
        read_rows=lambda: read_training_snapshot(engine),
        log_run=log_run,
        log=log,
    )


def main(argv=None) -> int:
    from .config import create_db_engine, load_settings

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch_id")
    parser.add_argument(
        "inputs", help="Trusted module:factory returning EvaluationInputs"
    )
    parser.add_argument("--etl-run-id")
    args = parser.parse_args(argv)
    module, separator, factory = args.inputs.partition(":")
    if not separator or not module or not factory:
        parser.error("inputs must be module:factory")
    settings = load_settings()
    inputs = getattr(import_module(module), factory)()
    engine = create_db_engine(settings)
    try:
        report = run_evaluation_mode(
            engine,
            inputs,
            batch_id=args.batch_id,
            etl_run_id=args.etl_run_id,
            log_run=make_mlflow_logger(settings.mlflow_experiment_name),
        )
    finally:
        engine.dispose()
    return report.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
