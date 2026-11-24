"""Per-batch automatic retraining of every candidate model (PBI #25).

`retrain(...)` reads the training rows once, fits each model in the manifest on them
with its stored hyperparameters (no search), compares each with the current champion,
and keeps going when one model fails. It never promotes a model. The collaborators
(data reader, per-model fitter, champion lookup) are passed in, so the orchestration is
testable without RDS or MLflow; `run_evaluation_mode` wires the real reader.

Evaluation mode (holdout on) uses only rows before the #6 validation window. Release
mode (after the #23 final evaluation) is not implemented here: it waits for #23.
"""

import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import pandas as pd

MAX_ARTIFACT_BYTES = 50 * 1024 * 1024


@dataclass(frozen=True)
class ModelSpec:
    """One entry of the training list: a name, its owner, and a factory with stored parameters."""

    name: str
    owner: str
    build: Callable[[], object]


@dataclass
class RetrainReport:
    etl_run_id: str
    batch_id: str
    rows_read: int
    train_rows: int
    excluded_rows: int
    results: dict[str, dict] = field(default_factory=dict)
    failures: dict[str, str] = field(default_factory=dict)

    @property
    def exit_code(self) -> int:
        return 1 if self.failures else 0


def new_etl_run_id() -> str:
    return uuid.uuid4().hex


def run_name(etl_run_id: str, batch_id: str, model: str) -> str:
    """Names contain both IDs, so one pipeline run filters with `name LIKE '%<id>%'`."""
    return f"{model}-{batch_id}-{etl_run_id}"


def eligible_rows(
    all_rows: pd.DataFrame, validation_start, date_column: str = "order_date"
) -> tuple[pd.DataFrame, int]:
    """Rows dated before the validation window, and how many later rows were excluded.

    Rows from any batch qualify by date alone, so a batch is never excluded as a whole.
    """
    dates = pd.to_datetime(all_rows[date_column])
    keep = dates < pd.Timestamp(validation_start)
    return all_rows[keep], int((~keep).sum())


def validate_manifest(models: Sequence[ModelSpec]) -> None:
    names = [m.name for m in models]
    if not models:
        raise ValueError("The training list is empty.")
    if len(set(names)) != len(names):
        raise ValueError("Model names in the training list must be unique.")
    if len({m.owner for m in models}) < 2:
        raise ValueError("The training list needs at least one model per engineer.")


def retrain(
    models: Sequence[ModelSpec],
    *,
    batch_id: str,
    read_rows: Callable[[], pd.DataFrame],
    validation_start,
    fit_and_log: Callable[[ModelSpec, pd.DataFrame, str], dict],
    champion_comparison: Callable[[dict], dict],
    etl_run_id: str | None = None,
    log: Callable[[str], None] = lambda message: None,
) -> RetrainReport:
    """Retrain every model on the shared training rows; see the module docstring.

    `fit_and_log(spec, train_rows, run_name)` fits, scores on the frozen validation
    window and logs the run, returning metrics with `model_size_bytes` and
    `fit_seconds`. `champion_comparison(metrics)` returns the comparison to log.
    """
    validate_manifest(models)
    etl_run_id = etl_run_id or new_etl_run_id()
    rows = read_rows()  # once per run, shared by all models
    train_rows, excluded = eligible_rows(rows, validation_start)
    report = RetrainReport(etl_run_id, batch_id, len(rows), len(train_rows), excluded)
    log(
        f"etl_run_id={etl_run_id} batch={batch_id} read={len(rows)} "
        f"train={len(train_rows)} excluded_in_or_after_validation={excluded}"
    )
    for spec in models:
        name = run_name(etl_run_id, batch_id, spec.name)
        try:
            metrics = fit_and_log(spec, train_rows, name)
            if metrics.get("model_size_bytes", 0) > MAX_ARTIFACT_BYTES:
                raise ValueError("model artifact is over 50 MB")
            metrics["champion_comparison"] = champion_comparison(metrics)
            report.results[spec.name] = metrics
        except Exception as error:  # noqa: BLE001 - one failed model must not stop the rest
            report.failures[spec.name] = f"{type(error).__name__}: {error}"
            log(f"FAILED {spec.name}: {report.failures[spec.name]}")
    return report


def run_evaluation_mode(
    engine, models, *, batch_id, fit_and_log, champion_comparison, log=print
):
    """Evaluation mode against the real database: one `purpose="analysis"` read."""
    from .rds_reads import read_labelled_orders
    from .splitting import load_policy

    policy = load_policy()
    return retrain(
        models,
        batch_id=batch_id,
        read_rows=lambda: read_labelled_orders(engine, purpose="analysis"),
        validation_start=policy.validation_start,
        fit_and_log=fit_and_log,
        champion_comparison=champion_comparison,
        log=log,
    )
