"""Per-batch evaluation-mode retraining (PBI #25)."""

import argparse
import hashlib
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from importlib import import_module

import pandas as pd

from .champion import SIZE_NOTE_BYTES
from .rds_reads import read_release_snapshot, read_training_snapshot
from .splitting import split_orders

TRAIN_AND_LOG_SECONDS = 85
MAX_ARTIFACT_BYTES = SIZE_NOTE_BYTES  # 50,000,000 bytes, as in D15-03
MODES = ("evaluation", "release")


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
    k: int
    champion_run_id: str
    champion_model: object
    seed: int | None = None


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
    train_candidate: Callable,
    score_rows: Callable,
    log: Callable[[str], None],
    mode: str = "evaluation",
) -> RetrainReport:
    if mode not in MODES:
        raise ValueError(f"Unknown retraining mode: {mode}")
    release = mode == "release"
    validate_manifest(inputs.models, inputs.engineers)
    validation, split = split_orders(inputs.validation_rows, purpose="validation")
    if (
        (not release and not split["test_locked"])
        or not inputs.champion_run_id
        or validation.empty
        or split["excluded_rows"]
        or not validation["is_fraud"].isin([0, 1]).all()
    ):
        raise ValueError("Supply a labelled frozen #6 validation window and champion.")
    rows, exclusions = read_rows()
    train_rows = rows.loc[rows["is_fraud"].isin([0, 1])]
    new_rows = int(train_rows["batch_id"].eq(batch_id).sum())
    counts = {
        **exclusions,
        "train_old_rows": len(train_rows) - new_rows,
        "train_new_rows": new_rows,
        "excluded_invalid_label_rows": len(rows) - len(train_rows),
        "eligible_train_orders": len(train_rows),
        "excluded_rows": exclusions["excluded_date_rows"] + len(rows) - len(train_rows),
    }
    if release and not new_rows:
        raise ValueError(
            f"Release mode found no eligible labelled rows for batch {batch_id}; "
            "a date cut-off or missing labels would silently exclude it."
        )
    report = RetrainReport(
        etl_run_id, batch_id, {**counts, "train_orders": len(train_rows)}
    )
    log(f"mode={mode} etl_run_id={etl_run_id} batch_id={batch_id} {counts}")
    started = time.perf_counter()
    validation_hash = hashlib.sha256(
        validation.to_csv(index=False).encode()
    ).hexdigest()
    champion = score_rows(inputs.champion_model, validation, k=inputs.k)
    champion_caught = champion.at_k.fraud_caught
    champion_ap = champion.average_precision
    if champion_ap is None:
        raise ValueError("Frozen validation must support average precision.")

    def comparison(result):
        # Release models also fit the frozen validation rows, so their scores are
        # in-sample (tagged scores_out_of_sample=False): the comparison is logged for
        # the record but cannot support promotion.
        caught = result.report.at_k.fraud_caught
        ap = result.report.average_precision
        return {
            **counts,
            "champion_fraud_caught_at_k": champion_caught,
            "champion_average_precision": champion_ap,
            "fraud_caught_at_k_delta": caught - champion_caught,
            "average_precision_delta": ap - champion_ap,
            "beats_champion": int((caught, ap) > (champion_caught, champion_ap)),
        }

    for spec in inputs.models:
        name = f"{spec.name}-{batch_id}-{etl_run_id}"
        try:
            estimator = spec.build().set_params(**spec.params)
            run_id, result = train_candidate(
                estimator,
                k=inputs.k,
                seed=inputs.seed,
                labelled_windows=(train_rows, validation),
                max_model_size_bytes=MAX_ARTIFACT_BYTES,
                extra_metrics=comparison,
                tags={
                    "mlflow.runName": name,
                    "champion_eligible": "False",
                    "retrain_mode": mode,
                    "scores_out_of_sample": str(not release),
                    "model": spec.name,
                    "owner": spec.owner,
                    "etl_run_id": etl_run_id,
                    "batch_id": batch_id,
                    "champion_run_id": inputs.champion_run_id,
                    "validation_sha256": validation_hash,
                    "model_params": repr(dict(spec.params)),
                },
            )
            report.results[spec.name] = {
                **comparison(result),
                "fraud_caught_at_k": result.report.at_k.fraud_caught,
                "average_precision": result.report.average_precision,
                "fit_seconds": result.fit_seconds,
                "mlflow_run_id": run_id,
                "scores_out_of_sample": not release,
            }
        except Exception as error:  # noqa: BLE001 - isolate each candidate failure
            report.failures[spec.name] = f"{type(error).__name__}: {error}"
            log(f"FAILED {name}: {report.failures[spec.name]}")
    log(
        f"Train and log: {time.perf_counter() - started:.2f}s / {TRAIN_AND_LOG_SECONDS}s"
    )
    return report


def run_evaluation_mode(
    engine, inputs, *, batch_id, train_candidate, score_rows, etl_run_id=None, log=print
) -> RetrainReport:
    """Read the #154 training snapshot once and share it across all candidates."""
    return retrain(
        inputs,
        batch_id=batch_id,
        etl_run_id=uuid.uuid4().hex if etl_run_id is None else etl_run_id,
        read_rows=lambda: read_training_snapshot(engine),
        train_candidate=train_candidate,
        score_rows=score_rows,
        log=log,
    )


def run_release_mode(
    engine, inputs, *, batch_id, train_candidate, score_rows, etl_run_id=None, log=print
) -> RetrainReport:
    """Retrain on all eligible labelled rows, including later batches (after #23).

    Refuses to run while the #6 holdout setting is on. The #23 results and champion
    artifact are not touched; runs are logged separately under their own names.
    """
    return retrain(
        inputs,
        batch_id=batch_id,
        etl_run_id=uuid.uuid4().hex if etl_run_id is None else etl_run_id,
        read_rows=lambda: read_release_snapshot(engine),
        train_candidate=train_candidate,
        score_rows=score_rows,
        log=log,
        mode="release",
    )


def main(argv=None) -> int:
    from . import config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch_id")
    parser.add_argument(
        "inputs", help="Trusted module:factory returning EvaluationInputs"
    )
    parser.add_argument("--etl-run-id")
    parser.add_argument(
        "--mode",
        choices=MODES,
        default="evaluation",
        help="release trains on all eligible rows once the #6 holdout setting is off",
    )
    args = parser.parse_args(argv)
    module, separator, factory = args.inputs.partition(":")
    if not separator or not module or not factory:
        parser.error("inputs must be module:factory")
    from .training import score_labelled_rows, train

    settings = config.load_settings()
    inputs = getattr(import_module(module), factory)()
    engine = config.create_db_engine(settings)
    try:
        run = run_release_mode if args.mode == "release" else run_evaluation_mode
        report = run(
            engine,
            inputs,
            batch_id=args.batch_id,
            etl_run_id=args.etl_run_id,
            train_candidate=train,
            score_rows=score_labelled_rows,
        )
    finally:
        engine.dispose()
    return report.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
