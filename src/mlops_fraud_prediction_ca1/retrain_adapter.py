"""Fit stored candidate configurations on in-memory rows and log evaluated artifacts."""

import hashlib
import pickle
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from .splitting import split_orders

FEATURE_COLUMNS = ("collect_type", "payment_method", "order_value", "num_items_ordered")
MAX_ARTIFACT_BYTES = 50 * 1024 * 1024


def model_inputs(rows: pd.DataFrame) -> pd.DataFrame:
    """Match the order-only candidate harness's numeric and categorical inputs."""
    frame = rows[list(FEATURE_COLUMNS)].copy()
    for name in ("order_value", "num_items_ordered"):
        frame[name] = pd.to_numeric(frame[name], errors="coerce").astype("float64")
    for name in ("collect_type", "payment_method"):
        frame[name] = (
            frame[name].astype(object).where(frame[name].notna(), float("nan"))
        )
    return frame


def _evaluate(inputs, model, validation, features):
    scores = model.predict_proba(features)[:, list(model.classes_).index(1)]
    result = inputs.evaluate_top_k(validation.copy(), scores, inputs.k)
    if result.average_precision is None:
        raise ValueError("Frozen validation must support average precision.")
    return result.at_k.fraud_caught, result.average_precision


def make_fit_and_log(inputs, log_run):
    validation, split = split_orders(inputs.validation_rows, purpose="validation")
    if (
        not split["test_locked"]
        or not inputs.champion_run_id
        or validation.empty
        or split["excluded_rows"]
        or not validation["is_fraud"].isin([0, 1]).all()
    ):
        raise ValueError(
            "Supply a labelled frozen #6 validation window and champion in evaluation mode."
        )
    features = model_inputs(validation)
    validation_hash = hashlib.sha256(
        validation.to_csv(index=False).encode()
    ).hexdigest()
    champion_caught, champion_ap = _evaluate(
        inputs, inputs.champion_model, validation, features
    )

    def fit_and_log(spec, train_rows, run_name, counts, batch_id, etl_run_id):
        model = spec.build()
        if spec.params:
            model.set_params(**spec.params)
        start = time.perf_counter()
        model.fit(model_inputs(train_rows), train_rows["is_fraud"].astype(int))
        fit_seconds = time.perf_counter() - start
        caught, ap = _evaluate(inputs, model, validation, features)
        artifact = pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)
        size = len(artifact)
        if not 0 < size <= MAX_ARTIFACT_BYTES:
            raise ValueError(f"model artifact is {size} bytes, outside the 50 MB limit")
        metrics = {
            **counts,
            "fraud_caught_at_k": caught,
            "average_precision": ap,
            "fit_seconds": fit_seconds,
            "model_size_bytes": size,
            "validation_orders": len(validation),
            "champion_fraud_caught_at_k": champion_caught,
            "champion_average_precision": champion_ap,
            "fraud_caught_at_k_delta": caught - champion_caught,
            "average_precision_delta": ap - champion_ap,
            "beats_champion": int((caught, ap) > (champion_caught, champion_ap)),
        }
        params = {
            "model": spec.name,
            "owner": spec.owner,
            "k": inputs.k,
            "batch_id": batch_id,
            "etl_run_id": etl_run_id,
            "champion_run_id": inputs.champion_run_id,
            "validation_sha256": validation_hash,
            **{
                f"model.{key}": value
                for key, value in model.get_params(deep=True).items()
            },
        }
        run_id = log_run(run_name, params, metrics, artifact)
        return {**metrics, "mlflow_run_id": run_id}

    return fit_and_log


def make_mlflow_logger(experiment_name):
    import mlflow

    def log_run(name, params, metrics, artifact):
        with TemporaryDirectory() as folder:
            path = Path(folder, "model.pkl")
            path.write_bytes(artifact)
            experiment = mlflow.set_experiment(experiment_name)
            with mlflow.start_run(
                run_name=name, experiment_id=experiment.experiment_id
            ) as run:
                mlflow.log_params(params)
                mlflow.log_metrics(metrics)
                mlflow.log_artifact(str(path))
                return run.info.run_id

    return log_run
