"""TEMPORARY adapter so #25 can fit models on rows read from RDS (PBI #25).

Remove when #151's `training.fit_and_score` accepts pre-read windows (see the suggestion
on PR #151); `make_fit_and_log` then becomes a thin call to it. Until then this repeats
a little of the harness: fit on the feature columns of the training rows, score the
frozen validation rows with the top-K evaluator, and report one run's metrics.
`evaluate_top_k` and `log_run` are passed in, so this file imports neither #148 nor MLflow.
"""

import pickle
import time
from collections.abc import Callable

import pandas as pd

# The four order columns D4-01 allows as model inputs.
FEATURE_COLUMNS = ("collect_type", "payment_method", "order_value", "num_items_ordered")
FORBIDDEN = ("is_fraud", "refund_value", "order_id", "customer_id", "batch_id")


def model_inputs(rows: pd.DataFrame) -> pd.DataFrame:
    """Feature columns as the harness gives models: floats and strings, NaN for missing."""
    frame = rows[list(FEATURE_COLUMNS)].copy()
    for name in ("order_value", "num_items_ordered"):
        frame[name] = pd.to_numeric(frame[name], errors="coerce").astype("float64")
    for name in ("collect_type", "payment_method"):
        frame[name] = (
            frame[name].astype(object).where(frame[name].notna(), float("nan"))
        )
    return frame


def make_fit_and_log(
    *,
    k: int,
    seed: int,
    validation_rows: pd.DataFrame,
    evaluate_top_k: Callable,
    log_run: Callable[[str, dict, dict, object], str],
) -> Callable:
    """Build the `fit_and_log(spec, train_rows, run_name)` that `retrain.retrain` calls."""
    x_validation = model_inputs(validation_rows)

    def fit_and_log(spec, train_rows: pd.DataFrame, run_name: str) -> dict:
        assert not set(FORBIDDEN) & set(FEATURE_COLUMNS)
        model = spec.build()
        model.set_params(
            **{n: seed for n in model.get_params() if n.endswith("random_state")}
        )
        labelled = train_rows[train_rows["is_fraud"].isin([0, 1])]
        start = time.perf_counter()
        model.fit(model_inputs(labelled), labelled["is_fraud"].astype(int))
        fit_seconds = time.perf_counter() - start
        report = evaluate_top_k(
            validation_rows, model.predict_proba(x_validation)[:, 1], k
        )
        metrics = {
            "fraud_caught_at_k": report.at_k.fraud_caught,
            "average_precision": report.average_precision,
            "fit_seconds": fit_seconds,
            "model_size_bytes": len(pickle.dumps(model)),
            "train_orders": len(labelled),
            "validation_orders": len(validation_rows),
        }
        params = {"model": spec.name, "owner": spec.owner, "k": k, "seed": seed}
        metrics["mlflow_run_id"] = log_run(run_name, params, metrics, model)
        return metrics

    return fit_and_log


def compare_with_champion(metrics: dict, champion: dict | None) -> dict:
    """Compare one retrained model with the champion's logged validation metrics.

    Reports only; nothing is promoted. With no champion yet, `better` is None.
    """
    if not champion:
        return {"champion": None, "better": None}
    caught = metrics["fraud_caught_at_k"] - champion["fraud_caught_at_k"]
    precision = metrics["average_precision"] - champion["average_precision"]
    return {
        "champion": champion.get("run_id"),
        "fraud_caught_at_k_delta": caught,
        "average_precision_delta": precision,
        "better": (caught, precision) > (0, 0),
    }
