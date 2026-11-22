"""The shared training harness (issue #13): fit a model on the training window, score the
validation window with the top-K evaluator and log the run to MLflow.

Data comes only through the protected reader (`data.load_base_data`), once for the
training window and once for the validation window, so the locked test window is never
loaded. `cleaning.clean_frame` then cleans each table. Labels join orders by country and
order ID through `relationships.link_orders_customers`, and orders without a valid label
are left out.
The features are the order columns `leakage.select_feature_inputs` allows. An exploratory
run adds the seven customer columns (decision D4-02), withholding them for orders whose
customer dates are inconsistent or cannot be checked (D4-03).

`uv run train baseline --k K` trains and logs the baseline model, `uv run train
boosted-trees --k K` logs the boosted-trees candidate with and without the customer
columns under one parent run (`compare_customer_columns`), and `uv run train rerun
RUN_ID` re-runs a logged run to check it reproduces. docs/training.md says what a run
records.
"""

import argparse
import hashlib
import os
import subprocess
import sys
import tempfile
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import skops.io
from dotenv import load_dotenv
from sklearn.base import clone

from mlops_fraud_prediction_ca1.cleaning import CLEANING_POLICY_VERSION, clean_frame
from mlops_fraud_prediction_ca1.data import load_base_data
from mlops_fraud_prediction_ca1.leakage import (
    ASSUMED_LABEL_DELAY_DAYS,
    customer_timing_audit,
    historical_order_mask,
    select_feature_inputs,
)
from mlops_fraud_prediction_ca1.models import baseline_model, boosted_trees_model
from mlops_fraud_prediction_ca1.relationships import link_orders_customers
from mlops_fraud_prediction_ca1.splitting import load_policy
from mlops_fraud_prediction_ca1.top_k import TopKReport, evaluate_top_k, format_report

# A column whose name contains one of these is never a feature, which also catches
# derived columns such as raw_refund_value.
FORBIDDEN_COLUMNS = ("is_fraud", "refund_value", "order_id", "customer_id", "batch_id")
TABLES = ("customers", "orders", "labels")
_CUTOFFS = ("train_start", "validation_start", "test_start", "test_end_exclusive")
_KEYS = ["order_date", "country_code", "order_id", "customer_id", "is_fraud"]


@dataclass(frozen=True)
class FitResult:
    """A fitted model, its validation results and the facts a run logs about it."""

    model: object
    report: TopKReport
    fit_seconds: float
    validation_customer_overlap: float
    train_orders: int
    validation_orders: int


def check_feature_columns(features: pd.DataFrame) -> None:
    """Raise `ValueError` if the target, a refund, an identifier or a batch ID is a feature."""
    leaked = [
        str(column)
        for column in features.columns
        if any(name in str(column) for name in FORBIDDEN_COLUMNS)
    ]
    if leaked:
        raise ValueError(f"These columns must not be features: {', '.join(leaked)}.")


def check_label_history(order_dates: pd.Series, label_order_dates: pd.Series) -> None:
    """Raise `ValueError` if a customer-history feature read a label too new for its order.

    Each row pairs a scored order's `order_date` with the `order_date` of one labelled
    order that the order's history feature read; the two Series share an index. A label
    is usable only when its order date plus the label delay
    (`leakage.ASSUMED_LABEL_DELAY_DAYS`, decision D4-04) is before the scored order's
    date. Missing or invalid dates fail too. Dates are datetimes or order-date text.
    """
    order_dates, label_order_dates = map(_date_text, (order_dates, label_order_dates))
    for current, prior in label_order_dates.groupby(order_dates, dropna=False):
        if not historical_order_mask(
            prior, current, label_delay_days=ASSUMED_LABEL_DELAY_DAYS
        ).all():
            raise ValueError(
                f"A history feature for an order on {current} uses a label less than"
                f" {ASSUMED_LABEL_DELAY_DAYS} days older than the order."
            )


def _date_text(dates: pd.Series) -> pd.Series:
    # historical_order_mask parses order-date text only.
    if pd.api.types.is_datetime64_any_dtype(dates):
        return dates.dt.strftime("%Y-%m-%d")
    return dates


def _load_window(
    data_dir: str | Path, purpose: str, exploratory: bool
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The labelled orders of one window: their keys and label, and their features."""
    frames = load_base_data(data_dir, purpose=purpose, raw_strings=True)
    raw = dict(zip(TABLES, frames, strict=True))
    # clean_frame adds raw_<column> copies; the harness keeps only the cleaned columns.
    customers, orders, labels = (
        clean_frame(raw[name])[raw[name].columns] for name in TABLES
    )
    joined, _ = link_orders_customers(orders, labels, customers)
    joined = joined.loc[
        joined.label_link.eq("both") & joined.is_fraud.notna()
    ].reset_index(drop=True)
    features = select_feature_inputs(joined[orders.columns], source="orders")
    if exploratory:
        extra = select_feature_inputs(
            joined[customers.columns], source="customers", exploratory=True
        )
        timing, _ = customer_timing_audit(*(raw[name] for name in TABLES))
        flagged = timing.loc[
            timing.date_evidence.isin(["before_first_order", "unevaluable"]),
            ["country_code", "order_id"],
        ]
        withheld = pd.MultiIndex.from_frame(
            joined[["country_code", "order_id"]].astype(str)
        ).isin(pd.MultiIndex.from_frame(flagged.astype(str)))
        extra.loc[withheld, :] = None
        features = pd.concat([features, extra], axis=1)
    keys = joined[_KEYS].astype({"is_fraud": int})
    return keys, _model_inputs(features)


def _model_inputs(features: pd.DataFrame) -> pd.DataFrame:
    """Nullable columns as plain NumPy types, with NaN for every missing value."""
    result = features.copy()
    for name, column in result.items():
        if pd.api.types.is_numeric_dtype(column) or pd.api.types.is_bool_dtype(column):
            result[name] = column.astype("float64")
        elif not pd.api.types.is_datetime64_any_dtype(column):
            result[name] = column.astype(object).where(column.notna(), np.nan)
    return result


def fit_and_score(
    estimator, data_dir: str | Path, *, k: int, seed: int, exploratory: bool = False
) -> FitResult:
    """Fit `estimator` on the training window and evaluate it on the validation window.

    `estimator` is a scikit-learn classifier or pipeline with `predict_proba`; it gets a
    DataFrame of the feature columns. `seed` goes to every `random_state` parameter it
    has. `k` is the daily review capacity for the top-K evaluator, set before any result
    is seen.
    """
    estimator.set_params(
        **{
            name: seed
            for name in estimator.get_params()
            if name.endswith("random_state")
        }
    )
    train_keys, train_x = _load_window(data_dir, "train", exploratory)
    validation_keys, validation_x = _load_window(data_dir, "validation", exploratory)
    check_feature_columns(train_x)
    start = time.perf_counter()
    estimator.fit(train_x, train_keys.is_fraud)
    fit_seconds = time.perf_counter() - start
    report = evaluate_top_k(
        validation_keys, estimator.predict_proba(validation_x)[:, 1], k
    )
    customer_keys = ["country_code", "customer_id"]
    seen = pd.MultiIndex.from_frame(train_keys[customer_keys].astype(str))
    overlap = validation_keys.customer_id.notna() & pd.MultiIndex.from_frame(
        validation_keys[customer_keys].astype(str)
    ).isin(seen)
    return FitResult(
        estimator,
        report,
        fit_seconds,
        float(overlap.mean()),
        len(train_keys),
        len(validation_keys),
    )


def train(
    estimator,
    data_dir: str | Path,
    *,
    k: int,
    seed: int,
    exploratory: bool = False,
    tags: dict[str, str] | None = None,
) -> tuple[str, FitResult]:
    """Run `fit_and_score` and log the run to MLflow; return the run ID and the result.

    MLflow reads MLFLOW_TRACKING_URI from the environment or `.env`, and the run goes to
    the experiment MLFLOW_EXPERIMENT_NAME. The author is git's `user.name`. A run from a
    tree with uncommitted changes is logged with `git_dirty` set to true and a warning,
    because its commit alone does not reproduce it. `tags` are added to the run's tags.
    If a run is active, this run is logged as its child.
    """
    _use_experiment()
    hashes = _dataset_hashes(data_dir)
    dirty = bool(_git("status", "--porcelain", "--untracked-files=no"))
    if dirty:
        warnings.warn(
            "The working tree has uncommitted changes, so the logged commit alone does"
            " not reproduce this run.",
            stacklevel=2,
        )
    result = fit_and_score(estimator, data_dir, k=k, seed=seed, exploratory=exploratory)
    policy = load_policy()
    model_params = {
        name: value
        for name, value in estimator.get_params().items()
        if value is None or isinstance(value, str | int | float | bool)
    }
    metrics = {
        "average_precision": result.report.average_precision,
        "validation_customer_overlap": result.validation_customer_overlap,
        "fit_seconds": result.fit_seconds,
        "train_orders": result.train_orders,
        "validation_orders": result.validation_orders,
    }
    for suffix, at in (("at_k", result.report.at_k), ("at_2k", result.report.at_2k)):
        for name in ("fraud_caught", "fraud_catchable", "selected", "total_fraud"):
            metrics[f"{name}_{suffix}"] = getattr(at, name)
        metrics[f"precision_{suffix}"] = at.precision
        metrics[f"recall_{suffix}"] = at.recall

    with mlflow.start_run(nested=True) as run:
        mlflow.log_params(
            {
                **model_params,
                "model_class": type(estimator).__name__,
                "k": k,
                "seed": seed,
                "exploratory": exploratory,
                "label_delay_days": ASSUMED_LABEL_DELAY_DAYS,
                "cleaning_policy_version": CLEANING_POLICY_VERSION,
                **{name: getattr(policy, name).isoformat() for name in _CUTOFFS},
                **hashes,
            }
        )
        mlflow.set_tags(
            {
                "git_commit": _git("rev-parse", "HEAD"),
                "git_dirty": dirty,
                "author": _git("config", "user.name"),
                **(tags or {}),
            }
        )
        # An undefined ratio (no order selected, or no fraud) is left out, not logged as 0.
        mlflow.log_metrics(
            {name: value for name, value in metrics.items() if value is not None}
        )
        # MLflow saves with skops, which loads only the types listed here. The model was
        # just fitted in this process, so every type in it is trusted.
        trusted = skops.io.get_untrusted_types(data=skops.io.dumps(result.model))
        with tempfile.TemporaryDirectory() as folder:
            mlflow.sklearn.save_model(
                result.model, Path(folder, "model"), skops_trusted_types=trusted
            )
            size = sum(
                path.stat().st_size
                for path in Path(folder).rglob("*")
                if path.is_file()
            )
        mlflow.log_metric("model_size_bytes", size)
        mlflow.sklearn.log_model(
            result.model, name="model", skops_trusted_types=trusted
        )
    return run.info.run_id, result


def compare_customer_columns(
    estimator, data_dir: str | Path, *, k: int, seed: int, tags: dict[str, str]
) -> tuple[str, tuple[str, FitResult], tuple[str, FitResult]]:
    """Train `estimator` without and with the unconfirmed customer columns (issue #14).

    `train` logs both runs, from copies of `estimator`, as children of one parent run
    named after the model. The with-columns run is exploratory and tagged
    `champion_eligible` false. The parent logs the with-columns result minus the
    without result for fraud caught at K and 2K and for average precision. `tags` go on
    all three runs. Returns the parent run ID and `train`'s (run ID, result) for the
    run without, then with, the columns.
    """
    _use_experiment()
    model = estimator[-1] if hasattr(estimator, "steps") else estimator
    with mlflow.start_run(
        run_name=f"{type(model).__name__}: customer columns compared",
        tags={
            **tags,
            "git_commit": _git("rev-parse", "HEAD"),
            "author": _git("config", "user.name"),
        },
    ) as parent:
        without, with_columns = (
            train(
                clone(estimator),
                data_dir,
                k=k,
                seed=seed,
                exploratory=exploratory,
                tags={
                    **tags,
                    "mlflow.runName": label,
                    "champion_eligible": str(not exploratory),
                },
            )
            for exploratory, label in (
                (False, "without customer columns"),
                (True, "with customer columns (exploratory)"),
            )
        )
        before, after = without[1].report, with_columns[1].report
        mlflow.log_metrics(
            {
                "fraud_caught_at_k_difference": after.at_k.fraud_caught
                - before.at_k.fraud_caught,
                "fraud_caught_at_2k_difference": after.at_2k.fraud_caught
                - before.at_2k.fraud_caught,
                "average_precision_difference": after.average_precision
                - before.average_precision,
            }
        )
    return parent.info.run_id, without, with_columns


def rerun(run_id: str, data_dir: str | Path) -> dict:
    """Re-run a logged run and check it reproduces.

    The model is rebuilt from the logged model's parameters (a fresh, unfitted copy) and
    refitted with the logged seed, K and feature choice on the same input files. It
    reproduces when fraud caught at K is within 1% of the logged count and average
    precision within 0.005. Raises `ValueError` if an input file's SHA-256 differs from
    the logged one.
    """
    load_dotenv()
    run = mlflow.get_run(run_id)
    params, metrics = run.data.params, run.data.metrics
    changed = [
        name
        for name, digest in _dataset_hashes(data_dir).items()
        if params[name] != digest
    ]
    if changed:
        raise ValueError(
            f"The input files differ from the logged run's: {', '.join(changed)}."
        )
    model = clone(mlflow.sklearn.load_model(f"runs:/{run_id}/model"))
    result = fit_and_score(
        model,
        data_dir,
        k=int(params["k"]),
        seed=int(params["seed"]),
        exploratory=params["exploratory"] == "True",
    )
    caught, logged_caught = (
        result.report.at_k.fraud_caught,
        metrics["fraud_caught_at_k"],
    )
    precision, logged_precision = (
        result.report.average_precision,
        metrics["average_precision"],
    )
    return {
        "fraud_caught_at_k": caught,
        "logged_fraud_caught_at_k": logged_caught,
        "average_precision": precision,
        "logged_average_precision": logged_precision,
        "same_commit": _git("rev-parse", "HEAD") == run.data.tags["git_commit"],
        "reproduced": abs(caught - logged_caught) <= 0.01 * logged_caught
        and abs(precision - logged_precision) <= 0.005,
    }


def _use_experiment() -> None:
    load_dotenv()
    experiment = os.environ.get("MLFLOW_EXPERIMENT_NAME")
    if not experiment or not os.environ.get("MLFLOW_TRACKING_URI"):
        raise RuntimeError(
            "Set MLFLOW_TRACKING_URI and MLFLOW_EXPERIMENT_NAME in .env (see .env.example)."
        )
    mlflow.set_experiment(experiment)


def _dataset_hashes(data_dir: str | Path) -> dict[str, str]:
    hashes = {}
    for name in TABLES:
        with Path(data_dir, f"{name}_base.csv").open("rb") as file:
            hashes[f"dataset_sha256_{name}"] = hashlib.file_digest(
                file, "sha256"
            ).hexdigest()
    return hashes


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        capture_output=True,
        text=True,
        check=True,
        cwd=Path(__file__).parent,
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="train", description="Train and log models with the shared harness."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    baseline = commands.add_parser(
        "baseline", help="Train the baseline model and log the run to MLflow."
    )
    baseline.add_argument(
        "--exploratory",
        action="store_true",
        help="Add the seven unconfirmed customer columns (never for a champion).",
    )
    trees = commands.add_parser(
        "boosted-trees",
        help="Log the boosted-trees model without and with the customer columns.",
    )
    trees.add_argument(
        "--k-status",
        required=True,
        help="Where K comes from, such as 'confirmed' or 'provisional: <reason>'.",
    )
    for command in (baseline, trees):
        command.add_argument(
            "--k",
            type=int,
            required=True,
            help="Daily review capacity K, in orders per day, fixed before seeing"
            " results.",
        )
        command.add_argument("--seed", type=int, default=0)
    again = commands.add_parser("rerun", help="Re-run a logged run and compare.")
    again.add_argument("run_id")
    for command in (baseline, trees, again):
        command.add_argument(
            "--data-dir", default="data", help="Folder with the three base CSV files."
        )
    args = parser.parse_args()
    # MLflow prints an emoji when a run ends, which Windows cannot encode when piped.
    sys.stdout.reconfigure(errors="replace")
    if args.command == "baseline":
        run_id, result = train(
            baseline_model(),
            args.data_dir,
            k=args.k,
            seed=args.seed,
            exploratory=args.exploratory,
        )
        print(format_report(result.report))
        print(f"Logged run {run_id}.")
    elif args.command == "boosted-trees":
        parent_id, *children = compare_customer_columns(
            boosted_trees_model(),
            args.data_dir,
            k=args.k,
            seed=args.seed,
            tags={"k_status": args.k_status},
        )
        for label, (run_id, result) in zip(("Without", "With"), children, strict=True):
            print(f"{label} the customer columns:\n{format_report(result.report)}")
            print(f"Logged run {run_id}.")
        print(f"Logged parent run {parent_id}. Re-run a child run, not the parent.")
    else:
        for name, value in rerun(args.run_id, args.data_dir).items():
            print(f"{name}: {value}")


if __name__ == "__main__":
    main()
