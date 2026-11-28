# Comparisons and CLI stay together with the harness's private provenance helpers.
"""The shared training harness (issue #13): fit a model on the training window, score the
validation window with the top-K evaluator and log the run to MLflow.

In CSV mode, data comes through the protected reader (`data.load_base_data`), once for the
training window and once for the validation window, so the locked test window is never
loaded. `cleaning.clean_frame` then cleans each table. Labels join orders by country and
order ID through `relationships.link_orders_customers`, and orders without a valid label
are left out. Batch retraining supplies cleaned, joined windows from the protected
RDS reader and trains on the order-only feature set; scoring adds the country for a
model fitted with it.
The features are the order columns `leakage.select_feature_inputs` allows. An exploratory
run adds the seven customer columns (decision D4-02), withholding them for orders whose
customer dates are inconsistent or cannot be checked (D4-03). A run with
`country_input` also gives the model the order's `country_code` (D22-03).

`uv run train baseline` trains and logs the baseline model, `uv run train
boosted-trees` logs the boosted-trees candidate with and without the customer
columns under one parent run (`compare_customer_columns`), `uv run train
boosted-trees-country` logs it without and with the country input
(`compare_country`), and `uv run train rerun RUN_ID` re-runs a logged run to check it
reproduces. docs/training.md says what a run records.

The file is over the 500-line target. The comparisons and the command line call the
harness's private experiment and provenance helpers, which is why they stay together
with it; moving them out needs those helpers made public, a refactor of its own.
"""

import argparse
import hashlib
import os
import subprocess
import sys
import tempfile
import time
import warnings
from collections.abc import Callable
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
from mlops_fraud_prediction_ca1.customer_history import (
    HISTORY_COLUMNS,
    customer_history_features,
)
from mlops_fraud_prediction_ca1.data import load_base_data
from mlops_fraud_prediction_ca1.diagnostics import (
    Diagnostics,
    build_diagnostics,
    log_diagnostics,
)
from mlops_fraud_prediction_ca1.feature_pipeline import FEATURE_GROUPS, build_pipeline
from mlops_fraud_prediction_ca1.leakage import (
    ASSUMED_LABEL_DELAY_DAYS,
    customer_timing_audit,
    historical_order_mask,
    select_feature_inputs,
)
from mlops_fraud_prediction_ca1.models import (
    baseline_model,
    boosted_trees_model,
    extra_trees_model,
    random_forest_model,
)
from mlops_fraud_prediction_ca1.relationships import link_orders_customers
from mlops_fraud_prediction_ca1.splitting import load_policy
from mlops_fraud_prediction_ca1.top_k import (
    K,
    TopKReport,
    evaluate_top_k,
    format_report,
)

# A column whose name contains one of these is never a feature, which also catches
# derived columns such as raw_refund_value.
FORBIDDEN_COLUMNS = ("is_fraud", "refund_value", "order_id", "customer_id", "batch_id")
TABLES = ("customers", "orders", "labels")
_CUTOFFS = ("train_start", "validation_start", "test_start", "test_end_exclusive")
_KEYS = ["order_date", "country_code", "order_id", "customer_id", "is_fraud"]
RECENCY_HALF_LIFE_DAYS = 14.0


@dataclass(frozen=True)
class FitResult:
    """A fitted model, its validation results and the facts a run logs about it."""

    model: object
    report: TopKReport
    fit_seconds: float
    validation_customer_overlap: float
    train_orders: int
    validation_orders: int
    diagnostics: Diagnostics | None = None


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
        # The model pipelines drop date columns, so the customer's first order becomes
        # a number: whole days from it to this order. Order dates have no time of day.
        first_order = extra.pop("first_order_datetime").dt.normalize()
        extra["days_since_first_order"] = (
            pd.to_datetime(joined.order_date) - first_order
        ).dt.days
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
        else:
            result[name] = column.astype(object).where(column.notna(), np.nan)
    return result


def _labelled_window(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The keys and the order-column features of cleaned, joined RDS rows."""
    features = select_feature_inputs(rows, source="orders")
    for column in ("order_value", "num_items_ordered"):
        # MySQL represents money as Decimal, which pandas stores as object.
        features[column] = pd.to_numeric(features[column])
    return rows[_KEYS].astype({"is_fraud": int}), _model_inputs(features)


def _add_history(
    train_keys: pd.DataFrame,
    train_x: pd.DataFrame,
    validation_keys: pd.DataFrame,
    validation_x: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add the v2 customer-history counts (decision D22-01) to both windows.

    Training orders read the training labels and validation orders read the training
    and validation labels; either way only earlier orders count, with the label delay.
    """
    seen = pd.concat([train_keys, validation_keys], ignore_index=True)
    return (
        pd.concat([train_x, customer_history_features(train_keys, train_keys)], axis=1),
        pd.concat(
            [validation_x, customer_history_features(validation_keys, seen)], axis=1
        ),
    )


def _score_window(estimator, keys, features, k: int) -> TopKReport:
    return evaluate_top_k(keys, estimator.predict_proba(features)[:, 1], k)


def _with_country(keys: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """`features` plus each order's `country_code` from `keys`, matched by position."""
    return features.assign(country_code=keys.country_code.astype(object).to_numpy())


def _validate_half_life(half_life_days: float) -> float:
    """Validate that a recency half-life is a positive, finite number (not bool)."""
    if (
        isinstance(half_life_days, bool)
        or not np.isfinite(half_life_days)
        or half_life_days <= 0
    ):
        raise ValueError("recency_half_life_days must be a positive finite number.")
    return float(half_life_days)


def _recency_weights(order_dates: pd.Series, half_life_days: float) -> np.ndarray:
    """Exponential decay weights from the latest training `order_date` (D22-03)."""
    validated = _validate_half_life(half_life_days)
    dates = pd.to_datetime(order_dates, errors="coerce")
    if dates.empty or dates.isna().any():
        raise ValueError(
            "Training rows need valid order_date values for recency weighting."
        )
    age_days = (dates.max() - dates).dt.days.to_numpy(dtype=float)
    return np.power(0.5, age_days / validated)


def _fit_kwargs(estimator, sample_weight: np.ndarray | None) -> dict[str, object]:
    """Route `sample_weight` to the final step of a pipeline or directly to an estimator."""
    if sample_weight is None:
        return {}
    if hasattr(estimator, "steps"):
        return {f"{estimator.steps[-1][0]}__sample_weight": sample_weight}
    return {"sample_weight": sample_weight}


def score_labelled_rows(estimator, rows: pd.DataFrame, *, k: int) -> TopKReport:
    """Score an already fitted model on cleaned, labelled rows.

    A model fitted with the country input (D22-03) also gets each row's `country_code`.
    """
    keys, features = _labelled_window(rows)
    if "country_code" in getattr(estimator, "feature_names_in_", ()):
        features = _with_country(keys, features)
    return _score_window(estimator, keys, features, k)


def fit_and_score(
    estimator,
    data_dir: str | Path | None = None,
    *,
    k: int,
    seed: int | None,
    exploratory: bool = False,
    labelled_windows: tuple[pd.DataFrame, pd.DataFrame] | None = None,
    history: bool = False,
    country_input: bool = False,
    recency_half_life_days: float | None = None,
) -> FitResult:
    """Fit `estimator` on the training window and evaluate it on the validation window.

    Supply a CSV directory or the cleaned, joined train and validation frames.
    Labelled frames don't support the exploratory customer columns. The classifier receives a DataFrame
    and must implement `predict_proba`. A seed overrides every `random_state`
    parameter when supplied. Set daily review capacity `k` before seeing any result.
    `history` adds the three v2 customer-history counts (decision D22-01).
    `country_input` adds the order's `country_code` as a text column (D22-03).
    `recency_half_life_days` weights training rows by exponential decay from the
    latest training `order_date` without weighting validation rows (D22-03).
    """
    if (data_dir is None) == (labelled_windows is None):
        raise ValueError("Supply exactly one of data_dir and labelled_windows.")
    if recency_half_life_days is not None:
        _validate_half_life(recency_half_life_days)
    if labelled_windows is None:
        train_keys, train_x = _load_window(data_dir, "train", exploratory)
        validation_keys, validation_x = _load_window(
            data_dir, "validation", exploratory
        )
    else:
        if exploratory:
            raise ValueError("Labelled windows support order-only training.")
        (train_keys, train_x), (validation_keys, validation_x) = map(
            _labelled_window, labelled_windows
        )
    if history:
        train_x, validation_x = _add_history(
            train_keys, train_x, validation_keys, validation_x
        )
    if country_input:
        train_x = _with_country(train_keys, train_x)
        validation_x = _with_country(validation_keys, validation_x)
    sample_weight = (
        _recency_weights(train_keys.order_date, recency_half_life_days)
        if recency_half_life_days is not None
        else None
    )
    if seed is not None:
        estimator.set_params(
            **{
                name: seed
                for name in estimator.get_params()
                if name.endswith("random_state")
            }
        )
    check_feature_columns(train_x)
    start = time.perf_counter()
    estimator.fit(train_x, train_keys.is_fraud, **_fit_kwargs(estimator, sample_weight))
    fit_seconds = time.perf_counter() - start
    scores = estimator.predict_proba(validation_x)[:, 1]
    report = evaluate_top_k(validation_keys, scores, k)
    diagnostics = build_diagnostics(validation_keys, validation_x, scores, k)
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
        diagnostics,
    )


def train(
    estimator,
    data_dir: str | Path | None = None,
    *,
    k: int,
    seed: int | None,
    exploratory: bool = False,
    tags: dict[str, str] | None = None,
    labelled_windows: tuple[pd.DataFrame, pd.DataFrame] | None = None,
    run_name: str | None = None,
    extra_params: dict[str, object] | None = None,
    extra_metrics: Callable[[FitResult], dict[str, float]] | None = None,
    max_model_size_bytes: int | None = None,
    history: bool = False,
    country_input: bool = False,
    recency_half_life_days: float | None = None,
) -> tuple[str, FitResult]:
    """Run `fit_and_score` and log the run to MLflow; return the run ID and the result.

    MLflow reads MLFLOW_TRACKING_URI from the environment or `.env`, and the run goes to
    the experiment MLFLOW_EXPERIMENT_NAME. The author is git's `user.name`, or "unknown"
    when git has none. A run from a
    tree with uncommitted changes is logged with `git_dirty` set to true and a warning,
    because its commit alone does not reproduce it. Provenance takes precedence over
    caller-supplied `tags`.
    If a run is active, this run is logged as its child. Extra parameters and
    metrics record caller context. An optional size limit is checked before upload.
    """
    _use_experiment()
    provenance = _provenance_tags()
    if provenance["git_dirty"] == "True":
        warnings.warn(
            "The working tree has uncommitted changes, so the logged commit alone does"
            " not reproduce this run.",
            stacklevel=2,
        )
    data_params = (
        _dataset_hashes(data_dir)
        if data_dir is not None and labelled_windows is None
        else {"data_source": "rds_labelled_windows"}
    )
    result = fit_and_score(
        estimator,
        data_dir,
        k=k,
        seed=seed,
        exploratory=exploratory,
        labelled_windows=labelled_windows,
        history=history,
        country_input=country_input,
        recency_half_life_days=recency_half_life_days,
    )
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

    if extra_metrics is not None:
        metrics.update(extra_metrics(result))

    with mlflow.start_run(nested=True, run_name=run_name) as run:
        mlflow.log_params(
            {
                **model_params,
                "model_class": type(estimator).__name__,
                "k": k,
                "seed": seed,
                "exploratory": exploratory,
                "history_features": history,
                "country_input": country_input,
                "recency_half_life_days": recency_half_life_days,
                "label_delay_days": ASSUMED_LABEL_DELAY_DAYS,
                "cleaning_policy_version": CLEANING_POLICY_VERSION,
                **{name: getattr(policy, name).isoformat() for name in _CUTOFFS},
                **data_params,
                **(extra_params or {}),
            }
        )
        mlflow.set_tags({**(tags or {}), **provenance})
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
        log_diagnostics(result.diagnostics)
        if max_model_size_bytes is not None and not 0 < size <= max_model_size_bytes:
            raise ValueError(
                f"Saved model has {size:,} bytes. Expected 1–{max_model_size_bytes:,} bytes."
            )
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
        tags={**tags, **_provenance_tags()},
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


def compare_history(
    estimator, data_dir: str | Path, *, k: int, seed: int, tags: dict[str, str]
) -> tuple[str, tuple[str, FitResult], tuple[str, FitResult]]:
    """Train `estimator` on the order columns (v1), then with the v2 history counts (#22).

    Both runs are children of one parent run, whose description (`mlflow.note.content`)
    states the weakness, the change and the validation change in the primary measure
    (D22-01, docs/model-v2-diagnosis.md). The parent logs the differences too. Returns
    the parent run ID and `train`'s (run ID, result) for v1, then v2.
    """
    _use_experiment()
    model = estimator[-1] if hasattr(estimator, "steps") else estimator
    with mlflow.start_run(
        run_name=f"{type(model).__name__}: v2 customer history compared",
        tags={**tags, **_provenance_tags()},
    ) as parent:
        v1, v2 = (
            train(
                clone(estimator),
                data_dir,
                k=k,
                seed=seed,
                history=history,
                # Neither run is exploratory, so both are champion candidates, as the
                # non-exploratory run of compare_customer_columns is.
                tags={**tags, "mlflow.runName": label, "champion_eligible": "True"},
            )
            for history, label in ((False, "v1 order columns"), (True, "v2 history"))
        )
        before, after = v1[1].report, v2[1].report
        caught = after.at_k.fraud_caught - before.at_k.fraud_caught
        precision = after.average_precision - before.average_precision
        mlflow.log_metrics(
            {
                "fraud_caught_at_k_difference": caught,
                "average_precision_difference": precision,
            }
        )
        mlflow.set_tag(
            "mlflow.note.content",
            "Weakness: v1 ranks by country and payment prior and cannot see a"
            " customer's own earlier fraud. Change: add "
            f"{', '.join(HISTORY_COLUMNS)} (labels used only when more than"
            f" {ASSUMED_LABEL_DELAY_DAYS} days old, D22-01). Validation, K = {k}:"
            f" fraud caught {before.at_k.fraud_caught} -> {after.at_k.fraud_caught}"
            f" ({caught:+d}); average precision {before.average_precision:.4f} ->"
            f" {after.average_precision:.4f} ({precision:+.4f}).",
        )
    return parent.info.run_id, v1, v2


def compare_country(
    estimator,
    data_dir: str | Path,
    *,
    k: int,
    seed: int,
    tags: dict[str, str],
    recency_half_life_days: float | None = RECENCY_HALF_LIFE_DAYS,
) -> tuple[str, list[tuple[str, str, FitResult]]]:
    """Train `estimator` on the order columns (v1), then with country input and optional recency weighting (#22).

    Children are logged under one parent run whose description (`mlflow.note.content`)
    states the weaknesses, the changes, the ablation in fraud caught at K and average
    precision, and what the change costs the country with v1's lowest precision at K
    (D22-03). A country v1 gave no place has no precision, so it is never the one named.
    Returns the parent run ID and a list of `(label, run_id, result)` tuples.
    """
    _use_experiment()
    if recency_half_life_days is not None:
        _validate_half_life(recency_half_life_days)
    model = estimator[-1] if hasattr(estimator, "steps") else estimator
    stages: list[tuple[str, bool, float | None]] = [
        ("v1 order columns", False, None),
        ("v2 country input", True, None),
    ]
    if recency_half_life_days is not None:
        stages.append(
            (
                f"v2 country input + {recency_half_life_days:g}d recency",
                True,
                recency_half_life_days,
            )
        )
    run_title = (
        f"{type(model).__name__}: v2 country input and recency weighting compared"
        if recency_half_life_days is not None
        else f"{type(model).__name__}: v2 country input compared"
    )
    with mlflow.start_run(
        run_name=run_title,
        tags={**tags, **_provenance_tags()},
    ) as parent:
        children: list[tuple[str, str, FitResult]] = []
        for label, country_input, half_life in stages:
            run_id, result = train(
                clone(estimator),
                data_dir,
                k=k,
                seed=seed,
                country_input=country_input,
                recency_half_life_days=half_life,
                tags={**tags, "mlflow.runName": label, "champion_eligible": "True"},
            )
            children.append((label, run_id, result))
        before = children[0][2]
        country_only = children[1][2]
        after = children[-1][2]
        caught = after.report.at_k.fraud_caught - before.report.at_k.fraud_caught
        precision = after.report.average_precision - before.report.average_precision
        parent_metrics = {
            "fraud_caught_at_k_difference": caught,
            "average_precision_difference": precision,
        }
        if recency_half_life_days is not None:
            parent_metrics["country_only_fraud_caught_at_k_difference"] = (
                country_only.report.at_k.fraud_caught - before.report.at_k.fraud_caught
            )
            parent_metrics["recency_fraud_caught_at_k_difference"] = (
                after.report.at_k.fraud_caught - country_only.report.at_k.fraud_caught
            )
        mlflow.log_metrics(parent_metrics)
        was, now = (
            result.diagnostics.slices["country"].set_index("country")
            for result in (before, after)
        )
        # A country with no selected order has no precision, and idxmin skips it.
        worst = was.precision_at_k.idxmin()
        if recency_half_life_days is not None:
            change_text = (
                "Weakness: v1 has no country input, so its daily list holds orders"
                f" from low-fraud countries ({worst} had the lowest precision at K:"
                f" v1 caught {int(was.caught[worst])} fraud with"
                f" {int(was.selected[worst])} places), and uniform training weights"
                " lag behind the mid-March fraud surge across countries."
                " Change: add the order's country_code as a model input and apply"
                f" training-only {recency_half_life_days:g}-day exponential recency"
                " decay weights (D22-03)."
                f" Validation, K = {k}: fraud caught"
                f" {before.report.at_k.fraud_caught} ->"
                f" {country_only.report.at_k.fraud_caught} (country) ->"
                f" {after.report.at_k.fraud_caught} (country + recency,"
                f" {caught:+d} total); average precision"
                f" {before.report.average_precision:.4f} ->"
                f" {country_only.report.average_precision:.4f} ->"
                f" {after.report.average_precision:.4f} ({precision:+.4f})."
                f" Cost: {worst} gets {int(now.selected[worst])} places"
                f" ({int(now.caught[worst])} fraud caught), because one shared K"
                " (D5-01) favours the countries with the most recent fraud."
            )
        else:
            change_text = (
                "Weakness: v1 has no country input, so its daily list holds orders"
                f" from low-fraud countries. {worst} had the lowest precision at K:"
                f" v1 caught {int(was.caught[worst])} fraud with"
                f" {int(was.selected[worst])} places."
                " Change: add the order's country_code as a model input (D22-03)."
                f" Validation, K = {k}: fraud caught"
                f" {before.report.at_k.fraud_caught} ->"
                f" {after.report.at_k.fraud_caught} ({caught:+d}); average precision"
                f" {before.report.average_precision:.4f} ->"
                f" {after.report.average_precision:.4f} ({precision:+.4f})."
                f" Cost: {worst} gets {int(now.selected[worst])} places"
                f" ({int(now.caught[worst])} fraud caught), because one shared K"
                " (D5-01) favours the countries with the most fraud."
            )
        mlflow.set_tag("mlflow.note.content", change_text)
    return parent.info.run_id, children


def rerun(run_id: str, data_dir: str | Path) -> dict:
    """Re-run a logged run and check it reproduces.

    The model is rebuilt from the logged model's parameters (a fresh, unfitted copy) and
    refitted with the logged seed, K and feature choice on the same input files. It
    reproduces when fraud caught at K is within 1% of the logged count and average
    precision within 0.005. Raises `ValueError` if an input file's SHA-256 differs from
    the logged one, and `RuntimeError` if MLflow's settings are missing, as `train` does.
    """
    _use_experiment()
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
    raw_half_life = params.get("recency_half_life_days")
    recency_half_life_days = (
        None if raw_half_life in (None, "", "None") else float(raw_half_life)
    )
    model = clone(mlflow.sklearn.load_model(f"runs:/{run_id}/model"))
    result = fit_and_score(
        model,
        data_dir,
        k=int(params["k"]),
        seed=int(params["seed"]),
        exploratory=params["exploratory"] == "True",
        history=params.get("history_features") == "True",
        country_input=params.get("country_input") == "True",
        recency_half_life_days=recency_half_life_days,
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


def _provenance_tags() -> dict[str, str]:
    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": str(bool(_git("status", "--porcelain", "--untracked-files=no"))),
        # CI runners and new laptops often have no git identity.
        "author": _git("config", "--default", "unknown", "user.name"),
    }


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        capture_output=True,
        text=True,
        check=True,
        cwd=Path(__file__).parent,
    ).stdout.strip()


def compare_groups(
    data_dir: str | Path, *, k: int, seed: int = 0, class_weight: str | None = None
) -> pd.DataFrame:
    """Train the baseline classifier on each feature group; log one MLflow run each.

    Each run is scored on the validation window by the harness, so the groups are
    compared on the primary measure (fraud caught at K) out of sample.
    """
    rows = []
    for group in FEATURE_GROUPS:
        run_id, result = train(
            build_pipeline(baseline_model(class_weight=class_weight), group),
            data_dir,
            k=k,
            seed=seed,
        )
        rows.append(
            {
                "group": group,
                "run_id": run_id,
                "fraud_caught_at_k": result.report.at_k.fraud_caught,
                "average_precision": result.report.average_precision,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    cli = argparse.ArgumentParser(
        prog="train", description="Train and log models with the shared harness."
    )
    commands = cli.add_subparsers(dest="command", required=True)
    baseline = commands.add_parser(
        "baseline", help="Train the baseline model and log the run to MLflow."
    )
    history_parser = commands.add_parser(
        "boosted-trees-v2",
        help="Log boosted trees on the order columns (v1) and with the v2 history counts.",
    )
    country_parser = commands.add_parser(
        "boosted-trees-country",
        help="Log boosted trees on the order columns (v1), with country input, and with recency weighting.",
    )
    recency_group = country_parser.add_mutually_exclusive_group()
    recency_group.add_argument(
        "--recency-half-life",
        type=float,
        default=RECENCY_HALF_LIFE_DAYS,
        help="Half-life in days for training-only exponential recency weights"
        " (default: %(default)g, D22-03).",
    )
    recency_group.add_argument(
        "--no-recency-weighting",
        action="store_true",
        help="Compare v1 against country_code alone with uniform training weights.",
    )
    comparison = commands.add_parser(
        "compare-feature-groups",
        help="Train and compare the baseline on each order feature group.",
    )
    baseline.add_argument(
        "--exploratory",
        action="store_true",
        help="Add the seven unconfirmed customer columns (never for a champion).",
    )
    candidates = {
        "boosted-trees": boosted_trees_model,
        "random-forest": random_forest_model,
        "extra-trees": extra_trees_model,
    }
    candidate_parsers = []
    for name in candidates:
        parser = commands.add_parser(
            name, help=f"Log the {name} model without and with the customer columns."
        )
        parser.add_argument(
            "--k-status",
            help="Where K comes from, tagged on every run. Required when --k differs"
            f" from D5-01's K (default: D5-01: product owner assumption, K = {K}).",
        )
        parser.add_argument(
            "--tag",
            action="append",
            default=[],
            metavar="KEY=VALUE",
            help="Extra tag on every run (repeatable), such as pr=164. "
            "git_commit, git_dirty, and author are set by the harness.",
        )
        candidate_parsers.append(parser)
    for command in (
        baseline,
        comparison,
        history_parser,
        country_parser,
        *candidate_parsers,
    ):
        command.add_argument(
            "--weighting",
            choices=("balanced",),
            help="Balance class weights using the training labels.",
        )
        command.add_argument(
            "--k",
            type=int,
            default=K,
            help="Daily review capacity K, in orders per day (default: %(default)s,"
            " from decision D5-01).",
        )
        command.add_argument("--seed", type=int, default=0)
    again = commands.add_parser("rerun", help="Re-run a logged run and compare.")
    again.add_argument("run_id")
    for command in (
        baseline,
        comparison,
        history_parser,
        country_parser,
        *candidate_parsers,
        again,
    ):
        command.add_argument(
            "--data-dir", default="data", help="Folder with the three base CSV files."
        )
    args = cli.parse_args()
    if args.command in candidates and args.k_status is None:
        if args.k != K:
            cli.error(f"--k {args.k} differs from D5-01's K = {K}: give --k-status")
        args.k_status = f"D5-01: product owner assumption, K = {K}"
    # MLflow prints an emoji when a run ends, which Windows cannot encode when piped.
    sys.stdout.reconfigure(errors="replace")
    if args.command == "baseline":
        run_id, result = train(
            baseline_model(class_weight=args.weighting),
            args.data_dir,
            k=args.k,
            seed=args.seed,
            exploratory=args.exploratory,
        )
        print(format_report(result.report))
        print(f"Logged run {run_id}.")
    elif args.command in candidates:
        parent_id, *children = compare_customer_columns(
            candidates[args.command](class_weight=args.weighting),
            args.data_dir,
            k=args.k,
            seed=args.seed,
            tags={
                "k_status": args.k_status,
                **dict(tag.split("=", 1) for tag in args.tag),
            },
        )
        for label, (run_id, result) in zip(("Without", "With"), children, strict=True):
            print(f"{label} the customer columns:\n{format_report(result.report)}")
            print(f"Logged run {run_id}.")
        print(f"Logged parent run {parent_id}. Re-run a child run, not the parent.")
    elif args.command == "boosted-trees-v2":
        parent_id, *children = compare_history(
            boosted_trees_model(class_weight=args.weighting),
            args.data_dir,
            k=args.k,
            seed=args.seed,
            tags={"k_status": f"D5-01: product owner assumption, K = {K}"}
            if args.k == K
            else {},
        )
        for label, (run_id, result) in zip(("v1", "v2"), children, strict=True):
            print(f"{label}:\n{format_report(result.report)}")
            print(f"Logged run {run_id}.")
        print(f"Logged parent run {parent_id}. Re-run a child run, not the parent.")
    elif args.command == "boosted-trees-country":
        half_life = None if args.no_recency_weighting else args.recency_half_life
        parent_id, children = compare_country(
            boosted_trees_model(class_weight=args.weighting),
            args.data_dir,
            k=args.k,
            seed=args.seed,
            tags={"k_status": f"D5-01: product owner assumption, K = {K}"}
            if args.k == K
            else {},
            recency_half_life_days=half_life,
        )
        for label, run_id, result in children:
            print(f"{label}:\n{format_report(result.report)}")
            print(f"Logged run {run_id}.")
        print(f"Logged parent run {parent_id}. Re-run a child run, not the parent.")
    elif args.command == "compare-feature-groups":
        table = compare_groups(
            args.data_dir, k=args.k, seed=args.seed, class_weight=args.weighting
        )
        print(table.to_string(index=False))
    else:
        for name, value in rerun(args.run_id, args.data_dir).items():
            print(f"{name}: {value}")


if __name__ == "__main__":
    main()
