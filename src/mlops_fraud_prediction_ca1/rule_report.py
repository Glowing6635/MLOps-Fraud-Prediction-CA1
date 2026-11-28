"""Run the #18 comparison on the validation window from a file of model scores.

    uv run rule-report --data-dir data \\
        --scores scores.csv --k 250 --grid training-quantiles

`scores.csv` has `country_code`, `order_id` and `score` for every labelled validation
order (higher means riskier). Without `--threshold`, X is fitted on the validation
window and reported as a validation-tuned proxy.
"""

import argparse
import sys

import numpy as np
import pandas as pd

from mlops_fraud_prediction_ca1.cleaning import clean_frame, raw_column_names
from mlops_fraud_prediction_ca1.data import load_base_data
from mlops_fraud_prediction_ca1.loyalty import loyalty_status
from mlops_fraud_prediction_ca1.relationships import link_orders_customers
from mlops_fraud_prediction_ca1.rule_comparison import (
    THRESHOLD_GRID,
    compare,
    format_comparison,
    select_threshold,
)
from mlops_fraud_prediction_ca1.rules import account_age_days
from mlops_fraud_prediction_ca1.temporal import parse_date_rows

_TABLES = ("customers", "orders", "labels")


def _clean(data_dir, purpose, config_path=None):
    customers, orders, labels = load_base_data(
        data_dir, purpose=purpose, config_path=config_path, raw_strings=True
    )
    raw_column_names(customers.columns)
    first_order = customers["first_order_datetime"]
    # Keep the raw timestamp for link disambiguation and account-age diagnostics.
    parsed = parse_date_rows(first_order, kind="customer")
    cleaned_customers = clean_frame(customers.drop(columns="first_order_datetime"))
    return {
        "customers": cleaned_customers.assign(
            first_order_datetime=parsed["date"], raw_first_order_datetime=first_order
        ),
        "orders": clean_frame(orders),
        "labels": clean_frame(labels),
    }


TRAINING_QUANTILES = (0.90, 0.95, 0.99, 0.995, 0.999, 0.9999)


def training_quantile_grid(data_dir) -> tuple[float, ...]:
    """The D18-03 grid: quantiles of the training window's order_value, no labels."""
    value = pd.to_numeric(_clean(data_dir, "train")["orders"]["order_value"])
    grid = value.dropna().quantile(list(TRAINING_QUANTILES)).round(2)
    return tuple(sorted(set(grid.tolist())))


def validation_inputs(
    data_dir, purpose="validation", config_path=None, history_purposes=("train",)
) -> dict:
    """Labelled orders of one window with account age and loyalty, ready to compare.

    The default is the validation window with the training window as loyalty history.
    PBI #23 passes the final window, with every earlier window as history.
    """
    history = [_clean(data_dir, p, config_path) for p in history_purposes]
    validation = _clean(data_dir, purpose, config_path)
    train = {
        name: pd.concat([h[name] for h in history], ignore_index=True)
        for name in _TABLES
    }
    joined, _ = link_orders_customers(
        validation["orders"], validation["labels"], validation["customers"]
    )
    joined = joined.loc[
        joined.label_link.eq("both") & joined.is_fraud.notna()
    ].reset_index(drop=True)
    joined["is_fraud"] = joined["is_fraud"].astype(int)
    age = account_age_days(
        joined["order_date"],
        joined["first_order_datetime"],
        raw_order_date=joined.get("raw_order_date"),
        raw_first_order_datetime=joined.get("raw_first_order_datetime"),
    )
    both = {
        name: pd.concat([train[name], validation[name]], ignore_index=True)
        for name in _TABLES
    }
    loyal = loyalty_status(
        both["orders"],
        joined[validation["orders"].columns],
        both["labels"],
        both["customers"],
        config_path=config_path,
    )
    return {
        "orders": joined,
        "age": age["account_age_days"],
        "age_status": age["account_age_status"].value_counts().to_dict(),
        "loyalty": loyal["loyalty_status"].to_numpy(),
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--scores", required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument(
        "--grid",
        choices=("original", "training-quantiles"),
        default="original",
        help=(
            "Candidate X values: the predeclared D18-01 grid or the post-hoc "
            "D18-03 training-quantile grid adopted in D18-04."
        ),
    )
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--threshold-source")
    args = parser.parse_args(argv)
    if (args.threshold is None) != (args.threshold_source is None):
        sys.exit("--threshold and --threshold-source go together.")
    inputs = validation_inputs(args.data_dir)
    orders = inputs["orders"]
    if orders.empty:
        sys.exit(
            "No labelled validation orders. No threshold was fitted or verdict produced."
        )

    scores = pd.read_csv(
        args.scores,
        dtype={"country_code": str, "order_id": str, "score": str},
        keep_default_na=False,
    )
    duplicated = scores.duplicated(["country_code", "order_id"])
    if duplicated.any():
        sys.exit(
            f"{int(duplicated.sum())} duplicate (country_code, order_id) keys in "
            "the scores file; each order needs exactly one score."
        )
    merged = (
        orders[["country_code", "order_id"]]
        .astype(str)
        .merge(
            scores, on=["country_code", "order_id"], how="left", validate="one_to_one"
        )
    )
    if merged["score"].isna().any():
        sys.exit(f"{int(merged['score'].isna().sum())} orders have no score.")
    numeric_scores = pd.to_numeric(merged["score"], errors="coerce")
    invalid_count = int((~np.isfinite(numeric_scores)).sum())
    if invalid_count:
        sys.exit(
            f"{invalid_count} orders have an invalid score; scores must be finite numbers."
        )

    candidates = None
    if args.threshold is None:
        if args.grid == "training-quantiles":
            grid = training_quantile_grid(args.data_dir)
            source = (
                "validation-tuned proxy, post-hoc D18-03 training-quantile grid "
                "adopted in D18-04"
            )
        else:
            grid = THRESHOLD_GRID
            source = "validation-tuned proxy, predeclared D18-01 original grid"
        fit = select_threshold(orders, inputs["age"], args.k, grid)
        candidates = fit["candidates"]
        threshold = fit["x"]
    else:
        threshold, source = args.threshold, args.threshold_source
    result = compare(
        orders,
        numeric_scores,
        inputs["age"],
        inputs["loyalty"],
        args.k,
        threshold,
        source,
        candidates=candidates,
    )
    print(f"Account age status counts: {inputs['age_status']}")  # noqa: T201
    print(format_comparison(result))  # noqa: T201


if __name__ == "__main__":
    main()
