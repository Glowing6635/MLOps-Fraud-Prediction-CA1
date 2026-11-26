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


def clean_window(data_dir, purpose, *, config_path=None) -> dict:
    """The cleaned customers, orders and labels of one window, as `load_base_data`
    reads them for `purpose`, with the raw first-order timestamp kept."""
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
    value = pd.to_numeric(clean_window(data_dir, "train")["orders"]["order_value"])
    grid = value.dropna().quantile(list(TRAINING_QUANTILES)).round(2)
    return tuple(sorted(set(grid.tolist())))


def validation_inputs(data_dir) -> dict:
    """Labelled validation orders with account age and loyalty, ready to compare."""
    train = clean_window(data_dir, "train")
    validation = clean_window(data_dir, "validation")
    return window_inputs([train, validation], validation)


def labelled_orders(window: dict) -> pd.DataFrame:
    """A `clean_window` result's orders that have a valid label, joined to their label
    and customer, with `is_fraud` as 0 or 1."""
    joined, _ = link_orders_customers(
        window["orders"], window["labels"], window["customers"]
    )
    joined = joined.loc[
        joined.label_link.eq("both") & joined.is_fraud.notna()
    ].reset_index(drop=True)
    joined["is_fraud"] = joined["is_fraud"].astype(int)
    return joined


def window_inputs(history: list[dict], window: dict) -> dict:
    """Labelled orders of `window` with account age and loyalty, ready to compare.

    `window` and each item of `history` are `clean_window` results. Loyalty counts the
    earlier orders in `history`, which includes `window` itself.
    """
    joined = labelled_orders(window)
    age = account_age_days(
        joined["order_date"],
        joined["first_order_datetime"],
        raw_order_date=joined.get("raw_order_date"),
        raw_first_order_datetime=joined.get("raw_first_order_datetime"),
    )
    every = {
        name: pd.concat([part[name] for part in history], ignore_index=True)
        for name in _TABLES
    }
    loyal = loyalty_status(
        every["orders"],
        joined[window["orders"].columns],
        every["labels"],
        every["customers"],
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
