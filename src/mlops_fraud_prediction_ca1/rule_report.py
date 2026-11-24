"""Run the #18 comparison on the validation window from a file of model scores.

    uv run python -m mlops_fraud_prediction_ca1.rule_report --data-dir data \\
        --scores scores.csv --k 250 [--threshold X --threshold-source "..."]

`scores.csv` has `country_code`, `order_id` and `score` for every labelled validation
order (higher means riskier). Without `--threshold`, X is fitted on the validation
window and reported as a validation-tuned proxy.
"""

import argparse
import sys

import pandas as pd

from mlops_fraud_prediction_ca1.cleaning import clean_frame
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

_TABLES = ("customers", "orders", "labels")


def _clean(data_dir, purpose):
    frames = load_base_data(data_dir, purpose=purpose, raw_strings=True)
    return {
        name: clean_frame(frame) for name, frame in zip(_TABLES, frames, strict=True)
    }


TRAINING_QUANTILES = (0.90, 0.95, 0.99, 0.995, 0.999, 0.9999)


def training_quantile_grid(data_dir) -> tuple[float, ...]:
    """The D18-03 grid: quantiles of the training window's order_value, no labels."""
    value = pd.to_numeric(_clean(data_dir, "train")["orders"]["order_value"])
    grid = value.dropna().quantile(list(TRAINING_QUANTILES)).round(2)
    return tuple(sorted(set(grid.tolist())))


def validation_inputs(data_dir) -> dict:
    """Labelled validation orders with account age and loyalty, ready to compare."""
    train, validation = _clean(data_dir, "train"), _clean(data_dir, "validation")
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
        help="Candidate X values: D18-01's fixed grid or D18-03's training quantiles.",
    )
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--threshold-source")
    args = parser.parse_args(argv)
    if (args.threshold is None) != (args.threshold_source is None):
        sys.exit("--threshold and --threshold-source go together.")
    inputs = validation_inputs(args.data_dir)
    orders = inputs["orders"]
    scores = pd.read_csv(args.scores, dtype={"country_code": str, "order_id": str})
    merged = (
        orders[["country_code", "order_id"]]
        .astype(str)
        .merge(
            scores, on=["country_code", "order_id"], how="left", validate="one_to_one"
        )
    )
    if merged["score"].isna().any():
        sys.exit(f"{int(merged['score'].isna().sum())} orders have no score.")
    if args.threshold is None:
        grid = (
            training_quantile_grid(args.data_dir)
            if args.grid == "training-quantiles"
            else THRESHOLD_GRID
        )
        fit = select_threshold(orders, inputs["age"], args.k, grid)
        print(f"Candidate X values ({args.grid} grid, validation-tuned proxy):")  # noqa: T201
        print(fit["candidates"].to_string(index=False))  # noqa: T201
        threshold, source = fit["x"], "validation-tuned proxy, not the deployed value"
    else:
        threshold, source = args.threshold, args.threshold_source
    result = compare(
        orders,
        merged["score"],
        inputs["age"],
        inputs["loyalty"],
        args.k,
        threshold,
        source,
    )
    print(f"Account age status counts: {inputs['age_status']}")  # noqa: T201
    print(format_comparison(result))  # noqa: T201


if __name__ == "__main__":
    main()
