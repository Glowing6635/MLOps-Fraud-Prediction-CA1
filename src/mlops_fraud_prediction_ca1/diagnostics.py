"""Diagnostic tables every logged model version carries (issue #22).

For the validation window a run logs a precision-recall curve, a confusion matrix at K
and the errors by country, payment method and order-value band. All of it comes from the
same top-K selection the evaluator makes (`top_k.select_top_k`), so a slice's "caught"
adds up to the run's `fraud_caught_at_k`. `training.train` logs these for v1 and v2 alike.

Errors: a missed fraud is a fraud order not selected; a false alarm is a selected order
that is not fraud.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from mlops_fraud_prediction_ca1.top_k import select_top_k

# Order-value bands in the data's currency unit (median 5.45, 99th percentile 30.7), as
# docs/model-v2-diagnosis.md states them: lower bound included, upper bound excluded.
ORDER_VALUE_EDGES = (0, 3, 5, 8, 13, 30, np.inf)
ORDER_VALUE_BANDS = ("[0,3)", "[3,5)", "[5,8)", "[8,13)", "[13,30)", "[30,∞)")
# The curve is thinned to this many points for the logged table.
CURVE_POINTS = 500
SLICES = ("country", "payment_method", "order_value_band")


@dataclass(frozen=True)
class Diagnostics:
    """The tables of one run: curve, confusion matrix at K and the three slice tables."""

    k: int
    curve: pd.DataFrame
    confusion: pd.DataFrame
    slices: dict[str, pd.DataFrame]


def order_value_band(order_value: pd.Series) -> pd.Series:
    """The band each order value falls in; a missing value is "missing"."""
    band = pd.cut(
        pd.to_numeric(order_value),
        bins=list(ORDER_VALUE_EDGES),
        labels=list(ORDER_VALUE_BANDS),
        right=False,
    )
    return band.astype(object).where(band.notna(), "missing")


def build_diagnostics(
    keys: pd.DataFrame, features: pd.DataFrame, scores, k: int
) -> Diagnostics:
    """Build the tables for validation orders.

    `keys` holds `order_date`, `country_code`, `order_id` and `is_fraud`; `features` holds
    `payment_method` and `order_value`; `scores` is the model's fraud score per order.
    """
    scores = np.asarray(scores, dtype=float)
    fraud = keys["is_fraud"].to_numpy().astype(bool)
    selected = select_top_k(keys, scores, k)
    caught = selected & fraud
    precision, recall, _ = precision_recall_curve(fraud, scores)
    # Thin evenly along the curve, always keeping its two ends.
    take = np.unique(np.linspace(0, len(precision) - 1, CURVE_POINTS).astype(int))
    curve = pd.DataFrame({"recall": recall[take], "precision": precision[take]})
    tp, fp = int(caught.sum()), int((selected & ~fraud).sum())
    fn, tn = int((~selected & fraud).sum()), int((~selected & ~fraud).sum())
    confusion = pd.DataFrame(
        {
            "actual_fraud": [tp, fn],
            "actual_not_fraud": [fp, tn],
        },
        index=pd.Index(["selected", "not_selected"], name=f"at_k_{k}"),
    ).reset_index()
    columns = {
        "country": keys["country_code"].astype(object),
        "payment_method": features["payment_method"].astype(object),
        "order_value_band": order_value_band(features["order_value"]),
    }
    slices = {
        name: _slice_table(name, values, fraud, selected, caught)
        for name, values in columns.items()
    }
    return Diagnostics(k, curve, confusion, slices)


def _slice_table(name, values, fraud, selected, caught) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            name: pd.Series(values).fillna("missing").to_numpy(),
            "orders": 1,
            "fraud": fraud.astype(int),
            "selected": selected.astype(int),
            "caught": caught.astype(int),
        }
    )
    table = frame.groupby(name, sort=False).sum().reset_index()
    table["missed_fraud"] = table["fraud"] - table["caught"]
    table["false_alarms"] = table["selected"] - table["caught"]
    table["fraud_rate"] = table["fraud"] / table["orders"]
    table["precision_at_k"] = (table["caught"] / table["selected"]).where(
        table["selected"] > 0
    )
    table["recall_at_k"] = (table["caught"] / table["fraud"]).where(table["fraud"] > 0)
    if name == "order_value_band":
        order = {band: i for i, band in enumerate((*ORDER_VALUE_BANDS, "missing"))}
        return table.sort_values(name, key=lambda s: s.map(order), ignore_index=True)
    return table.sort_values("orders", ascending=False, ignore_index=True)


def log_diagnostics(diagnostics: Diagnostics) -> None:
    """Log the tables to the active MLflow run, as table artifacts under `diagnostics/`."""
    import mlflow

    mlflow.log_table(diagnostics.curve, "diagnostics/precision_recall_curve.json")
    mlflow.log_table(
        diagnostics.confusion, f"diagnostics/confusion_matrix_at_k{diagnostics.k}.json"
    )
    for name, table in diagnostics.slices.items():
        mlflow.log_table(table, f"diagnostics/errors_by_{name}.json")
    mlflow.set_tag(
        "order_value_bands", ", ".join(ORDER_VALUE_BANDS) + " (lower bound included)"
    )
