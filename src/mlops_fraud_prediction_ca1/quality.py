"""Data-quality evidence and semantic checks without cleaning source values."""

import pandas as pd

from .categories import standardise_categories
from .logic import DEFAULT_NUMERIC_COLUMNS
from .temporal import fraud_rates


CATEGORY_COLUMNS = {
    "customers": ("country_code", "mobile_verified"),
    "orders": ("country_code", "collect_type", "payment_method"),
    "labels": ("country_code", "is_fraud"),
}
COUNT_COLUMNS = {
    "num_orders_last_50days", "num_cancelled_orders_last_50days",
    "num_refund_orders_last_50days", "num_associated_customers", "num_items_ordered",
}


def quality_reports(datasets: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Profile inferred types, missingness, copies, numeric validity and labels.

    Use raw text category reports separately when original CSV spelling matters.
    Missingness here follows the loader's pandas inference. Numeric coercions
    exist only in temporary check Series; failures and infinities are counted.
    A count must be finite and integral; negativity is reported independently.
    The spec defines mobile verification as True/False and labels as 0/1, but
    provides examples rather than closed collection/payment category domains.
    """
    schema, numeric, missing, copies = [], [], [], []
    for name, frame in datasets.items():
        copies.append({"dataset": name, "rows": len(frame),
                       "extra_copies": int(frame.duplicated().sum()),
                       "rows_in_copy_groups": int(frame.duplicated(keep=False).sum())})
        for column in frame:
            values = frame[column]
            expected = (
                "text identifier" if column.endswith("_id") else
                "calendar date" if column == "order_date" else
                "timestamp" if column == "first_order_datetime" else
                "boolean True/False" if column == "mobile_verified" else
                "binary target 0/1" if column == "is_fraud" else
                "integer count" if column in COUNT_COLUMNS else
                "decimal amount" if column in DEFAULT_NUMERIC_COLUMNS.get(name, ()) else
                "category"
            )
            schema.append({"dataset": name, "column": column,
                           "inferred_dtype": str(values.dtype), "meaning": expected})
            count = int(values.isna().sum())
            missing.append({"dataset": name, "column": column, "rows": len(values),
                            "missing": count,
                            "missing_rate": count / len(values) if len(values) else float("nan")})
            if column not in DEFAULT_NUMERIC_COLUMNS.get(name, ()):
                continue
            converted = pd.to_numeric(values, errors="coerce")
            infinite = converted.isin([float("inf"), -float("inf")])
            finite = converted.notna() & ~infinite
            numeric.append({"dataset": name, "column": column, "rows": len(values),
                            "missing": count,
                            "conversion_failures": int((values.notna() & converted.isna()).sum()),
                            "infinite": int(infinite.sum()),
                            "negative": int((finite & converted.lt(0)).sum()),
                            "nonintegral_count": int((finite & converted.mod(1).ne(0)).sum())
                            if column in COUNT_COLUMNS else None})
    labels = datasets["labels"][["is_fraud"]].copy()
    labels["source_row"] = range(len(labels))
    labels["population"] = "raw label rows"
    distribution = fraud_rates(labels, ["population"], entity_keys=["source_row"])
    distribution["nonfraud_labels"] = distribution.valid_labels - distribution.fraud_labels
    return {"types": pd.DataFrame(schema), "missing": pd.DataFrame(missing),
            "copies": pd.DataFrame(copies), "numeric_validity": pd.DataFrame(numeric),
            "class_balance": distribution}


def raw_category_reports(raw_categories: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Inventory all raw values and formatting proposals, without semantic aliases.

    Read the selected CSV columns as strings with keep_default_na=False. Empty
    fields remain visible and are classified as blank. Only spec-defined binary
    domains are validated; country/collection/payment semantic validity remains
    unchecked. In particular 'yes' and 'on Delivery' are never silently remapped.
    """
    reports = []
    for name, columns in CATEGORY_COLUMNS.items():
        for column in columns:
            allowed = {"true", "false"} if column == "mobile_verified" else (
                {"0", "1"} if column == "is_fraud" else None)
            preview = standardise_categories(
                raw_categories[name][column],
                case="upper" if column == "country_code" else "lower",
                allowed_values=allowed,
            )
            preview.insert(0, "column", column)
            preview.insert(0, "dataset", name)
            reports.append(preview)
    return pd.concat(reports, ignore_index=True)
