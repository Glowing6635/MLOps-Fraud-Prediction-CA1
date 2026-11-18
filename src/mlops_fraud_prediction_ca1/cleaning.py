"""Deterministic, row-preserving cleaning of already-authorized input frames.

No reads, joins, split selection, fitted transformations or feature approval.
Consumers retain raw alongside cleaned values and select safe features later.
"""

from copy import deepcopy
from types import MappingProxyType

import pandas as pd

from mlops_fraud_prediction_ca1.categories import standardise_categories
from mlops_fraud_prediction_ca1.temporal import parse_dates

# Versioned engineering policy from the supplied field definitions and the recorded PBI #3 inventory.
# This is an explicit vocabulary, not a business-authoritative domain or a fit.
# See docs/cleaning.md for provenance and the prior full-extract EDA limitation.
CLEANING_POLICY_VERSION = "pbi12-v3"
BASE_CATEGORY_VALUES = MappingProxyType(
    {
        "country_code": frozenset({"MY", "PH", "PK", "TH", "BD"}),
        "collect_type": frozenset({"delivery", "pickup", "self-pick-up"}),
        "payment_method": frozenset(
            {
                "payment on delivery",
                "cybersource",
                "online banking",
                "credit card",
                "antfinancial gcash",
                "generic creditcard",
                "balance",
                "antfinancial bkash",
                "antfinancial truemoney",
                "boost wallet",
                "jazzcash wallet",
                "invoice",
                "adyen hpp molpay",
                "no payment",
                "cybersource applepay",
                "paypal",
                "xendit directdebit",
            }
        ),
    }
)


_CATEGORIES = {"country_code", "collect_type", "payment_method", "mobile_verified"}
_NUMBERS = {
    "num_orders_last_50days",
    "num_cancelled_orders_last_50days",
    "num_refund_orders_last_50days",
    "num_associated_customers",
    "num_items_ordered",
    "total_payment_last_50days",
    "order_value",
    "refund_value",
}
_DATES = {"order_date": "order", "first_order_datetime": "customer"}


def clean_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return cleaned columns and exact originals in raw_<column> columns.

    Keep every row, order and index, including duplicate indices, without
    mutating the source. Duplicate columns and generated-name collisions raise.
    Country text becomes uppercase, other categories lowercase, with boundary
    spaces removed. Frozen engineering vocabularies do not learn from batches
    or assert Operations-approved domains. Missing, blank and unseen values
    become pd.NA; 'yes' and 'on Delivery' remain unknown. Mobile verification
    accepts normalized true/false and uses nullable boolean.

    Dates accept only the explicit parser's string formats, including English
    month tokens irrespective of case. Invalid, native and padded dates become
    NaT. Numeric fields use nullable Float64; invalid, nonfinite and magnitudes
    at or above 2**53 become unknown. Labels accept literal/numeric zero and one
    as nullable Int8. Finite negatives, fractions and refunds within the range
    remain unchanged. IDs and other fields retain their supplied values.
    """
    if not frame.columns.is_unique:
        raise ValueError("Cleaning requires unique column names")
    raw_names = {column: f"raw_{column}" for column in frame.columns}
    if set(raw_names.values()) & set(frame.columns):
        raise ValueError("Input columns conflict with generated raw_ columns")
    raw = frame.copy(deep=True)
    cleaned = frame.reset_index(drop=True).copy(deep=True)
    for column in cleaned.columns:
        source = cleaned[column].copy()
        if column in _CATEGORIES:
            # Normalize unique textual values through the shared category policy,
            # then map them back positionally, preserving duplicate caller indices.
            # Convert before grouping so True and numeric 1 remain distinct tokens.
            text = source.astype("string")
            preview = standardise_categories(
                text,
                case="upper" if column == "country_code" else "lower",
                allowed_values=(
                    {"true", "false"}
                    if column == "mobile_verified"
                    else BASE_CATEGORY_VALUES[column]
                ),
            ).set_index("original_value")
            outcomes = text.map(preview["status"])
            normalized = text.map(preview["proposed_value"]).astype("string")
            known = outcomes.eq("allowed")
            if column == "mobile_verified":
                cleaned[column] = normalized.map({"true": True, "false": False}).astype(
                    "boolean"
                )
            else:
                cleaned[column] = normalized.where(known, pd.NA)
        elif column in _DATES:
            parsed = parse_dates(source, kind=_DATES[column])
            cleaned[column] = parsed["date"]
        elif column in _NUMBERS:
            text = source.astype("string").str.strip()
            numbers = pd.to_numeric(
                text, errors="coerce", dtype_backend="numpy_nullable"
            ).astype("Float64")
            nonfinite = numbers.isin([float("inf"), float("-inf")])
            precision = numbers.abs().ge(2**53).fillna(False) & ~nonfinite
            cleaned[column] = numbers.mask(nonfinite | precision)
        elif column == "is_fraud":
            # Text matching rejects Boolean values and arbitrary numeric labels.
            text = source.astype("string").str.strip()
            labels = text.map({"0": 0, "1": 1, "0.0": 0, "1.0": 1})
            cleaned[column] = labels.astype("Int8")
    cleaned.index = frame.index.copy()
    result = pd.concat([cleaned, raw.rename(columns=raw_names)], axis=1)
    result.attrs = deepcopy(frame.attrs)
    return result
