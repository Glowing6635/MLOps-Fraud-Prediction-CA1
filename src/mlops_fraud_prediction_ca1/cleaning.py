"""Deterministic, row-preserving cleaning of already-authorized input frames.

No reads, joins, split selection, fitted transformations or feature approval.
Consumers retain raw alongside cleaned values and select safe features later.
"""

from collections.abc import Collection, Mapping
from dataclasses import dataclass
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


@dataclass
class CleaningResult:
    """Aligned raw/cleaned copies and per-cell diagnostic status.

    Every row and its original index are retained, including duplicate indices.
    Source and batch identity belong to the caller. No rows are rejected.
    """

    raw: pd.DataFrame
    cleaned: pd.DataFrame
    status: pd.DataFrame


def _category_policy(
    category_values: Mapping[str, Collection[str]],
) -> dict[str, frozenset[str]]:
    policy = {"mobile_verified": frozenset({"true", "false"})}
    for column, values in category_values.items():
        if column not in _CATEGORIES - {"mobile_verified"}:
            raise ValueError(f"Unsupported configurable category: {column}")
        if isinstance(values, str):
            raise TypeError("Category vocabulary must be a collection, not a string")
        entries = frozenset(values)
        if "" in entries:
            raise ValueError("Vocabulary entries must be nonempty strings")
        standardise_categories(
            pd.Series([], dtype="string"),
            case="upper" if column == "country_code" else "lower",
            allowed_values=entries,
        )
        if column == "payment_method" and "on delivery" in entries:
            raise ValueError(
                "'on delivery' is unresolved; no semantic alias is approved"
            )
        policy[column] = entries
    return policy


def clean_table(
    frame: pd.DataFrame,
    *,
    category_values: Mapping[str, Collection[str]] = BASE_CATEGORY_VALUES,
) -> CleaningResult:
    """Clean copies with a fixed caller-supplied vocabulary, never batch statistics.

    Every present category except mobile_verified needs a canonical vocabulary;
    missing configuration raises rather than learning from this batch. These
    sets are consumer configuration, not assertions of an authoritative domain.
    Country codes become uppercase, other category text lowercase, with boundary
    whitespace removed. Missing/blank/unseen values become pd.NA. 'yes' and
    'on Delivery' remain unknown. Mobile verification becomes nullable Boolean after canonical normalization.

    Dates use the committed explicit parser; invalid/missing/unrecognized dates
    become NaT with separate statuses. Only strings in its documented formats
    are accepted, including case-insensitive English month tokens. Native
    datetimes and padded dates remain unsupported. Numeric columns use nullable
    Float64 for direct model conversion; invalid, nonfinite and values outside
    the exact integer range (-2**53, 2**53) become unknown with separate statuses.
    Original values always remain in raw. Labels accept only literal/numeric
    zero and one, using nullable Int8. Finite negatives, fractions and refunds
    inside the supported range remain unchanged. No rows are rejected.
    """
    if not frame.columns.is_unique:
        raise ValueError("Cleaning requires unique column names")
    policy = _category_policy(category_values)
    missing = (_CATEGORIES & set(frame.columns)) - policy.keys()
    if missing:
        raise ValueError(f"Explicit category vocabularies required: {sorted(missing)}")
    raw = frame.copy(deep=True)
    cleaned = frame.reset_index(drop=True).copy(deep=True)
    status = pd.DataFrame("unchanged", index=cleaned.index, columns=cleaned.columns)
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
                allowed_values=policy[column],
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
            status[column] = outcomes.map(
                {
                    "allowed": "normalized",
                    "unexpected": "unknown_unseen",
                    "blank": "unknown_blank",
                    "missing": "unknown_missing",
                }
            )
        elif column in _DATES:
            parsed = parse_dates(source, kind=_DATES[column])
            cleaned[column] = parsed["date"]
            status[column] = parsed["status"].map(
                {
                    "valid": "normalized",
                    "invalid": "unknown_invalid",
                    "unrecognized": "unknown_unrecognized",
                    "missing": "unknown_missing",
                }
            )
            blank = source.astype("string").str.strip().eq("").fillna(False)
            status.loc[blank, column] = "unknown_blank"
        elif column in _NUMBERS:
            text = source.astype("string").str.strip()
            numbers = pd.to_numeric(
                text, errors="coerce", dtype_backend="numpy_nullable"
            ).astype("Float64")
            nonfinite = numbers.isin([float("inf"), float("-inf")])
            precision = numbers.abs().ge(2**53).fillna(False) & ~nonfinite
            cleaned[column] = numbers.mask(nonfinite | precision)
            status[column] = "normalized"
            status.loc[numbers.isna(), column] = "unknown_invalid"
            status.loc[nonfinite, column] = "unknown_nonfinite"
            status.loc[precision, column] = "unknown_precision"
            status.loc[text.eq("").fillna(False), column] = "unknown_blank"
            status.loc[source.isna(), column] = "unknown_missing"
        elif column == "is_fraud":
            # Text matching rejects Boolean values and arbitrary numeric labels.
            text = source.astype("string").str.strip()
            labels = text.map({"0": 0, "1": 1, "0.0": 0, "1.0": 1})
            cleaned[column] = labels.astype("Int8")
            status[column] = "normalized"
            status.loc[labels.isna(), column] = "unknown_invalid"
            status.loc[text.eq("").fillna(False), column] = "unknown_blank"
            status.loc[source.isna(), column] = "unknown_missing"
    cleaned.index = frame.index.copy()
    status.index = frame.index.copy()
    return CleaningResult(raw, cleaned, status)


def clean_frame(
    frame: pd.DataFrame,
    *,
    category_values: Mapping[str, Collection[str]] = BASE_CATEGORY_VALUES,
) -> pd.DataFrame:
    """Return one cleaned table with original values in raw_<column> columns.

    Keeps every row and its index. Use the same frozen policy for base and
    incoming batches. Generated-column collisions raise instead of overwriting
    supplied data. clean_table additionally exposes per-cell diagnostic status.
    """
    raw_names = {column: f"raw_{column}" for column in frame.columns}
    if set(raw_names.values()) & set(frame.columns):
        raise ValueError("Input columns conflict with generated raw_ columns")
    result = clean_table(frame, category_values=category_values)
    return pd.concat([result.cleaned, result.raw.rename(columns=raw_names)], axis=1)


@dataclass
class CleanedBatch:
    """Three row-preserving tables cleaned with the same versioned policy."""

    policy_version: str
    tables: dict[str, CleaningResult]


def clean_batch(tables: Mapping[str, pd.DataFrame]) -> CleanedBatch:
    """Apply the frozen policy independently to three authorized input tables.

    Missing customer/label links do not discard orders: there are no joins.
    The caller retains source/batch provenance. Native database dates require
    serialization at the consumer boundary before applying this string policy.
    """
    if set(tables) != {"customers", "orders", "labels"}:
        raise ValueError(
            "Supply customers, orders and labels separately, including empty tables"
        )
    return CleanedBatch(
        CLEANING_POLICY_VERSION,
        {
            name: clean_table(tables[name], category_values=BASE_CATEGORY_VALUES)
            for name in ("customers", "orders", "labels")
        },
    )


def load_clean_base_data(
    data_dir, *, purpose="analysis", config_path=None
) -> CleanedBatch:
    """Guarded literal CSV adapter for analysis or a future #13 caller.

    Split selection happens before cleaning, through #6's public reader. The
    adapter returns every exposed row per table without inner joins. It does
    not demonstrate adoption by an external harness or a database loader.
    """
    from mlops_fraud_prediction_ca1.data import load_base_data

    frames = load_base_data(
        data_dir, purpose=purpose, config_path=config_path, raw_strings=True
    )
    return clean_batch(
        dict(zip(("customers", "orders", "labels"), frames, strict=True)),
    )


def summarize_batch(batch: CleanedBatch) -> pd.DataFrame:
    """Before/after counts for supplied rows, with unknown cells by column.

    Unknown totals are cells, not distinct rows. Reader exclusions and downstream
    join decisions are separate from this row-preserving transformation.
    """
    records = []
    for name, result in batch.tables.items():
        unknowns = {
            column: int(result.status[column].str.startswith("unknown_").sum())
            for column in result.status
        }
        records.append(
            {
                "table": name,
                "before_rows": len(result.raw),
                "after_rows": len(result.cleaned),
                "unknown_cells": sum(unknowns.values()),
                "unknown_by_column": ", ".join(
                    f"{column}: {count}" for column, count in unknowns.items() if count
                )
                or "none",
            }
        )
    return pd.DataFrame(records)
