"""Deterministic, row-preserving cleaning of already-authorized input frames.

No reads, joins, split selection, fitted transformations or feature approval.
Consumers retain raw alongside cleaned values and select safe features later.
"""

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import pandas as pd

from mlops_fraud_prediction_ca1.temporal import parse_dates

# Versioned engineering policy from A1 examples and the recorded PBI #3 inventory.
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
    """Aligned copies; source_positions identifies rows even with duplicate indices.

    Positions are local to this call, not globally unique ingestion identifiers.
    Retain the source batch identifier separately. No rows are rejected. status
    gives per-cell outcomes; audit aggregates them without influencing cleaning.
    Identifiers and unrecognized columns are copied unchanged; labels use Int8.
    """

    raw: pd.DataFrame
    cleaned: pd.DataFrame
    status: pd.DataFrame
    source_positions: tuple[int, ...]
    audit: pd.DataFrame


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
        for entry in entries:
            if not isinstance(entry, str) or not entry:
                raise ValueError("Vocabulary entries must be nonempty strings")
            normalized = (
                entry.strip().upper()
                if column == "country_code"
                else entry.strip().lower()
            )
            if normalized != entry:
                raise ValueError(
                    "Vocabulary entries must already use canonical case/whitespace"
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
    Country codes become uppercase, other categories lowercase, with boundary
    whitespace removed. Missing/blank/unseen values become pd.NA. 'yes' and
    'on Delivery' remain unknown. Boolean mobile values normalize textually.

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
            text = source.astype("string").str.strip()
            text = text.str.upper() if column == "country_code" else text.str.lower()
            known = text.isin(policy[column])
            cleaned[column] = text.where(known, pd.NA)
            status[column] = "unknown_unseen"
            status.loc[known, column] = "normalized"
            status.loc[text.eq("").fillna(False), column] = "unknown_blank"
            status.loc[source.isna(), column] = "unknown_missing"
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
    records = [
        {"column": column, "outcome": outcome, "rows": int(count), "rejected_rows": 0}
        for column in status.columns
        for outcome, count in status[column].value_counts(sort=False).items()
    ]
    audit = pd.DataFrame(
        records, columns=["column", "outcome", "rows", "rejected_rows"]
    )
    cleaned.index = frame.index.copy()
    status.index = frame.index.copy()
    return CleaningResult(raw, cleaned, status, tuple(range(len(frame))), audit)


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
    """Three independent row-preserving tables with caller-supplied provenance.

    A durable row identity is (source, batch_id, table_name, source_position).
    This is not a database load, a join, a deduplication or feature preparation.
    """

    batch_id: str
    source: str
    policy_version: str
    tables: dict[str, CleaningResult]


def clean_batch(
    tables: Mapping[str, pd.DataFrame],
    *,
    batch_id: str,
    source: str,
) -> CleanedBatch:
    """Apply the frozen v3 policy to literal, already-authorized tables.

    Missing labels/customer links never discard order rows: there are no joins.
    New ingestion batches can use this without applying base evaluation dates.
    Native database dates must be explicitly serialized to documented formats
    at that consumer's boundary; passing them here is not supported by v3.
    """
    for name, value in (("batch_id", batch_id), ("source", source)):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a nonblank string")
    if set(tables) != {"customers", "orders", "labels"}:
        raise ValueError(
            "Supply customers, orders and labels separately, including empty tables"
        )
    return CleanedBatch(
        batch_id,
        source,
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
        batch_id="base",
        source=str(data_dir),
    )


def complete_key_mask(result: CleaningResult, keys: Sequence[str]) -> pd.Series:
    """Mark complete cleaned keys without dropping rows or allowing null joins.

    Consumers must still enforce unique composite keys and temporal eligibility.
    IDs stay opaque: whitespace is checked for blankness, never stripped from
    a nonblank identifier, and '001' remains different from '1'.
    """
    if not keys or len(set(keys)) != len(keys):
        raise ValueError("Specify distinct key columns")
    complete = pd.Series(True, index=result.cleaned.index)
    for key in keys:
        values = result.cleaned[key]
        complete &= values.map(
            lambda value: isinstance(value, str) and bool(value.strip())
        )
    return complete


def summarize_batch(batch: CleanedBatch) -> pd.DataFrame:
    """Before/after counts for supplied rows, with unknown cells by column.

    Unknown totals are cells, not distinct rows. Rejection counts describe
    cleaning only, not rows withheld earlier by a reader or later by a join.
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
                "rejected_rows": len(result.raw) - len(result.cleaned),
                "unknown_cells": sum(unknowns.values()),
                "unknown_by_column": ", ".join(
                    f"{column}: {count}" for column, count in unknowns.items() if count
                )
                or "none",
            }
        )
    return pd.DataFrame(records)
