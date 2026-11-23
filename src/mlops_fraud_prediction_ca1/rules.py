"""Baseline-rule inputs: account age in whole calendar days (PBI #18, #123)."""

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype

from mlops_fraud_prediction_ca1.temporal import parse_dates


def _dates(values: pd.Series, kind: str) -> tuple[pd.Series, pd.Series]:
    """Return (midnight date, status) for raw text or already-cleaned datetimes."""
    if is_datetime64_any_dtype(values):
        status = pd.Series(
            values.notna().map({True: "valid", False: "missing"}), index=values.index
        )
        return values.dt.normalize(), status
    parsed = parse_dates(values, kind=kind)
    return parsed["date"], parsed["status"]


def account_age_days(order_date: pd.Series, first_order_datetime: pd.Series):
    """Whole days from the date part of first_order_datetime to order_date.

    Accepts raw strings (parsed with the explicit formats of ``parse_dates``) or
    the datetimes produced by ``clean_frame``; times of day are ignored, so a
    same-day order has age 0 whatever the clock times. Returns a frame with
    ``account_age_days`` (nullable Int64) and ``account_age_status``:
    ``valid``, ``missing_order_date``, ``missing_first_order``,
    ``unusable_order_date``, ``unusable_first_order`` (unrecognised or invalid
    text) or ``negative``.

    Policy: a missing, unusable or negative age is NA, never zero and never
    repaired; the reason is kept in the status. A negative age means the first
    order is dated after this order, so the dates disagree and, as in D4-03, the
    customer value is withheld. An NA age cannot satisfy an "account younger
    than N days" condition, so the rule does not flag on it. Inputs unchanged.
    """
    if not order_date.index.equals(first_order_datetime.index):
        raise ValueError("order_date and first_order_datetime must share an index")
    order, order_status = _dates(order_date, "order")
    first, first_status = _dates(first_order_datetime, "customer")
    age = (order - first.dt.normalize()).dt.days
    status = pd.Series("valid", index=order_date.index, dtype=object)
    status[age < 0] = "negative"
    status[first_status.eq("unrecognized") | first_status.eq("invalid")] = (
        "unusable_first_order"
    )
    status[first_status.eq("missing")] = "missing_first_order"
    status[order_status.eq("unrecognized") | order_status.eq("invalid")] = (
        "unusable_order_date"
    )
    status[order_status.eq("missing")] = "missing_order_date"
    age = age.astype("Int64").mask(status.ne("valid"), pd.NA)
    return pd.DataFrame({"account_age_days": age, "account_age_status": status})
