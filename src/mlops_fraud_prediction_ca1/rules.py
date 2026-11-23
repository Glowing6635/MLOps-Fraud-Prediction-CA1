"""Baseline-rule inputs: account age in whole calendar days (PBI #18, #123)."""

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype

from mlops_fraud_prediction_ca1.temporal import parse_dates


def _days(values: pd.Series) -> pd.Series:
    """Calendar day number (Int64, NA for NaT) of datetimes, without normalising.

    Floors epoch seconds to whole days, so no midnight value is built and times
    of day on the earliest or latest representable day cannot overflow.
    """
    seconds = values.dt.as_unit("s")
    days = seconds.astype("int64") // 86_400
    return days.astype("Int64").mask(values.isna())


def _dates(
    values: pd.Series, kind: str, raw: pd.Series | None
) -> tuple[pd.Series, pd.Series]:
    """Return (day number, status) for raw text or already-cleaned datetimes.

    Cleaned datetimes turn missing, unrecognised and impossible dates into the
    same NaT. ``raw`` (``clean_frame``'s ``raw_<column>``) restores the reason;
    without it, or if it contradicts the NaT, the status is ``unavailable``.
    """
    if not is_datetime64_any_dtype(values):
        parsed = parse_dates(values, kind=kind)
        return _days(parsed["date"]), parsed["status"]
    status = pd.Series("valid", index=values.index, dtype=object)
    status[values.isna()] = "unavailable"
    if raw is not None:
        if not raw.index.equals(values.index):
            raise ValueError("raw dates must share the index of the cleaned dates")
        reason = parse_dates(raw, kind=kind)["status"]
        known = values.isna() & reason.isin(["missing", "unrecognized", "invalid"])
        status[known] = reason[known]
    return _days(values), status


def account_age_days(
    order_date: pd.Series,
    first_order_datetime: pd.Series,
    *,
    raw_order_date: pd.Series | None = None,
    raw_first_order_datetime: pd.Series | None = None,
):
    """Whole days from the date part of first_order_datetime to order_date.

    Accepts raw strings (parsed with the explicit formats of ``parse_dates``) or
    the datetimes produced by ``clean_frame``; times of day are ignored, so a
    same-day order has age 0 whatever the clock times. Returns a frame with
    ``account_age_days`` (nullable Int64) and ``account_age_status``:
    ``valid``, ``missing_order_date``, ``missing_first_order``,
    ``unusable_order_date``, ``unusable_first_order`` (unrecognised or invalid
    text), ``unavailable_order_date``, ``unavailable_first_order`` or
    ``negative``.

    Cleaned datetimes cannot tell a missing date from a malformed one (both are
    NaT). Pass ``clean_frame``'s ``raw_order_date`` / ``raw_first_order_datetime``
    to recover the reason; otherwise a NaT is ``unavailable_*``: no age, and no
    claim that the value was truly missing. Ages use calendar day numbers, so the
    whole parser range (1677-09-21 to 2262-04-11) is safe.

    Policy: a missing, unusable or negative age is NA, never zero and never
    repaired; the reason is kept in the status. A negative age means the first
    order is dated after this order, so the dates disagree and, as in D4-03, the
    customer value is withheld. An NA age cannot satisfy an "account younger
    than N days" condition, so the rule does not flag on it. Inputs unchanged.
    """
    if not order_date.index.equals(first_order_datetime.index):
        raise ValueError("order_date and first_order_datetime must share an index")
    order, order_status = _dates(order_date, "order", raw_order_date)
    first, first_status = _dates(
        first_order_datetime, "customer", raw_first_order_datetime
    )
    age = order - first
    status = pd.Series("valid", index=order_date.index, dtype=object)
    status[age.lt(0).fillna(False).astype(bool)] = "negative"
    status[first_status.eq("unrecognized") | first_status.eq("invalid")] = (
        "unusable_first_order"
    )
    status[first_status.eq("missing")] = "missing_first_order"
    status[first_status.eq("unavailable")] = "unavailable_first_order"
    status[order_status.eq("unrecognized") | order_status.eq("invalid")] = (
        "unusable_order_date"
    )
    status[order_status.eq("missing")] = "missing_order_date"
    status[order_status.eq("unavailable")] = "unavailable_order_date"
    age = age.mask(status.ne("valid"), pd.NA)
    return pd.DataFrame({"account_age_days": age, "account_age_status": status})
