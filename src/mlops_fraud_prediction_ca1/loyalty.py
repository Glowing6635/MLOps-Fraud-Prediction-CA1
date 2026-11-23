"""Historical loyal-customer status for the #18 comparison report (#125).

Implements the frozen definition in docs/baseline-rule.md (criterion 3). This
only labels orders from history for reporting; it is never a model input and
computes no comparison result.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from mlops_fraud_prediction_ca1.relationships import link_orders_customers
from mlops_fraud_prediction_ca1.splitting import _calendar_dates, load_policy

LOYAL_MIN_PRIOR_ORDERS = 2
_ORDER_KEYS = ["country_code", "order_id"]
_DAY_OFFSET = 1 << 18  # shifts parser-range day numbers (about +-1e5) positive
_DAY_SPAN = 1 << 20  # exceeds any shifted day number, so customer blocks never mix


def _day_numbers(dates: pd.Series) -> np.ndarray:
    """Whole days since the epoch, shifted positive; -1 where the date is NaT."""
    seconds = dates.dt.as_unit("s").astype("int64") // 86_400
    return np.where(dates.isna().to_numpy(), -1, seconds.to_numpy() + _DAY_OFFSET)


def _usable_dates(dates: pd.Series) -> pd.Series:
    """Calendar dates as ``_calendar_dates`` reads them; out-of-range ones are NaT.

    A value outside the datetime64[ns] range (for example 9999-12-31) makes the
    shared parser raise for the whole batch. Bisect so only those rows become NaT
    and every other row is read exactly as before (same formats, same timezone
    rule), without a Python-level pass over rows that are fine.
    """
    dates = dates.reset_index(drop=True)
    try:
        return _calendar_dates(dates)
    except pd.errors.OutOfBoundsDatetime:
        if len(dates) == 1:
            return pd.Series(pd.NaT, index=dates.index, dtype="datetime64[ns]")
    mid = len(dates) // 2
    return pd.concat(
        [_usable_dates(dates.iloc[:mid]), _usable_dates(dates.iloc[mid:])],
        ignore_index=True,
    )


def _safe_orders(orders, labels, customers) -> pd.DataFrame:
    if "customer_id" in orders:
        raise ValueError("orders must not carry customer_id; it comes from labels")
    linked, _ = link_orders_customers(orders, labels, customers, normalize_country=True)
    linked = linked.reset_index(drop=True)
    linked["_day"] = _day_numbers(_usable_dates(linked["order_date"]))
    return linked


def _normalised_keys(frame: pd.DataFrame) -> pd.DataFrame:
    keys = frame[_ORDER_KEYS].copy()
    keys["country_code"] = keys["country_code"].astype("string").str.strip().str.upper()
    return keys


def _cross_input_conflicts(history, comparison) -> pd.MultiIndex:
    """Keys whose records differ between the inputs (identical overlaps are fine).

    Each input is already reduced to safe keys on its own; the same key may still
    carry different records (for example another date) in the other input, so an
    earlier version of an order could count as its own prior. Records are compared
    on the columns both inputs share, with country and date canonicalised.
    """
    shared = [c for c in history.columns if c in comparison.columns]
    parts = []
    for frame in (history, comparison):
        part = frame[shared].copy().reset_index(drop=True)
        part[_ORDER_KEYS] = _normalised_keys(part)
        part["order_date"] = _usable_dates(part["order_date"])
        parts.append(part)
    both = pd.concat(parts, ignore_index=True)
    both = both.loc[both[_ORDER_KEYS].notna().all(axis=1)].drop_duplicates()
    conflict = both.duplicated(_ORDER_KEYS, keep=False)
    return pd.MultiIndex.from_frame(both.loc[conflict, _ORDER_KEYS].drop_duplicates())


def _in_keys(frame: pd.DataFrame, keys: pd.MultiIndex) -> np.ndarray:
    return pd.MultiIndex.from_frame(_normalised_keys(frame)).isin(keys)


def loyalty_status(
    history: pd.DataFrame,
    comparison: pd.DataFrame,
    labels: pd.DataFrame,
    customers: pd.DataFrame,
    *,
    config_path: str | Path | None = None,
) -> pd.DataFrame:
    """Return one loyalty row per ``comparison`` row, in order, with its index.

    ``history`` and ``comparison`` are already-scoped order frames with
    ``country_code``, ``order_id`` and ``order_date`` (raw text or dates);
    ``labels`` supplies ``customer_id`` per order and ``customers`` the known
    (``country_code``, ``customer_id``) pairs. The customer is that safe pair,
    linked with ``link_orders_customers(normalize_country=True)``. A prior order
    is a distinct safe history order of the same customer dated on or after the
    frozen training start and strictly before the comparison order's date.

    Columns: ``country_code`` (normalised), ``order_id``, ``customer_id``,
    ``prior_order_count`` (Int64, NA when unknown), ``loyalty_status``
    (``loyal``, ``not_loyal``, ``unknown``) and ``reason`` (empty unless
    unknown: ``unsafe_order_key``, ``no_label_link``, ``no_customer_link`` or
    ``order_date_unavailable``). Unknown is never loyal and rows are never
    dropped or multiplied; exact duplicate comparison rows share one result.
    The two-prior-order threshold is the frozen definition and is not a parameter.
    An order key whose records differ between ``history`` and ``comparison`` is
    unsafe: it counts as no one's prior and its comparison rows are unknown.
    """
    start = pd.Timestamp(load_policy(config_path).train_start)
    hist = _safe_orders(history, labels, customers)
    comp = _safe_orders(comparison, labels, customers)
    unsafe_keys = _cross_input_conflicts(history, comparison)
    hist = hist.loc[~_in_keys(hist, unsafe_keys)]

    prior = hist.loc[
        hist.customer_link.eq("both")
        & hist._day.ge(_day_numbers(pd.Series([start]))[0]),
        ["country_code", "customer_id", "_day"],
    ]
    cust_cols = ["country_code", "customer_id"]
    codes, _ = pd.factorize(
        pd.MultiIndex.from_frame(pd.concat([prior[cust_cols], comp[cust_cols]]))
    )
    prior_code = codes[: len(prior)].astype("int64")
    comp_code = codes[len(prior) :].astype("int64")
    prior_keys = np.sort(prior_code * _DAY_SPAN + prior["_day"].to_numpy())
    comp_day = comp["_day"].to_numpy()
    block = comp_code * _DAY_SPAN
    count = np.searchsorted(prior_keys, block + comp_day, "left") - np.searchsorted(
        prior_keys, block, "left"
    )

    reason = pd.Series("", index=comp.index, dtype=object)
    reason[comp.customer_link.ne("both")] = "no_customer_link"
    reason[comp.label_link.ne("both")] = "no_label_link"
    reason[comp["_day"].lt(0)] = "order_date_unavailable"
    reason[_in_keys(comp, unsafe_keys)] = "unsafe_order_key"
    known = reason.eq("")
    safe = pd.DataFrame(
        {
            "country_code": comp["country_code"],
            "order_id": comp["order_id"],
            "customer_id": comp["customer_id"],
            "prior_order_count": pd.Series(count, dtype="Int64").mask(~known),
            "reason": reason,
        }
    )
    safe["loyalty_status"] = np.where(
        ~known,
        "unknown",
        np.where(
            safe.prior_order_count.ge(LOYAL_MIN_PRIOR_ORDERS).fillna(False),
            "loyal",
            "not_loyal",
        ),
    )

    keys = comparison[["country_code", "order_id"]].copy()
    keys["country_code"] = keys["country_code"].astype("string").str.strip().str.upper()
    keys["_row"] = np.arange(len(keys))
    out = keys.merge(
        safe, on=_ORDER_KEYS, how="left", validate="many_to_one"
    ).sort_values("_row")
    unsafe = out["loyalty_status"].isna()
    out.loc[unsafe, "loyalty_status"] = "unknown"
    out.loc[unsafe, "reason"] = "unsafe_order_key"
    out = out.drop(columns="_row")
    out.index = comparison.index
    return out[
        [*_ORDER_KEYS, "customer_id", "prior_order_count", "loyalty_status", "reason"]
    ]
