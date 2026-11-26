"""The shared top-K evaluator: how much fraud a model catches within the daily review capacity K.

Each selection group ranks its orders by model score, highest first, and selects the top K.
A group is one `order_date`; when K is given per country it is one `order_date` and
`country_code`. Equal scores within a group are ordered by `country_code`, then `order_id`,
so a selection never depends on row order. A group with fewer than K orders selects them all.
`select_top_k` makes this selection and needs no labels, so other comparisons can share it;
it can also limit the selection to some orders, or select fewer than K in some groups.

The primary measure is the labelled fraud caught, shown next to the most fraud catchable
at K (the sum over groups of the smaller of K and the group's fraud count). Precision,
recall and the confusion matrix come from the same selection. Every result is reported at
K and 2K. Models that catch the same fraud at K are ordered by average precision over the
whole window, ignoring groups.

A ratio whose denominator is zero is undefined: it is `None` in results and "N/A" in the
text report, never 0 or NaN.
"""

import math
import numbers
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import groupby

import numpy as np
import pandas as pd

_KEY_COLUMNS = ("order_date", "country_code", "order_id")
ORDER_COLUMNS = (*_KEY_COLUMNS, "is_fraud")

# The daily review capacity, a product owner assumption with its reason in D5-01 of
# docs/decision-log.md. Change it only together with that entry.
K = 250


@dataclass(frozen=True)
class TopKResult:
    """The selection at one K, as counts over the whole window."""

    k: int | Mapping[str, int]
    fraud_caught: int
    fraud_catchable: int
    selected: int
    total_fraud: int
    total_orders: int

    @property
    def precision(self) -> float | None:
        """Fraud caught over orders selected, or `None` when nothing is selected."""
        return _ratio(self.fraud_caught, self.selected)

    @property
    def recall(self) -> float | None:
        """Fraud caught over all fraud in the window, or `None` when there is no fraud."""
        return _ratio(self.fraud_caught, self.total_fraud)


@dataclass(frozen=True)
class TopKReport:
    """One model's results at K and 2K, and its average precision over the whole window."""

    at_k: TopKResult
    at_2k: TopKResult
    average_precision: float | None


def evaluate_top_k(
    orders: Mapping[str, Sequence], scores: Sequence[float], k: int | Mapping[str, int]
) -> TopKReport:
    """Evaluate one model's `scores` (higher means riskier) on `orders`, at K and 2K.

    The arguments are as for `select_top_k`, and `orders` also needs an `is_fraud`
    column of 0 or 1 (or a bool).

    Raises `ValueError` for any input `select_top_k` rejects, a missing `is_fraud`
    column, an `is_fraud` column of another length, or a label other than 0 or 1.
    """
    _require_columns(orders, ORDER_COLUMNS)
    at_k = select_top_k(orders, scores, k)
    labels = list(orders["is_fraud"])
    if len(labels) != len(scores):
        raise ValueError(
            "The order columns and the scores must all have the same length."
        )
    for order_id, label in zip(orders["order_id"], labels, strict=True):
        if label is pd.NA or label not in (0, 1):
            raise ValueError(
                f"is_fraud must be 0 or 1, but order {order_id} has {label}."
            )
    is_fraud = np.array(labels, dtype=bool)

    def result(capacity: int | Mapping[str, int], selected: np.ndarray) -> TopKResult:
        # Selecting among the fraud alone gives each group the smaller of its K and its
        # fraud count.
        catchable = select_top_k(orders, scores, capacity, eligible=is_fraud)
        return TopKResult(
            capacity,
            fraud_caught=int((selected & is_fraud).sum()),
            fraud_catchable=int(catchable.sum()),
            selected=int(selected.sum()),
            total_fraud=int(is_fraud.sum()),
            total_orders=len(labels),
        )

    double_k = (
        2 * k
        if isinstance(k, int)
        else {country: 2 * value for country, value in k.items()}
    )
    return TopKReport(
        at_k=result(k, at_k),
        at_2k=result(double_k, select_top_k(orders, scores, double_k)),
        average_precision=_average_precision(
            list(zip(scores, is_fraud.tolist(), strict=True))
        ),
    )


def select_top_k(
    orders: Mapping[str, Sequence],
    scores: Sequence[float],
    k: int | Mapping[str, int],
    *,
    eligible: Sequence[bool] | None = None,
    group_limits: Mapping[object, int] | None = None,
) -> np.ndarray:
    """Select each group's top K orders by `scores` (higher means riskier).

    `orders` is a pandas DataFrame or a dict of equal-length columns holding at least
    `order_date`, `country_code` and `order_id`; `scores` has one score per order, in
    the same order, finite for every order that can be selected. `k` is the daily
    review capacity, a Python int of orders per day, or a mapping from `country_code`
    to that country's capacity per day.

    Returns a numpy array of bools, True for each selected order, by position in
    `orders` whatever its index.

    `eligible`, if given, holds one bool per order by position: only True orders can be
    selected, and the others take no place in the top K and need no finite score.

    `group_limits`, if given, selects fewer than K in some groups. Its key is the group:
    with one K, the `order_date` value; with a K per country, an
    (`order_date`, `country_code`) tuple. The values must equal the column's values,
    types included. With `k=250`, `{date(2024, 3, 1): 0}` selects nothing on 1 March
    2024; with `k={"SG": 250, "MY": 100}`, `{(date(2024, 3, 1), "SG"): 40}` selects at
    most 40 of that day's SG orders. A limit is a whole number (numpy ones too, not a
    bool). A group not in the mapping keeps its K.

    Raises `ValueError` for a K that is not a Python int of at least 1, a country with
    no K, a missing column, a missing date, country or order ID (None, NA, NaN or NaT),
    columns of different lengths, an `order_id` repeated within one `country_code`, a
    score that is not finite (only an eligible order's, when `eligible` is given), an
    `eligible` of another length or not all bools, or a group limit for no group in the
    orders, or that is not a whole number from 0 to the group's K.
    """
    _require_columns(orders, _KEY_COLUMNS)
    capacities = list(k.values()) if isinstance(k, Mapping) else [k]
    if any(
        isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1
        for capacity in capacities
    ):
        raise ValueError(
            f"K must be a Python int of at least 1, or a mapping from country_code to"
            f" such ints, but it is {k!r}."
        )
    if len({len(scores), *(len(orders[column]) for column in _KEY_COLUMNS)}) > 1:
        raise ValueError(
            "The order columns and the scores must all have the same length."
        )
    if eligible is None:
        eligible = np.ones(len(scores), dtype=bool)
    else:
        eligible = np.asarray(eligible)
        if eligible.dtype != bool or eligible.shape != (len(scores),):
            raise ValueError(
                f"eligible must hold one True or False per order ({len(scores)} in"
                f" all), but it holds {eligible.size} values of type {eligible.dtype}."
            )

    # Each group holds (sort key, position); the key sorts the highest score first.
    groups = defaultdict(list)
    # An order is its country and ID: the same order_id can appear in two countries.
    orders_seen = set()
    for position, (order_date, country_code, order_id, score) in enumerate(
        zip(*(orders[column] for column in _KEY_COLUMNS), scores, strict=True)
    ):
        if any(
            value is None or value is pd.NA or value != value  # noqa: PLR0124 - NaN and NaT
            for value in (order_date, country_code, order_id)
        ):
            raise ValueError(
                f"Every order needs an order_date, country_code and order_id, but order"
                f" {order_id} in country {country_code} on {order_date} is missing one."
            )
        if eligible[position] and (
            score is None or score is pd.NA or not math.isfinite(score)
        ):
            raise ValueError(
                f"Order {order_id} has a score that is not finite: {score}."
            )
        if (country_code, order_id) in orders_seen:
            raise ValueError(
                f"Order {order_id} in country {country_code} appears more than once."
            )
        orders_seen.add((country_code, order_id))
        # The group is made even when the order is not eligible, so a limit can name it.
        ranked = groups[
            order_date if isinstance(k, int) else (order_date, country_code)
        ]
        if eligible[position]:
            ranked.append(((-score, country_code, order_id), position))
    if not isinstance(k, int):
        countries = {country_code for _, country_code in groups}
        no_k = sorted(countries - set(k), key=str)
        if no_k:
            raise ValueError(f"K is not set for country {', '.join(map(str, no_k))}.")

    limits = {group: k if isinstance(k, int) else k[group[1]] for group in groups}
    for group, limit in (group_limits or {}).items():
        if group not in limits:
            raise ValueError(
                f"group_limits names {group!r}, which is no group in the orders."
            )
        if (
            isinstance(limit, bool)
            or not isinstance(limit, numbers.Integral)
            or not 0 <= limit <= limits[group]
        ):
            raise ValueError(
                f"The limit for group {group!r} must be a whole number from 0 to its K,"
                f" {limits[group]}, but it is {limit!r}."
            )
        limits[group] = int(limit)

    selected = np.zeros(len(scores), dtype=bool)
    for group, ranked in groups.items():
        ranked.sort()
        for _, position in ranked[: limits[group]]:
            selected[position] = True
    return selected


def rank_models(reports: Mapping[str, TopKReport]) -> list[str]:
    """Model names, best first: most fraud caught at K, then the higher average precision.

    Raises `ValueError` unless every report has the same K and the same order and fraud
    counts, a sign that the models were evaluated on the same orders.
    """
    results = [report.at_k for report in reports.values()]
    window = [(result.k, result.total_orders, result.total_fraud) for result in results]
    if any(setting != window[0] for setting in window):
        raise ValueError(
            "Models can only be ranked when evaluated with the same K on the same orders."
        )

    # Equal fraud counts mean every average precision is defined, or none is.
    return sorted(
        reports,
        key=lambda name: (
            -reports[name].at_k.fraud_caught,
            -(reports[name].average_precision or 0),
        ),
    )


def format_report(report: TopKReport) -> str:
    """The report as plain text, giving each ratio's denominator and N/A when undefined."""
    lines = []
    for label, result in (("K", report.at_k), ("2K", report.at_2k)):
        missed = result.total_fraud - result.fraud_caught
        false_alarms = result.selected - result.fraud_caught
        cleared = result.total_orders - result.selected - missed
        lines += [
            f"At {label} = {result.k}:",
            (
                f"  Fraud caught: {result.fraud_caught} of {result.fraud_catchable}"
                f" catchable ({result.total_fraud} fraud in {result.total_orders} orders)"
            ),
            f"  Precision: {_show_ratio(result.fraud_caught, result.selected, 'orders selected')}",
            f"  Recall: {_show_ratio(result.fraud_caught, result.total_fraud, 'fraud orders')}",
            "  Confusion matrix (selected / not selected):",
            f"    Fraud: {result.fraud_caught} / {missed}",
            f"    Not fraud: {false_alarms} / {cleared}",
        ]
    average_precision = report.average_precision
    shown = (
        "N/A (no fraud in the window)"
        if average_precision is None
        else f"{average_precision:.3f}"
    )
    lines.append(f"Average precision over the window: {shown}")
    return "\n".join(lines)


def _require_columns(orders: Mapping[str, Sequence], columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in orders]
    if missing:
        raise ValueError(f"The orders have no {', '.join(missing)} column.")


def _average_precision(labelled: list[tuple[float, bool]]) -> float | None:
    """Average precision of (score, is_fraud) pairs, or `None` when none is fraud.

    The sum over score thresholds of precision times the recall gained there; orders with
    equal scores share one threshold, matching scikit-learn's `average_precision_score`.
    """
    total_fraud = sum(is_fraud for _, is_fraud in labelled)
    if not total_fraud:
        return None
    labelled = sorted(labelled, key=lambda pair: pair[0], reverse=True)
    average_precision = 0.0
    caught = seen = 0
    for _, tied in groupby(labelled, key=lambda pair: pair[0]):
        caught_before = caught
        for _, is_fraud in tied:
            caught += is_fraud
            seen += 1
        average_precision += (caught - caught_before) / total_fraud * caught / seen
    return average_precision


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _show_ratio(numerator: int, denominator: int, denominator_name: str) -> str:
    ratio = _ratio(numerator, denominator)
    shown = "N/A" if ratio is None else f"{ratio:.3f}"
    return f"{shown} ({numerator} / {denominator} {denominator_name})"
