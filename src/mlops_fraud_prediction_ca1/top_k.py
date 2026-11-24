"""The shared top-K evaluator: how much fraud a model catches within the daily review capacity K.

Each selection group ranks its orders by model score, highest first, and selects the top K.
A group is one `order_date`; when K is given per country it is one `order_date` and
`country_code`. Equal scores within a group are ordered by `country_code`, then `order_id`,
so a selection never depends on row order. A group with fewer than K orders selects them all.

The primary measure is the labelled fraud caught, shown next to the most fraud catchable
at K (the sum over groups of the smaller of K and the group's fraud count). Precision,
recall and the confusion matrix come from the same selection. Every result is reported at
K and 2K. Models that catch the same fraud at K are ordered by average precision over the
whole window, ignoring groups.

A ratio whose denominator is zero is undefined: it is `None` in results and "N/A" in the
text report, never 0 or NaN.
"""

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import groupby

ORDER_COLUMNS = ("order_date", "country_code", "order_id", "is_fraud")

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

    `orders` is a pandas DataFrame or a dict of equal-length columns holding at least
    `ORDER_COLUMNS`, with `is_fraud` as 0 or 1 (or a bool); `scores` has one finite score
    per order, in the same order. `k` is the daily review capacity, a whole number of
    orders per day, or a mapping from `country_code` to that country's capacity per day.

    Raises `ValueError` for a K that is not a whole number of at least 1, a country with
    no K, a missing column, a missing date, country or order ID (None, NaN or NaT),
    columns of different lengths, an `order_id` repeated within one `country_code`, a
    label other than 0 or 1, or a score that is not finite.
    """
    missing = [column for column in ORDER_COLUMNS if column not in orders]
    if missing:
        raise ValueError(f"The orders have no {', '.join(missing)} column.")
    capacities = list(k.values()) if isinstance(k, Mapping) else [k]
    if any(
        isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1
        for capacity in capacities
    ):
        raise ValueError(
            f"K must be a whole number of orders, at least 1, but it is {k}."
        )
    if len({len(scores), *(len(orders[column]) for column in ORDER_COLUMNS)}) > 1:
        raise ValueError(
            "The order columns and the scores must all have the same length."
        )

    # Each group holds (sort key, is_fraud); the key sorts the highest score first.
    groups = defaultdict(list)
    labelled = []
    # An order is its country and ID: the same order_id can appear in two countries.
    orders_seen = set()
    for order_date, country_code, order_id, is_fraud, score in zip(
        *(orders[column] for column in ORDER_COLUMNS), scores, strict=True
    ):
        if any(
            value is None or value != value  # noqa: PLR0124 - only NaN and NaT are unequal to themselves.
            for value in (order_date, country_code, order_id)
        ):
            raise ValueError(
                f"Every order needs an order_date, country_code and order_id, but order"
                f" {order_id} in country {country_code} on {order_date} is missing one."
            )
        if is_fraud not in (0, 1):
            raise ValueError(
                f"is_fraud must be 0 or 1, but order {order_id} has {is_fraud}."
            )
        if not math.isfinite(score):
            raise ValueError(
                f"Order {order_id} has a score that is not finite: {score}."
            )
        if (country_code, order_id) in orders_seen:
            raise ValueError(
                f"Order {order_id} in country {country_code} appears more than once."
            )
        orders_seen.add((country_code, order_id))
        labelled.append((score, bool(is_fraud)))
        group = order_date if isinstance(k, int) else (order_date, country_code)
        groups[group].append(((-score, country_code, order_id), bool(is_fraud)))
    if not isinstance(k, int):
        countries = {country_code for _, country_code in groups}
        no_k = sorted(countries - set(k), key=str)
        if no_k:
            raise ValueError(f"K is not set for country {', '.join(map(str, no_k))}.")
    for ranked in groups.values():
        ranked.sort()

    double_k = (
        2 * k
        if isinstance(k, int)
        else {country: 2 * value for country, value in k.items()}
    )
    return TopKReport(
        at_k=_select(groups, k),
        at_2k=_select(groups, double_k),
        average_precision=_average_precision(labelled),
    )


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


def _select(groups: Mapping[object, list], k: int | Mapping[str, int]) -> TopKResult:
    """Count the top K of each ranked group against the labels."""
    fraud_caught = fraud_catchable = selected = total_fraud = total_orders = 0
    for group, ranked in groups.items():
        group_k = k if isinstance(k, int) else k[group[1]]
        group_fraud = sum(is_fraud for _, is_fraud in ranked)
        top = ranked[:group_k]
        fraud_caught += sum(is_fraud for _, is_fraud in top)
        fraud_catchable += min(group_k, group_fraud)
        selected += len(top)
        total_fraud += group_fraud
        total_orders += len(ranked)
    return TopKResult(
        k, fraud_caught, fraud_catchable, selected, total_fraud, total_orders
    )


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
