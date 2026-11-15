"""Tests for the top-K evaluator on small made-up orders whose results are worked out by hand."""

import math

import pytest

from mlops_fraud_prediction_ca1.top_k import (
    TopKReport,
    TopKResult,
    evaluate_top_k,
    format_report,
    rank_models,
)


def _orders(*rows):
    """Columns and scores from (order_date, country_code, order_id, is_fraud, score) rows."""
    columns = ["order_date", "country_code", "order_id", "is_fraud", "score"]
    table = {column: [row[i] for row in rows] for i, column in enumerate(columns)}
    return table, table.pop("score")


@pytest.mark.parametrize(
    ("rows", "fraud_caught"),
    [
        # Equal scores: the lower country_code is selected first.
        ([("d1", "SG", "o2", 1, 0.5), ("d1", "MY", "o9", 0, 0.5)], 0),
        # Same country: the lower order_id is selected first, whatever the row order.
        ([("d1", "SG", "o1", 0, 0.5), ("d1", "SG", "o2", 1, 0.5)], 0),
        ([("d1", "SG", "o2", 0, 0.5), ("d1", "SG", "o1", 1, 0.5)], 1),
    ],
)
def test_equal_scores_are_ordered_by_country_then_order_id(rows, fraud_caught):
    assert evaluate_top_k(*_orders(*rows), k=1).at_k.fraud_caught == fraud_caught


def test_each_day_selects_its_top_k_and_reports_k_and_2k():
    orders, scores = _orders(
        ("d1", "SG", "a1", 1, 0.9),
        ("d1", "SG", "a2", 0, 0.8),
        ("d1", "SG", "a3", 1, 0.1),
        ("d2", "SG", "b1", 1, 0.7),  # Fewer orders than K: all are selected.
        ("d3", "MY", "c1", 0, 0.6),  # A day with no fraud: nothing to catch.
        ("d3", "MY", "c2", 0, 0.2),
    )
    assert evaluate_top_k(orders, scores, k=2) == TopKReport(
        at_k=TopKResult(
            k=2,
            fraud_caught=2,
            fraud_catchable=3,
            selected=5,
            total_fraud=3,
            total_orders=6,
        ),
        at_2k=TopKResult(
            k=4,
            fraud_caught=3,
            fraud_catchable=3,
            selected=6,
            total_fraud=3,
            total_orders=6,
        ),
        # Ranked F N F N N F over the window: (1/1 + 2/3 + 3/6) / 3.
        average_precision=pytest.approx(13 / 18),
    )


def test_per_country_k_selects_within_each_day_and_country():
    orders, scores = _orders(
        ("d1", "SG", "s1", 1, 0.9),
        ("d1", "SG", "s2", 1, 0.8),
        ("d1", "SG", "s3", 0, 0.7),
        ("d1", "MY", "m1", 1, 0.1),
        ("d1", "MY", "m2", 0, 0.05),
    )
    report = evaluate_top_k(orders, scores, k={"SG": 2, "MY": 1})
    assert (report.at_k, report.at_2k) == (
        TopKResult(
            k={"SG": 2, "MY": 1},
            fraud_caught=3,
            fraud_catchable=3,
            selected=3,
            total_fraud=3,
            total_orders=5,
        ),
        TopKResult(
            k={"SG": 4, "MY": 2},
            fraud_caught=3,
            fraud_catchable=3,
            selected=5,
            total_fraud=3,
            total_orders=5,
        ),
    )
    # One K for the whole day picks s1, s2 and s3, missing MY's fraud.
    assert evaluate_top_k(orders, scores, k=3).at_k.fraud_caught == 2


def test_negative_scores_still_rank_highest_first():
    orders, scores = _orders(
        ("d1", "SG", "n1", 0, -3.0),
        ("d1", "SG", "n2", 1, -0.5),
        ("d1", "SG", "n3", 0, -2.0),
    )
    report = evaluate_top_k(orders, scores, k=1)
    assert (report.at_k.fraud_caught, report.average_precision) == (1, 1.0)


def test_ratios_without_a_denominator_are_shown_as_not_available():
    orders, scores = _orders(("d1", "SG", "x1", 0, 0.4), ("d1", "SG", "x2", 0, 0.3))
    assert format_report(evaluate_top_k(orders, scores, k=1)) == (
        "At K = 1:\n"
        "  Fraud caught: 0 of 0 catchable (0 fraud in 2 orders)\n"
        "  Precision: 0.000 (0 / 1 orders selected)\n"
        "  Recall: N/A (0 / 0 fraud orders)\n"
        "  Confusion matrix (selected / not selected):\n"
        "    Fraud: 0 / 0\n"
        "    Not fraud: 1 / 1\n"
        "At 2K = 2:\n"
        "  Fraud caught: 0 of 0 catchable (0 fraud in 2 orders)\n"
        "  Precision: 0.000 (0 / 2 orders selected)\n"
        "  Recall: N/A (0 / 0 fraud orders)\n"
        "  Confusion matrix (selected / not selected):\n"
        "    Fraud: 0 / 0\n"
        "    Not fraud: 2 / 0\n"
        "Average precision over the window: N/A (no fraud in the window)"
    )


def test_an_empty_window_has_no_precision_or_recall():
    report = evaluate_top_k(*_orders(), k=1)
    assert (report.at_k.precision, report.at_k.recall, report.average_precision) == (
        None,
        None,
        None,
    )
    assert "Precision: N/A (0 / 0 orders selected)" in format_report(report)


def test_models_catching_the_same_fraud_are_ranked_by_average_precision():
    def report(o1, o2, o3):
        return evaluate_top_k(
            *_orders(
                ("d1", "SG", "o1", 1, o1),
                ("d1", "SG", "o2", 0, o2),
                ("d1", "SG", "o3", 1, o3),
            ),
            k=1,
        )

    reports = {
        "falls_back": report(0.9, 0.5, 0.1),  # Catches 1; F N F: (1 + 2/3) / 2 = 5/6.
        "misses": report(0.1, 0.9, 0.5),  # Catches 0; N F F: (1/2 + 2/3) / 2 = 7/12.
        # Catches o1 by order_id; equal scores share one threshold: 2/3.
        "all_tied": report(0.5, 0.5, 0.5),
        "perfect": report(0.9, 0.1, 0.5),  # Catches 1; F F N: 1.
    }
    assert [reports[name].average_precision for name in reports] == pytest.approx(
        [5 / 6, 7 / 12, 2 / 3, 1.0]
    )
    assert rank_models(reports) == ["perfect", "falls_back", "all_tied", "misses"]


def test_fraud_caught_at_k_outranks_average_precision():
    orders, _ = _orders(
        ("d1", "SG", "a", 1, 0),
        ("d1", "SG", "b", 0, 0),
        ("d2", "SG", "c", 0, 0),
        ("d2", "SG", "e", 1, 0),
        ("d3", "SG", "x", 0, 0),
    )
    reports = {
        # Catches a and e; window ranked N F F: (1/2 + 2/3) / 2 = 7/12.
        "catches_more": evaluate_top_k(orders, [0.2, 0.1, 0.05, 0.15, 0.9], k=1),
        # Catches only e; window ranked F N F: (1 + 2/3) / 2 = 5/6.
        "ranks_better": evaluate_top_k(orders, [0.45, 0.5, 0.05, 0.6, 0.01], k=1),
    }
    assert [
        (report.at_k.fraud_caught, report.average_precision)
        for report in reports.values()
    ] == [(2, pytest.approx(7 / 12)), (1, pytest.approx(5 / 6))]
    assert rank_models(reports) == ["catches_more", "ranks_better"]


def test_models_evaluated_with_different_k_are_not_ranked():
    orders, scores = _orders(("d1", "SG", "o1", 1, 0.9), ("d1", "SG", "o2", 0, 0.1))
    reports = {
        "a": evaluate_top_k(orders, scores, k=1),
        "b": evaluate_top_k(orders, scores, k=2),
    }
    with pytest.raises(ValueError, match="same K"):
        rank_models(reports)


ORDER = _orders(("d1", "SG", "o1", 1, 0.5))[0]


@pytest.mark.parametrize(
    ("orders", "scores", "k", "message"),
    [
        (ORDER, [0.5], 0, "K must be a whole number"),
        (ORDER, [0.5], 0.5, "K must be a whole number"),
        (ORDER, [0.5], True, "K must be a whole number"),
        (ORDER, [0.5], {"SG": 1.0}, "K must be a whole number"),
        (ORDER, [0.5], {"MY": 1}, "K is not set for country SG"),
        (
            {"order_date": ["d1"], "is_fraud": [1]},
            [0.5],
            1,
            "no country_code, order_id",
        ),
        (ORDER, [0.5, 0.4], 1, "same length"),
        (_orders(("d1", "SG", "o1", 2, 0.5))[0], [0.5], 1, "is_fraud must be 0 or 1"),
        (ORDER, [math.nan], 1, "not finite"),
        (
            _orders(("d1", "SG", "o1", 1, 0.5), ("d2", "SG", "o1", 0, 0.4))[0],
            [0.5, 0.4],
            1,
            "appears more than once",
        ),
    ],
)
def test_invalid_input_is_rejected(orders, scores, k, message):
    with pytest.raises(ValueError, match=message):
        evaluate_top_k(orders, scores, k)
