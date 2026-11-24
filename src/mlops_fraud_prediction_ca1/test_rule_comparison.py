"""The rule-versus-model comparison matches flag counts per group and reports #18's counts."""

import unittest

import pandas as pd

from mlops_fraud_prediction_ca1.rule_comparison import (
    compare,
    cut_rule_flags,
    format_comparison,
    match_model_to_rule,
    select_threshold,
    top_k_selection,
)


def _orders(rows):
    return pd.DataFrame(
        rows,
        columns=["order_date", "country_code", "order_id", "is_fraud", "order_value"],
    )


DAY1 = [
    ("2023-03-27", "SG", "a", 1, 900.0),
    ("2023-03-27", "SG", "b", 0, 800.0),
    ("2023-03-27", "SG", "c", 1, 700.0),
    ("2023-03-27", "SG", "d", 0, 50.0),
]
DAY2 = [
    ("2023-03-28", "SG", "e", 1, 600.0),
    ("2023-03-28", "SG", "f", 0, 10.0),
]


class CutTests(unittest.TestCase):
    def test_rule_flags_over_k_are_ranked_by_order_value_and_cut(self):
        orders = _orders(DAY1 + DAY2)
        flags = [True, True, True, False, True, False]
        cut = cut_rule_flags(orders, flags, 2)
        self.assertEqual(cut.tolist(), [True, True, False, False, True, False])

    def test_value_ties_break_by_country_then_order_id(self):
        orders = _orders(
            [
                ("2023-03-27", "SG", "z", 0, 500.0),
                ("2023-03-27", "MY", "y", 0, 500.0),
                ("2023-03-27", "SG", "a", 0, 500.0),
            ]
        )
        cut = cut_rule_flags(orders, [True, True, True], 2)
        self.assertEqual(cut.tolist(), [False, True, True])

    def test_per_country_k_cuts_within_country(self):
        orders = _orders(DAY1[:2] + [("2023-03-27", "MY", "m", 0, 5.0)])
        cut = cut_rule_flags(orders, [True, True, True], {"SG": 1, "MY": 1})
        self.assertEqual(cut.tolist(), [True, False, True])

    def test_bad_k_is_refused(self):
        with self.assertRaises(ValueError):
            cut_rule_flags(_orders(DAY1), [True] * 4, 0)


class MatchTests(unittest.TestCase):
    def test_model_is_cut_to_the_rule_count_in_each_group(self):
        orders = _orders(DAY1 + DAY2)
        scores = [0.1, 0.2, 0.9, 0.3, 0.5, 0.4]
        rule = [True, False, False, False, True, True]  # 1 in day 1, 2 in day 2
        matched = match_model_to_rule(orders, scores, rule, 3)
        self.assertEqual(matched.tolist(), [False, False, True, False, True, True])

    def test_group_with_no_rule_flags_gets_no_model_flags(self):
        orders = _orders(DAY1 + DAY2)
        matched = match_model_to_rule(orders, [1] * 6, [False] * 4 + [True, False], 3)
        self.assertEqual(matched[:4].sum(), 0)
        self.assertEqual(matched[4:].sum(), 1)

    def test_top_k_selection_takes_k_per_day(self):
        orders = _orders(DAY1 + DAY2)
        sel = top_k_selection(orders, [0.1, 0.2, 0.9, 0.3, 0.5, 0.4], 2)
        self.assertEqual(sel.tolist(), [False, False, True, True, True, True])

    def test_non_finite_scores_are_refused(self):
        with self.assertRaises(ValueError):
            top_k_selection(_orders(DAY2), [float("nan"), 1.0], 1)


class ThresholdTests(unittest.TestCase):
    def test_threshold_maximises_fraud_caught_then_fewer_flags_then_larger_x(self):
        orders = _orders(DAY1)
        age = pd.Series([0, 0, 0, 0], dtype="Int64")
        chosen = select_threshold(orders, age, 5, grid=(50, 100, 600, 800))
        # X=50, 100 and 600 all flag a, b, c (2 fraud): tie on flags, so the larger X wins
        self.assertEqual(chosen["x"], 600)
        self.assertEqual(chosen["candidates"]["fraud_caught"].tolist(), [2, 2, 2, 1])

    def test_equal_everything_picks_the_larger_x(self):
        orders = _orders([("2023-03-27", "SG", "a", 1, 900.0)])
        age = pd.Series([0], dtype="Int64")
        self.assertEqual(select_threshold(orders, age, 5, grid=(50, 100))["x"], 100)


class CompareTests(unittest.TestCase):
    def setUp(self):
        self.orders = _orders(DAY1 + DAY2)
        self.age = pd.Series([0, 0, 0, 0, 0, 3], dtype="Int64")
        self.loyal = ["loyal", "not_loyal", "unknown", "not_loyal", "loyal", "loyal"]

    def test_counts_exposure_false_flags_and_loyal_false_flags(self):
        result = compare(
            self.orders,
            [0.9, 0.1, 0.8, 0.2, 0.7, 0.3],
            self.age,
            self.loyal,
            k=2,
            threshold=500,
            threshold_source="validation-tuned proxy",
        )
        # Rule flags a, b, c, e; day 1 cut to 2 by value -> a, b; plus e.
        self.assertEqual(result["rule_flags_before_cut"], 4)
        self.assertEqual(
            result["rule"],
            {
                "flags": 3,
                "fraud_caught": 2,
                "fraud_exposure": 1500.0,
                "false_flags": 1,
                "false_flags_on_loyal": 0,
            },
        )
        self.assertEqual(result["model_matched"]["fraud_caught"], 3)
        self.assertEqual(result["model_matched"]["fraud_exposure"], 2200.0)
        self.assertEqual(result["model_at_k"]["flags"], 4)
        self.assertTrue(result["beats_rule"])
        self.assertIn("validation-tuned proxy", format_comparison(result))
        self.assertIn("did beat", format_comparison(result))

    def test_model_that_ties_the_rule_does_not_beat_it(self):
        result = compare(
            self.orders,
            [0.9, 0.8, 0.1, 0.2, 0.7, 0.3],
            self.age,
            self.loyal,
            k=2,
            threshold=500,
            threshold_source="confirmed",
        )
        self.assertFalse(result["beats_rule"])
        self.assertIn("did not beat", format_comparison(result))

    def test_unusable_age_rows_are_never_flagged_and_counted(self):
        age = pd.Series([pd.NA] * 6, dtype="Int64")
        result = compare(
            self.orders, [1] * 6, age, self.loyal, 2, 500, "validation-tuned proxy"
        )
        self.assertEqual(result["rule"]["flags"], 0)
        self.assertEqual(result["rule_status_counts"]["unusable_age"], 6)
        self.assertEqual(result["model_matched"]["flags"], 0)


if __name__ == "__main__":
    unittest.main()
