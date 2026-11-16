"""Relationship reports preserve evidence and never multiply ambiguous joins."""

import unittest

import pandas as pd
from pandas.testing import assert_frame_equal

from mlops_fraud_prediction_ca1.relationships import (
    bidirectional_matches,
    country_agreement,
    key_profile,
    link_orders_customers,
    unique_records,
)


class RelationshipTests(unittest.TestCase):
    def test_profile_and_safe_records_separate_copies_conflicts_and_nulls(self):
        frame = pd.DataFrame(
            {
                "country_code": ["MY"] * 6,
                "order_id": [1, 1, 2, 2, None, None],
                "value": [4, 4, 5, 6, 7, 7],
            }
        )
        original = frame.copy(deep=True)
        safe, report = unique_records(frame, ["country_code", "order_id"])
        self.assertEqual(safe.order_id.tolist(), [1])
        self.assertEqual(report["duplicate_groups"], 2)
        self.assertEqual(report["duplicate_rows"], 4)
        self.assertEqual(report["extra_key_rows"], 2)
        self.assertEqual(report["null_key_rows"], 2)
        self.assertEqual(report["conflicting_keys"], 1)
        self.assertEqual(report["conflicting_source_rows"], 2)
        self.assertEqual(report["raw_exact_extra_copies"], 2)
        assert_frame_equal(frame, original)

    def test_bidirectional_rates_are_row_weighted_without_null_matching(self):
        a = pd.DataFrame({"id": [1, 1, 2, None]})
        b = pd.DataFrame({"id": [1, 3, None]})
        report = bidirectional_matches(a, b, ["id"])
        self.assertEqual(report.loc["left_to_right", "matched_rows"], 2)
        self.assertEqual(report.loc["left_to_right", "row_match_rate"], 2 / 3)
        self.assertEqual(report.loc["left_to_right", "key_match_rate"], 0.5)
        self.assertEqual(report.loc["right_to_left", "row_match_rate"], 0.5)
        self.assertEqual(report.loc["left_to_right", "null_key_rows"], 1)
        empty = bidirectional_matches(a.iloc[:0], b, ["id"])
        self.assertTrue(pd.isna(empty.loc["left_to_right", "row_match_rate"]))

    def test_uniqueness_alone_and_composite_and_country_formatting(self):
        a = pd.DataFrame({"country_code": ["MY", "PK"], "id": [1, 1]})
        self.assertFalse(key_profile(a, ["id"])["unique_complete_keys"])
        self.assertTrue(key_profile(a, ["country_code", "id"])["unique_complete_keys"])
        b = pd.DataFrame({"country_code": [" my ", "pk"], "id": [1, 1]})
        raw = bidirectional_matches(a, b, ["country_code", "id"])
        normalized = bidirectional_matches(
            a, b, ["country_code", "id"], normalize_country=True
        )
        self.assertEqual(raw.loc["left_to_right", "matched_rows"], 0)
        self.assertEqual(normalized.loc["left_to_right", "matched_rows"], 2)
        self.assertEqual(b.country_code.iloc[0], " my ")

    def test_country_sets_expose_conflicts_without_cartesian_rows(self):
        a = pd.DataFrame(
            {
                "id": [1, 2, 2, 3, 4, None],
                "country_code": ["MY", "MY", "PK", "MY", None, "MY"],
            }
        )
        b = pd.DataFrame(
            {
                "id": [1, 2, 2, 3, 4, None],
                "country_code": ["my", "MY", "PK", "PK", None, "MY"],
            }
        )
        report = country_agreement(a, b, "id", normalize_country=True).set_index("id")
        self.assertEqual(len(report), 4)
        self.assertEqual(
            report.status.to_dict(),
            {1: "agree", 2: "ambiguous", 3: "disagree", 4: "missing_country"},
        )
        self.assertTrue(report.loc[2, "sets_equal"])
        self.assertEqual(report.loc[2, "shared_country_count"], 2)

    def test_partial_missing_country_is_not_reported_as_agreement(self):
        left = pd.DataFrame({"id": [1, 1], "country_code": ["MY", None]})
        right = pd.DataFrame({"id": [1], "country_code": ["MY"]})
        row = country_agreement(left, right, "id").iloc[0]
        self.assertEqual(row.status, "missing_country")
        self.assertFalse(row.sets_equal)
        self.assertTrue(row.left_missing_country)

    def test_bridge_excludes_conflicts_and_collapses_only_exact_copies(self):
        orders = pd.DataFrame(
            {"country_code": ["my"] * 5, "order_id": [1, 1, 2, 3, None]}
        )
        labels = pd.DataFrame(
            {
                "country_code": ["MY"] * 6,
                "order_id": [1, 1, 2, 2, 3, None],
                "customer_id": [10, 10, 20, 21, None, 10],
                "is_fraud": [1, 1, 0, 0, 0, 1],
            }
        )
        customers = pd.DataFrame(
            {
                "country_code": ["MY"] * 3,
                "customer_id": [10, 10, None],
                "value": [2, 2, 4],
            }
        )
        joined, audit = link_orders_customers(
            orders, labels, customers, normalize_country=True
        )
        self.assertEqual(len(joined), 3)
        self.assertEqual(audit["orders_with_label"], 2)
        self.assertEqual(audit["orders_with_customer"], 1)
        self.assertEqual(audit["labels"]["conflicting_keys"], 1)
        self.assertEqual(
            joined.loc[joined.order_id.eq(2), "label_link"].item(), "left_only"
        )
        self.assertEqual(
            joined.loc[joined.order_id.eq(3), "customer_link"].item(), "left_only"
        )

    def test_empty_tables(self):
        frame = pd.DataFrame(columns=["country_code", "order_id"])
        safe, audit = unique_records(frame, ["country_code", "order_id"])
        self.assertTrue(safe.empty)
        self.assertEqual(audit["safe_rows"], 0)
        self.assertTrue(country_agreement(frame, frame, "order_id").empty)
