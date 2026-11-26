"""Loyalty status follows the documented examples on small synthetic frames."""

import unittest

import pandas as pd

from mlops_fraud_prediction_ca1.cleaning import clean_frame
from mlops_fraud_prediction_ca1.loyalty import loyalty_status

CUSTOMERS = pd.DataFrame({"country_code": ["MY", "SG"], "customer_id": [1, 1]})


def _orders(rows):
    return pd.DataFrame(rows, columns=["country_code", "order_id", "order_date"])


def _labels(rows):
    return pd.DataFrame(rows, columns=["country_code", "order_id", "customer_id"])


def _run(prior, current, links=None, customers=CUSTOMERS, **kwargs):
    """Prior rows are (country, order_id, date); every order links to customer 1."""
    history = _orders(prior)
    comparison = _orders(current)
    both = pd.concat([history, comparison])
    labels = _labels(
        links
        if links is not None
        else [(c, o, 1) for c, o in zip(both.country_code, both.order_id, strict=True)]
    )
    return loyalty_status(history, comparison, labels, customers, **kwargs)


NOW = [("MY", 100, "2023-04-01")]


class LoyaltyTests(unittest.TestCase):
    def test_documented_threshold_examples(self):
        cases = {
            "loyal": [("MY", 1, "2023-03-10"), ("MY", 2, "2023-03-31")],
            "not_loyal": [("MY", 1, "2023-03-31")],
        }
        for expected, prior in cases.items():
            result = _run(prior, NOW)
            self.assertEqual(result.loyalty_status.tolist(), [expected])

    def test_same_day_orders_never_count_but_one_earlier_day_counts_twice(self):
        same = _run([("MY", 1, "2023-04-01"), ("MY", 2, "2023/04/01")], NOW)
        self.assertEqual(same.prior_order_count.tolist(), [0])
        self.assertEqual(same.loyalty_status.tolist(), ["not_loyal"])
        two = _run([("MY", 1, "2023-03-31"), ("MY", 2, "2023-03-31")], NOW)
        self.assertEqual(two.loyalty_status.tolist(), ["loyal"])

    def test_future_rows_and_pre_start_rows_do_not_count(self):
        result = _run(
            [
                ("MY", 1, "2023-04-02"),
                ("MY", 2, "2023-05-01"),
                ("MY", 3, "2022-11-11"),  # before the frozen training start
                ("MY", 4, "2022-11-12"),
            ],
            NOW,
        )
        self.assertEqual(result.prior_order_count.tolist(), [1])

    def test_exact_copies_collapse_and_conflicts_are_excluded(self):
        copy_plus = _run(
            [
                ("MY", 1, "2023-03-30"),
                ("my ", 1, "2023-03-30"),
                ("MY", 2, "2023-03-31"),
            ],
            NOW,
        )
        self.assertEqual(copy_plus.prior_order_count.tolist(), [2])
        self.assertEqual(copy_plus.loyalty_status.tolist(), ["loyal"])
        copy_only = _run([("MY", 1, "2023-03-30"), ("MY", 1, "2023-03-30")], NOW)
        self.assertEqual(copy_only.prior_order_count.tolist(), [1])
        conflict = _run(
            [("MY", 1, "2023-03-30"), ("MY", 1, "2023-03-29"), ("MY", 2, "2023-03-31")],
            NOW,
        )
        self.assertEqual(conflict.prior_order_count.tolist(), [1])
        self.assertEqual(conflict.loyalty_status.tolist(), ["not_loyal"])

    def test_unusable_or_missing_prior_dates_are_not_counted(self):
        result = _run(
            [("MY", 1, "2023-03-31"), ("MY", 2, "31/03/2023"), ("MY", 3, None)], NOW
        )
        self.assertEqual(result.prior_order_count.tolist(), [1])

    def test_same_customer_id_in_another_country_is_a_different_customer(self):
        result = _run(
            [("SG", 1, "2023-03-30"), ("SG", 2, "2023-03-31")],
            NOW,
            links=[("SG", 1, 1), ("SG", 2, 1), ("MY", 100, 1)],
        )
        self.assertEqual(result.prior_order_count.tolist(), [0])
        self.assertEqual(result.loyalty_status.tolist(), ["not_loyal"])
        sg = _run(
            [("SG", 1, "2023-03-30"), ("SG", 2, "2023-03-31")],
            [("SG", 100, "2023-04-01")],
            links=[("SG", 1, 1), ("SG", 2, 1), ("SG", 100, 1)],
        )
        self.assertEqual(sg.loyalty_status.tolist(), ["loyal"])

    def test_unknown_for_missing_conflicting_or_unlisted_links_and_bad_dates(self):
        prior = [("MY", 1, "2023-03-30"), ("MY", 2, "2023-03-31")]
        current = [
            ("MY", 100, "2023-04-01"),  # missing customer_id
            ("MY", 101, "2023-04-01"),  # conflicting labels
            ("MY", 102, "2023-04-01"),  # customer not in the customer table
            ("MY", 103, "not a date"),
            ("MY", 104, "2023-04-01"),  # no label
            (None, 105, "2023-04-01"),  # unsafe key
        ]
        links = [
            ("MY", 1, 1),
            ("MY", 2, 1),
            ("MY", 100, None),
            ("MY", 101, 1),
            ("MY", 101, 2),
            ("MY", 102, 9),
            ("MY", 103, 1),
        ]
        result = _run(prior, current, links=links)
        self.assertEqual(result.loyalty_status.eq("unknown").tolist(), [True] * 6)
        self.assertTrue(result.prior_order_count.isna().all())
        self.assertEqual(
            result.reason.tolist(),
            [
                "no_customer_link",
                "no_label_link",
                "no_customer_link",
                "order_date_unavailable",
                "no_label_link",
                "unsafe_order_key",
            ],
        )

    def test_unlinked_prior_orders_do_not_count(self):
        result = _run(
            [("MY", 1, "2023-03-30"), ("MY", 2, "2023-03-31")],
            NOW,
            links=[("MY", 1, 1), ("MY", 100, 1)],
        )
        self.assertEqual(result.prior_order_count.tolist(), [1])

    def test_rows_are_neither_lost_nor_multiplied_and_keep_index_and_order(self):
        prior = [("MY", 1, "2023-03-30"), ("MY", 2, "2023-03-31")]
        history = _orders(prior)
        comparison = _orders(
            [
                ("MY", 100, "2023-04-01"),
                ("MY", 100, "2023-04-01"),
                ("MY", 7, "2023-04-01"),
                ("MY", 200, "2023-03-31"),
            ]
        )
        comparison.index = [5, 5, 3, 9]
        labels = _labels([("MY", o, 1) for o in (1, 2, 100, 200)])
        before = comparison.copy()
        result = loyalty_status(history, comparison, labels, CUSTOMERS)
        pd.testing.assert_frame_equal(comparison, before)
        self.assertEqual(result.index.tolist(), [5, 5, 3, 9])
        self.assertEqual(result.order_id.tolist(), [100, 100, 7, 200])
        self.assertEqual(
            result.loyalty_status.tolist(), ["loyal", "loyal", "unknown", "not_loyal"]
        )
        self.assertEqual(result.prior_order_count.tolist(), [2, 2, pd.NA, 1])

    def test_key_with_different_records_across_inputs_is_unsafe(self):
        # Finding 1: order 1 is dated 03-29 in history but 04-01 in comparison,
        # so the earlier version must not count as a prior of the later one.
        history = _orders(
            [("MY", 1, "2023-03-29"), ("MY", 2, "2023-03-30"), ("MY", 3, "2023-03-31")]
        )
        comparison = _orders(
            [
                ("MY", 1, "2023-04-01"),  # differs from history: unsafe
                ("MY", 2, "2023-03-30"),  # exact overlap: still valid
                ("MY", 100, "2023-04-01"),  # neighbour
                ("MY", 100, "2023-04-01"),  # identical repeat
            ]
        )
        comparison.index = [7, 3, 7, 1]
        labels = _labels([("MY", o, 1) for o in (1, 2, 3, 100)])
        result = loyalty_status(history, comparison, labels, CUSTOMERS)
        self.assertEqual(result.index.tolist(), [7, 3, 7, 1])
        self.assertEqual(result.order_id.tolist(), [1, 2, 100, 100])
        self.assertEqual(
            result.loyalty_status.tolist(), ["unknown", "not_loyal", "loyal", "loyal"]
        )
        self.assertEqual(result.reason.tolist(), ["unsafe_order_key", "", "", ""])
        # The unsafe key is excluded from every prior count: only orders 2 and 3.
        self.assertEqual(result.prior_order_count.tolist(), [pd.NA, 0, 2, 2])

    def test_exact_overlap_between_inputs_stays_valid_in_both_roles(self):
        history = _orders([("MY", 1, "2023-03-30"), ("MY", 2, "2023-03-31")])
        comparison = _orders([("my ", 2, pd.Timestamp("2023-03-31")), *NOW])
        labels = _labels([("MY", o, 1) for o in (1, 2, 100)])
        result = loyalty_status(history, comparison, labels, CUSTOMERS)
        self.assertEqual(result.loyalty_status.tolist(), ["not_loyal", "loyal"])
        self.assertEqual(result.prior_order_count.tolist(), [1, 2])

    def test_out_of_range_dates_do_not_abort_the_batch(self):
        # Finding 2: 9999-12-31 is outside the supported range in either input.
        for bad in ("9999-12-31", __import__("datetime").date(9999, 12, 31)):
            history = _orders(
                [
                    ("MY", 1, "2023-03-30"),
                    ("MY", 2, "2023-03-31"),
                    ("MY", 3, bad),
                    ("MY", 4, "2023-03-29"),
                ]
            )
            comparison = _orders(
                [("MY", 100, "2023-04-01"), ("MY", 101, bad), ("MY", 102, "2023-03-31")]
            )
            labels = _labels([("MY", o, 1) for o in (1, 2, 3, 4, 100, 101, 102)])
            result = loyalty_status(history, comparison, labels, CUSTOMERS)
            self.assertEqual(
                result.loyalty_status.tolist(), ["loyal", "unknown", "loyal"]
            )
            self.assertEqual(result.reason.tolist(), ["", "order_date_unavailable", ""])
            self.assertEqual(result.prior_order_count.tolist()[::2], [3, 2])

    def test_conflict_inside_one_input_survives_a_narrower_other_input(self):
        # s40 finding 2: history versions differ only in a column the comparison
        # lacks, so the shared-column check saw one identical record.
        for version in ([1, 2], [1, None]):
            history = _orders(
                [
                    ("MY", 1, "2023-03-30"),
                    ("MY", 1, "2023-03-30"),
                    ("MY", 2, "2023-03-31"),
                ]
            ).assign(extra=[*version, 0])
            comparison = _orders([("MY", 100, "2023-04-01"), ("my ", 1, "2023-03-30")])
            labels = _labels([("MY", o, 1) for o in (1, 2, 100)])
            result = loyalty_status(history, comparison, labels, CUSTOMERS)
            self.assertEqual(result.loyalty_status.tolist(), ["not_loyal", "unknown"])
            self.assertEqual(result.reason.tolist(), ["", "unsafe_order_key"])
            # The conflicting version confers no loyalty on the later order.
            self.assertEqual(result.prior_order_count.tolist()[0], 1)

    def test_country_format_copies_count_the_same_before_and_after_clean_frame(self):
        # s40 finding 3: clean_frame's raw_country_code made copies look different.
        history = _orders(
            [("MY", 1, "2023-03-30"), ("my ", 1, "2023-03-30"), ("MY", 2, "2023-03-31")]
        )
        comparison = _orders([("MY", 100, "2023-04-01")])
        labels = _labels([("MY", o, 1) for o in (1, 2, 100)])
        raw = loyalty_status(history, comparison, labels, CUSTOMERS)
        cleaned = loyalty_status(
            clean_frame(history), clean_frame(comparison), labels, CUSTOMERS
        )
        self.assertIn("raw_country_code", clean_frame(history))
        self.assertEqual(raw.prior_order_count.tolist(), [2])
        pd.testing.assert_frame_equal(cleaned, raw)
        # A genuine conflict is still excluded after cleaning.
        conflict = history.assign(order_date=["2023-03-30", "2023-03-29", "2023-03-31"])
        after = loyalty_status(
            clean_frame(conflict), clean_frame(comparison), labels, CUSTOMERS
        )
        self.assertEqual(after.prior_order_count.tolist(), [1])

    def test_threshold_is_fixed_by_the_definition(self):
        # Finding 3: the two-prior-order threshold cannot be lowered by callers.
        with self.assertRaises(TypeError):
            _run([("MY", 1, "2023-03-31")], NOW, min_prior_orders=1)
        self.assertEqual(
            _run([("MY", 1, "2023-03-31")], NOW).loyalty_status.tolist(), ["not_loyal"]
        )

    def test_empty_inputs(self):
        empty = _orders([])
        result = loyalty_status(empty, empty, _labels([]), CUSTOMERS)
        self.assertTrue(result.empty)

    def test_customer_id_on_orders_is_rejected(self):
        orders = _orders(NOW).assign(customer_id=1)
        with self.assertRaises(ValueError):
            loyalty_status(orders, orders, _labels([]), CUSTOMERS)


if __name__ == "__main__":
    unittest.main()
