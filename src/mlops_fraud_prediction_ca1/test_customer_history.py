"""History counts follow the strict-date and 30-day label-delay rules."""

import unittest

import pandas as pd

from mlops_fraud_prediction_ca1.customer_history import (
    HISTORY_COLUMNS,
    customer_history_features,
)
from mlops_fraud_prediction_ca1.leakage import historical_order_mask


def _rows(*rows):
    return pd.DataFrame(
        rows, columns=["order_date", "country_code", "customer_id", "is_fraud"]
    )


class CustomerHistoryTests(unittest.TestCase):
    def test_counts_follow_strict_date_and_delay(self):
        history = _rows(
            ("2023-01-01", "PK", "a", 1),
            ("2023-01-01", "PK", "a", 0),
            ("2023-01-20", "PK", "a", 1),
            ("2023-02-10", "PK", "a", 1),
        )
        orders = _rows(
            ("2023-01-01", "PK", "a", 0),  # same day: nothing before
            ("2023-01-20", "PK", "a", 0),  # two Jan-1 orders, too recent for labels
            ("2023-01-31", "PK", "a", 0),  # Jan 1 is exactly 30 days: not yet mature
            ("2023-02-01", "PK", "a", 0),  # Jan 1 is 31 days: 2 labelled, 1 fraud
            ("2023-03-15", "PK", "a", 0),  # all four matured, 3 fraud
        )
        out = customer_history_features(orders, history)
        self.assertEqual(list(out.columns), HISTORY_COLUMNS)
        self.assertEqual(out.n_prior_orders.tolist(), [0, 2, 3, 3, 4])
        self.assertEqual(out.n_prior_labelled_30d.tolist(), [0, 0, 0, 2, 4])
        self.assertEqual(out.n_prior_fraud_30d.tolist(), [0, 0, 0, 1, 3])

    def test_matches_historical_order_mask(self):
        history = _rows(*[(f"2023-01-{d:02d}", "PH", "c", d % 2) for d in range(1, 29)])
        orders = _rows(*[(f"2023-02-{d:02d}", "PH", "c", 0) for d in range(1, 28)])
        out = customer_history_features(orders, history)
        for i, day in enumerate(orders.order_date):
            matured = historical_order_mask(
                history.order_date, day, label_delay_days=30
            )
            self.assertEqual(out.n_prior_labelled_30d.iloc[i], matured.sum())
            self.assertEqual(
                out.n_prior_fraud_30d.iloc[i], history.is_fraud[matured].sum()
            )

    def test_customers_are_separate_by_country_and_missing_ids_get_zeros(self):
        history = _rows(
            ("2022-01-01", "PK", "a", 1),
            ("2022-01-01", "BD", "a", 0),
        )
        orders = _rows(
            ("2023-01-01", "PK", "a", 0),
            ("2023-01-01", "BD", "a", 0),
            ("2023-01-01", "MY", "a", 0),
            ("2023-01-01", "PK", None, 0),
        )
        out = customer_history_features(orders, history)
        self.assertEqual(out.n_prior_fraud_30d.tolist(), [1, 0, 0, 0])
        self.assertEqual(out.n_prior_orders.tolist(), [1, 1, 0, 0])

    def test_index_preserved_inputs_untouched_and_no_history(self):
        orders = _rows(("2023-01-01", "PK", "a", 0), ("2023-01-02", "PK", "b", 0))
        orders.index = [10, 5]
        history = _rows(("2022-01-01", "PK", "b", 1))
        before = orders.copy()
        out = customer_history_features(orders, history)
        self.assertEqual(out.index.tolist(), [10, 5])
        self.assertEqual(out.n_prior_fraud_30d.tolist(), [0, 1])
        pd.testing.assert_frame_equal(orders, before)
        empty = customer_history_features(orders, history.iloc[0:0])
        self.assertTrue((empty == 0).all().all())

    def test_bad_inputs_raise(self):
        good = _rows(("2023-01-01", "PK", "a", 0))
        bad_date = _rows(("not a date", "PK", "a", 0))
        with self.assertRaises(ValueError):
            customer_history_features(bad_date, good)
        with self.assertRaises(ValueError):
            customer_history_features(good, bad_date)
        with self.assertRaises(ValueError):
            customer_history_features(good, good, label_delay_days=-1)
        with self.assertRaises(ValueError):
            customer_history_features(good, good.drop(columns="is_fraud"))


if __name__ == "__main__":
    unittest.main()
