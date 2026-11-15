"""Business checks use disclosed populations and calendar-day comparisons."""

import unittest

import pandas as pd
from pandas.testing import assert_frame_equal

from mlops_fraud_prediction_ca1.logic import business_logic_report


class BusinessLogicTests(unittest.TestCase):
    def test_checks_counts_and_safe_calendar_dates(self):
        customers = pd.DataFrame({"country_code": ["MY", "MY"], "customer_id": [1, 2],
                                  "num_orders_last_50days": [1, None], "num_cancelled_orders_last_50days": [2, 1],
                                  "num_refund_orders_last_50days": [2, 0], "total_payment_last_50days": [-1, 1],
                                  "num_associated_customers": [0, 0],
                                  "first_order_datetime": ["2024-01-02 23:00:00", "2024-01-03 00:00:00"]})
        orders = pd.DataFrame({"country_code": ["MY"] * 6, "order_id": [1, 1, 2, 3, 4, None],
                               "refund_value": [11, 11, -1, 0, 0, 0], "order_value": [10, 10, 5, float("inf"), 1, 1],
                               "num_items_ordered": [1, 1, 0, 1, 1, 1],
                               "order_date": ["2024-01-02", "2024-01-02", "2024-01-02", "bad", "2024-01-01", "2024-01-01"]})
        labels = pd.DataFrame({"country_code": ["MY"] * 4, "order_id": [1, 2, 4, 4],
                               "customer_id": [1, 2, 1, 2], "is_fraud": [0, 1, 0, 0]})
        originals = [frame.copy(deep=True) for frame in (customers, orders, labels)]
        report = business_logic_report(customers, orders, labels)
        self.assertEqual(report.loc["refund_above_order_value", "flagged"], 2)
        self.assertEqual(report.loc["refund_above_order_value", "evaluable"], 5)
        self.assertEqual(report.loc["cancelled_above_total_orders", "flagged"], 1)
        self.assertEqual(report.loc["refunded_above_total_orders", "excluded"], 1)
        self.assertEqual(report.loc["negative:orders.refund_value", "flagged"], 1)
        self.assertEqual(report.loc["orders_without_label_key", "flagged"], 1)
        self.assertEqual(report.loc["orders_without_label_key", "excluded"], 1)
        self.assertEqual(report.loc["orders_without_safe_label", "flagged"], 2)
        self.assertEqual(report.loc["order_before_customer_first_order", "evaluable"], 2)
        self.assertEqual(report.loc["order_before_customer_first_order", "flagged"], 1)
        self.assertEqual(report.attrs["link_audit"]["orders"]["raw_exact_extra_copies"], 1)
        for frame, original in zip((customers, orders, labels), originals):
            assert_frame_equal(frame, original)
