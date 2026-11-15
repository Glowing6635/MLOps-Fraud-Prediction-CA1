"""Check the full preview's join and temporal denominators on a tiny batch."""

import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from scripts.preview_pbi3 import _serializable, build_preview


class EdaPreviewTests(unittest.TestCase):
    def test_full_preview_preserves_sources_and_exposes_exclusions(self):
        customers = pd.DataFrame({
            "country_code": ["my", "MY"], "customer_id": ["c1", "c2"],
            "mobile_verified": ["True", "yes"], "num_orders_last_50days": [2, 1],
            "num_cancelled_orders_last_50days": [0, 0], "num_refund_orders_last_50days": [0, 2],
            "total_payment_last_50days": [10, 20], "num_associated_customers": [1, 1],
            "first_order_datetime": ["2023-01-02 12:00:00", None],
        })
        orders = pd.DataFrame({
            "country_code": ["MY"] * 3, "order_id": ["o1", "o2", "o3"],
            "collect_type": ["Delivery"] * 3, "payment_method": ["cash", None, "cash"],
            "order_value": ["bad", 5, 20], "refund_value": [0, 6, 0], "num_items_ordered": [1, 2, 3],
            "order_date": ["2023/01/02", "2023/Jan/09", "2023-01-16"],
        })
        labels = pd.DataFrame({"country_code": ["my", "MY", "MY"],
                               "order_id": ["o1", "o2", "orphan"],
                               "customer_id": ["c1", "c2", "absent"], "is_fraud": [1, 0, 1]})
        with tempfile.TemporaryDirectory(prefix="pbi3-preview-test-") as directory:
            root = Path(directory)
            for name, frame in (("customers", customers), ("orders", orders), ("labels", labels)):
                frame.to_csv(root / f"{name}_base.csv", index=False)
            before = {file.name: file.read_bytes() for file in root.iterdir()}
            result = build_preview(root)
            after = {file.name: file.read_bytes() for file in root.iterdir()}
            self.assertEqual(before, after)
        logic = result["logic"].set_index("check")
        self.assertEqual(logic.loc["refund_above_order_value", "flagged"], 1)
        self.assertEqual(logic.loc["refunded_above_total_orders", "flagged"], 1)
        self.assertEqual(logic.loc["orders_without_label_key", "flagged"], 1)
        self.assertEqual(logic.loc["order_before_customer_first_order", "flagged"], 0)
        self.assertEqual(logic.loc["order_before_customer_first_order", "evaluable"], 1)
        rates = result["temporal"]["country_rates"].iloc[0]
        self.assertEqual((rates.rows, rates.valid_labels, rates.missing_labels), (3, 2, 1))
        self.assertEqual(rates.fraud_rate, .5)
        weekly = result["temporal"]["weekly_overall_rates"]
        self.assertEqual(weekly.valid_labels.sum(), 2)
        self.assertEqual(weekly.fraud_labels.sum(), 1)
        distribution = result["quality"]["numeric_distributions"].set_index("feature").loc["order_value"]
        self.assertEqual(distribution.source_missing_count, 0)
        self.assertEqual(distribution.conversion_failures, 1)
        self.assertEqual(distribution.missing_or_unparseable_count, 1)
        # Strict JSON also guarantees undefined rates do not leak NaN tokens.
        json.dumps(_serializable(result), allow_nan=False)

    def test_header_only_batch_produces_empty_evidence(self):
        columns = {
            "customers": ["country_code", "customer_id", "mobile_verified", "num_orders_last_50days",
                          "num_cancelled_orders_last_50days", "num_refund_orders_last_50days",
                          "total_payment_last_50days", "num_associated_customers", "first_order_datetime"],
            "orders": ["country_code", "order_id", "collect_type", "payment_method", "order_value",
                       "num_items_ordered", "refund_value", "order_date"],
            "labels": ["country_code", "order_id", "customer_id", "is_fraud"],
        }
        with tempfile.TemporaryDirectory(prefix="pbi3-preview-empty-") as directory:
            root = Path(directory)
            for name, names in columns.items():
                pd.DataFrame(columns=names).to_csv(root / f"{name}_base.csv", index=False)
            result = build_preview(root)
        self.assertEqual(result["link_audit"]["safe_orders"], 0)
        self.assertTrue(result["temporal"]["country_rates"].empty)
        self.assertTrue(result["quality"]["numeric_distributions"].finite_count.eq(0).all())
        json.dumps(_serializable(result), allow_nan=False)


if __name__ == "__main__":
    unittest.main()
