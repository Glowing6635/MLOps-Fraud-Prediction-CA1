"""Exercise loader, joins and analysis together without a second analysis script."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from mlops_fraud_prediction_ca1.data import load_base_data
from mlops_fraud_prediction_ca1.logic import business_logic_report
from mlops_fraud_prediction_ca1.quality import quality_reports
from mlops_fraud_prediction_ca1.relationships import link_orders_customers
from mlops_fraud_prediction_ca1.temporal import fraud_rates, weekly_fraud_rates


class EdaIntegrationTests(unittest.TestCase):
    def test_full_preview_preserves_sources_and_exposes_exclusions(self):
        customers = pd.DataFrame(
            {
                "country_code": ["my", "MY"],
                "customer_id": ["c1", "c2"],
                "mobile_verified": ["True", "yes"],
                "num_orders_last_50days": [2, 1],
                "num_cancelled_orders_last_50days": [0, 0],
                "num_refund_orders_last_50days": [0, 2],
                "total_payment_last_50days": [10, 20],
                "num_associated_customers": [1, 1],
                "first_order_datetime": ["2023-01-02 12:00:00", None],
            }
        )
        orders = pd.DataFrame(
            {
                "country_code": ["MY"] * 3,
                "order_id": ["o1", "o2", "o3"],
                "collect_type": ["Delivery"] * 3,
                "payment_method": ["cash", None, "cash"],
                "order_value": ["bad", 5, 20],
                "refund_value": [0, 6, 0],
                "num_items_ordered": [1, 2, 3],
                "order_date": ["2023/01/02", "2023/Jan/09", "2023-01-16"],
            }
        )
        labels = pd.DataFrame(
            {
                "country_code": ["my", "MY", "MY"],
                "order_id": ["o1", "o2", "orphan"],
                "customer_id": ["c1", "c2", "absent"],
                "is_fraud": [1, 0, 1],
            }
        )
        secret_order = orders.iloc[[0]].copy()
        secret_order["order_id"] = "secret"
        secret_order["order_date"] = "2023-04-17"
        secret_order["payment_method"] = "TEST_SECRET"
        secret_customer = customers.iloc[[0]].copy()
        secret_customer["customer_id"] = "secret_customer"
        secret_customer["mobile_verified"] = "TEST_SECRET"
        orders = pd.concat([orders, secret_order], ignore_index=True)
        customers = pd.concat([customers, secret_customer], ignore_index=True)
        labels = pd.concat(
            [
                labels,
                pd.DataFrame(
                    {
                        "country_code": ["MY"],
                        "order_id": ["secret"],
                        "customer_id": ["secret_customer"],
                        "is_fraud": [1],
                    }
                ),
            ],
            ignore_index=True,
        )
        with tempfile.TemporaryDirectory(prefix="pbi3-preview-test-") as directory:
            root = Path(directory)
            for name, frame in (
                ("customers", customers),
                ("orders", orders),
                ("labels", labels),
            ):
                frame.to_csv(root / f"{name}_base.csv", index=False)
            before = {file.name: file.read_bytes() for file in root.iterdir()}
            customers, orders, labels = load_base_data(root)
            for frame in (customers, orders, labels):
                self.assertNotIn("TEST_SECRET", frame.to_json())
            linked, _ = link_orders_customers(
                orders, labels, customers, normalize_country=True
            )
            logic = business_logic_report(
                customers, orders, labels, normalize_country=True
            )
            country_rates = fraud_rates(linked, ["country_code"])
            weekly = weekly_fraud_rates(linked)
            quality = quality_reports(
                {"customers": customers, "orders": orders, "labels": labels}
            )
            after = {file.name: file.read_bytes() for file in root.iterdir()}
            self.assertEqual(before, after)
        self.assertEqual(logic.loc["refund_above_order_value", "flagged"], 1)
        self.assertEqual(logic.loc["refunded_above_total_orders", "flagged"], 1)
        self.assertEqual(logic.loc["orders_without_label_key", "flagged"], 1)
        self.assertEqual(logic.loc["order_before_customer_first_order", "flagged"], 0)
        self.assertEqual(logic.loc["order_before_customer_first_order", "evaluable"], 1)
        rates = country_rates.iloc[0]
        self.assertEqual(
            (rates.rows, rates.valid_labels, rates.missing_labels), (3, 2, 1)
        )
        self.assertEqual(rates.fraud_rate, 0.5)
        self.assertEqual(weekly.valid_labels.sum(), 2)
        self.assertEqual(weekly.fraud_labels.sum(), 1)
        numeric = quality["numeric_validity"].set_index("column").loc["order_value"]
        self.assertEqual(numeric.conversion_failures, 1)

    def test_header_only_batch_produces_empty_evidence(self):
        columns = {
            "customers": [
                "country_code",
                "customer_id",
                "mobile_verified",
                "num_orders_last_50days",
                "num_cancelled_orders_last_50days",
                "num_refund_orders_last_50days",
                "total_payment_last_50days",
                "num_associated_customers",
                "first_order_datetime",
            ],
            "orders": [
                "country_code",
                "order_id",
                "collect_type",
                "payment_method",
                "order_value",
                "num_items_ordered",
                "refund_value",
                "order_date",
            ],
            "labels": ["country_code", "order_id", "customer_id", "is_fraud"],
        }
        with tempfile.TemporaryDirectory(prefix="pbi3-preview-empty-") as directory:
            root = Path(directory)
            for name, names in columns.items():
                pd.DataFrame(columns=names).to_csv(
                    root / f"{name}_base.csv", index=False
                )
            customers, orders, labels = load_base_data(root)
            linked, _ = link_orders_customers(
                orders, labels, customers, normalize_country=True
            )
            self.assertTrue(linked.empty)
            self.assertTrue(weekly_fraud_rates(linked).empty)
            self.assertTrue(fraud_rates(linked, ["country_code"]).empty)
