"""Check semantic categories and label denominators against small fixtures."""

import unittest

import pandas as pd
from pandas.testing import assert_frame_equal

from mlops_fraud_prediction_ca1.quality import quality_reports, raw_category_reports


class QualityReportTests(unittest.TestCase):
    def test_raw_categories_do_not_invent_semantic_mappings(self):
        raw = {
            "customers": pd.DataFrame({"country_code": ["my", "MY"], "mobile_verified": ["True", "yes"]}),
            "orders": pd.DataFrame({"country_code": ["MY", "MY"], "collect_type": ["Delivery", "pickup"],
                                    "payment_method": ["", "on Delivery"]}),
            "labels": pd.DataFrame({"country_code": ["my", "MY"], "is_fraud": ["0", "1"]}),
        }
        before = {name: frame.copy(deep=True) for name, frame in raw.items()}
        result = raw_category_reports(raw)
        yes = result.loc[result.original_value.eq("yes")].iloc[0]
        self.assertEqual(yes.status, "unexpected")
        self.assertEqual(yes.proposed_value, "yes")
        self.assertEqual(result.loc[result.original_value.eq(""), "status"].iloc[0], "blank")
        self.assertEqual(result.loc[result.original_value.eq("on Delivery"), "status"].iloc[0], "unchecked")
        for name in raw:
            assert_frame_equal(raw[name], before[name])

    def test_quality_reports_separate_invalid_labels_and_numeric_failures(self):
        datasets = {
            "customers": pd.DataFrame({"customer_id": ["a", "b", "c"],
                                       "num_orders_last_50days": [1, 1.5, -1]}),
            "orders": pd.DataFrame({"order_id": ["a", "b", "c"],
                                    "num_items_ordered": ["bad", None, "2"],
                                    "order_value": [1, float("inf"), 3]}),
            "labels": pd.DataFrame({"is_fraud": [1, 0, None, "unknown", True]}),
        }
        result = quality_reports(datasets)
        label = result["class_balance"].iloc[0]
        self.assertEqual((label.valid_labels, label.fraud_labels, label.missing_labels, label.invalid_labels), (2, 1, 1, 2))
        self.assertEqual(label.fraud_rate, .5)
        numeric = result["numeric_validity"].set_index("column")
        self.assertEqual(numeric.loc["num_items_ordered", "conversion_failures"], 1)
        self.assertEqual(numeric.loc["num_items_ordered", "missing"], 1)
        self.assertEqual(numeric.loc["order_value", "infinite"], 1)
        self.assertEqual(numeric.loc["num_orders_last_50days", "nonintegral_count"], 1)
        self.assertEqual(numeric.loc["num_orders_last_50days", "negative"], 1)


if __name__ == "__main__":
    unittest.main()
