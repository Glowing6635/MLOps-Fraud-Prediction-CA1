"""Check the skew/outlier transform on synthetic rows only."""

import math
import unittest

import pandas as pd
from pandas.testing import assert_frame_equal

from mlops_fraud_prediction_ca1.cleaning import clean_frame
from mlops_fraud_prediction_ca1.leakage import ORDER_FEATURE_COLUMNS
from mlops_fraud_prediction_ca1.skew_outliers import (
    TRANSFORMED_FEATURES,
    TREATMENTS,
    transform_skewed_features,
)


def _orders(values: list, items: list) -> pd.DataFrame:
    return clean_frame(
        pd.DataFrame(
            {
                "order_id": [f"{i:03d}" for i in range(len(values))],
                "order_value": values,
                "num_items_ordered": items,
            },
            index=[10 + i for i in range(len(values))],
        )
    )


class SkewOutlierTests(unittest.TestCase):
    def test_reason_recorded_for_every_relevant_feature(self) -> None:
        for feature in (*ORDER_FEATURE_COLUMNS, "refund_value"):
            self.assertIn(feature, TREATMENTS)
        self.assertTrue(all(TREATMENTS[f] for f in TRANSFORMED_FEATURES))

    def test_extremes_compressed_but_raw_retained(self) -> None:
        frame = _orders(["10", "0", "1000000"], ["1", "2", "1600"])
        out = transform_skewed_features(frame)
        self.assertAlmostEqual(out["num_items_ordered_log1p"].iloc[2], math.log1p(1600))
        self.assertEqual(out["order_value_log1p"].iloc[1], 0.0)
        assert_frame_equal(out[list(frame.columns)], frame)  # raw/keys/index intact
        self.assertEqual(list(out.index), [10, 11, 12])

    def test_missing_infinite_and_negative_inputs(self) -> None:
        frame = _orders(["5", None, "-3"], [None, "x", "4"])
        frame.loc[10, "order_value"] = float("inf")
        out = transform_skewed_features(frame)
        self.assertTrue(pd.isna(out["order_value_log1p"].iloc[0]))
        self.assertTrue(pd.isna(out["order_value_log1p"].iloc[1]))
        self.assertAlmostEqual(out["order_value_log1p"].iloc[2], -math.log1p(3))
        self.assertTrue(out["num_items_ordered_log1p"].iloc[:2].isna().all())
        self.assertEqual(str(out["order_value_log1p"].dtype), "Float64")

    def test_heldout_rows_unaffected_by_other_rows(self) -> None:
        train = _orders(["1", "2", "3"], ["1", "1", "2"])
        held = _orders(["9999"], ["500"])
        alone = transform_skewed_features(held)
        combined = transform_skewed_features(pd.concat([train, held]))
        self.assertEqual(
            alone["order_value_log1p"].iloc[0], combined["order_value_log1p"].iloc[-1]
        )

    def test_input_not_mutated_and_bad_input_rejected(self) -> None:
        frame = _orders(["1"], ["1"])
        before = frame.copy(deep=True)
        transform_skewed_features(frame)
        assert_frame_equal(frame, before)
        with self.assertRaises(KeyError):
            transform_skewed_features(frame.drop(columns="order_value"))
        with self.assertRaises(TypeError):
            transform_skewed_features(frame.assign(order_value=["a"]))
        with self.assertRaises(ValueError):
            transform_skewed_features(transform_skewed_features(frame))


if __name__ == "__main__":
    unittest.main()
