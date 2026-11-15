"""Verify numerical screening arithmetic, denominators, and edge cases."""

import math
import unittest

import pandas as pd
from pandas.testing import assert_frame_equal

from mlops_fraud_prediction_ca1.numeric import profile_numeric_features


class NumericProfileTests(unittest.TestCase):
    """Check evidence against independently calculable small distributions."""

    def test_fences_exclude_missing_and_infinite_values(self) -> None:
        """Count both tails using finite observations without modifying input."""
        frame: pd.DataFrame = pd.DataFrame({
            "amount": [-100, -2, -1, 0, 1, 2, 100, float("inf"), None],
        })
        original: pd.DataFrame = frame.copy(deep=True)
        row: pd.Series = profile_numeric_features(
            {"orders": frame}, {"orders": ["amount"]}
        ).loc[("orders", "amount")]
        for key, value in {
            "rows": 9, "finite_count": 7, "missing_count": 1,
            "infinite_count": 1, "q1": -1.5, "q3": 1.5, "iqr": 3,
            "lower_fence": -6, "upper_fence": 6, "below_fence": 1,
            "above_fence": 1, "flagged_count": 2, "negative_count": 3,
            "median": 0, "mean": 0, "skewness": 0,
        }.items():
            with self.subTest(statistic=key):
                self.assertEqual(row[key], value)
        self.assertAlmostEqual(row["flagged_percent"], 2 / 7 * 100)
        self.assertAlmostEqual(row["zero_percent"], 1 / 7 * 100)
        assert_frame_equal(frame, original)

    def test_zero_iqr_is_explicit_and_fences_are_inclusive(self) -> None:
        """Expose zero-heavy screening and retain observations on the fences."""
        frame: pd.DataFrame = pd.DataFrame({
            "zero_heavy": [0, 0, 0, 0, 5],
            "on_fences": [-3, 0, 1, 2, 5],
        })
        result: pd.DataFrame = profile_numeric_features(
            {"example": frame}, {"example": list(frame.columns)}
        )
        self.assertTrue(result.loc[("example", "zero_heavy"), "zero_iqr"])
        self.assertEqual(result.loc[("example", "zero_heavy"), "flagged_count"], 1)
        self.assertEqual(result.loc[("example", "zero_heavy"), "flagged_percent"], 20)
        self.assertEqual(result.loc[("example", "on_fences"), "flagged_count"], 0)

    def test_undefined_distributions_are_not_reported_as_symmetric(self) -> None:
        """Represent constant, tiny, and entirely missing skewness as undefined."""
        for values in ([4.0, 4.0, 4.0], [1.0, 2.0], [float("nan")], []):
            with self.subTest(values=values):
                frame: pd.DataFrame = pd.DataFrame({"value": pd.Series(values, dtype="float64")})
                row: pd.Series = profile_numeric_features(
                    {"example": frame}, {"example": ["value"]}
                ).iloc[0]
                self.assertTrue(math.isnan(row["skewness"]))
                if row["finite_count"] == 0:
                    self.assertTrue(math.isnan(row["flagged_percent"]))

    def test_rejects_unsupported_features_and_rules(self) -> None:
        """Avoid silently coercing text/categories or accepting invalid fences."""
        for values in (["1", "2"], [True, False], [1 + 2j, 3 + 4j]):
            with self.subTest(values=values):
                with self.assertRaises(TypeError):
                    profile_numeric_features(
                        {"example": pd.DataFrame({"value": values})},
                        {"example": ["value"]},
                    )
        for multiplier in (0, -1, float("inf"), float("nan")):
            with self.subTest(multiplier=multiplier):
                with self.assertRaises(ValueError):
                    profile_numeric_features(
                        {"example": pd.DataFrame({"value": [1, 2, 3]})},
                        {"example": ["value"]}, iqr_multiplier=multiplier,
                    )
        with self.assertRaises(ValueError):
            profile_numeric_features({}, {})
