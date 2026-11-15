"""Temporal reports use deterministic dates and explicit fraud denominators."""

import unittest

import pandas as pd
from pandas.testing import assert_series_equal

from mlops_fraud_prediction_ca1.temporal import (
    date_profile, fraud_rates, largest_weekly_change, parse_dates, weekly_fraud_rates,
)


class TemporalTests(unittest.TestCase):
    def test_explicit_formats_and_missing_invalid_unrecognized(self):
        values = pd.Series(["2024/01/02", "2024-01-03", "2024/Feb/29", "2023/Feb/29", "2024/Foo/01", "01/02/2024", None, "", "2024-1-02"])
        original = values.copy(deep=True)
        parsed = parse_dates(values)
        self.assertEqual(parsed.status.tolist(), ["valid"] * 3 + ["invalid", "invalid", "unrecognized", "missing", "unrecognized", "unrecognized"])
        self.assertEqual(parsed.date.iloc[2], pd.Timestamp("2024-02-29"))
        self.assertEqual(parsed["format"].iloc[2], "%Y/%b/%d")
        profile = date_profile(values)
        self.assertEqual(profile["valid"], 3)
        self.assertEqual(profile["minimum"], pd.Timestamp("2024-01-02"))
        self.assertEqual(set(profile["formats"]), {"%Y/%m/%d", "%Y-%m-%d", "%Y/%b/%d"})
        assert_series_equal(values, original)
        self.assertEqual(parse_dates(pd.Series(["2024-01-01 23:59:59"]), kind="customer").status.item(), "valid")

    def test_rates_disclose_invalid_and_missing_denominators(self):
        rows = pd.DataFrame({"country_code": ["MY"] * 7 + ["PK"], "order_id": range(8),
                             "is_fraud": [0, 1, "1", None, 2, "yes", True, None]})
        result = fraud_rates(rows, ["country_code"]).set_index("country_code")
        self.assertEqual(result.loc["MY", "valid_labels"], 3)
        self.assertEqual(result.loc["MY", "fraud_labels"], 2)
        self.assertEqual(result.loc["MY", "missing_labels"], 1)
        self.assertEqual(result.loc["MY", "invalid_labels"], 3)
        self.assertEqual(result.loc["MY", "fraud_rate"], 2 / 3)
        self.assertTrue(pd.isna(result.loc["PK", "fraud_rate"]))
        with self.assertRaises(ValueError):
            fraud_rates(pd.concat([rows, rows.iloc[:1]]), ["country_code"])
        with self.assertRaises(ValueError):
            fraud_rates(rows.assign(order_id=None), ["country_code"])

    def test_weeks_boundaries_and_dated_descriptive_change(self):
        rows = pd.DataFrame({"country_code": ["MY"] * 6, "order_id": range(6),
                             "order_date": ["2024-01-01", "2024-01-07", "2024-01-08", "2024-01-14", "2024-01-15", "bad"],
                             "is_fraud": [0, 0, 1, 1, 0, 1]})
        weekly = weekly_fraud_rates(rows)
        self.assertEqual(weekly.week_start.tolist(), list(pd.to_datetime(["2024-01-01", "2024-01-08", "2024-01-15"])))
        self.assertEqual(weekly.partial_week.tolist(), [False, False, True])
        self.assertEqual(weekly.attrs["excluded_date_rows"], 1)
        change = largest_weekly_change(weekly)
        self.assertEqual(change.week_start.item(), pd.Timestamp("2024-01-08"))
        self.assertEqual(change.rate_change.item(), 1)
        self.assertEqual(change.previous_valid_labels.item(), 2)
        self.assertFalse(change.partial_week.item())

    def test_missing_weeks_are_not_adjacent_and_empty_is_valid(self):
        rows = pd.DataFrame({"country_code": ["MY", "MY"], "order_id": [1, 2],
                             "order_date": ["2024-01-01", "2024-01-15"], "is_fraud": [0, 1]})
        self.assertTrue(largest_weekly_change(weekly_fraud_rates(rows)).empty)
        empty = weekly_fraud_rates(rows.iloc[:0])
        self.assertTrue(empty.empty)
        self.assertEqual(empty.attrs["excluded_date_rows"], 0)
