"""Temporal reports use deterministic dates and explicit fraud denominators."""

import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal

from mlops_fraud_prediction_ca1.temporal import (
    daily_order_counts,
    date_profile,
    fraud_rates,
    largest_weekly_change,
    parse_dates,
    weekly_country_changes,
    weekly_fraud_rates,
)


class TemporalTests(unittest.TestCase):
    def test_before_after_scan_finds_early_and_late_movement_with_pooled_counts(self):
        weeks = pd.date_range("2022-11-14", periods=25, freq="7D")
        rows = []
        for country, boundary, before, after in (
            ("BD", 2, 10, 30),
            ("PK", 16, 20, 40),
            ("PH", 21, 20, 5),
        ):
            for i, week in enumerate(weeks):
                rows.append(
                    {
                        "country_code": country,
                        "week_start": week,
                        "valid_labels": 100,
                        "fraud_labels": before if i < boundary else after,
                        "partial_week": False,
                    }
                )
        rates = pd.DataFrame(rows)
        original = rates.copy(deep=True)
        changes = weekly_country_changes(rates)
        for country, boundary in (("BD", 2), ("PK", 16), ("PH", 21)):
            candidates = changes.loc[changes.country_code.eq(country)]
            best = candidates.loc[candidates.difference_pp.abs().idxmax()]
            self.assertEqual(best.boundary_week, weeks[boundary])
            self.assertEqual(best.before_valid + best.after_valid, 2500)
            self.assertEqual(best.before_weeks + best.after_weeks, 25)
        self.assertEqual(len(changes), 3 * 22)
        pd.testing.assert_frame_equal(rates, original)
        self.assertTrue(weekly_country_changes(rates.iloc[:2]).empty)
        self.assertTrue(
            weekly_country_changes(rates.drop(index=3)).country_code.ne("BD").all()
        )
        with self.assertRaises(ValueError):
            weekly_country_changes(pd.concat([rates, rates.iloc[:1]]))
        with self.assertRaises(ValueError):
            weekly_country_changes(rates, minimum_weeks=0)

    def test_daily_counts_reconcile_gaps_exclusions_and_boundaries(self):
        rows = pd.DataFrame(
            {
                "country_code": ["MY"] * 6,
                "order_id": list("abcdef"),
                "order_date": [
                    "2024-01-01",
                    "2024/Jan/03",
                    "2024/01/03",
                    None,
                    "2024-02-30",
                    "bad",
                ],
            }
        )
        original = rows.copy(deep=True)
        result = daily_order_counts(rows)
        self.assertEqual(result.orders.tolist(), [1, 0, 2])
        self.assertEqual(result.potential_partial_day.tolist(), [True, False, True])
        self.assertEqual(result.attrs["excluded_date_rows"], 3)
        self.assertEqual(
            result.orders.sum() + result.attrs["excluded_date_rows"], len(rows)
        )
        pd.testing.assert_frame_equal(rows, original)
        with self.assertRaises(ValueError):
            daily_order_counts(pd.concat([rows, rows.iloc[:1]]))

    def test_daily_counts_empty_and_all_undated(self):
        rows = pd.DataFrame(
            {"country_code": ["MY"], "order_id": ["a"], "order_date": [None]}
        )
        for sample in [rows, rows.iloc[:0]]:
            result = daily_order_counts(sample)
            self.assertTrue(result.empty)
            self.assertEqual(result.attrs["excluded_date_rows"], len(sample))

    def test_explicit_formats_and_missing_invalid_unrecognized(self):
        values = pd.Series(
            [
                "2024/01/02",
                "2024-01-03",
                "2024/Feb/29",
                "2023/Feb/29",
                "2024/Foo/01",
                "01/02/2024",
                None,
                "",
                "2024-1-02",
            ]
        )
        original = values.copy(deep=True)
        parsed = parse_dates(values)
        self.assertEqual(
            parsed.status.tolist(),
            ["valid"] * 3
            + [
                "invalid",
                "invalid",
                "unrecognized",
                "missing",
                "unrecognized",
                "unrecognized",
            ],
        )
        self.assertEqual(parsed.date.iloc[2], pd.Timestamp("2024-02-29"))
        self.assertEqual(parsed["format"].iloc[2], "%Y/%b/%d")
        profile = date_profile(values)
        self.assertEqual(profile["valid"], 3)
        self.assertEqual(profile["minimum"], pd.Timestamp("2024-01-02"))
        self.assertEqual(set(profile["formats"]), {"%Y/%m/%d", "%Y-%m-%d", "%Y/%b/%d"})
        assert_series_equal(values, original)
        self.assertEqual(
            parse_dates(
                pd.Series(["2024-01-01 23:59:59"]), kind="customer"
            ).status.item(),
            "valid",
        )

    def test_month_names_are_case_insensitive_and_numeric_dates_are_year_first(self):
        values = pd.Series(
            ["2024/FEB/01", "2024/feb/01", "2024/Feb/01", "2024/01/02", "2024-01-02"]
        )
        parsed = parse_dates(values)
        self.assertEqual(parsed.status.tolist(), ["valid"] * 5)
        self.assertEqual(
            parsed.date.tolist(),
            [pd.Timestamp("2024-02-01")] * 3 + [pd.Timestamp("2024-01-02")] * 2,
        )

    def test_rates_disclose_invalid_and_missing_denominators(self):
        rows = pd.DataFrame(
            {
                "country_code": ["MY"] * 7 + ["PK"],
                "order_id": range(8),
                "is_fraud": [0, 1, "1", None, 2, "yes", True, None],
            }
        )
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
        rows = pd.DataFrame(
            {
                "country_code": ["MY"] * 6,
                "order_id": range(6),
                "order_date": [
                    "2024-01-01",
                    "2024-01-07",
                    "2024-01-08",
                    "2024-01-14",
                    "2024-01-15",
                    "bad",
                ],
                "is_fraud": [0, 0, 1, 1, 0, 1],
            }
        )
        weekly = weekly_fraud_rates(rows)
        self.assertEqual(
            weekly.week_start.tolist(),
            list(pd.to_datetime(["2024-01-01", "2024-01-08", "2024-01-15"])),
        )
        self.assertEqual(weekly.partial_week.tolist(), [False, False, True])
        self.assertEqual(weekly.attrs["excluded_date_rows"], 1)
        change = largest_weekly_change(weekly)
        self.assertEqual(change.week_start.item(), pd.Timestamp("2024-01-08"))
        self.assertEqual(change.rate_change.item(), 1)
        self.assertEqual(change.previous_valid_labels.item(), 2)
        self.assertFalse(change.partial_week.item())

    def test_object_numeric_labels_accept_numpy_scalars_but_reject_booleans(
        self,
    ) -> None:
        """Count numeric zero/one labels consistently across scalar representations."""
        rows = pd.DataFrame(
            {
                "country_code": ["MY"] * 8,
                "order_id": range(8),
                "is_fraud": pd.Series(
                    [
                        np.int64(1),
                        np.int32(0),
                        np.float32(1),
                        np.float64(0),
                        True,
                        np.bool_(True),
                        np.int64(2),
                        None,
                    ],
                    dtype=object,
                ),
            }
        )
        result = fraud_rates(rows, ["country_code"]).iloc[0]
        self.assertEqual(result.valid_labels, 4)
        self.assertEqual(result.fraud_labels, 2)
        self.assertEqual(result.invalid_labels, 3)
        self.assertEqual(result.missing_labels, 1)
        self.assertEqual(result.fraud_rate, 0.5)

    def test_largest_change_ignores_dataframe_index_labels(self) -> None:
        """Weekly comparisons depend on dates, including when row indices repeat."""
        rows = pd.DataFrame(
            {
                "country_code": ["MY"] * 3,
                "order_id": range(3),
                "order_date": ["2024-01-01", "2024-01-08", "2024-01-15"],
                "is_fraud": [0, 1, 1],
            }
        )
        weekly = weekly_fraud_rates(rows)
        weekly.index = [0, 0, 0]
        change = largest_weekly_change(weekly)
        self.assertEqual(change.previous_week.item(), pd.Timestamp("2024-01-01"))
        self.assertEqual(change.week_start.item(), pd.Timestamp("2024-01-08"))
        self.assertEqual(change.rate_change.item(), 1)

    def test_missing_weeks_are_not_adjacent_and_empty_is_valid(self):
        rows = pd.DataFrame(
            {
                "country_code": ["MY", "MY"],
                "order_id": [1, 2],
                "order_date": ["2024-01-01", "2024-01-15"],
                "is_fraud": [0, 1],
            }
        )
        self.assertTrue(largest_weekly_change(weekly_fraud_rates(rows)).empty)
        empty = weekly_fraud_rates(rows.iloc[:0])
        self.assertTrue(empty.empty)
        self.assertEqual(empty.attrs["excluded_date_rows"], 0)
