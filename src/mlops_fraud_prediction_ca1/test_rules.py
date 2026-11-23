"""Account-age calculation uses real input representations and flags bad ages."""

import unittest

import pandas as pd

from mlops_fraud_prediction_ca1.cleaning import clean_frame
from mlops_fraud_prediction_ca1.rules import account_age_days


def _age(order, first):
    return account_age_days(pd.Series(order), pd.Series(first, dtype=object))


class AccountAgeTests(unittest.TestCase):
    def test_same_day_is_zero_whatever_the_clock_times(self):
        result = _age(
            ["2022/11/14"] * 2, ["2022-11-14 00:00:00", "2022-11-14 23:59:59"]
        )
        self.assertEqual(result.account_age_days.tolist(), [0, 0])
        self.assertTrue(result.account_age_status.eq("valid").all())

    def test_calendar_boundaries_count_dates_not_elapsed_hours(self):
        result = _age(
            ["2022-11-15", "2022/Mar/01", "2023/01/01", "2022-11-16"],
            [
                "2022-11-14 23:59:59",  # one minute before midnight: 1 day
                "2022-02-28 12:00:00",  # non-leap February: 1 day
                "2022-12-31 00:00:01",  # year boundary: 1 day
                "2022-11-14 00:00:00",  # 2 days
            ],
        )
        self.assertEqual(result.account_age_days.tolist(), [1, 1, 1, 2])

    def test_leap_day_boundary(self):
        result = _age(["2024/03/01"], ["2024-02-28 10:00:00"])
        self.assertEqual(result.account_age_days.tolist(), [2])

    def test_missing_and_unusable_dates_are_na_with_reason(self):
        result = _age(
            [None, "2022/11/14", "2022/11/14", "2022/13/45", "2022/11/14"],
            [
                "2022-11-01 00:00:00",
                None,
                "not a date",
                "2022-11-01 00:00:00",
                "2022-02-30 00:00:00",
            ],
        )
        self.assertTrue(result.account_age_days.isna().all())
        self.assertEqual(
            result.account_age_status.tolist(),
            [
                "missing_order_date",
                "missing_first_order",
                "unusable_first_order",
                "unusable_order_date",
                "unusable_first_order",
            ],
        )

    def test_negative_age_is_withheld_not_repaired_or_zero(self):
        result = _age(["2022/11/14"], ["2022-11-15 00:00:01"])
        self.assertTrue(result.account_age_days.isna().all())
        self.assertEqual(result.account_age_status.tolist(), ["negative"])

    def test_cleaned_datetimes_give_same_answers_as_raw_text(self):
        raw = pd.DataFrame(
            {
                "order_date": ["2022/Nov/15", "2022/11/14", None, "2022/11/14"],
                "first_order_datetime": [
                    "2022-11-14 23:59:59",
                    "2022-11-15 00:00:00",
                    "2022-11-01 00:00:00",
                    "2022-11-14 08:00:00",
                ],
            }
        )
        cleaned = clean_frame(raw)
        from_raw = account_age_days(raw.order_date, raw.first_order_datetime)
        from_clean = account_age_days(
            cleaned.order_date,
            cleaned.first_order_datetime,
            raw_order_date=cleaned.raw_order_date,
            raw_first_order_datetime=cleaned.raw_first_order_datetime,
        )
        pd.testing.assert_frame_equal(from_raw, from_clean)
        self.assertEqual(from_raw.account_age_days.tolist(), [1, pd.NA, pd.NA, 0])

    def test_whole_parser_range_does_not_overflow(self):
        result = _age(
            ["2100-01-01", "2262-04-11", "2022-11-14"],
            [
                "1800-01-01 00:00:00",
                "1677-09-21 12:00:00",  # midnight is below the datetime64[ns] floor
                "1677-09-21 12:00:00",
            ],
        )
        self.assertEqual(result.account_age_days.tolist()[0], 109573)
        self.assertTrue(result.account_age_status.eq("valid").all())
        raw = pd.DataFrame(
            {
                "order_date": ["2100-01-01", "2022-11-14"],
                "first_order_datetime": [
                    "1800-01-01 00:00:00",
                    "1677-09-21 12:00:00",
                ],
            }
        )
        cleaned = clean_frame(raw)
        pd.testing.assert_frame_equal(
            account_age_days(raw.order_date, raw.first_order_datetime),
            account_age_days(cleaned.order_date, cleaned.first_order_datetime),
        )

    def test_cleaned_nat_without_raw_is_unavailable_not_missing(self):
        raw = pd.DataFrame(
            {
                "order_date": [None, "2022/13/45", "2022/11/14", "2022/11/14"],
                "first_order_datetime": [
                    "2022-11-01 00:00:00",
                    "2022-11-01 00:00:00",
                    None,
                    "bad",
                ],
            },
            index=[5, 6, 7, 8],
        )
        cleaned = clean_frame(raw)
        bare = account_age_days(cleaned.order_date, cleaned.first_order_datetime)
        self.assertEqual(
            bare.account_age_status.tolist(),
            [
                "unavailable_order_date",
                "unavailable_order_date",
                "unavailable_first_order",
                "unavailable_first_order",
            ],
        )
        self.assertEqual(bare.index.tolist(), [5, 6, 7, 8])
        restored = account_age_days(
            cleaned.order_date,
            cleaned.first_order_datetime,
            raw_order_date=cleaned.raw_order_date,
            raw_first_order_datetime=cleaned.raw_first_order_datetime,
        )
        direct = account_age_days(raw.order_date, raw.first_order_datetime)
        pd.testing.assert_frame_equal(restored, direct)
        self.assertEqual(
            restored.account_age_status.tolist(),
            [
                "missing_order_date",
                "unusable_order_date",
                "missing_first_order",
                "unusable_first_order",
            ],
        )

    def test_mixed_batch_with_boundary_dates_keeps_each_reason(self):
        raw = pd.DataFrame(
            {
                "order_date": ["2100-01-01", "2022-11-14", None, "2022-11-14", "x"],
                "first_order_datetime": [
                    "1800-01-01 00:00:00",
                    "2022-11-15 00:00:01",
                    "1677-09-21 12:00:00",
                    "2022-02-30 00:00:00",
                    None,
                ],
            }
        )
        cleaned = clean_frame(raw)
        from_raw = account_age_days(raw.order_date, raw.first_order_datetime)
        from_clean = account_age_days(
            cleaned.order_date,
            cleaned.first_order_datetime,
            raw_order_date=cleaned.raw_order_date,
            raw_first_order_datetime=cleaned.raw_first_order_datetime,
        )
        pd.testing.assert_frame_equal(from_raw, from_clean)
        self.assertEqual(
            from_raw.account_age_status.tolist(),
            [
                "valid",
                "negative",
                "missing_order_date",
                "unusable_first_order",
                "unusable_order_date",
            ],
        )
        self.assertEqual(from_raw.account_age_days.tolist()[0], 109573)
        with self.assertRaises(ValueError):
            account_age_days(
                cleaned.order_date,
                cleaned.first_order_datetime,
                raw_order_date=cleaned.raw_order_date.reset_index(drop=True).rename(
                    lambda i: i + 10
                ),
            )

    def test_index_is_preserved_and_mismatch_raises(self):
        order = pd.Series(["2022/11/14"], index=[7])
        first = pd.Series(["2022-11-13 01:00:00"], index=[7])
        self.assertEqual(account_age_days(order, first).index.tolist(), [7])
        with self.assertRaises(ValueError):
            account_age_days(order, first.reset_index(drop=True))


if __name__ == "__main__":
    unittest.main()
