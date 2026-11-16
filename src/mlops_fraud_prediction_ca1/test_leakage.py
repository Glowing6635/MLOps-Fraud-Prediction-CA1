"""Leakage audits fail closed and distinguish observation from confirmation."""

import unittest

import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal

from mlops_fraud_prediction_ca1.leakage import (
    ASSUMED_LABEL_DELAY_DAYS,
    CUSTOMER_FEATURE_COLUMNS,
    CUSTOMER_WINDOW_COLUMNS,
    ORDER_FEATURE_COLUMNS,
    availability_inventory,
    customer_timing_audit,
    historical_order_mask,
    select_feature_inputs,
)


class LeakageTests(unittest.TestCase):
    def setUp(self):
        self.customers = pd.DataFrame(
            {
                "country_code": ["my", "MY"],
                "customer_id": [1, 2],
                "first_order_datetime": ["2024-01-02 23:00:00", None],
            }
        )
        self.orders = pd.DataFrame(
            {
                "country_code": ["MY"] * 6,
                "order_id": [1, 2, 3, 4, 5, 6],
                "order_date": [
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-03",
                    "bad",
                    "2024-01-04",
                    "2024-01-04",
                ],
            }
        )
        self.labels = pd.DataFrame(
            {
                "country_code": ["MY"] * 5,
                "order_id": [1, 2, 3, 4, 5],
                "customer_id": [1, 1, 1, 1, 2],
                "is_fraud": [0, 1, 0, 1, 0],
            }
        )

    def test_inventory_covers_actual_columns_and_never_confirms_timing(self):
        customers = self.customers.assign(**{c: 0 for c in CUSTOMER_WINDOW_COLUMNS})
        orders = self.orders.assign(refund_value=0, new_field="unknown")
        frames = (customers, orders, self.labels)
        originals = [f.copy(deep=True) for f in frames]
        report = availability_inventory(*frames, reviewed_on="2026-11-15")
        self.assertEqual(len(report), sum(len(f.columns) for f in frames))
        self.assertEqual(report.column.eq("country_code").sum(), 3)
        self.assertTrue(report.status.eq("unconfirmed").all())
        self.assertEqual(
            report.loc[report.column.eq("refund_value"), "role"].item(),
            "evaluation only",
        )
        self.assertEqual(
            report.loc[report.column.eq("is_fraud"), "role"].item(), "target"
        )
        self.assertEqual(
            report.loc[report.column.eq("new_field"), "role"].item(), "unknown"
        )
        for frame, original in zip(frames, originals, strict=True):
            assert_frame_equal(frame, original)

    def test_customer_evidence_is_per_order_and_calendar_day_only(self):
        originals = [
            f.copy(deep=True) for f in (self.customers, self.orders, self.labels)
        ]
        rows, summary = customer_timing_audit(self.customers, self.orders, self.labels)
        self.assertEqual(
            rows.date_evidence.tolist(),
            [
                "before_first_order",
                "same_day_intraday_unknown",
                "after_first_order_day",
                "unevaluable",
                "unevaluable",
                "unevaluable",
            ],
        )
        self.assertEqual(summary["date_evaluable_orders"], 3)
        self.assertEqual(summary["customers_with_multiple_observed_orders"], 1)
        self.assertEqual(summary["orders_for_multiple_order_customers"], 4)
        self.assertEqual(summary["orders_strictly_after_earliest_observed_day"], 2)
        self.assertTrue(
            rows.loc[rows.customer_id.eq(1), "observed_order_count"].eq(4).all()
        )
        self.assertTrue(summary["snapshot_anchor"].startswith("unknown"))
        for frame, original in zip(
            (self.customers, self.orders, self.labels), originals, strict=True
        ):
            assert_frame_equal(frame, original)

    def test_exact_copies_collapse_conflicting_bridge_is_not_used(self):
        orders = pd.concat([self.orders, self.orders.iloc[[0]]], ignore_index=True)
        labels = pd.concat(
            [self.labels, self.labels.iloc[[0]].assign(customer_id=2)],
            ignore_index=True,
        )
        rows, summary = customer_timing_audit(self.customers, orders, labels)
        self.assertEqual(len(rows), 6)
        self.assertEqual(
            rows.loc[rows.order_id.eq(1), "date_evidence"].item(), "unevaluable"
        )
        self.assertEqual(summary["link_audit"]["orders"]["raw_exact_extra_copies"], 1)

    def test_empty_inputs(self):
        rows, summary = customer_timing_audit(
            self.customers.iloc[:0],
            self.orders.iloc[:0],
            self.labels.iloc[:0],
        )
        self.assertTrue(rows.empty)
        self.assertEqual(summary["safe_orders"], 0)
        self.assertEqual(summary["customers_with_multiple_observed_orders"], 0)

    def test_history_strict_delay_boundary_invalid_dates_and_duplicate_index(self):
        prior = pd.Series(
            ["2024-01-01", "2024/Jan/02", "2024/01/03", "2024-01-04", "bad", None],
            index=[0, 0, 1, 1, 2, 2],
        )
        original = prior.copy(deep=True)
        assert_series_equal(
            historical_order_mask(prior, "2024-01-04", label_delay_days=0),
            pd.Series([True, True, True, False, False, False], index=prior.index),
        )
        assert_series_equal(
            historical_order_mask(prior, "2024-01-04", label_delay_days=2),
            pd.Series([True, False, False, False, False, False], index=prior.index),
        )
        assert_series_equal(prior, original)

    def test_history_rejects_implicit_or_invalid_timing(self):
        for delay in (-1, 1.5, True, "2", float("nan")):
            with self.subTest(delay=delay), self.assertRaises(ValueError):
                historical_order_mask(
                    pd.Series(["2024-01-01"]), "2024-01-04", label_delay_days=delay
                )
        with self.assertRaises(ValueError):
            historical_order_mask(
                pd.Series(["2024-01-01"]), "2024-02-30", label_delay_days=0
            )

    def test_champion_excludes_customer_group_and_all_forbidden_inputs(self) -> None:
        """Champion selection withholds risky columns, including future fields."""
        columns = [
            "country_code",
            "customer_id",
            "order_id",
            "is_fraud",
            "refund_value",
            "order_date",
            "future_field",
            *CUSTOMER_FEATURE_COLUMNS,
            *ORDER_FEATURE_COLUMNS,
        ]
        frame = pd.DataFrame([[0] * len(columns)], columns=columns, index=[42])
        original = frame.copy(deep=True)
        for source in ("customers", "labels"):
            selected = select_feature_inputs(frame, source=source)
            self.assertEqual(selected.shape, (1, 0))
            self.assertEqual(selected.index.tolist(), [42])
        selected = select_feature_inputs(frame, source="orders")
        self.assertEqual(selected.columns.tolist(), list(ORDER_FEATURE_COLUMNS))
        selected.iloc[0, 0] = 99
        assert_frame_equal(frame, original)

    def test_exploratory_selects_customer_group_only_in_customer_source(self) -> None:
        """The opt-in keeps seven fields together and cannot admit a label."""
        frame = pd.DataFrame(
            columns=[
                "is_fraud",
                *reversed(CUSTOMER_FEATURE_COLUMNS),
                "new_snapshot_field",
            ]
        )
        self.assertEqual(
            select_feature_inputs(
                frame, source="customers", exploratory=True
            ).columns.tolist(),
            list(reversed(CUSTOMER_FEATURE_COLUMNS)),
        )
        for source in ("orders", "labels"):
            self.assertEqual(
                select_feature_inputs(frame, source=source, exploratory=True).shape,
                (0, 0),
            )

    def test_feature_selector_rejects_ambiguous_source_and_options(self) -> None:
        """Typos and truthy strings must not opt a caller into risky features."""
        with self.assertRaises(ValueError):
            select_feature_inputs(pd.DataFrame(), source="order")
        with self.assertRaises(TypeError):
            select_feature_inputs(
                pd.DataFrame(), source="customers", exploratory="False"
            )
        with self.assertRaises(ValueError):
            select_feature_inputs(
                pd.DataFrame(columns=["order_value"] * 2), source="orders"
            )

    def test_history_requires_explicit_delay(self) -> None:
        with self.assertRaises(TypeError):
            historical_order_mask(pd.Series(["2024-01-01"]), "2024-02-01")

    def test_recorded_label_delay_strict_boundary_and_no_label_mode(self) -> None:
        """Thirty-day assumption excludes equality, future, and invalid dates."""
        prior = pd.Series(
            [
                "2024-01-01",
                "2024-01-02",
                "2024-01-03",
                "2024-02-01",
                "2024-02-02",
                "bad",
                None,
            ],
            name="order_date",
        )
        self.assertEqual(ASSUMED_LABEL_DELAY_DAYS, 30)
        assert_series_equal(
            historical_order_mask(
                prior,
                "2024-02-01",
                label_delay_days=ASSUMED_LABEL_DELAY_DAYS,
            ),
            pd.Series(
                [True, False, False, False, False, False, False], name=prior.name
            ),
        )
        assert_series_equal(
            historical_order_mask(prior, "2024-02-01", label_delay_days=0),
            pd.Series([True, True, True, False, False, False, False], name=prior.name),
        )


if __name__ == "__main__":
    unittest.main()
