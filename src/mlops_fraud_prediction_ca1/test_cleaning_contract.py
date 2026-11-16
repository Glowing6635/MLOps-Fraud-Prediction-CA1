"""Consumer contracts; actual #13/#20 adoption is tracked separately."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
from pandas.testing import assert_frame_equal

from mlops_fraud_prediction_ca1.cleaning import (
    BASE_CATEGORY_VALUES,
    CLEANING_POLICY_VERSION,
    clean_batch,
    clean_table,
    complete_key_mask,
    load_clean_base_data,
    summarize_batch,
)


class BatchContractTests(unittest.TestCase):
    def test_frozen_policy_covers_recorded_formats_without_semantic_aliases(self):
        self.assertEqual(CLEANING_POLICY_VERSION, "pbi12-v3")
        with self.assertRaises(TypeError):
            BASE_CATEGORY_VALUES["country_code"] = frozenset({"ZZ"})
        self.assertNotIn("on delivery", BASE_CATEGORY_VALUES["payment_method"])
        result = clean_table(
            pd.DataFrame(
                {
                    "country_code": ["th", "bd", "ZZ"],
                    "collect_type": ["pickup", "SELF-PICK-UP", "spaceship"],
                    "payment_method": [
                        "Cybersource",
                        "xendit directdebit",
                        "on Delivery",
                    ],
                }
            ),
            category_values=BASE_CATEGORY_VALUES,
        )
        self.assertEqual(result.cleaned.country_code.iloc[:2].tolist(), ["TH", "BD"])
        self.assertEqual(
            result.cleaned.collect_type.iloc[:2].tolist(), ["pickup", "self-pick-up"]
        )
        self.assertEqual(
            result.cleaned.payment_method.iloc[:2].tolist(),
            ["cybersource", "xendit directdebit"],
        )
        self.assertTrue(result.cleaned.iloc[2].isna().all())
        # Every frozen entry recognizes itself independently of batch contents.
        for column, values in BASE_CATEGORY_VALUES.items():
            for value in values:
                one = clean_table(
                    pd.DataFrame({column: [value]}),
                    category_values=BASE_CATEGORY_VALUES,
                )
                self.assertEqual(one.cleaned.iloc[0, 0], value)

    def test_missing_links_unknown_keys_and_future_batches_keep_all_rows(self):
        tables = {
            "customers": pd.DataFrame(columns=["country_code", "customer_id"]),
            "orders": pd.DataFrame(
                {
                    "country_code": ["my", "ZZ", "MY"],
                    "order_id": ["001", "1", "no-label"],
                    "order_date": ["2023-04-18", "2030/Jan/01", "bad"],
                    "refund_value": ["999999", "-5", "bad"],
                },
                index=[7, 7, 2],
            ),
            "labels": pd.DataFrame(
                {
                    "country_code": ["MY"],
                    "order_id": ["001"],
                    "customer_id": ["missing-customer"],
                    "is_fraud": ["1"],
                }
            ),
        }
        result = clean_batch(tables, batch_id="future-01", source="synthetic")
        self.assertEqual(
            (result.batch_id, result.source, result.policy_version),
            ("future-01", "synthetic", "pbi12-v3"),
        )
        for name, frame in tables.items():
            assert_frame_equal(result.tables[name].raw, frame)
            self.assertEqual(len(result.tables[name].cleaned), len(frame))
            self.assertEqual(
                result.tables[name].source_positions, tuple(range(len(frame)))
            )
        orders = result.tables["orders"]
        self.assertEqual(orders.cleaned.order_id.tolist(), ["001", "1", "no-label"])
        self.assertEqual(orders.cleaned.refund_value.iloc[:2].tolist(), [999999, -5])
        self.assertEqual(
            complete_key_mask(orders, ["country_code", "order_id"]).tolist(),
            [True, False, True],
        )
        self.assertEqual(orders.cleaned.order_date.iloc[1], pd.Timestamp("2030-01-01"))
        self.assertEqual(len(result.tables["labels"].cleaned), 1)

    def test_batch_metadata_and_key_contract_fail_closed(self):
        tables = {name: pd.DataFrame() for name in ("customers", "orders", "labels")}
        for metadata in (
            {"batch_id": "", "source": "fixture"},
            {"batch_id": "base", "source": " "},
        ):
            with self.assertRaises(ValueError):
                clean_batch(tables, **metadata)
        with self.assertRaises(ValueError):
            clean_batch({"orders": pd.DataFrame()}, batch_id="base", source="fixture")
        result = clean_table(
            pd.DataFrame(
                {"country_code": ["MY"] * 4, "order_id": ["001", "1", " ", None]}
            ),
            category_values=BASE_CATEGORY_VALUES,
        )
        self.assertEqual(
            complete_key_mask(result, ["country_code", "order_id"]).tolist(),
            [True, True, False, False],
        )
        for keys in ([], ["order_id", "order_id"]):
            with self.assertRaises(ValueError):
                complete_key_mask(result, keys)


class GuardedCleaningTests(unittest.TestCase):
    def test_real_reader_boundary_withholds_sentinels_and_preserves_literal_values(
        self,
    ):
        from mlops_fraud_prediction_ca1.leakage import select_feature_inputs

        with TemporaryDirectory(prefix="pbi12-contract-") as tmp:
            root = Path(tmp)
            pd.DataFrame(
                {
                    "country_code": ["MY", "MY", "MY"],
                    "customer_id": ["001", "1", "cv"],
                    "mobile_verified": ["yes", "TEST_SECRET", "False"],
                    "first_order_datetime": ["", "TEST_SECRET", "2023-03-01 01:02:03"],
                }
            ).to_csv(root / "customers_base.csv", index=False)
            pd.DataFrame(
                {
                    "country_code": ["my", "MY", "MY", "MY", "MY"],
                    "order_id": ["010", "10", "v", "unlabelled", "missing-customer"],
                    "order_date": [
                        "2023/Jan/01",
                        "2023-04-17",
                        "2023/03/27",
                        "2023-01-02",
                        "2023-01-03",
                    ],
                    "payment_method": [
                        "",
                        "TEST_SECRET",
                        "on Delivery",
                        "crypto",
                        "credit card",
                    ],
                    "collect_type": ["Delivery"] * 5,
                    "refund_value": ["999", "TEST_SECRET", "-2", "bad", "0"],
                    "order_value": ["1", "TEST_SECRET", "2", "3", "4"],
                }
            ).to_csv(root / "orders_base.csv", index=False)
            pd.DataFrame(
                {
                    "country_code": ["MY"] * 4,
                    "order_id": ["010", "10", "v", "missing-customer"],
                    "customer_id": ["001", "1", "cv", "absent"],
                    "is_fraud": ["0", "1", "1", "0"],
                }
            ).to_csv(root / "labels_base.csv", index=False)
            before = {p.name: p.read_bytes() for p in root.iterdir()}
            for purpose, expected in (
                ("analysis", ["010", "v", "unlabelled", "missing-customer"]),
                ("train", ["010", "unlabelled", "missing-customer"]),
                ("validation", ["v"]),
            ):
                with self.subTest(purpose=purpose):
                    batch = load_clean_base_data(root, purpose=purpose)
                    self.assertEqual(
                        batch.tables["orders"].raw.order_id.tolist(), expected
                    )
                    for result in batch.tables.values():
                        for attr in ("raw", "cleaned", "status", "audit"):
                            self.assertNotIn(
                                "TEST_SECRET", str(getattr(result, attr).to_dict())
                            )
                        self.assertEqual(len(result.raw), len(result.cleaned))
                    order = batch.tables["orders"]
                    if purpose != "validation":
                        self.assertEqual(
                            batch.tables["customers"].raw.customer_id.tolist(),
                            ["001", "cv"] if purpose == "analysis" else ["001"],
                        )
                        self.assertEqual(order.raw.payment_method.iloc[0], "")
                        self.assertTrue(pd.isna(order.cleaned.payment_method.iloc[0]))
                        self.assertEqual(order.cleaned.refund_value.iloc[0], 999)
                    features = select_feature_inputs(order.cleaned, source="orders")
                    self.assertFalse(
                        {
                            "refund_value",
                            "is_fraud",
                            "order_id",
                            "customer_id",
                            "batch_id",
                        }
                        & set(features.columns)
                    )
            batch = load_clean_base_data(root)
            report = summarize_batch(batch).set_index("table")
            self.assertEqual(report.loc["orders", "before_rows"], 4)
            self.assertEqual(report.loc["orders", "after_rows"], 4)
            self.assertEqual(report.loc["orders", "unknown_cells"], 4)
            self.assertEqual(report.loc["customers", "unknown_cells"], 2)
            self.assertTrue(report.rejected_rows.eq(0).all())
            self.assertEqual(before, {p.name: p.read_bytes() for p in root.iterdir()})
