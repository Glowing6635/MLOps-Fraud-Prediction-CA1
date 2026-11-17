"""Consumer contracts; actual #13/#20 adoption is tracked separately."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
from pandas.testing import assert_frame_equal

from mlops_fraud_prediction_ca1.cleaning import (
    BASE_CATEGORY_VALUES,
    CLEANING_POLICY_VERSION,
    clean_frame,
)
from mlops_fraud_prediction_ca1.data import load_base_data


class CleaningContractTests(unittest.TestCase):
    def test_frozen_policy_covers_recorded_formats_without_semantic_aliases(self):
        self.assertEqual(CLEANING_POLICY_VERSION, "pbi12-v3")
        with self.assertRaises(TypeError):
            BASE_CATEGORY_VALUES["country_code"] = frozenset({"ZZ"})
        self.assertNotIn("on delivery", BASE_CATEGORY_VALUES["payment_method"])
        result = clean_frame(
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
        )
        self.assertEqual(result.country_code.iloc[:2].tolist(), ["TH", "BD"])
        self.assertEqual(
            result.collect_type.iloc[:2].tolist(), ["pickup", "self-pick-up"]
        )
        self.assertEqual(
            result.payment_method.iloc[:2].tolist(),
            ["cybersource", "xendit directdebit"],
        )
        self.assertTrue(
            result.loc[:, ["country_code", "collect_type", "payment_method"]]
            .iloc[2]
            .isna()
            .all()
        )
        # Every frozen entry recognizes itself independently of batch contents.
        for column, values in BASE_CATEGORY_VALUES.items():
            for value in values:
                one = clean_frame(
                    pd.DataFrame({column: [value]}),
                )
                self.assertEqual(one.iloc[0, 0], value)

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
        result = {name: clean_frame(frame) for name, frame in tables.items()}
        for name, frame in tables.items():
            assert_frame_equal(
                result[name].filter(regex="^raw_").rename(columns=lambda c: c[4:]),
                frame,
            )
            self.assertEqual(len(result[name]), len(frame))
        orders = result["orders"]
        self.assertEqual(orders.order_id.tolist(), ["001", "1", "no-label"])
        self.assertEqual(orders.refund_value.iloc[:2].tolist(), [999999, -5])
        self.assertTrue(pd.isna(orders.country_code.iloc[1]))
        self.assertEqual(orders.order_date.iloc[1], pd.Timestamp("2030-01-01"))
        self.assertEqual(len(result["labels"]), 1)


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
                    frames = load_base_data(root, purpose=purpose, raw_strings=True)
                    tables = dict(
                        zip(("customers", "orders", "labels"), frames, strict=True)
                    )
                    cleaned = {
                        name: clean_frame(frame) for name, frame in tables.items()
                    }
                    self.assertEqual(tables["orders"].order_id.tolist(), expected)
                    for name, frame in tables.items():
                        self.assertNotIn("TEST_SECRET", str(frame.to_dict()))
                        self.assertNotIn("TEST_SECRET", str(cleaned[name].to_dict()))
                        self.assertEqual(len(frame), len(cleaned[name]))
                        self.assertEqual(cleaned[name].attrs, frame.attrs)
                        self.assertTrue(frame.attrs["split_audit"]["test_locked"])
                        assert_frame_equal(
                            cleaned[name]
                            .filter(regex="^raw_")
                            .rename(columns=lambda c: c[4:]),
                            frame,
                        )
                    order = cleaned["orders"]
                    if purpose != "validation":
                        self.assertEqual(
                            tables["customers"].customer_id.tolist(),
                            ["001", "cv"] if purpose == "analysis" else ["001"],
                        )
                        self.assertEqual(order.raw_payment_method.iloc[0], "")
                        self.assertTrue(pd.isna(order.payment_method.iloc[0]))
                        self.assertEqual(order.refund_value.iloc[0], 999)
                    features = select_feature_inputs(order, source="orders")
                    self.assertFalse(
                        {
                            "refund_value",
                            "is_fraud",
                            "order_id",
                            "customer_id",
                        }
                        & set(features.columns)
                    )
            self.assertEqual(before, {p.name: p.read_bytes() for p in root.iterdir()})
