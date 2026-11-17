"""Synthetic contracts for row-preserving cleaning."""

import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from mlops_fraud_prediction_ca1.cleaning import clean_frame, clean_table

POLICY = {
    "country_code": {"MY", "PK", "PH"},
    "collect_type": {"delivery", "self-pick-up"},
    "payment_method": {"credit card", "online banking", "payment on delivery"},
}


class CleaningTests(unittest.TestCase):
    def clean(self, frame):
        return clean_table(frame, category_values=POLICY)

    def test_category_case_whitespace_and_unseen_kept_unknown(self):
        frame = pd.DataFrame(
            {
                "country_code": ["my", " PK ", "ph", "ZZ"],
                "collect_type": ["Delivery", " delivery ", "self-pick-up", "spaceship"],
                "payment_method": [
                    "Online Banking",
                    "payment on delivery",
                    "credit card",
                    "crypto",
                ],
            }
        )
        result = self.clean(frame)
        self.assertEqual(
            result.cleaned.country_code.iloc[:3].tolist(), ["MY", "PK", "PH"]
        )
        self.assertEqual(
            result.cleaned.collect_type.iloc[:3].tolist(),
            ["delivery", "delivery", "self-pick-up"],
        )
        self.assertEqual(
            result.cleaned.payment_method.iloc[:3].tolist(),
            ["online banking", "payment on delivery", "credit card"],
        )
        self.assertTrue(result.cleaned.iloc[3].isna().all())
        self.assertTrue(result.status.iloc[3].eq("unknown_unseen").all())
        self.assertEqual(len(result.cleaned), 4)

    def test_unapproved_semantics_and_blanks(self):
        frame = pd.DataFrame(
            {
                "mobile_verified": ["yes", True, " FALSE ", None],
                "payment_method": ["on Delivery", "", "   ", None],
                "first_order_datetime": ["", "   ", None, "2023-01-01 12:13:14"],
            }
        )
        result = self.clean(frame)
        self.assertTrue(pd.isna(result.cleaned.mobile_verified.iloc[0]))
        self.assertEqual(
            result.cleaned.mobile_verified.iloc[1:3].tolist(), [True, False]
        )
        self.assertTrue(result.cleaned.payment_method.isna().all())
        self.assertEqual(
            result.status.payment_method.tolist(),
            ["unknown_unseen", "unknown_blank", "unknown_blank", "unknown_missing"],
        )
        self.assertTrue(result.cleaned.first_order_datetime.iloc[:3].isna().all())
        self.assertEqual(
            result.cleaned.first_order_datetime.iloc[3],
            pd.Timestamp("2023-01-01 12:13:14"),
        )
        assert_frame_equal(result.raw, frame)

    def test_mobile_verification_nullable_boolean_and_originals(self):
        frame = pd.DataFrame(
            {
                "mobile_verified": [
                    "true",
                    " FALSE ",
                    True,
                    False,
                    "yes",
                    "",
                    " ",
                    None,
                    pd.NA,
                    1,
                ]
            },
            index=[7] * 10,
        )
        original = frame.copy(deep=True)
        result = clean_frame(frame)
        self.assertEqual(str(result.mobile_verified.dtype), "boolean")
        self.assertEqual(
            result.mobile_verified.iloc[:4].tolist(), [True, False, True, False]
        )
        self.assertTrue(result.mobile_verified.iloc[4:].isna().all())
        self.assertEqual(result.index.tolist(), frame.index.tolist())
        assert_frame_equal(
            result[["raw_mobile_verified"]].rename(
                columns={"raw_mobile_verified": "mobile_verified"}
            ),
            original,
        )
        assert_frame_equal(frame, original)

    def test_all_evidenced_date_formats_and_invalid_dates(self):
        frame = pd.DataFrame(
            {
                "order_date": [
                    "2024/02/29",
                    "2024-02-29",
                    "2024/Feb/29",
                    "2023/Feb/29",
                    "2024/Foo/01",
                    "29/02/2024",
                    None,
                    "",
                ]
            }
        )
        result = self.clean(frame)
        self.assertTrue(
            result.cleaned.order_date.iloc[:3].eq(pd.Timestamp("2024-02-29")).all()
        )
        self.assertTrue(result.cleaned.order_date.iloc[3:].isna().all())
        self.assertEqual(
            result.status.order_date.tolist(),
            ["normalized"] * 3
            + [
                "unknown_invalid",
                "unknown_invalid",
                "unknown_unrecognized",
                "unknown_missing",
                "unknown_blank",
            ],
        )
        customer = self.clean(
            pd.DataFrame({"first_order_datetime": ["2023-02-30 10:00:00"]})
        )
        self.assertEqual(customer.status.iloc[0, 0], "unknown_invalid")

    def test_case_insensitive_months_and_parser_limitations_are_explicit(self):
        result = self.clean(
            pd.DataFrame(
                {
                    "order_date": [
                        "2024/feb/29",
                        " 2024-02-29 ",
                        pd.Timestamp("2024-02-29"),
                    ]
                }
            )
        )
        self.assertEqual(result.cleaned.order_date.iloc[0], pd.Timestamp("2024-02-29"))
        self.assertTrue(result.cleaned.order_date.iloc[1:].isna().all())
        self.assertEqual(
            result.status.order_date.tolist(),
            ["normalized", "unknown_unrecognized", "unknown_unrecognized"],
        )

    def test_numeric_unknowns_and_finite_values_no_clamping(self):
        frame = pd.DataFrame(
            {
                "order_value": ["2.5", "bad", "", None, "inf", "-7", True],
                "num_items_ordered": [
                    "1",
                    "2",
                    None,
                    " ",
                    float("-inf"),
                    "1.5",
                    np.bool_(True),
                ],
                "refund_value": ["999999999", "-5", "0", None, "inf", "bad", 1],
            }
        )
        result = self.clean(frame)
        self.assertEqual(result.cleaned.order_value.iloc[0], 2.5)
        self.assertEqual(result.cleaned.order_value.iloc[5], -7)
        self.assertEqual(result.cleaned.refund_value.iloc[0], 999999999)
        self.assertEqual(result.cleaned.refund_value.iloc[1], -5)
        self.assertEqual(result.cleaned.num_items_ordered.iloc[5], 1.5)
        self.assertTrue(result.cleaned.order_value.iloc[[1, 2, 3, 4, 6]].isna().all())
        self.assertTrue(pd.isna(result.cleaned.num_items_ordered.iloc[6]))
        self.assertEqual(len(result.cleaned), len(frame))

    def test_large_numbers_are_flagged_without_rounding_originals(self):
        for values in (["9007199254740993"], ["9007199254740993", "bad", None, "1.5"]):
            frame = pd.DataFrame({"order_value": values})
            result = self.clean(frame)
            self.assertTrue(pd.isna(result.cleaned.order_value.iloc[0]))
            self.assertEqual(result.raw.order_value.iloc[0], "9007199254740993")
            self.assertEqual(result.status.order_value.iloc[0], "unknown_precision")
            self.assertEqual(str(result.cleaned.order_value.dtype), "Float64")

    def test_simple_dataframe_api_preserves_originals_and_numeric_labels(self):
        frame = pd.DataFrame(
            {
                "order_id": ["001", "1", "x"],
                "order_value": [" 1.5 ", "-2", "bad"],
                "is_fraud": ["1", "0", "yes"],
            },
            index=[7, 7, 2],
        )
        result = clean_frame(frame)
        self.assertEqual(result.index.tolist(), frame.index.tolist())
        self.assertEqual(result.order_value.iloc[:2].tolist(), [1.5, -2])
        self.assertEqual(result.is_fraud.iloc[:2].tolist(), [1, 0])
        self.assertTrue(pd.isna(result.is_fraud.iloc[2]))
        self.assertEqual(str(result.is_fraud.dtype), "Int8")
        self.assertTrue(pd.api.types.is_numeric_dtype(result.order_value))
        original = result.filter(regex="^raw_").rename(columns=lambda c: c[4:])
        assert_frame_equal(original, frame)
        with self.assertRaises(ValueError):
            clean_frame(pd.DataFrame({"order_value": [1], "raw_order_value": [2]}))

    def test_empty_frames_and_all_empty_columns(self):
        for frame in (
            pd.DataFrame(),
            pd.DataFrame(columns=["order_date", "payment_method", "order_value"]),
            pd.DataFrame(
                {
                    "order_date": [None, None],
                    "payment_method": [None, ""],
                    "order_value": [None, ""],
                }
            ),
        ):
            with self.subTest(shape=frame.shape):
                result = self.clean(frame)
                self.assertEqual(result.cleaned.shape, frame.shape)
                self.assertTrue(result.cleaned.isna().all(axis=None))

    def test_no_input_mutation_raw_copy_duplicate_index_and_identity(self):
        frame = pd.DataFrame(
            {
                "country_code": ["my", "pk"],
                "order_id": ["same", "same"],
                "order_date": ["2023/Jan/01", "bad"],
                "is_fraud": ["1", "0"],
                "other": ["a", "b"],
            },
            index=[7, 7],
        )
        original = frame.copy(deep=True)
        result = self.clean(frame)
        assert_frame_equal(frame, original)
        assert_frame_equal(result.raw, original)
        self.assertEqual(result.cleaned.index.tolist(), [7, 7])
        self.assertEqual(result.cleaned.order_id.tolist(), ["same", "same"])
        self.assertEqual(result.cleaned.is_fraud.tolist(), [1, 0])
        self.assertTrue(result.status.other.eq("unchanged").all())
        result.raw.iloc[0, 0] = "changed"
        result.cleaned.iloc[1, 0] = "changed"
        assert_frame_equal(frame, original)

    def test_determinism_batch_independence_and_label_independence(self):
        frame = pd.DataFrame(
            {
                "country_code": ["my", "ph", "ZZ"],
                "payment_method": ["credit card", "", "new"],
                "order_date": ["2023/Jan/01", "2023-01-02", "bad"],
                "order_value": ["1", "99999999", "bad"],
                "is_fraud": [0, 1, 1],
            },
            index=[9, 9, 2],
        )
        first, second = self.clean(frame), self.clean(frame)
        for attr in ("raw", "cleaned", "status"):
            assert_frame_equal(getattr(first, attr), getattr(second, attr))
        for position in range(len(frame)):
            single = self.clean(frame.iloc[[position]])
            assert_frame_equal(
                first.cleaned.iloc[[position]], single.cleaned, check_dtype=False
            )
            assert_frame_equal(first.status.iloc[[position]], single.status)
        shuffled = self.clean(frame.iloc[[2, 0, 1]])
        assert_frame_equal(first.cleaned.iloc[[2, 0, 1]], shuffled.cleaned)
        altered = self.clean(frame.assign(is_fraud=[1, 0, 0]))
        assert_frame_equal(
            first.cleaned.drop(columns="is_fraud"),
            altered.cleaned.drop(columns="is_fraud"),
        )
        assert_frame_equal(first.status, altered.status)

    def test_configuration_is_explicit_fixed_and_validated(self):
        with self.assertRaises(ValueError):
            clean_table(pd.DataFrame({"payment_method": ["new"]}), category_values={})
        for policy in (
            {"payment_method": {"Credit Card"}},
            {"payment_method": {"on delivery"}},
            {"mobile_verified": {"yes"}},
            {"payment_method": {""}},
        ):
            with self.subTest(policy=policy), self.assertRaises(ValueError):
                clean_table(pd.DataFrame(), category_values=policy)
        with self.assertRaises(TypeError):
            clean_table(pd.DataFrame(), category_values={"country_code": "MY"})
        with self.assertRaises(ValueError):
            self.clean(pd.DataFrame([[1, 2]], columns=["a", "a"]))
        result = clean_table(
            pd.DataFrame({"payment_method": ["credit card"]}),
            category_values={"payment_method": set()},
        )
        self.assertTrue(pd.isna(result.cleaned.iloc[0, 0]))


if __name__ == "__main__":
    unittest.main()
