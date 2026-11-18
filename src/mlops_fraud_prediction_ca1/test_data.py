"""Verify the base-data loader's file checks and return contract."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd
from pandas.testing import assert_frame_equal

from mlops_fraud_prediction_ca1.data import load_base_data


class LoadBaseDataTests(unittest.TestCase):
    """Exercise loading with small, independent CSV fixtures."""

    def test_loads_in_documented_order_and_preserves_files(self) -> None:
        """Preserve every row, column, declared dtype, and source file."""
        with TemporaryDirectory() as temporary:
            directory: Path = Path(temporary)
            originals: dict[str, bytes] = {
                "customers_base.csv": (
                    b"customer_id,country_code\ncustomer-A,SG\ncustomer-B,MY\n"
                ),
                "orders_base.csv": (
                    b"order_id,order_value\norder-A,-2.0\norder-B,0.0\norder-C,12.5\n"
                ),
                "labels_base.csv": (
                    b"order_id,is_fraud\norder-A,1\norder-B,0\norder-C,0\n"
                ),
            }
            expected: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] = (
                pd.DataFrame(
                    {
                        "customer_id": ["customer-A", "customer-B"],
                        "country_code": ["SG", "MY"],
                    }
                ),
                pd.DataFrame(
                    {
                        "order_id": ["order-A", "order-B", "order-C"],
                        "order_value": [-2.0, 0.0, 12.5],
                    }
                ),
                pd.DataFrame(
                    {
                        "order_id": ["order-A", "order-B", "order-C"],
                        "is_fraud": [1, 0, 0],
                    }
                ),
            )
            for name, content in originals.items():
                (directory / name).write_bytes(content)

            with self.assertLogs(
                "mlops_fraud_prediction_ca1.data", level="INFO"
            ) as captured:
                actual: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] = (
                    load_base_data(str(directory))
                )

            for name, frame, expected_frame in zip(
                originals, actual, expected, strict=True
            ):
                with self.subTest(dataset=name):
                    expected_frame = expected_frame.astype(
                        {
                            column: "string"
                            for column in expected_frame
                            if column
                            in {"customer_id", "order_id", "country_code", "is_fraud"}
                        }
                    )
                    assert_frame_equal(frame, expected_frame, check_exact=True)
                    dataset_logs: str = "\n".join(
                        record.getMessage()
                        for record in captured.records
                        if name in record.getMessage()
                    )
                    self.assertIn(name, dataset_logs)
                    self.assertRegex(
                        dataset_logs, rf"\b{len(frame)}\b.*\b{len(frame.columns)}\b"
                    )
            for name, content in originals.items():
                self.assertEqual((directory / name).read_bytes(), content)

    def test_digit_only_ids_and_flag_spellings_survive_loading(self) -> None:
        """Keep leading zeros and mixed flag tokens across every input file."""
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "customers_base.csv").write_text(
                "customer_id,mobile_verified,first_order_datetime\n"
                "0001,True,2023-01-01 00:00:00\n"
                "0002,False,\n0003,yes,2023-01-02 00:00:00\n"
            )
            (directory / "orders_base.csv").write_text(
                "order_id,order_date,order_value\n0007,2023/01/02,12.5\n"
            )
            (directory / "labels_base.csv").write_text(
                "order_id,customer_id,is_fraud\n0007,0001,1\n"
            )
            customers, orders, labels = load_base_data(directory)
            self.assertEqual(customers.customer_id.tolist(), ["0001", "0002", "0003"])
            self.assertEqual(
                customers.mobile_verified.tolist(), ["True", "False", "yes"]
            )
            self.assertTrue(pd.isna(customers.first_order_datetime.iloc[1]))
            self.assertEqual(orders.order_id.item(), "0007")
            self.assertEqual(labels.order_id.item(), "0007")
            self.assertEqual(labels.customer_id.item(), "0001")
            self.assertEqual(labels.is_fraud.item(), "1")
            self.assertEqual(orders.order_value.item(), 12.5)

    def test_reports_all_missing_files_before_reading(self) -> None:
        """Report every absent file without starting a partial load."""
        with TemporaryDirectory() as temporary:
            directory: Path = Path(temporary)
            (directory / "customers_base.csv").write_text("customer_id\nA\n")
            with patch("mlops_fraud_prediction_ca1.data.pd.read_csv") as read_csv:
                with self.assertRaises(FileNotFoundError) as raised:
                    load_base_data(directory)
                read_csv.assert_not_called()
            self.assertIn("orders_base.csv", str(raised.exception))
            self.assertIn("labels_base.csv", str(raised.exception))

    def test_propagates_invalid_csv_errors(self) -> None:
        """Keep parse failures visible to the notebook caller."""
        with TemporaryDirectory() as temporary:
            directory: Path = Path(temporary)
            for name in ("customers", "orders", "labels"):
                (directory / f"{name}_base.csv").write_text("")
            with self.assertRaises(pd.errors.EmptyDataError):
                load_base_data(directory)
