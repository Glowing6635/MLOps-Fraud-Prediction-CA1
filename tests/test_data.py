"""Verify the base-data loader's file checks and return contract."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd
from pandas.testing import assert_frame_equal

from mlops_fraud_prediction_ca1.data import load_base_data


class LoadBaseDataTests(unittest.TestCase):
    """Exercise loading with small, independent CSV fixtures."""

    def test_loads_in_documented_order_and_preserves_files(self) -> None:
        """Preserve every row, column, inferred dtype, and source file."""
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
                pd.DataFrame({
                    "customer_id": ["customer-A", "customer-B"],
                    "country_code": ["SG", "MY"],
                }),
                pd.DataFrame({
                    "order_id": ["order-A", "order-B", "order-C"],
                    "order_value": [-2.0, 0.0, 12.5],
                }),
                pd.DataFrame({
                    "order_id": ["order-A", "order-B", "order-C"],
                    "is_fraud": [1, 0, 0],
                }),
            )
            for name, content in originals.items():
                (directory / name).write_bytes(content)

            with self.assertLogs("mlops_fraud_prediction_ca1.data", level="INFO") as captured:
                actual: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] = load_base_data(
                    str(directory)
                )

            for name, frame, expected_frame in zip(originals, actual, expected, strict=True):
                with self.subTest(dataset=name):
                    assert_frame_equal(frame, expected_frame, check_exact=True)
                    dataset_logs: str = "\n".join(
                        record.getMessage() for record in captured.records
                        if name in record.getMessage()
                    )
                    self.assertIn(name, dataset_logs)
                    self.assertRegex(dataset_logs, rf"\b{len(frame)}\b.*\b{len(frame.columns)}\b")
            for name, content in originals.items():
                self.assertEqual((directory / name).read_bytes(), content)

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
