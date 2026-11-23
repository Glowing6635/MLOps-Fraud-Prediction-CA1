"""Verify the synthetic rehearsal batches: columns, joins and distinct cases."""

import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from mlops_fraud_prediction_ca1 import rehearsal
from mlops_fraud_prediction_ca1.cleaning import BASE_CATEGORY_VALUES

BATCH_IDS = (*rehearsal.VALID_BATCHES, rehearsal.EDGE_BATCH)


def _read(directory: Path, batch_id: str) -> dict[str, pd.DataFrame]:
    return {
        name: pd.read_csv(
            directory / batch_id / f"{name}_{batch_id}.csv",
            dtype=str,
            keep_default_na=False,
        )
        for name in rehearsal.TABLE_COLUMNS
    }


class RehearsalBatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._temporary = TemporaryDirectory()
        cls.out = Path(cls._temporary.name)
        result = subprocess.run(
            [sys.executable, "-m", "mlops_fraud_prediction_ca1.rehearsal"]
            + ["--out", str(cls.out)],
            capture_output=True,
            text=True,
            check=True,
        )
        cls.stdout = result.stdout
        cls.frames = {batch: _read(cls.out, batch) for batch in BATCH_IDS}

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temporary.cleanup()

    def test_command_writes_three_suffixed_files_per_batch(self) -> None:
        for batch in BATCH_IDS:
            names = sorted(p.name for p in (self.out / batch).iterdir())
            self.assertEqual(
                names,
                sorted(f"{t}_{batch}.csv" for t in rehearsal.TABLE_COLUMNS),
            )
            self.assertIn(batch, self.stdout)

    def test_exact_columns(self) -> None:
        for batch, tables in self.frames.items():
            for name, frame in tables.items():
                self.assertEqual(
                    list(frame.columns), list(rehearsal.TABLE_COLUMNS[name]), batch
                )

    def test_valid_batches_join_completely_and_use_known_categories(self) -> None:
        for batch in rehearsal.VALID_BATCHES:
            t = self.frames[batch]
            for name in ("customers", "orders", "labels"):
                key = "customer_id" if name == "customers" else "order_id"
                self.assertTrue(t[name][key].is_unique, (batch, name))
            order_keys = set(
                zip(t["orders"].country_code, t["orders"].order_id, strict=True)
            )
            label_keys = set(
                zip(t["labels"].country_code, t["labels"].order_id, strict=True)
            )
            self.assertEqual(order_keys, label_keys)
            customers = set(
                zip(
                    t["customers"].country_code, t["customers"].customer_id, strict=True
                )
            )
            linked = set(
                zip(t["labels"].country_code, t["labels"].customer_id, strict=True)
            )
            self.assertTrue(linked <= customers)
            self.assertTrue(
                t["orders"]
                .payment_method.isin(BASE_CATEGORY_VALUES["payment_method"])
                .all()
            )
            self.assertTrue(t["labels"].is_fraud.isin({"0", "1"}).all())
            self.assertTrue(t["orders"].order_id.str.startswith("SYN-").all())

    def test_valid_batches_are_distinct(self) -> None:
        a, b = (self.frames[name] for name in rehearsal.VALID_BATCHES)
        for name in rehearsal.TABLE_COLUMNS:
            self.assertFalse(a[name].equals(b[name]), name)
        for key in ("order_id",):
            self.assertTrue(set(a["orders"][key]).isdisjoint(b["orders"][key]))
        self.assertTrue(
            set(a["customers"].customer_id).isdisjoint(b["customers"].customer_id)
        )

    def test_generation_is_deterministic(self) -> None:
        first = rehearsal.build_batches()
        second = rehearsal.build_batches()
        for x, y in zip(first, second, strict=True):
            self.assertEqual(x.tables(), y.tables())

    def test_edge_batch_has_novel_category_on_otherwise_valid_row(self) -> None:
        t = self.frames[rehearsal.EDGE_BATCH]
        novel = t["orders"][
            t["orders"].payment_method == rehearsal.NOVEL_PAYMENT_METHOD
        ]
        self.assertEqual(len(novel), 1)
        order_id = novel.order_id.iloc[0]
        self.assertNotIn(
            rehearsal.NOVEL_PAYMENT_METHOD, BASE_CATEGORY_VALUES["payment_method"]
        )
        label = t["labels"][t["labels"].order_id == order_id]
        self.assertEqual(len(label), 1)
        self.assertIn(label.customer_id.iloc[0], set(t["customers"].customer_id))

    def test_edge_batch_orphans_are_separately_identifiable(self) -> None:
        batch = rehearsal.build_edge_batch()
        t = self.frames[rehearsal.EDGE_BATCH]
        orders, customers = set(t["orders"].order_id), set(t["customers"].customer_id)
        self.assertEqual(len(batch.missing_order_label_ids), 2)
        for order_id in batch.missing_order_label_ids:
            self.assertNotIn(order_id, orders)
            self.assertIn(order_id, set(t["labels"].order_id))
        for order_id in batch.missing_customer_label_ids:
            row = t["labels"][t["labels"].order_id == order_id].iloc[0]
            self.assertIn(order_id, orders)
            self.assertNotIn(row.customer_id, customers)
        flagged = {
            *batch.missing_order_label_ids,
            *batch.missing_customer_label_ids,
        }
        self.assertTrue(flagged.isdisjoint(batch.novel_category_order_ids))
        unflagged = t["labels"][~t["labels"].order_id.isin(flagged)]
        self.assertTrue(unflagged.order_id.isin(orders).all())
        self.assertTrue(unflagged.customer_id.isin(customers).all())
        self.assertLess(len(t["orders"]), 20)


if __name__ == "__main__":
    unittest.main()
