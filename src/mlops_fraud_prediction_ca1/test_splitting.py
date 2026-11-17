"""Independent expected memberships and executed SQL parity at every edge."""

import sqlite3
import tempfile
import unittest
from datetime import UTC, date, datetime
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from pandas.testing import assert_frame_equal

from mlops_fraud_prediction_ca1.splitting import (
    load_policy,
    split_orders,
    sql_date_predicate,
)


class SplitTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="pbi6-final-config-")
        self.addCleanup(temporary.cleanup)
        self.final_config = Path(temporary.name) / "split.toml"
        self.final_config.write_text(
            (Path(__file__).resolve().parents[2] / "config/time_split.toml")
            .read_text()
            .replace(
                "final_evaluation_enabled = false", "final_evaluation_enabled = true"
            )
        )
        self.rows = pd.DataFrame(
            {
                "id": [
                    "early",
                    "start",
                    "slash",
                    "month",
                    "last_train",
                    "validation",
                    "last_val",
                    "test",
                    "last_test",
                    "end",
                    "future",
                    "bad",
                    "missing",
                    "native",
                    "datetime",
                    "tz",
                ],
                "order_date": [
                    "2022-11-11",
                    "2022-11-12",
                    "2023/01/01",
                    "2023/Feb/01",
                    "2023-03-26",
                    "2023-03-27",
                    "2023-04-16",
                    "2023-04-17",
                    "2023-05-10",
                    "2023-05-11",
                    "2025-01-01",
                    "2023-02-30",
                    None,
                    date(2023, 3, 1),
                    datetime(2023, 4, 17, 16),
                    datetime(2023, 1, 1, tzinfo=UTC),
                ],
            },
            index=[4] * 16,
        )
        self.expected = {
            "train": ["start", "slash", "month", "last_train", "native"],
            "validation": ["validation", "last_val"],
            "analysis": [
                "start",
                "slash",
                "month",
                "last_train",
                "validation",
                "last_val",
                "end",
                "future",
                "native",
            ],
            "final_evaluation": ["test", "last_test", "datetime"],
        }

    def test_edges_formats_indices_and_nonmutation(self):
        original = self.rows.copy(deep=True)
        for purpose, expected in self.expected.items():
            rows, audit = split_orders(
                self.rows,
                purpose=purpose,
                config_path=self.final_config
                if purpose == "final_evaluation"
                else None,
            )
            self.assertEqual(rows.id.tolist(), expected)
            self.assertEqual(audit["source_rows"], 16)
            self.assertEqual(audit["membership_counts"]["invalid_date"], 3)
            self.assertEqual(audit["returned_rows"] + audit["excluded_rows"], 16)
            self.assertFalse(
                any(isinstance(value, pd.DataFrame) for value in audit.values())
            )
        assert_frame_equal(self.rows, original)
        empty, audit = split_orders(self.rows.iloc[:0])
        self.assertTrue(empty.empty)
        self.assertEqual(audit["excluded_rows"], 0)

    def test_default_audit_records_resolved_cutoffs_for_each_purpose(self):
        expected = {
            "train_start": "2022-11-12",
            "validation_start": "2023-03-27",
            "test_start": "2023-04-17",
            "test_end_exclusive": "2023-05-11",
        }
        for purpose in self.expected:
            with self.subTest(purpose=purpose):
                _, audit = split_orders(
                    self.rows,
                    purpose=purpose,
                    config_path=self.final_config
                    if purpose == "final_evaluation"
                    else None,
                )
                self.assertEqual(audit["cutoffs"], expected)
                self.assertEqual(audit["purpose"], purpose)
                self.assertIs(audit["test_locked"], True)

    def test_custom_provenance_distinguishes_equal_counts_and_loads_once(self):
        rows = pd.DataFrame(
            {"id": ["old", "new"], "order_date": ["2023-03-26", "2023-03-27"]}
        )
        default_rows, default_audit = split_orders(rows, purpose="train")
        expected = {
            "train_start": "2023-03-27",
            "validation_start": "2023-03-28",
            "test_start": "2023-04-18",
            "test_end_exclusive": "2023-05-12",
        }
        with tempfile.TemporaryDirectory(prefix="pbi6-provenance-") as tmp:
            path = Path(tmp) / "split.toml"
            path.write_text(
                "\n".join(f'{key} = "{value}"' for key, value in expected.items())
                + "\ntest_locked = false\nfinal_evaluation_enabled = false\n"
            )
            with patch(
                "mlops_fraud_prediction_ca1.splitting.load_policy", wraps=load_policy
            ) as load:
                selected, audit = split_orders(rows, purpose="train", config_path=path)
                load.assert_called_once_with(path)
        self.assertEqual(default_rows.id.tolist(), ["old"])
        self.assertEqual(selected.id.tolist(), ["new"])
        self.assertEqual(default_audit["returned_rows"], audit["returned_rows"])
        self.assertNotEqual(default_audit["cutoffs"], audit["cutoffs"])
        self.assertEqual(audit["cutoffs"], expected)
        self.assertIs(audit["test_locked"], False)

    def test_shuffled_and_appended_batches(self):
        appended = pd.concat(
            [
                self.rows,
                pd.DataFrame(
                    {
                        "id": ["older", "newer"],
                        "order_date": ["2020-01-01", "2030-01-01"],
                    }
                ),
            ]
        ).sample(frac=1, random_state=9)
        for purpose, expected in self.expected.items():
            result, _ = split_orders(
                appended,
                purpose=purpose,
                config_path=self.final_config
                if purpose == "final_evaluation"
                else None,
            )
            self.assertEqual(set(result.id) - {"newer"}, set(expected))
            self.assertEqual("newer" in set(result.id), purpose == "analysis")

    def test_executed_sql_parity(self):
        # Canonical nullable DATE input, as required of future RDS adapters.
        rows = [
            ("early", "2022-11-11"),
            ("start", "2022-11-12"),
            ("last_train", "2023-03-26"),
            ("validation", "2023-03-27"),
            ("last_val", "2023-04-16"),
            ("test", "2023-04-17"),
            ("last_test", "2023-05-10"),
            ("end", "2023-05-11"),
            ("future", "2025-01-01"),
            ("missing", None),
        ]
        expected = {
            "train": {"start", "last_train"},
            "validation": {"validation", "last_val"},
            "analysis": {
                "start",
                "last_train",
                "validation",
                "last_val",
                "end",
                "future",
            },
            "final_evaluation": {"test", "last_test"},
        }
        with sqlite3.connect(":memory:") as db:
            db.execute("CREATE TABLE orders (id TEXT, order_date DATE)")
            db.executemany("INSERT INTO orders VALUES (?, ?)", rows)
            for purpose, ids in expected.items():
                clause, parameters = sql_date_predicate(
                    purpose=purpose,
                    config_path=self.final_config
                    if purpose == "final_evaluation"
                    else None,
                )
                result = {
                    row[0]
                    for row in db.execute(
                        "SELECT id FROM orders WHERE " + clause.replace("%s", "?"),
                        parameters,
                    )
                }
                self.assertEqual(result, ids)
                frame, _ = split_orders(
                    pd.DataFrame(rows, columns=["id", "order_date"]),
                    purpose=purpose,
                    config_path=self.final_config
                    if purpose == "final_evaluation"
                    else None,
                )
                self.assertEqual(set(frame.id), ids)
        clause, params = sql_date_predicate(purpose="train")
        self.assertIn("%s", clause)
        self.assertNotIn("2022-11-12", clause)
        self.assertEqual(params, ("2022-11-12", "2023-03-27"))

    def test_final_evaluation_requires_separate_setting_and_logs_access(self):
        with self.assertRaisesRegex(ValueError, "final_evaluation_enabled"):
            split_orders(self.rows, purpose="final_evaluation")
        with self.assertRaisesRegex(ValueError, "final_evaluation_enabled"):
            sql_date_predicate(purpose="final_evaluation")
        with self.assertLogs(
            "mlops_fraud_prediction_ca1.splitting", level="WARNING"
        ) as log:
            result, audit = split_orders(
                self.rows, purpose="final_evaluation", config_path=self.final_config
            )
        self.assertEqual(result.id.tolist(), self.expected["final_evaluation"])
        self.assertTrue(audit["final_evaluation_enabled"])
        self.assertTrue(audit["test_locked"])
        self.assertIn("2023-04-17", log.output[0])
        ordinary, _ = split_orders(self.rows, config_path=self.final_config)
        self.assertEqual(ordinary.id.tolist(), self.expected["analysis"])
        # An ordinary unlock must not implicitly enable final evaluation.
        with tempfile.TemporaryDirectory(prefix="pbi6-unlock-") as tmp:
            path = Path(tmp) / "split.toml"
            path.write_text(
                (Path(__file__).resolve().parents[2] / "config/time_split.toml")
                .read_text()
                .replace("test_locked = true", "test_locked = false")
            )
            with self.assertRaises(ValueError):
                split_orders(self.rows, purpose="final_evaluation", config_path=path)
            train, _ = split_orders(self.rows, purpose="train", config_path=path)
            self.assertEqual(train.id.tolist(), self.expected["train"])
            analysis, _ = split_orders(self.rows, config_path=path)
            self.assertEqual(
                set(analysis.id),
                set(self.expected["analysis"] + self.expected["final_evaluation"]),
            )
        self.assertFalse(load_policy().final_evaluation_enabled)

    def test_bad_configuration_and_purpose_fail_closed(self):
        text = (
            Path(__file__).resolve().parents[2] / "config/time_split.toml"
        ).read_text()
        bad = [
            text.replace("test_locked = true", 'test_locked = "false"'),
            text.replace("test_locked = true", ""),
            text.replace("2023-04-17", "2023-03-27"),
            text.replace("2023-04-17", "2023-02-30"),
            text.replace('"2023-04-17"', "2023-04-17"),
            text + "\nunknown = true\n",
            text.replace(
                "final_evaluation_enabled = false", 'final_evaluation_enabled = "true"'
            ),
            text.replace("final_evaluation_enabled = false", ""),
            text.replace("2023-04-17", "20230417"),
        ]
        with tempfile.TemporaryDirectory(prefix="pbi6-invalid-") as tmp:
            path = Path(tmp) / "split.toml"
            for value in bad:
                path.write_text(value)
                with self.assertRaises(ValueError):
                    split_orders(self.rows, config_path=path)
            with self.assertRaises(FileNotFoundError):
                load_policy(Path(tmp) / "absent")
        with self.assertRaises(ValueError):
            split_orders(self.rows, purpose="test")
        with self.assertRaises(ValueError):
            split_orders(self.rows, purpose="release_train")
