"""Prove protected payloads at the public CSV boundary."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from mlops_fraud_prediction_ca1.data import load_base_data, load_category_data


class LoadBaseDataTests(unittest.TestCase):
    def fixture(self, root):
        pd.DataFrame(
            {
                "country_code": ["MY", "MY", "SG", "MY"],
                "customer_id": ["safe", "secret", "collision", "orphan"],
                "mobile_verified": ["True", "SECRET", "True", "False"],
            }
        ).to_csv(root / "customers_base.csv", index=False)
        pd.DataFrame(
            {
                "country_code": ["my", "MY", "SG", "MY", "MY", "MY"],
                "order_id": ["a", "secret", "a", "bad", "cross", "cross"],
                "order_date": [
                    "2023-01-01",
                    "2023-04-17",
                    "2023-04-18",
                    "invalid",
                    "2023-01-01",
                    "2023-04-19",
                ],
                "collect_type": ["Delivery"] * 6,
                "payment_method": [""] * 6,
            }
        ).to_csv(root / "orders_base.csv", index=False)
        pd.DataFrame(
            {
                "country_code": [" MY ", "MY", "SG", "MY", "MY", "MY"],
                "order_id": ["a", "secret", "a", "bad", "orphan", "cross"],
                "customer_id": [
                    "safe",
                    "secret",
                    "collision",
                    "secret",
                    "orphan",
                    "secret",
                ],
                "is_fraud": [0, 1, 1, 1, 1, 1],
            }
        ).to_csv(root / "labels_base.csv", index=False)

    def test_guarded_payloads_preserve_sources_and_values(self):
        with TemporaryDirectory(prefix="pbi6-loader-") as tmp:
            root = Path(tmp)
            self.fixture(root)
            before = {p.name: p.read_bytes() for p in root.iterdir()}
            customers, orders, labels = load_base_data(root)
            self.assertEqual(orders.order_id.tolist(), ["a", "cross"])
            self.assertEqual(labels.order_id.tolist(), ["a"])
            self.assertEqual(customers.customer_id.tolist(), ["safe"])
            self.assertEqual(labels.country_code.tolist(), [" MY "])
            self.assertEqual(orders.country_code.tolist(), ["my", "MY"])
            self.assertEqual(
                orders.attrs["split_audit"]["membership_counts"]["test"], 3
            )
            for frame in (customers, orders, labels):
                self.assertNotIn("SECRET", str(frame.to_dict()))
                self.assertEqual(set(frame.attrs), {"split_audit"})
            raw = load_category_data(root)
            self.assertEqual(raw["orders"].payment_method.tolist(), ["", ""])
            self.assertEqual(raw["labels"].is_fraud.tolist(), ["0"])
            self.assertEqual(raw["customers"].mobile_verified.tolist(), ["True"])
            self.assertEqual(before, {p.name: p.read_bytes() for p in root.iterdir()})

    def test_resolved_policy_in_all_loader_and_category_attrs(self):
        expected_keys = {
            "purpose",
            "test_locked",
            "final_evaluation_enabled",
            "cutoffs",
            "source_rows",
            "returned_rows",
            "excluded_rows",
            "membership_counts",
        }
        default = {
            "train_start": "2022-11-12",
            "validation_start": "2023-03-27",
            "test_start": "2023-04-17",
            "test_end_exclusive": "2023-05-11",
        }
        custom = {
            "train_start": "2023-01-01",
            "validation_start": "2023-03-28",
            "test_start": "2023-04-18",
            "test_end_exclusive": "2023-05-12",
        }
        with TemporaryDirectory(prefix="pbi6-loader-provenance-") as tmp:
            root = Path(tmp)
            self.fixture(root)
            config = root / "split.toml"
            config.write_text(
                "\n".join(f'{key} = "{value}"' for key, value in custom.items())
                + "\ntest_locked = true\nfinal_evaluation_enabled = false\n"
            )
            for path, expected in ((None, default), (config, custom)):
                with self.subTest(config=path):
                    frames = load_base_data(root, config_path=path)
                    categories = load_category_data(root, config_path=path)
                    audits = [
                        frame.attrs["split_audit"]
                        for frame in (*frames, *categories.values())
                    ]
                    for audit in audits:
                        self.assertEqual(set(audit), expected_keys)
                        self.assertEqual(audit, audits[0])
                        self.assertEqual(audit["cutoffs"], expected)
                        self.assertEqual(audit["purpose"], "analysis")
                        self.assertIs(audit["test_locked"], True)
                        self.assertNotIn("SECRET", str(audit))
                    self.assertEqual(
                        set(audits[0]["membership_counts"]),
                        {
                            "train",
                            "validation",
                            "test",
                            "before_start",
                            "after_end",
                            "invalid_date",
                        },
                    )

    def test_numeric_looking_ids_cannot_expose_test_only_customers(self):
        with TemporaryDirectory(prefix="pbi6-opaque-ids-") as tmp:
            root = Path(tmp)
            pd.DataFrame(
                {
                    "country_code": ["MY", "MY"],
                    "customer_id": ["001", "1"],
                    "mobile_verified": ["True", "TEST_SECRET"],
                }
            ).to_csv(root / "customers_base.csv", index=False)
            pd.DataFrame(
                {
                    "country_code": ["MY", "MY"],
                    "order_id": ["train", "test"],
                    "order_date": ["2023-01-01", "2023-04-17"],
                    "collect_type": ["Delivery", "Delivery"],
                    "payment_method": ["Card", "TEST_SECRET"],
                }
            ).to_csv(root / "orders_base.csv", index=False)
            pd.DataFrame(
                {
                    "country_code": ["MY", "MY"],
                    "order_id": ["train", "test"],
                    "customer_id": ["001", "1"],
                    "is_fraud": [0, 1],
                }
            ).to_csv(root / "labels_base.csv", index=False)
            # First isolate customer leakage, then also exercise order-key
            # collisions. Neither ID type may be normalized through numbers.
            for order_ids in (["train", "test"], ["010", "10"]):
                for name in ("orders", "labels"):
                    path = root / f"{name}_base.csv"
                    frame = pd.read_csv(path, dtype="string", keep_default_na=False)
                    frame["order_id"] = order_ids
                    frame.to_csv(path, index=False)
                before = {p.name: p.read_bytes() for p in root.iterdir()}
                for raw_strings in (False, True):
                    with self.subTest(order_ids=order_ids, raw_strings=raw_strings):
                        customers, orders, labels = load_base_data(
                            root, raw_strings=raw_strings
                        )
                        self.assertEqual(orders.order_id.tolist(), [order_ids[0]])
                        self.assertEqual(labels.order_id.tolist(), [order_ids[0]])
                        self.assertEqual(labels.customer_id.tolist(), ["001"])
                        self.assertEqual(customers.customer_id.tolist(), ["001"])
                        self.assertNotIn("TEST_SECRET", str(customers.to_dict()))
                categories = load_category_data(root)
                self.assertEqual(
                    categories["customers"].mobile_verified.tolist(), ["True"]
                )
                self.assertNotIn("TEST_SECRET", str(categories))
                self.assertEqual(
                    before, {p.name: p.read_bytes() for p in root.iterdir()}
                )

    def test_explicit_final_read_does_not_unlock_ordinary_reads(self):
        with TemporaryDirectory(prefix="pbi6-final-") as tmp:
            root = Path(tmp)
            self.fixture(root)
            with self.assertRaises(ValueError):
                load_base_data(root, purpose="final_evaluation")
            config = root / "split.toml"
            config.write_text(
                (Path(__file__).resolve().parents[2] / "config/time_split.toml")
                .read_text()
                .replace(
                    "final_evaluation_enabled = false",
                    "final_evaluation_enabled = true",
                )
            )
            with self.assertLogs(
                "mlops_fraud_prediction_ca1.splitting", level="WARNING"
            ):
                final = load_base_data(
                    root, purpose="final_evaluation", config_path=config
                )
            self.assertEqual(final[1].order_id.tolist(), ["secret", "a", "cross"])
            self.assertEqual(final[2].order_id.tolist(), ["secret", "a"])
            ordinary = load_base_data(root)
            self.assertEqual(ordinary[2].order_id.tolist(), ["a"])

    def test_blank_and_missing_keys_cannot_link_related_rows(self):
        with TemporaryDirectory(prefix="pbi6-null-keys-") as tmp:
            root = Path(tmp)
            self.fixture(root)
            for name, identity in (
                ("orders", "order_id"),
                ("labels", "order_id"),
                ("customers", "customer_id"),
            ):
                frame = pd.read_csv(
                    root / f"{name}_base.csv", dtype="string", keep_default_na=False
                )
                extra = frame.iloc[[0]].copy()
                extra["country_code"] = " "
                extra[identity] = " "
                if name == "labels":
                    extra["customer_id"] = " "
                pd.concat([frame, extra], ignore_index=True).to_csv(
                    root / f"{name}_base.csv", index=False
                )
            for raw_strings in (False, True):
                customers, orders, labels = load_base_data(
                    root, raw_strings=raw_strings
                )
                self.assertEqual(len(orders), 3)
                self.assertEqual(labels.order_id.tolist(), ["a"])
                self.assertEqual(customers.customer_id.tolist(), ["safe"])

    def test_each_table_owns_nested_audit_metadata(self):
        with TemporaryDirectory(prefix="pbi6-audit-") as tmp:
            root = Path(tmp)
            self.fixture(root)
            customers, orders, labels = load_base_data(root)
            for frame in (customers, orders, labels):
                audit = frame.attrs["split_audit"]
                self.assertEqual(audit["cutoffs"]["test_start"], "2023-04-17")
                self.assertEqual(audit["membership_counts"]["test"], 3)
            customers.attrs["split_audit"]["cutoffs"]["test_start"] = "2030-01-01"
            customers.attrs["split_audit"]["membership_counts"]["test"] = -1
            customers.attrs["split_audit"]["purpose"] = "changed"
            for frame in (orders, labels):
                audit = frame.attrs["split_audit"]
                self.assertEqual(audit["cutoffs"]["test_start"], "2023-04-17")
                self.assertEqual(audit["membership_counts"]["test"], 3)
                self.assertEqual(audit["purpose"], "analysis")
            orders.attrs["split_audit"]["cutoffs"]["test_end_exclusive"] = "2031-01-01"
            self.assertEqual(
                labels.attrs["split_audit"]["cutoffs"]["test_end_exclusive"],
                "2023-05-11",
            )

    def test_missing_schema_fails_closed(self):
        with TemporaryDirectory(prefix="pbi6-schema-") as tmp:
            root = Path(tmp)
            for name in ("customers", "orders", "labels"):
                (root / f"{name}_base.csv").write_text("id\n1\n")
            with self.assertRaises(KeyError):
                load_base_data(root)
