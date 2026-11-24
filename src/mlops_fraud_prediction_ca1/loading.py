"""Load a checked batch into the team database, safely and in one transaction (issue #20).

`uv run load-batch <customers> <orders> <labels>` takes the three files as URLs or
local paths, checks them with `batch_files.read_batch` (issue #19), cleans each with
`cleaning.clean_frame` (issue #12, the only cleaning code here), and loads them.
The base files load the same way, as `data/customers_base.csv` and so on.

Rules, recorded in `docs/data-dictionary.md`, "Loading rules":

- Customers, orders and labels go in one transaction: any failure leaves all three
  tables as they were.
- A stored row is never changed. A row whose key is already stored is skipped, so
  loading the same batch again changes nothing.
- Rows that cannot be stored are rejected and counted, never a reason to stop: a
  blank ID, an unknown country on an order or label, rows sharing a key with
  different values, a label whose order or customer is missing, and a label that
  differs from the stored label for the same order.

The loading user needs only SELECT and INSERT (`rds-schema writer`).
"""

import argparse
import logging
from collections import Counter
from dataclasses import dataclass

import pandas as pd
from sqlalchemy import Connection, Engine
from sqlalchemy.exc import SQLAlchemyError

from mlops_fraud_prediction_ca1.batch_files import Batch, BatchError, read_batch
from mlops_fraud_prediction_ca1.cleaning import clean_frame
from mlops_fraud_prediction_ca1.config import (
    SettingError,
    create_db_engine,
    load_settings,
)

_LOGGER = logging.getLogger(__name__)
# Order matters: labels refer to orders, and every table refers to batches.
_TABLES = ("customers", "orders", "labels")
_KEYS = {
    "customers": ["customer_id"],
    "orders": ["country_code", "order_id"],
    "labels": ["country_code", "order_id"],
}
# IDs are stored as they arrive, so they have no raw_ copy.
_IDS = {"customer_id", "order_id"}
_RAW_IDS = {f"raw_{column}" for column in _IDS}
# Column sizes in schema.sql: longer values are rejected rather than stopping the load.
_ID_LENGTH = 64
_RAW_LENGTH = 255
# Strict mode makes a value that does not fit its column an error, not a silent cut.
_STRICT = (
    "SET SESSION sql_mode = 'STRICT_ALL_TABLES,NO_ZERO_IN_DATE,NO_ZERO_DATE,"
    "ERROR_FOR_DIVISION_BY_ZERO,NO_ENGINE_SUBSTITUTION'"
)
_CHUNK = 10_000


@dataclass(frozen=True)
class LoadReport:
    """What one load did: rows added and already stored per table, and rejections by reason."""

    batch_id: str
    loaded: dict[str, int]
    already_stored: dict[str, int]
    rejected: dict[tuple[str, str], int]


def load_files(engine: Engine, customers: str, orders: str, labels: str) -> LoadReport:
    """Check the three files, then load them; a `BatchError` means nothing was written."""
    return load_batch(engine, read_batch(customers, orders, labels))


def load_batch(engine: Engine, batch: Batch) -> LoadReport:
    """Clean and load a checked batch in one transaction, skipping rows already stored."""
    rejected = Counter()
    frames = {
        table: _accept(table, clean_frame(getattr(batch, table)), rejected)
        for table in _TABLES
    }
    customers, orders, labels = frames.values()
    with engine.begin() as connection:
        connection.exec_driver_sql(_STRICT)
        _add_batch(connection, batch.batch_id)
        stored_customers = _select(
            connection, "SELECT customer_id, batch_id FROM customers"
        )
        stored_orders = _select(connection, "SELECT country_code, order_id FROM orders")
        stored_labels = _select(
            connection, "SELECT country_code, order_id, is_fraud FROM labels"
        )

        in_this_batch = stored_customers["batch_id"].eq(batch.batch_id)
        new_customers = ~_id(customers["customer_id"]).isin(
            _id(stored_customers.loc[in_this_batch, "customer_id"])
        )
        new_orders = ~_key(orders, "orders").isin(_key(stored_orders, "orders"))

        label_keys = _key(labels, "labels")
        has_order = label_keys.isin(_key(orders, "orders")) | label_keys.isin(
            _key(stored_orders, "orders")
        )
        customer = _id(labels["customer_id"])
        has_customer = customer.isin(_id(customers["customer_id"])) | customer.isin(
            _id(stored_customers["customer_id"])
        )
        stored_label = pd.Series(
            stored_labels["is_fraud"].to_numpy(), index=_key(stored_labels, "labels")
        ).reindex(label_keys)
        stored_label.index = labels.index
        is_stored = label_keys.isin(_key(stored_labels, "labels"))
        same_label = stored_label.astype("Int8").eq(labels["is_fraud"]).fillna(
            False
        ) | (stored_label.isna() & labels["is_fraud"].isna())
        checks = [
            ("no matching order", ~has_order),
            ("no matching customer", has_order & ~has_customer),
            (
                "label differs from the stored label",
                has_order & has_customer & is_stored & ~same_label,
            ),
        ]
        for reason, mask in checks:
            if mask.any():
                rejected["labels", reason] += int(mask.sum())
        new_labels = has_order & has_customer & ~is_stored

        new = {"customers": new_customers, "orders": new_orders, "labels": new_labels}
        for table in _TABLES:
            _insert(
                connection,
                table,
                frames[table].loc[new[table]],
                batch.batch_id,
            )

    report = LoadReport(
        batch_id=batch.batch_id,
        loaded={table: int(new[table].sum()) for table in _TABLES},
        already_stored={
            "customers": int((~new_customers).sum()),
            "orders": int((~new_orders).sum()),
            "labels": int((has_order & has_customer & is_stored & same_label).sum()),
        },
        rejected=dict(rejected),
    )
    _LOGGER.info("Loaded batch %s | %s", batch.batch_id, report)
    return report


def _accept(table: str, frame: pd.DataFrame, rejected: Counter) -> pd.DataFrame:
    """Drop the rows of one table that cannot be stored, counting each by reason."""
    checks = []
    for column in _KEYS[table]:
        if column in _IDS:
            checks.append((f"blank {column}", frame[column].str.strip().eq("")))
            too_long = frame[column].str.len().gt(_ID_LENGTH)
            checks.append((f"{column} longer than {_ID_LENGTH} characters", too_long))
        else:
            checks.append((f"unknown {column}", frame[column].isna()))
    raw = [column for column in frame.columns if column.startswith("raw_")]
    long_value = frame[raw].apply(lambda values: values.str.len().gt(_RAW_LENGTH))
    checks.append(
        (f"a value longer than {_RAW_LENGTH} characters", long_value.any(axis=1))
    )
    cleaned = [column for column in frame.columns if not column.startswith("raw_")]
    # A repeated row whose cleaned values match is one record sent twice.
    checks.append(("copy of an earlier row", frame.duplicated(subset=cleaned)))
    keep = pd.Series(True, index=frame.index)
    for reason, mask in checks:
        mask = mask & keep
        if mask.any():
            rejected[table, reason] += int(mask.sum())
        keep &= ~mask
    frame = frame[keep.to_numpy()]
    # Rows that share a key but differ cannot say which is right, so none is kept.
    clash = _key(frame, table).duplicated(keep=False)
    if clash.any():
        rejected[table, "same key as another row with different values"] += int(
            clash.sum()
        )
    return frame[~clash].reset_index(drop=True)


def _add_batch(connection: Connection, batch_id: str) -> None:
    exists = connection.exec_driver_sql(
        "SELECT 1 FROM batches WHERE batch_id = %s", (batch_id,)
    ).first()
    if not exists:
        connection.exec_driver_sql(
            "INSERT INTO batches (batch_id) VALUES (%s)", (batch_id,)
        )


def _select(connection: Connection, query: str) -> pd.DataFrame:
    result = connection.exec_driver_sql(query)
    return pd.DataFrame(result.fetchall(), columns=list(result.keys()))


def _id(values: pd.Series) -> pd.Series:
    # The schema's ID collation ignores trailing spaces when it compares keys, so
    # "o1" and "o1 " are one key in MySQL. Compare them the same way here.
    return values.astype("string").str.rstrip(" ")


def _key(frame: pd.DataFrame, table: str) -> pd.MultiIndex:
    return pd.MultiIndex.from_arrays([_id(frame[column]) for column in _KEYS[table]])


def _insert(
    connection: Connection, table: str, frame: pd.DataFrame, batch_id: str
) -> None:
    columns = [column for column in frame.columns if column not in _RAW_IDS]
    rows = frame[columns].copy()
    for column in columns:
        if pd.api.types.is_datetime64_any_dtype(rows[column]):
            rows[column] = rows[column].dt.strftime("%Y-%m-%d %H:%M:%S")
    rows = rows.astype(object)
    rows = rows.where(rows.notna(), None)
    rows.insert(0, "batch_id", batch_id)
    statement = (
        f"INSERT INTO {table} ({', '.join(rows.columns)})"
        f" VALUES ({', '.join(['%s'] * len(rows.columns))})"
    )
    values = list(rows.itertuples(index=False, name=None))
    for start in range(0, len(values), _CHUNK):
        connection.exec_driver_sql(statement, values[start : start + _CHUNK])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="load-batch",
        description="Check a batch's three files and load them into the team database.",
    )
    for table in _TABLES:
        parser.add_argument(
            table, help=f"the {table}_<batch>.csv file: a URL or a path"
        )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        settings = load_settings()
        report = load_files(
            create_db_engine(settings), args.customers, args.orders, args.labels
        )
    except (SettingError, BatchError, SQLAlchemyError) as error:
        print(f"Batch not loaded: {error}")
        raise SystemExit(1) from error
    print(
        f"Batch {report.batch_id} loaded into {settings.db_name} on {settings.db_host}."
    )
    for table in _TABLES:
        print(
            f"{table}: {report.loaded[table]} added,"
            f" {report.already_stored[table]} already stored"
        )
    for (table, reason), count in sorted(report.rejected.items()):
        print(f"{table} rejected, {reason}: {count}")
