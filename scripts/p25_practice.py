"""Generate and optionally load the disposable PBI #25 practice inputs."""

import argparse
import hashlib
import json
import logging
import os
import re
import sqlite3
from contextlib import contextmanager
from decimal import Decimal
from pathlib import Path

import mlflow
import pandas as pd
from sqlalchemy import URL, create_engine

from mlops_fraud_prediction_ca1 import config, retrain_inputs
from mlops_fraud_prediction_ca1.batch_files import REQUIRED_COLUMNS
from mlops_fraud_prediction_ca1.loading import load_files
from mlops_fraud_prediction_ca1.models import baseline_model
from mlops_fraud_prediction_ca1.rds_reads import read_training_snapshot
from mlops_fraud_prediction_ca1.rds_schema import statements
from mlops_fraud_prediction_ca1.training import K, train

BATCH = "p25-practice"
EXPERIMENT = "p25-offline-retrain"
CHAMPION = "p25-offline-champion"
_LOG = logging.getLogger(__name__)


def _tables(batch, dates):
    rows = {table: [] for table in REQUIRED_COLUMNS}
    for i, day in enumerate(dates):
        customer = f"SYN25-{batch}-C{i:03d}"
        order = f"SYN25-{batch}-O{i:03d}"
        fraud = i % 2
        value = 10 + 2 * i + 20 * fraud
        rows["customers"].append(
            ("MY", customer, "True", 1, 0, 0, value, 0, "2022-10-01 00:00:00")
        )
        rows["orders"].append(
            ("MY", order, "delivery", "credit card", value, 1 + fraud, 0, day)
        )
        rows["labels"].append(("MY", order, customer, fraud))
    return {
        table: pd.DataFrame(values, columns=REQUIRED_COLUMNS[table])
        for table, values in rows.items()
    }


def generate(root):
    """Write synthetic old rows, an incoming batch and separate validation rows."""
    root.mkdir(parents=True, exist_ok=True)
    batches = {
        "base": ["2023-02-01"] * 12,
        BATCH: ["2023-03-26"] * 8 + ["2025-01-06"] * 3,
    }
    for batch, dates in batches.items():
        directory = root / batch
        directory.mkdir(exist_ok=True)
        for table, rows in _tables(batch, dates).items():
            rows.to_csv(directory / f"{table}_{batch}.csv", index=False)
    validation = _tables("validation", ["2023-04-01"] * 6)
    validation["orders"].merge(
        validation["labels"], on=["country_code", "order_id"]
    ).to_csv(root / "validation.csv", index=False)


class _FixtureConnection:
    def __init__(self, connection, snapshot_started):
        self._connection = connection
        self._snapshot_started = snapshot_started

    def begin(self):
        return self._connection.begin()

    def execution_options(self, *, isolation_level):
        if isolation_level != "REPEATABLE READ":
            raise ValueError(
                f"Unexpected snapshot isolation request: {isolation_level}"
            )
        self._snapshot_started()

    def exec_driver_sql(self, statement, parameters=()):
        # SQLite has no MySQL session SQL mode. Application SELECTs and inserts
        # run unchanged apart from the driver placeholder.
        if not statement.lstrip().upper().startswith("SET SESSION "):
            return self._connection.exec_driver_sql(
                statement.replace("%s", "?"), parameters
            )


class _FixtureEngine:
    def __init__(self, path):
        sqlite3.register_adapter(Decimal, str)
        self.engine = create_engine(
            URL.create("sqlite", database=str(path)),
            connect_args={"autocommit": False},
        )
        self.snapshots = 0

    def _snapshot_started(self):
        self.snapshots += 1
        _LOG.info("SQLite fixture training snapshots=%s", self.snapshots)

    @contextmanager
    def connect(self):
        with self.engine.connect() as connection:
            yield _FixtureConnection(connection, self._snapshot_started)

    @contextmanager
    def begin(self):
        with self.engine.begin() as connection:
            yield _FixtureConnection(connection, self._snapshot_started)

    def dispose(self):
        self.engine.dispose()


def _init_schema(engine):
    # Keep the application columns, constraints and keys. SQLite needs its
    # rowid primary key for batch order and has different DDL for secondary indexes.
    with engine.connect() as connection:
        connection.connection.driver_connection.autocommit = True
        connection.exec_driver_sql("PRAGMA foreign_keys = ON")
        for statement in statements():
            statement = re.sub(r"(?m)^    (?:UNIQUE )?KEY .*\n", "", statement)
            statement = re.sub(
                r" COLLATE utf8mb4_bin| ENGINE = InnoDB DEFAULT CHARSET = utf8mb4",
                "",
                statement,
            )
            statement = statement.replace(
                "batch_seq INT UNSIGNED NOT NULL AUTO_INCREMENT",
                "batch_seq INTEGER PRIMARY KEY",
            ).replace("PRIMARY KEY (batch_id),", "UNIQUE (batch_id)")
            connection.exec_driver_sql(statement)
        connection.connection.driver_connection.autocommit = False


def _load(engine, root, batch):
    paths = [str(root / batch / f"{table}_{batch}.csv") for table in REQUIRED_COLUMNS]
    report = load_files(engine, *paths)
    if report.rejected or any(report.already_stored.values()):
        raise ValueError(f"Unexpected fixture load counts: {report}")
    _LOG.info("Loaded %s: %s", batch, report.loaded)


def _tracking_uri(root):
    return f"sqlite:///{root / 'mlflow.db'}"


def prepare_offline(root):
    """Seed SQLite, save a synthetic fitted champion, and write pinned CLI settings."""
    if any(
        (root / name).exists() for name in ("training.db", "mlflow.db", "offline.env")
    ):
        raise ValueError("Choose a fresh output directory for the disposable stores.")
    generate(root)
    engine = _FixtureEngine(root / "training.db")
    try:
        _init_schema(engine.engine)
        _load(engine, root, "base")
        old, _ = read_training_snapshot(engine)
        _load(engine, root, BATCH)
    finally:
        engine.dispose()

    tracking_uri = _tracking_uri(root)
    os.environ.update(
        MLFLOW_TRACKING_URI=tracking_uri,
        MLFLOW_REGISTRY_URI=tracking_uri,
        MLFLOW_EXPERIMENT_NAME=CHAMPION,
    )
    mlflow.create_experiment(CHAMPION, artifact_location=(root / "champion").as_uri())
    mlflow.create_experiment(
        EXPERIMENT, artifact_location=(root / "candidates").as_uri()
    )
    validation_path = root / "validation.csv"
    validation = pd.read_csv(validation_path)
    run_id, _ = train(
        baseline_model(),
        k=K,
        seed=None,
        labelled_windows=(old, validation),
        tags={"champion_eligible": "False"},
    )
    version = mlflow.register_model(f"runs:/{run_id}/model", CHAMPION)
    settings = {
        "P25_OFFLINE_DIR": str(root),
        "MLFLOW_TRACKING_URI": tracking_uri,
        "MLFLOW_REGISTRY_URI": tracking_uri,
        "MLFLOW_EXPERIMENT_NAME": EXPERIMENT,
        "MLFLOW_TRACKING_USERNAME": "offline",
        "MLFLOW_TRACKING_PASSWORD": "offline",
        "DB_HOST": "offline",
        "DB_NAME": "offline",
        "DB_USER": "offline",
        "DB_PASSWORD": "offline",
        "RETRAIN_VALIDATION_CSV": str(validation_path),
        "RETRAIN_VALIDATION_SHA256": hashlib.sha256(
            validation_path.read_bytes()
        ).hexdigest(),
        "RETRAIN_CHAMPION_MODEL_NAME": CHAMPION,
        "RETRAIN_CHAMPION_MODEL_VERSION": str(version.version),
        "RETRAIN_CHAMPION_RUN_ID": run_id,
    }
    (root / "offline.env").write_text(
        "".join(f"{key}={json.dumps(value)}\n" for key, value in settings.items()),
        encoding="utf-8",
    )


def build_inputs():
    """Offline-only CLI factory that uses the real inputs and a local SQL adapter."""
    root = Path(os.environ["P25_OFFLINE_DIR"]).resolve()
    if (
        os.environ.get("MLFLOW_TRACKING_URI") != _tracking_uri(root)
        or os.environ.get("MLFLOW_REGISTRY_URI") != _tracking_uri(root)
        or os.environ.get("MLFLOW_EXPERIMENT_NAME") != EXPERIMENT
        or not (root / "training.db").is_file()
        or not (root / "mlflow.db").is_file()
    ):
        raise ValueError("Use the generated offline.env for the disposable stores.")
    inputs = retrain_inputs.build_inputs()
    # retrain.main creates its engine immediately after invoking the trusted
    # factory. Keep the substitution in this explicitly selected fixture module.
    config.create_db_engine = lambda _settings: _FixtureEngine(root / "training.db")
    logging.basicConfig(level=logging.INFO)
    return inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prepare-offline", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    root = args.out.resolve()
    if args.prepare_offline:
        prepare_offline(root)
    else:
        generate(root)


if __name__ == "__main__":
    main()
