import uuid
from datetime import date

import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow import MlflowClient
from sqlalchemy import create_engine
from sqlalchemy.exc import DBAPIError

from . import review_queue
from .config import load_settings
from .loading import load_files
from .models import baseline_model
from .review_queue import QueueEvaluation, evaluate_queue, write_queue
from .training import train

CUSTOMERS = ["MY,c1,true,1.0,0.0,0.0,10.0,0.0,2022-10-01 10:00:00"]
# o3 has no label, o4's label is invalid, which cleaning stores as missing, and o5 is in
# the locked test window.
ORDERS = [
    "MY,o1,Delivery,Credit Card,10.0,1.0,0.0,2023-01-05",
    "MY,o2,Delivery,Credit Card,30.0,1.0,0.0,2023-01-05",
    "MY,o3,Delivery,Credit Card,20.0,1.0,0.0,2023-01-05",
    "MY,o4,Delivery,Credit Card,50.0,1.0,0.0,2023-01-06",
    "MY,o5,Delivery,Credit Card,90.0,1.0,0.0,2023-04-20",
]
LABELS = ["MY,o1,c1,0", "MY,o2,c1,1", "MY,o4,c1,2", "MY,o5,c1,1"]


class ByValue:
    """A stand-in model whose fraud score is the order value in thousands."""

    def __init__(self, sign=1):
        self.sign = sign

    def predict_proba(self, features):
        score = self.sign * features["order_value"].to_numpy(dtype=float) / 1000
        return np.column_stack([1 - score, score])


def _queue_table(engine):
    with engine.connect() as connection:
        return connection.exec_driver_sql(
            "SELECT * FROM review_queue ORDER BY order_date, order_id"
        ).all()


def test_the_queue_keeps_each_days_top_k_once_without_reading_is_fraud(
    database, write_batch
):
    load_files(database, *write_batch("b1", CUSTOMERS, ORDERS, LABELS))
    # A user that may read every labels column but is_fraud: a query naming is_fraud,
    # or every column, fails for it.
    name, password = f"q{uuid.uuid4().hex[:8]}", "a long enough test password"
    db = database.url.database
    grants = [
        f"GRANT {privileges} ON `{db}`.{table} TO %s@'%%'"
        for privileges, table in [
            ("SELECT", "batches"),
            ("SELECT", "orders"),
            ("SELECT (country_code, order_id, customer_id)", "labels"),
            ("SELECT, INSERT", "review_queue"),
        ]
    ]
    with database.begin() as connection:
        connection.exec_driver_sql(
            "CREATE USER %s@'%%' IDENTIFIED BY %s", (name, password)
        )
        for grant in grants:
            connection.exec_driver_sql(grant, (name,))
    scorer = create_engine(database.url.set(username=name, password=password))
    try:
        with pytest.raises(DBAPIError, match="denied"), scorer.connect() as connection:
            connection.exec_driver_sql("SELECT is_fraud FROM labels")
        queue = write_queue(scorer, "b1", ByValue(), 7, k=2)
        rerun = write_queue(scorer, "b1", ByValue(sign=-1), 7, k=2)
    finally:
        scorer.dispose()
        with database.begin() as connection:
            connection.exec_driver_sql("DROP USER %s@'%%'", (name,))

    assert (queue.orders_in_batch, queue.orders_ranked, queue.orders_withheld) == (
        5,
        4,
        1,
    )
    assert queue.orders_without_label == 1
    assert queue.written
    assert queue.rows.order_id.tolist() == ["o2", "o3", "o4"]
    assert queue.rows.customer_id.tolist()[0] == "c1"
    assert pd.isna(queue.rows.customer_id[1])
    assert _queue_table(database) == [
        ("b1", 7, "MY", "o2", date(2023, 1, 5), pytest.approx(0.03)),
        ("b1", 7, "MY", "o3", date(2023, 1, 5), pytest.approx(0.02)),
        ("b1", 7, "MY", "o4", date(2023, 1, 6), pytest.approx(0.05)),
    ]
    # The rerun ranks the other way round, but the stored queue stays as it was, and
    # the rerun returns it.
    assert not rerun.written
    assert rerun.rows.order_id.tolist() == ["o2", "o3", "o4"]
    assert rerun.rows.score[0] == pytest.approx(0.03)
    assert rerun.rows.customer_id.tolist()[0] == "c1"
    assert len(_queue_table(database)) == 3
    # o2 is fraud; o3 has no label and o4's label is invalid.
    assert evaluate_queue(database, "b1", 7) == QueueEvaluation(3, 1, 1)
    assert evaluate_queue(database, "b1", 7).precision == 1.0


def _training_rows(start):
    return pd.DataFrame(
        {
            "order_date": pd.date_range(start, periods=20).date,
            "country_code": "MY",
            "order_id": [f"t{start}{i}" for i in range(20)],
            "customer_id": "c1",
            "is_fraud": [i % 2 for i in range(20)],
            "collect_type": "delivery",
            "payment_method": "credit card",
            "order_value": [100.0 if i % 2 else 10.0 for i in range(20)],
            "num_items_ordered": 1.0,
        }
    )


def test_the_command_scores_with_the_champion_and_logs_the_queue(
    database, write_batch, tmp_path, monkeypatch, capsys
):
    monkeypatch.chdir(tmp_path)  # MLflow stores artifacts under the working folder.
    tracking_uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "queue-test")
    settings = load_settings(
        {
            "DB_HOST": "db.example",
            "DB_NAME": "practice",
            "DB_USER": "user",
            "DB_PASSWORD": "password",
            "MLFLOW_TRACKING_URI": tracking_uri,
            "MLFLOW_EXPERIMENT_NAME": "queue-test",
            "MLFLOW_TRACKING_USERNAME": "user",
            "MLFLOW_TRACKING_PASSWORD": "password",
        }
    )
    monkeypatch.setattr(review_queue, "load_settings", lambda: settings)
    monkeypatch.setattr(review_queue, "create_db_engine", lambda settings: database)
    run_id, _ = train(
        baseline_model(),
        k=250,
        seed=0,
        labelled_windows=(_training_rows("2023-01-01"), _training_rows("2023-03-28")),
    )
    mlflow.register_model(f"runs:/{run_id}/model", "fraud-candidate")
    MlflowClient().set_registered_model_alias("fraud-candidate", "champion", "1")
    load_files(database, *write_batch("b1", CUSTOMERS, ORDERS, LABELS))
    capsys.readouterr()

    assert review_queue.main(["b1"]) == 0

    assert capsys.readouterr().out.splitlines() == [
        "Database host: db.example",
        "Database name: practice",
        "MLflow experiment: queue-test",
        "Champion: fraud-candidate version 1",
        "Batch b1: 4 orders ranked, 1 withheld by the holdout setting, 1 without a label",
        "Queue: 4 orders written, the top 250 per day",
        "Precision at K: 0.500 (1 fraud in 2 labelled of 4 queued orders)",
    ]
    experiment = mlflow.get_experiment_by_name("queue-test").experiment_id
    [run] = mlflow.search_runs(
        [experiment], "tags.batch_id = 'b1'", output_format="list"
    )
    assert run.info.run_name == "review-queue-b1-v1"
    assert run.data.metrics["orders_ranked"] == 4
    assert run.data.metrics["precision_at_k"] == 0.5
    folder = mlflow.artifacts.download_artifacts(
        run_id=run.info.run_id, artifact_path="review_queue_b1.csv"
    )
    logged = pd.read_csv(folder)
    assert logged.columns.tolist() == [
        "order_date",
        "country_code",
        "order_id",
        "customer_id",
        "score",
    ]
    assert sorted(logged.order_id) == ["o1", "o2", "o3", "o4"]
