"""The daily review queue (issue #33): rank a loaded batch's orders with the champion and
store each day's riskiest K for the fraud analysts.

`uv run review-queue BATCH_ID` loads the champion from the MLflow model registry by its
alias, with no retraining, and scores every order of the batch that the holdout setting
allows (`splitting.sql_date_predicate`, purpose "analysis"). Each selection group's top
K orders (`top_k.select_top_k`: one group per order date, K from `top_k.K`) go to the
`review_queue` table with the champion's model version. The command logs the queue to
MLflow as a CSV with the number of orders ranked. Then, as a separate step, it joins the
stored labels to the queue and logs precision at K to the same run.

Scoring never reads `is_fraud`: from `labels` it reads only `customer_id`, which goes
into the CSV for the analysts, and whether the order has a label. The champion is
scored on the `orders` columns it was trained on, as its `feature_names_in_` lists them.
Champions use order columns only (D4-02 in docs/decision-log.md), so an order without
a label is scored like any other, and the command counts them.

The queue for a batch and model version is written once. A rerun writes nothing, so
the table keeps the first queue, and logs that stored queue in a new MLflow run.
"""

import argparse
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from sqlalchemy import Engine
from sqlalchemy.exc import SQLAlchemyError

from .config import SettingError, create_db_engine, load_settings
from .splitting import sql_date_predicate
from .top_k import K, select_top_k
from .training import order_features

# TODO(#15): import the alias from champion.py (its CHAMPION) once the champion pull
# request (#171) reaches this branch. The model name stays here: champion.py takes it
# as the --register argument.
CHAMPION_MODEL = "fraud-candidate"
CHAMPION_ALIAS = "champion"

# `order_date` alone is the holdout predicate's column, and only `orders` has one.
_ORDERS = """
SELECT o.order_date, o.country_code, o.order_id, l.customer_id,
    l.order_id IS NOT NULL AS has_label{features}
FROM orders o
LEFT JOIN labels l ON l.country_code = o.country_code AND l.order_id = o.order_id
WHERE o.batch_id = %s AND {predicate}
ORDER BY o.order_date, o.country_code, o.order_id
"""
_STORED = """
SELECT q.order_date, q.country_code, q.order_id, l.customer_id, q.score
FROM review_queue q
LEFT JOIN labels l ON l.country_code = q.country_code AND l.order_id = q.order_id
WHERE q.batch_id = %s AND q.model_version = %s
ORDER BY q.order_date, q.country_code, q.order_id
"""
_QUEUE_COLUMNS = ["order_date", "country_code", "order_id", "customer_id", "score"]
_ORDER_KEYS = {"order_date", "country_code", "order_id"}


@dataclass(frozen=True)
class Queue:
    """One batch's queue, and the counts the command reports and logs."""

    orders_in_batch: int
    orders_ranked: int
    orders_without_label: int
    rows: pd.DataFrame
    written: bool

    @property
    def orders_withheld(self) -> int:
        """Orders the holdout setting kept out of the ranking."""
        return self.orders_in_batch - self.orders_ranked


@dataclass(frozen=True)
class QueueEvaluation:
    """How many queued orders have a label, and how many of those were fraud."""

    queued: int
    labelled: int
    fraud: int

    @property
    def precision(self) -> float | None:
        """Fraud over labelled queued orders, or `None` when none has a label."""
        return self.fraud / self.labelled if self.labelled else None


def write_queue(
    engine: Engine, batch_id: str, model, model_version: int, *, k: int
) -> Queue:
    """Score the batch's orders with `model` and store each day's top `k` once.

    `model` is a fitted classifier with `predict_proba`, and `feature_names_in_` naming
    the `orders` columns it was trained on, as scikit-learn records them for a model
    fitted on a DataFrame. The rows are stored in one transaction. If the table already
    holds a queue for this batch and model version, nothing is written, `written` is
    False and `rows` is the stored queue.

    Raises `ValueError` when the model does not name its input columns as plain column
    names, when the database holds no orders for the batch, or when the holdout setting
    withholds all of them: those in the locked test window, before the training data
    starts, or without a date.
    """
    inputs = [str(name) for name in getattr(model, "feature_names_in_", ())]
    if not inputs or not all(name.isidentifier() for name in inputs):
        raise ValueError(
            f"Model version {model_version} doesn't name its input columns as plain"
            " column names, so the orders columns to score it on are unknown."
        )
    predicate, parameters = sql_date_predicate(purpose="analysis")
    features = "".join(f", o.`{name}`" for name in inputs if name not in _ORDER_KEYS)
    query = _ORDERS.format(features=features, predicate=predicate)
    with engine.begin() as connection:
        in_batch = connection.exec_driver_sql(
            "SELECT COUNT(*) FROM orders WHERE batch_id = %s", (batch_id,)
        ).scalar()
        if not in_batch:
            raise ValueError(
                f"The database has no orders for batch {batch_id}. Load it first with"
                " uv run load-batch."
            )
        result = connection.exec_driver_sql(query, (batch_id, *parameters))
        rows = pd.DataFrame(result.fetchall(), columns=list(result.keys()))
        if rows.empty:
            raise ValueError(
                f"None of the {in_batch} orders of batch {batch_id} can be ranked. The"
                " holdout setting withholds orders in the locked test window, before"
                " the training data starts, or without a date."
            )
        rows["score"] = model.predict_proba(order_features(rows, inputs))[:, 1]
        queue = rows[select_top_k(rows, rows["score"].to_numpy(), k)]
        stored = connection.exec_driver_sql(
            "SELECT 1 FROM review_queue WHERE batch_id = %s AND model_version = %s"
            " LIMIT 1",
            (batch_id, model_version),
        ).first()
        if stored is not None:
            result = connection.exec_driver_sql(_STORED, (batch_id, model_version))
            queue = pd.DataFrame(result.fetchall(), columns=list(result.keys()))
        else:
            connection.exec_driver_sql(
                "INSERT INTO review_queue (batch_id, model_version, country_code,"
                " order_id, order_date, score) VALUES (%s, %s, %s, %s, %s, %s)",
                [
                    (batch_id, model_version, row.country_code, row.order_id)
                    + (row.order_date, float(row.score))
                    for row in queue.itertuples()
                ],
            )
    return Queue(
        orders_in_batch=in_batch,
        orders_ranked=len(rows),
        orders_without_label=int((rows["has_label"] == 0).sum()),
        rows=queue[_QUEUE_COLUMNS].reset_index(drop=True),
        written=stored is None,
    )


def evaluate_queue(
    engine: Engine, batch_id: str, model_version: int
) -> QueueEvaluation:
    """Join the stored labels to the stored queue, after the queue is written.

    A queued order whose label is missing or invalid counts as queued but not labelled.
    """
    with engine.connect() as connection:
        queued, labelled, fraud = connection.exec_driver_sql(
            "SELECT COUNT(*), COUNT(l.is_fraud), COALESCE(SUM(l.is_fraud), 0)"
            " FROM review_queue q LEFT JOIN labels l"
            " ON l.country_code = q.country_code AND l.order_id = q.order_id"
            " WHERE q.batch_id = %s AND q.model_version = %s",
            (batch_id, model_version),
        ).one()
    return QueueEvaluation(int(queued), int(labelled), int(fraud))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("batch_id", help="a batch already loaded into the database")
    args = parser.parse_args(argv)
    try:
        settings = load_settings()
    except SettingError as exc:
        sys.stderr.write(f"Configuration error: {exc}\n")
        return 1
    sys.stdout.write(
        f"Database host: {settings.db_host}\n"
        f"Database name: {settings.db_name}\n"
        f"MLflow experiment: {settings.mlflow_experiment_name}\n"
    )
    # MLflow prints an emoji when a run ends, which Windows cannot encode when piped.
    sys.stdout.reconfigure(errors="replace")
    engine = create_db_engine(settings)
    try:
        _run(engine, settings.mlflow_experiment_name, args.batch_id)
    except (SQLAlchemyError, MlflowException, ValueError) as error:
        sys.stderr.write(f"Review queue stopped: {error}\n")
        return 1
    finally:
        engine.dispose()
    return 0


def _run(engine: Engine, experiment: str, batch_id: str) -> None:
    # Load the version the alias names, so a move of the alias mid-run cannot mix two.
    version = int(
        MlflowClient()
        .get_model_version_by_alias(CHAMPION_MODEL, CHAMPION_ALIAS)
        .version
    )
    model = mlflow.sklearn.load_model(f"models:/{CHAMPION_MODEL}/{version}")
    sys.stdout.write(f"Champion: {CHAMPION_MODEL} version {version}\n")

    queue = write_queue(engine, batch_id, model, version, k=K)
    sys.stdout.write(
        f"Batch {batch_id}: {queue.orders_ranked} orders ranked,"
        f" {queue.orders_withheld} withheld by the holdout setting,"
        f" {queue.orders_without_label} without a label\n"
        + (
            f"Queue: {len(queue.rows)} orders written, the top {K} per day\n"
            if queue.written
            else "Queue: already stored for this batch and version, nothing written\n"
        )
    )
    mlflow.set_experiment(experiment)
    with mlflow.start_run(
        run_name=f"review-queue-{batch_id}-v{version}",
        tags={"batch_id": batch_id, "model_version": str(version)},
    ):
        mlflow.log_params({"batch_id": batch_id, "model_version": version, "k": K})
        mlflow.log_metrics(
            {
                "orders_ranked": queue.orders_ranked,
                "queue_orders": len(queue.rows),
                "orders_without_label": queue.orders_without_label,
                "orders_withheld": queue.orders_withheld,
            }
        )
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / f"review_queue_{batch_id}.csv"
            queue.rows.to_csv(path, index=False)
            mlflow.log_artifact(str(path))

        evaluation = evaluate_queue(engine, batch_id, version)
        if evaluation.precision is not None:
            mlflow.log_metric("precision_at_k", evaluation.precision)
        shown = "N/A" if evaluation.precision is None else f"{evaluation.precision:.3f}"
        sys.stdout.write(
            f"Precision at K: {shown} ({evaluation.fraud} fraud in"
            f" {evaluation.labelled} labelled of {evaluation.queued} queued orders)\n"
        )
