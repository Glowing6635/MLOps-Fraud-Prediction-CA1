"""Read labelled orders from the team database for training, validation and analysis.

Each order comes with its label and the customer's latest row loaded at or before
the order's batch, so a customer row from a later batch never supplies values to an
earlier order (issue #20). An order whose customer has no such row gets empty
customer values.

Values come back as the driver gives them: money as `Decimal`, `order_date` as a
date, and `is_fraud` as 0, 1 or None, where None means the source label was not
0 or 1. Training code converts the types it needs and drops or handles None labels.
"Latest" follows loading order, so batches must be loaded in the order they arrived.

Every read applies the holdout setting through `splitting.sql_date_predicate`
(issue #6): an ordinary read never returns test-window orders. Only cleaned values
are returned; the original `raw_` values stay in the database for audits.
"""

from pathlib import Path

import pandas as pd
from sqlalchemy import Connection, Engine

from mlops_fraud_prediction_ca1.splitting import load_policy, sql_date_predicate

_QUERY = """
SELECT
    o.country_code, o.order_id, o.batch_id, o.collect_type, o.payment_method,
    o.order_value, o.num_items_ordered, o.refund_value, o.order_date,
    l.customer_id, l.is_fraud,
    c.batch_id AS customer_batch_id, c.mobile_verified, c.num_orders_last_50days,
    c.num_cancelled_orders_last_50days, c.num_refund_orders_last_50days,
    c.total_payment_last_50days, c.num_associated_customers, c.first_order_datetime
FROM orders o
JOIN batches ob ON ob.batch_id = o.batch_id
JOIN labels l ON l.country_code = o.country_code AND l.order_id = o.order_id
LEFT JOIN customers c ON c.customer_id = l.customer_id AND c.batch_id = (
    SELECT c2.batch_id
    FROM customers c2
    JOIN batches cb ON cb.batch_id = c2.batch_id
    WHERE c2.customer_id = l.customer_id AND cb.batch_seq <= ob.batch_seq
    ORDER BY cb.batch_seq DESC
    LIMIT 1
)
WHERE {predicate}
ORDER BY o.order_date, o.country_code, o.order_id
"""


def read_labelled_orders(
    engine: Engine,
    *,
    purpose: str = "analysis",
    config_path: str | Path | None = None,
) -> pd.DataFrame:
    """Labelled orders allowed for `purpose`, with the customer values known at their batch.

    `purpose` and `config_path` mean what they mean in `splitting.split_orders`:
    "train", "validation" or "analysis", and "final_evaluation" only when the
    configuration enables it.
    """
    predicate, parameters = sql_date_predicate(purpose=purpose, config_path=config_path)
    with engine.connect() as connection:
        return _read(connection, predicate, parameters)


def _read(connection: Connection, predicate: str, parameters: tuple) -> pd.DataFrame:
    result = connection.exec_driver_sql(_QUERY.format(predicate=predicate), parameters)
    return pd.DataFrame(result.fetchall(), columns=list(result.keys()))


def read_training_snapshot(engine: Engine) -> tuple[pd.DataFrame, dict[str, int]]:
    """Read training rows and aggregate date exclusions from one MySQL snapshot."""
    predicate, parameters = sql_date_predicate(purpose="train")
    validation_start = load_policy().validation_start.isoformat()
    with engine.connect() as conn:
        conn.execution_options(isolation_level="REPEATABLE READ")
        with conn.begin():
            rows = _read(conn, predicate, parameters)
            excluded, late = conn.exec_driver_sql(
                f"""
                SELECT COUNT(*), COUNT(CASE WHEN o.order_date >= %s THEN 1 END)
                FROM orders o
                JOIN batches ob ON ob.batch_id = o.batch_id
                JOIN labels l ON l.country_code = o.country_code AND l.order_id = o.order_id
                WHERE NOT ({predicate})
                """,
                (validation_start, *parameters),
            ).one()
    return rows, {"excluded_date_rows": excluded, "excluded_late_rows": late}
