"""Fixtures for the loading and reading tests: made-up batch files, and a test database.

`database` is an empty copy of the team schema on a disposable MySQL server.

Set `TEST_MYSQL_URL` to an administrator URL on a throwaway MySQL 8.4 server, such
as `mysql+pymysql://root@127.0.0.1:3306`, to run the database tests; without it they
are skipped. Each test gets its own database, dropped afterwards. SQLite is not used
because its transactions, keys and types differ from MySQL's, which would let the
transaction and rerun tests pass on behaviour RDS does not have.
"""

import os
import uuid

import pytest
from sqlalchemy import create_engine, make_url

from mlops_fraud_prediction_ca1.batch_files import REQUIRED_COLUMNS
from mlops_fraud_prediction_ca1.rds_schema import create_tables


@pytest.fixture
def database():
    server = os.environ.get("TEST_MYSQL_URL")
    if not server:
        pytest.skip("Set TEST_MYSQL_URL to a disposable MySQL 8.4 server.")
    name = f"test_{uuid.uuid4().hex[:12]}"
    admin = create_engine(server, connect_args={"connect_timeout": 10})
    with admin.begin() as connection:
        connection.exec_driver_sql(f"CREATE DATABASE {name}")
    engine = create_engine(
        make_url(server).set(database=name), connect_args={"connect_timeout": 10}
    )
    try:
        create_tables(engine)
        yield engine
    finally:
        engine.dispose()
        with admin.begin() as connection:
            connection.exec_driver_sql(f"DROP DATABASE {name}")
        admin.dispose()


@pytest.fixture
def write_batch(tmp_path):
    """Write a batch's three CSV files from made-up rows and return their paths."""

    def write(batch, customers=(), orders=(), labels=()):
        paths = []
        for table, rows in [
            ("customers", customers),
            ("orders", orders),
            ("labels", labels),
        ]:
            path = tmp_path / f"{table}_{batch}.csv"
            header = ",".join(REQUIRED_COLUMNS[table])
            path.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")
            paths.append(str(path))
        return paths

    return write
