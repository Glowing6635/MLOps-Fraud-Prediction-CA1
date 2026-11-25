import uuid
from datetime import date
from decimal import Decimal

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.exc import DBAPIError

from mlops_fraud_prediction_ca1.batch_files import (
    REQUIRED_COLUMNS,
    BadFileNames,
    DownloadFailed,
    MissingColumns,
    NoRows,
    NotCsv,
)
from mlops_fraud_prediction_ca1.config import Settings, create_db_engine
from mlops_fraud_prediction_ca1.loading import LoadReport, load_files
from mlops_fraud_prediction_ca1.rds_schema import create_user

# Made-up rows. Columns are in batch_files.REQUIRED_COLUMNS order.
C1 = "my,c1,TRUE,3.0,0.0,1.0,45.5,1.0,2022-10-01 10:00:00"
C2 = "TH,c2,false,1.0,0.0,0.0,9.25,0.0,2022-12-24 08:30:00"
O1 = "my,o1,Delivery,Credit Card,20.5,2.0,0.0,2023/Jan/05"
O2 = "TH,o2,pickup,balance,7.0,1.0,7.0,2023-01-06"
L1 = "MY,o1,c1,0"
L2 = "TH,o2,c2,1"
TABLES = ("batches", "customers", "orders", "labels")


def counts(engine):
    with engine.connect() as connection:
        return {
            table: connection.exec_driver_sql(f"SELECT COUNT(*) FROM {table}").scalar()
            for table in TABLES
        }


def test_a_batch_stores_cleaned_and_original_values_with_its_batch_id(
    database, write_batch
):
    report = load_files(database, *write_batch("base", [C1, C2], [O1, O2], [L1, L2]))
    assert report == LoadReport(
        batch_id="base",
        loaded={"customers": 2, "orders": 2, "labels": 2},
        already_stored={"customers": 0, "orders": 0, "labels": 0},
        rejected={},
    )
    with database.connect() as connection:
        order = connection.exec_driver_sql(
            "SELECT * FROM orders WHERE order_id = 'o1'"
        ).one()
        customer = connection.exec_driver_sql(
            "SELECT batch_id, country_code, raw_country_code, mobile_verified,"
            " raw_mobile_verified, first_order_datetime FROM customers"
            " WHERE customer_id = 'c1'"
        ).one()
    assert order._asdict() == {
        "country_code": "MY",
        "order_id": "o1",
        "batch_id": "base",
        "collect_type": "delivery",
        "payment_method": "credit card",
        "order_value": Decimal("20.5"),
        "num_items_ordered": 2.0,
        "refund_value": Decimal(0),
        "order_date": date(2023, 1, 5),
        "raw_country_code": "my",
        "raw_collect_type": "Delivery",
        "raw_payment_method": "Credit Card",
        "raw_order_value": "20.5",
        "raw_num_items_ordered": "2.0",
        "raw_refund_value": "0.0",
        "raw_order_date": "2023/Jan/05",
    }
    assert tuple(customer)[:5] == ("base", "MY", "my", 1, "TRUE")


def test_loading_the_same_batch_twice_leaves_row_counts_unchanged(
    database, write_batch
):
    files = write_batch("base", [C1, C2], [O1, O2], [L1, L2])
    load_files(database, *files)
    before = counts(database)
    report = load_files(database, *files)
    assert (
        counts(database)
        == before
        == {
            "batches": 1,
            "customers": 2,
            "orders": 2,
            "labels": 2,
        }
    )
    assert report.loaded == {"customers": 0, "orders": 0, "labels": 0}
    assert report.already_stored == {"customers": 2, "orders": 2, "labels": 2}


def test_a_failure_on_the_third_table_leaves_all_three_unchanged_and_shows_no_rows(
    database, write_batch
):
    load_files(database, *write_batch("base", [C1], [O1], [L1]))
    before = counts(database)
    with database.begin() as connection:
        connection.exec_driver_sql(
            "CREATE TRIGGER fail_labels BEFORE INSERT ON labels FOR EACH ROW"
            " SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = 'labels insert failed'"
        )
    # The command's own engine, so the error text is what load-batch would print.
    url = database.url
    engine = create_db_engine(
        Settings(
            db_host=url.host,
            db_port=url.port or 3306,
            db_name=url.database,
            db_user=url.username,
            db_password=url.password or "",
            mlflow_tracking_uri="unused",
            mlflow_experiment_name="unused",
            mlflow_tracking_username="unused",
            mlflow_tracking_password="unused",
        )
    )

    # The test MySQL has a self-signed certificate, not RDS's, so connect without TLS.
    @event.listens_for(engine, "do_connect")
    def _without_rds_certificate(dialect, conn_rec, cargs, cparams):
        cparams.pop("ssl")

    with pytest.raises(DBAPIError, match="labels insert failed") as failure:
        load_files(engine, *write_batch("2", [C2], [O2], [L2]))
    engine.dispose()
    assert counts(database) == before
    assert "c2" not in str(failure.value)


@pytest.mark.parametrize(
    ("change", "error"),
    [
        ({"labels": "missing.csv"}, DownloadFailed),
        ({"orders": "<html>Sign in</html>"}, NotCsv),
        (
            {"labels": "country_code,order_id,customer,is_fraud\nTH,o2,c2,1"},
            MissingColumns,
        ),
        ({"customers": ",".join(REQUIRED_COLUMNS["customers"])}, NoRows),
        ({"name": "labels_3.csv"}, BadFileNames),
    ],
    ids=["download", "not-csv", "renamed-column", "no-rows", "names"],
)
def test_a_bad_batch_stops_before_any_database_change(
    database, write_batch, tmp_path, change, error
):
    load_files(database, *write_batch("base", [C1], [O1], [L1]))
    before = counts(database)
    files = write_batch("2", [C2], [O2], [L2])
    for position, table in enumerate(("customers", "orders", "labels")):
        if table not in change:
            continue
        if change[table] == "missing.csv":
            files[position] = str(tmp_path / "missing" / f"{table}_2.csv")
        else:
            (tmp_path / f"{table}_2.csv").write_text(change[table], encoding="utf-8")
    if "name" in change:
        files[2] = str((tmp_path / "labels_2.csv").rename(tmp_path / change["name"]))
    with pytest.raises(error):
        load_files(database, *files)
    assert counts(database) == before


def test_rows_that_cannot_be_stored_are_rejected_and_counted_without_stopping(
    database, write_batch
):
    report = load_files(
        database,
        *write_batch(
            "base",
            customers=[C1, C2, C2, "MY, ,true,1,0,0,1,0,2022-01-01 00:00:00"],
            orders=[
                O1,
                O2,
                "MY,o3,drone,barter,lots,1,0,2023/02/30",  # unseen and bad values
                "ZZ,o4,pickup,balance,1,1,0,2023-01-07",  # unknown country
                "MY, ,pickup,balance,1,1,0,2023-01-07",  # blank order_id
                "MY,o5,pickup,balance,1,1,0,2023-01-07",
                "MY,o5,pickup,balance,2,1,0,2023-01-07",  # same key, other value
                "MY,o6,pickup,balance,1,1,0,2023-01-07",
                "MY," + "x" * 65 + ",pickup,balance,1,1,0,2023-01-07",  # ID too long
                "MY,o7,pickup," + "y" * 256 + ",1,1,0,2023-01-07",  # value too long
            ],
            labels=[
                L1,
                L2,
                L2,  # an exact copy
                "MY,o3,c1,1",
                "MY,o3,c1,0",  # the same order with a different label
                "MY,o9,c1,1",  # no such order
                "MY,o6,c9,1",  # no such customer
            ],
        ),
    )
    assert report == LoadReport(
        batch_id="base",
        loaded={"customers": 2, "orders": 4, "labels": 2},
        already_stored={"customers": 0, "orders": 0, "labels": 0},
        rejected={
            ("customers", "blank customer_id"): 1,
            ("customers", "copy of an earlier row"): 1,
            ("orders", "unknown country_code"): 1,
            ("orders", "blank order_id"): 1,
            ("orders", "order_id longer than 64 characters"): 1,
            ("orders", "a value longer than 255 characters"): 1,
            ("orders", "same key as another row with different values"): 2,
            ("labels", "copy of an earlier row"): 1,
            ("labels", "same key as another row with different values"): 2,
            ("labels", "no matching order"): 1,
            ("labels", "no matching customer"): 1,
        },
    )
    with database.connect() as connection:
        o3 = connection.exec_driver_sql(
            "SELECT collect_type, payment_method, order_value, order_date,"
            " raw_collect_type, raw_order_value FROM orders WHERE order_id = 'o3'"
        ).one()
    assert tuple(o3) == (None, None, None, None, "drone", "lots")


def test_a_later_batch_keeps_stored_orders_and_labels_and_adds_customer_rows(
    database, write_batch
):
    o3 = "MY,o3,pickup,balance,5,1,0,2023-01-07"
    load_files(database, *write_batch("base", [C1], [O1, o3], [L1, "MY,o3,c1,"]))
    # MySQL compares IDs ignoring trailing spaces, so "o1 " is the stored o1.
    o1_again = O1.replace(",o1,", ",o1 ,")
    report = load_files(
        database,
        *write_batch(
            "2", [C1, C2], [o1_again, O2, o3], ["MY,o1 ,c1,1", L2, "MY,o3,c1,"]
        ),
    )
    assert report == LoadReport(
        batch_id="2",
        loaded={"customers": 2, "orders": 1, "labels": 1},
        already_stored={"customers": 0, "orders": 2, "labels": 1},
        rejected={("labels", "label differs from the stored label"): 1},
    )
    with database.connect() as connection:
        label = connection.exec_driver_sql(
            "SELECT order_id, batch_id, is_fraud FROM labels WHERE order_id = 'o1'"
        ).one()
    assert tuple(label) == ("o1", "base", 0)
    assert counts(database) == {"batches": 2, "customers": 3, "orders": 3, "labels": 3}


def test_the_writer_user_can_load_a_batch_but_not_change_stored_rows(
    database, write_batch
):
    name = f"t{uuid.uuid4().hex[:8]}"
    password = "a long enough test password"
    user = create_user(database, database.url.database, name, password, "writer")
    writer = create_engine(database.url.set(username=user, password=password))
    try:
        load_files(writer, *write_batch("base", [C1], [O1], [L1]))
        with pytest.raises(DBAPIError, match="denied"), writer.begin() as connection:
            connection.exec_driver_sql("UPDATE labels SET is_fraud = 1")
    finally:
        writer.dispose()
        with database.begin() as connection:
            connection.exec_driver_sql("DROP USER %s@'%%'", (user,))
    assert counts(database) == {"batches": 1, "customers": 1, "orders": 1, "labels": 1}
