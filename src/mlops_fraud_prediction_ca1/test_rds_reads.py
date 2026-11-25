from mlops_fraud_prediction_ca1.loading import load_files
from mlops_fraud_prediction_ca1.rds_reads import read_labelled_orders

# Made-up rows. config/time_split.toml puts 2023-01-05 in training, 2023-04-01 in
# validation and 2023-04-20 in the test window.


def customer(orders_last_50days):
    return f"MY,c1,true,{orders_last_50days},0,0,10,0,2022-10-01 10:00:00"


def order(order_id, order_date):
    return f"MY,{order_id},pickup,balance,5,1,0,{order_date}"


def test_each_order_gets_the_customer_row_from_its_batch_or_the_latest_before_it(
    database, write_batch
):
    load_files(
        database,
        *write_batch(
            "base", [customer(1)], [order("o1", "2023-01-05")], ["MY,o1,c1,0"]
        ),
    )
    load_files(
        database,
        *write_batch("2", [customer(5)], [order("o2", "2023-01-06")], ["MY,o2,c1,1"]),
    )
    load_files(
        database,
        *write_batch("3", [customer(9)], [order("o3", "2023-01-07")], ["MY,o3,c1,0"]),
    )
    rows = read_labelled_orders(database, purpose="train")
    assert rows[["order_id", "customer_batch_id", "num_orders_last_50days"]].to_dict(
        "records"
    ) == [
        {"order_id": "o1", "customer_batch_id": "base", "num_orders_last_50days": 1.0},
        {"order_id": "o2", "customer_batch_id": "2", "num_orders_last_50days": 5.0},
        {"order_id": "o3", "customer_batch_id": "3", "num_orders_last_50days": 9.0},
    ]


def test_every_read_applies_the_holdout_setting(database, write_batch):
    dates = {"train": "2023-01-05", "validation": "2023-04-01", "test": "2023-04-20"}
    load_files(
        database,
        *write_batch(
            "base",
            [customer(1)],
            [order(window, day) for window, day in dates.items()],
            [f"MY,{window},c1,0" for window in dates],
        ),
    )
    read = {
        purpose: read_labelled_orders(database, purpose=purpose)["order_id"].tolist()
        for purpose in ("train", "validation", "analysis")
    }
    assert read == {
        "train": ["train"],
        "validation": ["validation"],
        "analysis": ["train", "validation"],
    }
