import re

import pytest

from mlops_fraud_prediction_ca1.rds_schema import create_user, statements

MONEY = ["order_value", "refund_value", "total_payment_last_50days"]


def tables():
    found = {}
    for statement in statements():
        name = re.match(r"CREATE TABLE IF NOT EXISTS (\w+)", statement)
        assert name, statement
        found[name.group(1)] = statement
    return found


def test_every_statement_is_safe_to_rerun_and_every_table_has_a_batch_id():
    found = tables()
    assert set(found) == {"batches", "customers", "orders", "labels"}
    for statement in found.values():
        assert re.search(r"^\s+batch_id VARCHAR", statement, re.MULTILINE)


def test_money_is_decimal_and_ids_compare_exactly():
    columns = "\n".join(tables().values())
    for name in MONEY:
        assert re.search(rf"^\s+{name} DECIMAL\(", columns, re.MULTILINE), name
    for name in ["order_id", "customer_id", "batch_id"]:
        definitions = re.findall(rf"^\s+{name} VARCHAR.*$", columns, re.MULTILINE)
        assert definitions and all("utf8mb4_bin" in d for d in definitions), name


def test_keys_match_the_data_analysis():
    found = tables()
    assert "PRIMARY KEY (customer_id, batch_id)" in found["customers"]
    assert "PRIMARY KEY (country_code, order_id)" in found["orders"]
    assert "PRIMARY KEY (country_code, order_id)" in found["labels"]


@pytest.mark.parametrize(
    ("name", "password"),
    [("Richard", "long enough pw"), ("a'b", "long enough pw"), ("richard", "short")],
)
def test_users_reject_bad_names_and_short_passwords_before_connecting(name, password):
    with pytest.raises(ValueError):
        create_user(None, "fraud", name, password, "writer")
