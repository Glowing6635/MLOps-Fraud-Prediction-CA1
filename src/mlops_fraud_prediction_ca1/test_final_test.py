"""Tests for the final test-window scoring on small made-up base files and a local MLflow
store in a temporary folder."""

import mlflow
import mlflow.sklearn
import pandas as pd
import pytest
import skops.io
from mlflow.exceptions import MlflowException

from mlops_fraud_prediction_ca1 import final_test
from mlops_fraud_prediction_ca1.final_test import score_final_test
from mlops_fraud_prediction_ca1.models import baseline_model
from mlops_fraud_prediction_ca1.top_k import K
from mlops_fraud_prediction_ca1.training import fit_and_score

# Each test-window day: (order value, fraud, customer's first order on the order date).
# The rule flags a value over X = 13.36 from a customer whose first order is that day.
TEST_DAY = [(100.0, 1, True), (5.0, 1, True), (50.0, 0, True), (10.0, 0, False)]
TRAIN_AND_VALIDATION_ORDERS = 14


def _settings(**changes):
    settings = {
        "model_name": '"fraud-candidate"',
        "model_version": '"1"',
        "mode": '"as_trained"',
        "k": str(K),
        "loyal_min_prior_orders": "2",
        "threshold_x": "13.36",
    } | changes
    return "".join(f"{name} = {value}\n" for name, value in settings.items())


@pytest.fixture
def files(tmp_path, monkeypatch):
    """Base files with a test window, an enabled time-split configuration, an empty
    local MLflow store and a clean git tree; returns the folder and configuration."""
    orders, labels, customers = [], [], []

    def add(order_id, date, value, fraud, new_customer):
        orders.append(
            {
                "country_code": "MY",
                "order_id": order_id,
                "collect_type": "delivery",
                "payment_method": "credit card",
                "order_value": value,
                "num_items_ordered": 1,
                "refund_value": 0,
                "order_date": date,
            }
        )
        labels.append(
            {
                "country_code": "MY",
                "order_id": order_id,
                "customer_id": f"c{order_id}",
                "is_fraud": fraud,
            }
        )
        customers.append(
            {
                "country_code": "MY",
                "customer_id": f"c{order_id}",
                "mobile_verified": "true",
                "num_orders_last_50days": 1,
                "num_cancelled_orders_last_50days": 0,
                "num_refund_orders_last_50days": 0,
                "total_payment_last_50days": 10.0,
                "num_associated_customers": 1,
                "first_order_datetime": f"{date if new_customer else '2022-01-01'}"
                " 10:00:00",
            }
        )

    for i in range(10):
        add(f"train{i}", f"2022-12-{i + 1:02}", 20.0 + 60 * (i % 2), i % 2, False)
    for i in range(4):
        add(f"validation{i}", "2023-03-28", 20.0 + 60 * (i % 2), i % 2, False)
    for day in ("2023-04-18", "2023-04-19"):
        for i, row in enumerate(TEST_DAY):
            add(f"test{day[-2:]}{i}", day, *row)
    for name, rows in (
        ("customers", customers),
        ("orders", orders),
        ("labels", labels),
    ):
        pd.DataFrame(rows).to_csv(tmp_path / f"{name}_base.csv", index=False)

    config = tmp_path / "time_split.toml"
    config.write_text(
        (final_test.FREEZE_PATH.parent / "time_split.toml")
        .read_text(encoding="utf-8")
        .replace("final_evaluation_enabled = false", "final_evaluation_enabled = true"),
        encoding="utf-8",
        newline="\n",
    )
    monkeypatch.chdir(tmp_path)  # MLflow stores artifacts under the working folder.
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "final-test")
    monkeypatch.setattr(
        final_test, "_git", lambda *args: "" if args[0] == "status" else "abc123"
    )
    return tmp_path, config


@pytest.fixture
def store(files):
    """`files`, with `fraud-candidate` version 1 registered: a model fitted on the
    training window."""
    model = fit_and_score(baseline_model(), files[0], k=K, seed=0).model
    mlflow.set_experiment("final-test")
    with mlflow.start_run():
        mlflow.log_param("seed", 0)
        mlflow.sklearn.log_model(
            model,
            name="model",
            registered_model_name="fraud-candidate",
            skops_trusted_types=skops.io.get_untrusted_types(
                data=skops.io.dumps(model)
            ),
        )
    return files


def _score(files, **changes):
    folder, config = files
    freeze = folder / "final_test.toml"
    freeze.write_text(_settings(**changes), encoding="utf-8", newline="\n")
    return score_final_test(folder, freeze_path=freeze, config_path=config)


def _final_runs():
    return mlflow.search_runs(
        experiment_names=["final-test"],
        filter_string="tags.final_test = 'true'",
        output_format="list",
    )


def test_the_model_and_rule_are_scored_once(store):
    run_id, _ = _score(store)

    [run] = _final_runs()
    assert run.info.run_id == run_id
    assert run.data.tags["mode"] == "as_trained"
    metrics = run.data.metrics
    assert {name: metrics[name] for name in metrics if "at_k" in name} == {
        "fraud_caught_at_k": 4,
        "fraud_catchable_at_k": 4,
        "selected_at_k": 8,
        "total_fraud_at_k": 4,
        "precision_at_k": 0.5,
        "recall_at_k": 1.0,
        "rule_selected_at_k": 4,
        "rule_fraud_caught_at_k": 2,
        "rule_precision_at_k": 0.5,
        "rule_recall_at_k": 0.5,
    }
    with pytest.raises(RuntimeError, match=f"already scored, in run {run_id}"):
        _score(store)
    assert len(_final_runs()) == 1


def test_a_refit_uses_only_training_and_validation_rows(store):
    _score(store, mode='"refit"')

    [run] = _final_runs()
    assert run.data.tags["mode"] == "refit"
    assert run.data.metrics["refit_train_orders"] == TRAIN_AND_VALIDATION_ORDERS


@pytest.mark.parametrize(
    ("change", "error", "message"),
    [
        ({"k": "50"}, ValueError, f"the code uses {K}"),
        # The store has no registered model.
        ({}, MlflowException, "fraud-candidate"),
        ({"mode": '"latest"'}, ValueError, "as_trained"),
        ("dirty", RuntimeError, "uncommitted changes"),
        ("locked", ValueError, "final_evaluation_enabled"),
    ],
)
def test_a_failed_check_leaves_no_final_run(files, monkeypatch, change, error, message):
    if change == "dirty":
        monkeypatch.setattr(
            final_test, "_git", lambda *args: " M x.py" if args[0] == "status" else "a"
        )
    if change == "locked":
        config = files[1]
        config.write_text(
            config.read_text(encoding="utf-8").replace(
                "final_evaluation_enabled = true", "final_evaluation_enabled = false"
            ),
            encoding="utf-8",
            newline="\n",
        )
    with pytest.raises(error, match=message):
        _score(files, **(change if isinstance(change, dict) else {}))
    assert _final_runs() == []
