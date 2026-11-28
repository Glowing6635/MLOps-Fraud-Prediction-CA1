import mlflow
import numpy as np
import pandas as pd
import pytest

from mlops_fraud_prediction_ca1 import training
from mlops_fraud_prediction_ca1.diagnostics import (
    ORDER_VALUE_BANDS,
    build_diagnostics,
    order_value_band,
)
from mlops_fraud_prediction_ca1.models import boosted_trees_model
from mlops_fraud_prediction_ca1.test_training import data_dir  # noqa: F401
from mlops_fraud_prediction_ca1.top_k import evaluate_top_k


def _window():
    keys = pd.DataFrame(
        {
            "order_date": ["2023-03-28"] * 6,
            "country_code": ["PK", "PK", "PK", "MY", "MY", "MY"],
            "order_id": list("abcdef"),
            "is_fraud": [1, 1, 0, 0, 1, 0],
        }
    )
    features = pd.DataFrame(
        {
            "payment_method": ["cod", "cod", "card", "card", None, "cod"],
            "order_value": [1.0, 3.0, 12.99, 30.0, np.nan, 7.9],
        }
    )
    return keys, features, [0.9, 0.8, 0.7, 0.6, 0.5, 0.1]


def test_order_value_bands_include_the_lower_bound_and_name_missing():
    bands = order_value_band(pd.Series([0, 2.99, 3, 29.99, 30, 1e6, np.nan]))
    assert list(bands) == ["[0,3)", "[0,3)", "[3,5)", "[13,30)", "[30,∞)"] + [
        "[30,∞)",
        "missing",
    ]
    assert len(ORDER_VALUE_BANDS) == 6


def test_confusion_and_slices_agree_with_the_evaluator():
    keys, features, scores = _window()
    diagnostics = build_diagnostics(keys, features, scores, k=2)
    report = evaluate_top_k(keys, scores, 2)
    confusion = diagnostics.confusion.set_index("at_k_2")
    assert confusion.loc["selected", "actual_fraud"] == report.at_k.fraud_caught == 2
    assert confusion.to_numpy().sum() == len(keys)
    assert confusion.loc["not_selected", "actual_fraud"] == 1
    assert confusion.loc["selected", "actual_not_fraud"] == 0
    for name, table in diagnostics.slices.items():
        assert table["orders"].sum() == len(keys), name
        assert table["caught"].sum() == report.at_k.fraud_caught, name
        assert table["missed_fraud"].sum() == 1, name
        assert table["selected"].sum() == report.at_k.selected, name
    by_band = diagnostics.slices["order_value_band"].set_index("order_value_band")
    assert list(by_band.index) == [
        "[0,3)",
        "[3,5)",
        "[5,8)",
        "[8,13)",
        "[30,∞)",
        "missing",
    ]
    assert by_band.loc["missing", "missed_fraud"] == 1
    by_payment = diagnostics.slices["payment_method"].set_index("payment_method")
    assert "missing" in by_payment.index
    curve = diagnostics.curve
    assert curve.recall.iloc[0] == 1 and curve.recall.iloc[-1] == 0


def test_a_slice_with_nothing_selected_has_undefined_precision():
    keys, features, scores = _window()
    table = build_diagnostics(keys, features, scores, k=1).slices["country"]
    assert table.set_index("country").loc["MY", "precision_at_k"] != 0
    assert pd.isna(table.set_index("country").loc["MY", "precision_at_k"])


@pytest.mark.parametrize("history", [False, True])
def test_v1_and_v2_runs_log_the_same_diagnostics(
    data_dir,  # noqa: F811
    tmp_path,
    monkeypatch,
    history,
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "harness-test")
    run_id, result = training.train(
        boosted_trees_model(), data_dir, k=5, seed=0, history=history
    )
    client = mlflow.MlflowClient()
    logged = {a.path for a in client.list_artifacts(run_id, "diagnostics")}
    assert logged == {
        "diagnostics/precision_recall_curve.json",
        "diagnostics/confusion_matrix_at_k5.json",
        "diagnostics/errors_by_country.json",
        "diagnostics/errors_by_payment_method.json",
        "diagnostics/errors_by_order_value_band.json",
    }
    assert "[30,∞)" in mlflow.get_run(run_id).data.tags["order_value_bands"]
    caught = result.diagnostics.slices["country"]["caught"].sum()
    assert caught == result.report.at_k.fraud_caught
