from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from .retrain import ModelSpec, retrain
from .retrain_adapter import compare_with_champion, make_fit_and_log, model_inputs


def _lr():
    return make_pipeline(
        make_column_transformer(
            (
                SimpleImputer(strategy="median"),
                make_column_selector(dtype_include="number"),
            ),
            (
                make_pipeline(
                    SimpleImputer(strategy="constant", fill_value="missing"),
                    OneHotEncoder(handle_unknown="ignore"),
                ),
                make_column_selector(dtype_include=object),
            ),
        ),
        LogisticRegression(),
    )


def _rows(n, offset=0):
    rng = np.random.default_rng(offset)
    return pd.DataFrame(
        {
            "order_date": pd.date_range("2023-01-01", periods=n),
            "country_code": "MY",
            "order_id": [f"o{offset}-{i}" for i in range(n)],
            "collect_type": ["a", "b"] * (n // 2),
            "payment_method": ["x", "y", None, "x"] * (n // 4),
            "order_value": rng.random(n) * 100,
            "num_items_ordered": rng.integers(1, 5, n),
            "is_fraud": [0, 1] * (n // 2),
        }
    )


def test_model_inputs_use_only_allowed_columns_with_nan_for_missing():
    x = model_inputs(_rows(8))
    assert list(x.columns) == [
        "collect_type",
        "payment_method",
        "order_value",
        "num_items_ordered",
    ]
    assert x["payment_method"].isna().sum() == 2 and x["order_value"].dtype == "float64"


def test_adapter_runs_through_retrain_and_never_scores_training_rows():
    seen = {}

    def fake_eval(orders, scores, k):
        seen["orders"] = orders
        return SimpleNamespace(
            at_k=SimpleNamespace(fraud_caught=3), average_precision=0.5
        )

    logged = []
    fit = make_fit_and_log(
        k=2,
        seed=0,
        validation_rows=_rows(8, 1),
        evaluate_top_k=fake_eval,
        log_run=lambda name, p, m, model: logged.append(name) or "run1",
    )
    models = [
        ModelSpec("lr", "x", _lr),
        ModelSpec("dummy", "y", lambda: DummyClassifier(strategy="prior")),
    ]
    all_rows = pd.concat([_rows(8, 0), _rows(8, 1).assign(order_date="2023-06-01")])
    report = retrain(
        models,
        batch_id="B",
        read_rows=lambda: all_rows,
        validation_start="2023-03-27",
        fit_and_log=fit,
        champion_comparison=lambda m: compare_with_champion(m, None),
        etl_run_id="E",
    )
    assert report.exit_code == 0 and logged == ["lr-B-E", "dummy-B-E"]
    assert (
        seen["orders"]["order_id"].str.startswith("o1-").all()
    )  # validation rows only
    assert report.results["lr"]["train_orders"] == 8


def test_champion_comparison_reports_only():
    m = {"fraud_caught_at_k": 5, "average_precision": 0.4}
    assert compare_with_champion(m, None)["better"] is None
    c = {"fraud_caught_at_k": 4, "average_precision": 0.5, "run_id": "c"}
    out = compare_with_champion(m, c)
    assert out["fraud_caught_at_k_delta"] == 1 and out["better"] is True
