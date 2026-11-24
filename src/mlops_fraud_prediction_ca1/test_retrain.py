import sys
from dataclasses import replace
from types import ModuleType, SimpleNamespace

import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from . import config, retrain
from .loading import load_files
from .retrain import EvaluationInputs, ModelSpec, validate_manifest
from .test_rds_reads import customer, order


def _candidate():
    return make_pipeline(
        ColumnTransformer(
            [("kind", OneHotEncoder(handle_unknown="ignore"), ["collect_type"])],
            remainder="drop",
        ),
        LogisticRegression(),
    )


def _rows():
    return pd.DataFrame(
        {
            "order_id": ["a", "b", "c", "d"],
            "order_date": ["2023-04-01"] * 4,
            "batch_id": ["base", "new", "new", "new"],
            "collect_type": ["pickup", "delivery", "pickup", "delivery"],
            "payment_method": "balance",
            "order_value": [1, 2, 3, 4],
            "num_items_ordered": 1,
            "is_fraud": [0, 1, 0, 1],
        }
    )


@pytest.fixture
def inputs():
    validation = _rows()
    return EvaluationInputs(
        [
            ModelSpec(
                "first", "engineer-a", _candidate, {"logisticregression__C": 0.25}
            ),
            ModelSpec("last", "engineer-b", _candidate, {"logisticregression__C": 0.5}),
        ],
        ["engineer-a", "engineer-b"],
        validation,
        1,
        "champion-fixture",
        object(),
    )


def _snapshot():
    return _rows().assign(
        order_date="2023-01-01", order_id="training", is_fraud=[0, 1, 0, None]
    ), {"excluded_date_rows": 3, "excluded_late_rows": 2}


def test_cli_delegates_training_and_continues_after_failure(inputs, monkeypatch):
    def broken():
        raise RuntimeError("candidate failed")

    manifest = replace(
        inputs,
        models=[
            inputs.models[0],
            ModelSpec("broken", "engineer-a", broken),
            inputs.models[1],
        ],
    )
    module = ModuleType("representative_inputs")
    module.build = lambda: manifest
    monkeypatch.setitem(sys.modules, module.__name__, module)
    training = ModuleType(f"{retrain.__package__}.training")
    training.score_labelled_rows = _score_champion
    calls = []

    def train(estimator, **kwargs):
        result = SimpleNamespace(report=_score(1, 1.0), fit_seconds=0.1)
        calls.append((estimator, kwargs, kwargs["extra_metrics"](result)))
        return "candidate-run", result

    training.train = train
    monkeypatch.setitem(sys.modules, training.__name__, training)
    monkeypatch.setattr(
        config,
        "load_settings",
        lambda: SimpleNamespace(mlflow_experiment_name="fixture"),
    )
    monkeypatch.setattr(
        config, "create_db_engine", lambda _: SimpleNamespace(dispose=lambda: None)
    )
    snapshots = []

    def read(engine):
        snapshots.append(engine)
        return _snapshot()

    monkeypatch.setattr(retrain, "read_training_snapshot", read)
    assert (
        retrain.main(
            ["new", "representative_inputs:build", "--etl-run-id", "invocation"]
        )
        == 1
    )
    assert len(snapshots) == 1

    assert [kwargs["run_name"] for _, kwargs, _ in calls] == [
        "first-new-invocation",
        "last-new-invocation",
    ]
    assert calls[0][1]["labelled_windows"][0] is calls[1][1]["labelled_windows"][0]
    for (estimator, kwargs, metrics), expected_c in zip(
        calls, (0.25, 0.5), strict=True
    ):
        assert estimator.get_params()["logisticregression__C"] == expected_c
        passed_train, passed_validation = kwargs["labelled_windows"]
        assert len(passed_train) == 3
        pd.testing.assert_frame_equal(passed_validation, inputs.validation_rows)
        assert kwargs["k"] == 1
        assert kwargs["max_model_size_bytes"] == 52_428_800
        assert kwargs["tags"] == {"champion_eligible": "False"}
        assert kwargs["extra_params"]["etl_run_id"] == "invocation"
        assert kwargs["extra_params"]["champion_run_id"] == "champion-fixture"
        assert metrics["train_old_rows"] == 1
        assert metrics["train_new_rows"] == 2
        assert metrics["excluded_rows"] == 4
        assert metrics["excluded_late_rows"] == 2
        assert metrics["average_precision_delta"] == 0.5


def _score(caught, ap):
    return SimpleNamespace(
        at_k=SimpleNamespace(fraud_caught=caught), average_precision=ap
    )


def _score_champion(model, rows, *, k):
    assert len(rows) == 4
    assert k == 1
    return _score(0, 0.5)


def test_candidate_size_failures_and_unique_ids(inputs, monkeypatch):
    monkeypatch.setattr(retrain, "read_training_snapshot", lambda engine: _snapshot())

    def reject_oversize(estimator, *, max_model_size_bytes, **kwargs):
        assert max_model_size_bytes == 52_428_800
        raise ValueError("model exceeds size limit")

    def run():
        return retrain.run_evaluation_mode(
            None,
            inputs,
            batch_id="new",
            train_candidate=reject_oversize,
            score_rows=_score_champion,
        )

    first, second = run(), run()
    assert first.etl_run_id != second.etl_run_id
    assert not first.results
    assert first.exit_code == second.exit_code == 1
    assert set(first.failures) == {"first", "last"}
    assert all("size limit" in failure for failure in first.failures.values())


def test_manifest_requires_every_expected_engineer(inputs):
    with pytest.raises(ValueError, match="engineer-c"):
        validate_manifest(inputs.models, [*inputs.engineers, "engineer-c"])
    with pytest.raises(ValueError, match="unique"):
        validate_manifest([inputs.models[0]] * 2, inputs.engineers)


def test_rejects_validation_outside_frozen_window(inputs):
    wrong = replace(
        inputs, validation_rows=inputs.validation_rows.assign(order_date="2023-04-20")
    )
    with pytest.raises(ValueError, match="frozen"):
        retrain.run_evaluation_mode(
            None, wrong, batch_id="new", train_candidate=None, score_rows=None
        )


def test_training_snapshot_keeps_holdout_rows_in_db(database, write_batch):
    load_files(
        database,
        *write_batch(
            "base", [customer(1)], [order("old", "2023-01-05")], ["MY,old,c1,0"]
        ),
    )
    dates = {
        "new": "2023-03-26",
        "validation": "2023-03-27",
        "test": "2023-04-20",
        "late": "2023-05-11",
        "early": "2022-11-11",
    }
    load_files(
        database,
        *write_batch(
            "new",
            [customer(2)],
            [order(key, day) for key, day in dates.items()],
            [f"MY,{key},c1,1" for key in dates],
        ),
    )
    rows, counts = retrain.read_training_snapshot(database)
    assert rows[["order_id", "batch_id"]].values.tolist() == [
        ["old", "base"],
        ["new", "new"],
    ]
    assert counts == {"excluded_date_rows": 4, "excluded_late_rows": 3}
