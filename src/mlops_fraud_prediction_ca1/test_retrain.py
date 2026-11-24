import pickle
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import mlflow
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from . import config, retrain, retrain_adapter
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
    training = validation.assign(order_date="2023-02-01", order_id="historical")
    champion = DummyClassifier(strategy="prior").fit(
        retrain_adapter.model_inputs(training), training.is_fraud
    )

    def evaluate(rows, scores, k):
        pd.testing.assert_frame_equal(rows, validation)
        return SimpleNamespace(
            at_k=SimpleNamespace(fraud_caught=int(rows.is_fraud.iloc[scores.argmax()])),
            average_precision=average_precision_score(rows.is_fraud, scores),
        )

    return EvaluationInputs(
        [
            ModelSpec(
                "first", "engineer-a", _candidate, {"logisticregression__C": 0.25}
            ),
            ModelSpec("last", "engineer-b", _candidate, {"logisticregression__C": 0.5}),
        ],
        ["engineer-a", "engineer-b"],
        validation,
        evaluate,
        1,
        "champion-fixture",
        champion,
    )


def _snapshot():
    return _rows().assign(
        order_date="2023-01-01", order_id="training", is_fraud=[0, 1, 0, None]
    ), {"excluded_date_rows": 3, "excluded_late_rows": 2}


def test_cli_publishes_comparisons_and_continues_after_failure(
    inputs, tmp_path, monkeypatch
):
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
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'tracking.db'}")
    monkeypatch.chdir(tmp_path)
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

    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name("fixture")
    runs = {
        run.info.run_name: run for run in client.search_runs([experiment.experiment_id])
    }
    assert set(runs) == {"first-new-invocation", "last-new-invocation"}
    for name, expected_c in (("first", 0.25), ("last", 0.5)):
        run = runs[f"{name}-new-invocation"]
        assert run.info.status == "FINISHED"
        assert run.data.params["etl_run_id"] == "invocation"
        assert run.data.params["champion_run_id"] == "champion-fixture"
        assert run.data.metrics["train_old_rows"] == 1
        assert run.data.metrics["train_new_rows"] == 2
        assert run.data.metrics["excluded_rows"] == 4
        assert run.data.metrics["excluded_late_rows"] == 2
        assert run.data.metrics["average_precision_delta"] == 0.5
        artifact = Path(
            client.download_artifacts(run.info.run_id, "model.pkl")
        ).read_bytes()
        assert len(artifact) == run.data.metrics["model_size_bytes"]
        assert (
            pickle.loads(artifact).get_params()["logisticregression__C"] == expected_c
        )


def test_oversize_candidates_never_reach_logger(inputs, monkeypatch):
    monkeypatch.setattr(retrain_adapter, "MAX_ARTIFACT_BYTES", 1)
    logged = []
    monkeypatch.setattr(retrain, "read_training_snapshot", lambda engine: _snapshot())

    def run():
        return retrain.run_evaluation_mode(
            None, inputs, batch_id="new", log_run=lambda *args: logged.append(args)
        )

    first, second = run(), run()
    assert first.etl_run_id != second.etl_run_id
    assert not logged
    assert first.exit_code == second.exit_code == 1
    assert set(first.failures) == {"first", "last"}
    assert all("50 MB" in failure for failure in first.failures.values())


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
        retrain_adapter.make_fit_and_log(wrong, None)


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
