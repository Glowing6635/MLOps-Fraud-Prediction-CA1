import runpy
import sys
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from types import ModuleType, SimpleNamespace

import mlflow
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from . import champion, config, retrain, training
from .loading import load_files
from .retrain import EvaluationInputs, ModelSpec, validate_manifest
from .test_rds_reads import customer, order


def _candidate():
    return make_pipeline(
        ColumnTransformer(
            [("kind", OneHotEncoder(handle_unknown="ignore"), ["collect_type"])],
            remainder="drop",
        ),
        LogisticRegression(random_state=23),
    )


def _rows():
    return pd.DataFrame(
        {
            "order_id": ["a", "b", "c", "d"],
            "country_code": "MY",
            "customer_id": ["c1", "c2", "c3", "c4"],
            "order_date": ["2023-04-01"] * 4,
            "batch_id": ["base", "new", "new", "new"],
            "collect_type": ["pickup", "delivery", "pickup", "delivery"],
            "payment_method": "balance",
            "order_value": [Decimal("1.00"), Decimal("2.50"), Decimal("3.00"), None],
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
        order_date="2023-01-01",
        order_id=["t1", "t2", "t3", "t4"],
        is_fraud=[0, 1, 0, None],
    ), {"excluded_date_rows": 3, "excluded_late_rows": 2}


@pytest.fixture
def tracker(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "retrain-test")


def test_cli_trains_and_logs_real_models_after_candidate_failure(
    inputs, monkeypatch, tracker, capsys
):
    def broken():
        raise RuntimeError("candidate failed")

    champion = DummyClassifier(strategy="prior").fit([[0], [1]], [0, 1])
    manifest = replace(
        inputs,
        champion_model=champion,
        models=[
            inputs.models[0],
            ModelSpec("broken", "engineer-a", broken),
            inputs.models[1],
        ],
    )
    module = ModuleType("representative_inputs")
    module.build = lambda: manifest
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(config, "load_settings", lambda: SimpleNamespace())
    disposed = []
    engine = SimpleNamespace(dispose=lambda: disposed.append(True))
    monkeypatch.setattr(config, "create_db_engine", lambda _: engine)
    snapshots = []

    def read(engine):
        snapshots.append(engine)
        return _snapshot()

    monkeypatch.setattr(retrain, "read_training_snapshot", read)
    cli = Path(sys.executable).with_name("retrain-batch")
    monkeypatch.setattr(
        sys,
        "argv",
        [str(cli), "new", "representative_inputs:build", "--etl-run-id", "invocation"],
    )
    with pytest.raises(SystemExit) as status:
        runpy.run_path(str(cli), run_name="__main__")
    assert status.value.code == 1
    assert snapshots == [engine]
    assert disposed == [True]
    assert "FAILED broken-new-invocation: RuntimeError: candidate failed" in (
        capsys.readouterr().out
    )

    runs = mlflow.search_runs(experiment_names=["retrain-test"], output_format="list")
    runs = {run.info.run_name: run for run in runs}
    assert set(runs) == {"first-new-invocation", "last-new-invocation"}
    for name, expected_c in (("first", "0.25"), ("last", "0.5")):
        run = runs[f"{name}-new-invocation"]
        params, metrics = run.data.params, run.data.metrics
        assert run.info.status == "FINISHED"
        assert run.data.tags["champion_eligible"] == "False"
        tags = run.data.tags
        assert params["data_source"] == "labelled_windows"
        assert params["logisticregression__C"] == expected_c
        assert params["logisticregression__random_state"] == "23"
        assert params["k"] == "1"
        assert tags["etl_run_id"] == "invocation"
        assert tags["champion_run_id"] == "champion-fixture"
        assert len(tags["validation_sha256"]) == 64
        assert metrics["train_orders"] == 3
        assert metrics["validation_orders"] == 4
        assert metrics["train_old_rows"] == 1
        assert metrics["train_new_rows"] == 2
        assert metrics["excluded_rows"] == 4
        assert metrics["excluded_late_rows"] == 2
        assert metrics["average_precision_delta"] == 0.5
        assert 0 < metrics["model_size_bytes"] <= retrain.MAX_ARTIFACT_BYTES
        model = mlflow.sklearn.load_model(f"runs:/{run.info.run_id}/model")
        assert (
            training.score_labelled_rows(
                model, inputs.validation_rows, k=1
            ).at_k.fraud_caught
            == 1
        )


def test_shared_harness_stops_oversized_model_before_upload(inputs, tracker):
    rows, _ = _snapshot()
    with pytest.raises(ValueError, match="over the limit"):
        training.train(
            _candidate(),
            k=1,
            seed=42,
            labelled_windows=(rows.dropna(subset=["is_fraud"]), inputs.validation_rows),
            max_model_size_bytes=1,
        )

    assert mlflow.search_runs(experiment_names=["retrain-test"]).empty


def test_artifact_limit_matches_the_champion_rule_and_rejects_52_4_mb(
    inputs, tracker, monkeypatch
):
    assert retrain.MAX_ARTIFACT_BYTES == champion.SIZE_NOTE_BYTES == 50_000_000
    rows, _ = _snapshot()
    original = Path.stat

    class _Stat:
        st_size = 52_400_000
        st_mode = original(Path(".")).st_mode

    monkeypatch.setattr(
        training.Path, "is_file", lambda self: self.name == "model.skops"
    )
    monkeypatch.setattr(training.Path, "stat", lambda self, **kwargs: _Stat())
    with pytest.raises(ValueError, match="over the limit"):
        training.train(
            _candidate(),
            k=1,
            seed=42,
            labelled_windows=(rows.dropna(subset=["is_fraud"]), inputs.validation_rows),
            max_model_size_bytes=retrain.MAX_ARTIFACT_BYTES,
        )


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
        assert max_model_size_bytes == 50_000_000
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
