"""Roster and release-mode tests for PBI #25."""

from dataclasses import replace
from types import SimpleNamespace

import pandas as pd
import pytest

from mlops_fraud_prediction_ca1 import rds_reads, splitting
from mlops_fraud_prediction_ca1.loading import load_files

from . import retrain, retrain_roster
from .retrain import EvaluationInputs, ModelSpec
from .test_rds_reads import customer, order
from .test_retrain import (
    _candidate,
    _rows,
    _score,
    _score_champion,
    check_train_call,
)


def test_default_roster_loads_every_engineer_with_fresh_estimators():
    engineers, specs = retrain_roster.load_roster()
    assert {spec.owner for spec in specs} == set(engineers)
    assert specs[0].build() is not specs[0].build()
    assert retrain.TRAIN_AND_LOG_SECONDS == 85


def _toml(tmp_path, text):
    path = tmp_path / "roster.toml"
    path.write_text(text)
    return path


def test_roster_rejects_missing_engineer_and_unknown_factory(tmp_path):
    entry = 'name="a"\nowner="x"\nfactory="baseline_model"\nparams={}\n'
    with pytest.raises(ValueError, match="differ from roster"):
        retrain_roster.load_roster(
            _toml(tmp_path, f'engineers=["x","y"]\n[[models]]\n{entry}')
        )
    bad = 'name="a"\nowner="x"\nfactory="os.system"\nparams={}\n'
    with pytest.raises(ValueError, match="Unknown model factory"):
        retrain_roster.load_roster(
            _toml(tmp_path, f'engineers=["x"]\n[[models]]\n{bad}')
        )


def _inputs():
    return EvaluationInputs(
        [ModelSpec("m", "eng", _candidate)],
        ["eng"],
        _rows(),
        1,
        "champion-run",
        object(),
    )


def _run(mode, read_rows, calls, log=lambda _: None):
    def train(estimator, **kwargs):
        check_train_call(estimator, kwargs)
        result = SimpleNamespace(report=_score(1, 1.0), fit_seconds=0.1)
        calls.append((kwargs, kwargs["extra_metrics"](result)))
        return "run", result

    return retrain.retrain(
        _inputs(),
        batch_id="late",
        etl_run_id="etl1",
        read_rows=read_rows,
        train_candidate=train,
        score_rows=_score_champion,
        log=log,
        mode=mode,
    )


def _static(rows):
    return lambda: (rows, {"excluded_date_rows": 0, "excluded_late_rows": 0})


def _release_rows():
    return pd.DataFrame(
        {
            "order_id": ["x", "y", "z"],
            "order_date": ["2023-04-01", "2023-06-01", "2023-06-02"],
            "batch_id": ["base", "late", "late"],
            "is_fraud": [0, 1, 0],
        }
    )


@pytest.fixture
def open_holdout(monkeypatch):
    """Turn the #6 holdout setting off for the reader (release mode needs it)."""
    policy = replace(splitting.load_policy(), test_locked=False)
    monkeypatch.setattr(rds_reads, "load_policy", lambda *a, **k: policy)
    monkeypatch.setattr(splitting, "load_policy", lambda *a, **k: policy)


def _load(database, write_batch):
    load_files(
        database,
        *write_batch(
            "base",
            [customer(1)],
            [order("old", "2023-01-05"), order("ancient", "2022-10-01")],
            ["MY,old,c1,0", "MY,ancient,c1,0"],
        ),
    )
    load_files(
        database,
        *write_batch(
            "late",
            [customer(2)],
            [order("fresh", "2023-06-01"), order("held", "2023-04-20")],
            ["MY,fresh,c1,1", "MY,held,c1,0"],
        ),
    )


def test_release_through_reader_includes_later_batch_and_counts_old_rows(
    database, write_batch, open_holdout
):
    _load(database, write_batch)
    calls, lines = [], []
    report = _run(
        "release",
        lambda: rds_reads.read_release_snapshot(database),
        calls,
        lines.append,
    )
    kwargs, metrics = calls[0]
    train_rows = kwargs["labelled_windows"][0]
    assert report.exit_code == 0
    assert sorted(train_rows["order_id"]) == ["fresh", "held", "old"]
    assert report.counts["excluded_date_rows"] == 1  # "ancient", before train_start
    assert kwargs["tags"]["retrain_mode"] == "release"
    assert kwargs["tags"]["scores_out_of_sample"] == "False"
    assert kwargs["tags"]["mlflow.runName"] == "m-late-etl1"
    assert metrics["train_new_rows"] == 2
    # AC8: release runs still log the champion comparison, flagged in-sample.
    assert metrics["champion_average_precision"] == 0.5
    assert metrics["average_precision_delta"] == 0.5
    assert metrics["beats_champion"] == 1
    assert report.results["m"]["scores_out_of_sample"] is False
    assert "mode=release" in lines[0]


def test_evaluation_through_reader_is_out_of_sample_and_disjoint(database, write_batch):
    _load(database, write_batch)
    calls = []
    _run("evaluation", lambda: retrain.read_training_snapshot(database), calls)
    kwargs, metrics = calls[0]
    train_rows, validation = kwargs["labelled_windows"]
    assert not set(train_rows["order_id"]) & set(validation["order_id"])
    train_dates = pd.to_datetime(train_rows["order_date"])
    validation_dates = pd.to_datetime(validation["order_date"])
    assert len(train_dates) and len(validation_dates)
    assert not set(train_dates) & set(validation_dates)
    assert train_dates.max() < validation_dates.min()
    assert kwargs["tags"]["scores_out_of_sample"] == "True"
    assert metrics["average_precision_delta"] == 0.5
    assert "train_orders" not in metrics


def test_release_mode_refuses_a_batch_with_no_eligible_rows():
    rows = _release_rows().assign(batch_id="base")
    with pytest.raises(ValueError, match="silently exclude"):
        _run("release", _static(rows), [])


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="Unknown retraining mode"):
        _run("promote", _static(_release_rows()), [])


def test_release_read_requires_holdout_off(monkeypatch):
    from . import rds_reads

    with pytest.raises(ValueError, match="holdout setting off"):
        rds_reads.read_release_snapshot(None)
