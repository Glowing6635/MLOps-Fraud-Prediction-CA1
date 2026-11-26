"""Roster and release-mode tests for PBI #25."""

import pandas as pd
import pytest

from . import retrain, retrain_roster
from .retrain import EvaluationInputs, ModelSpec
from .test_retrain import _candidate, _rows, _score, _score_champion


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


def _run(mode, rows, calls, log=lambda _: None):
    def train(estimator, **kwargs):
        from types import SimpleNamespace

        result = SimpleNamespace(report=_score(1, 1.0), fit_seconds=0.1)
        calls.append((kwargs, kwargs["extra_metrics"](result)))
        return "run", result

    return retrain.retrain(
        _inputs(),
        batch_id="late",
        etl_run_id="etl1",
        read_rows=lambda: (rows, {"excluded_date_rows": 0, "excluded_late_rows": 0}),
        train_candidate=train,
        score_rows=_score_champion,
        log=log,
        mode=mode,
    )


def _release_rows():
    return pd.DataFrame(
        {
            "order_id": ["a", "b", "c"],
            "order_date": ["2023-04-01", "2023-06-01", "2023-06-02"],
            "batch_id": ["base", "late", "late"],
            "is_fraud": [0, 1, 0],
        }
    )


def test_release_mode_includes_later_batch_and_labels_scores_in_sample():
    calls, lines = [], []
    report = _run("release", _release_rows(), calls, lines.append)
    kwargs, metrics = calls[0]
    assert report.exit_code == 0
    assert len(kwargs["labelled_windows"][0]) == 3
    assert kwargs["tags"]["retrain_mode"] == "release"
    assert kwargs["tags"]["scores_out_of_sample"] == "False"
    assert kwargs["tags"]["mlflow.runName"] == "m-late-etl1"
    assert metrics["train_new_rows"] == 2
    assert "average_precision_delta" not in metrics
    assert "average_precision" not in report.results["m"]
    assert "mode=release" in lines[0]


def test_release_mode_refuses_a_batch_with_no_eligible_rows():
    rows = _release_rows().assign(batch_id="base")
    with pytest.raises(ValueError, match="silently exclude"):
        _run("release", rows, [])


def test_evaluation_mode_keeps_out_of_sample_comparison():
    calls = []
    _run("evaluation", _release_rows(), calls)
    kwargs, metrics = calls[0]
    assert kwargs["tags"]["scores_out_of_sample"] == "True"
    assert metrics["average_precision_delta"] == 0.5
    assert "train_orders" not in metrics


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="Unknown retraining mode"):
        _run("promote", _release_rows(), [])


def test_release_read_requires_holdout_off(monkeypatch):
    from . import rds_reads

    with pytest.raises(ValueError, match="holdout setting off"):
        rds_reads.read_release_snapshot(None)
