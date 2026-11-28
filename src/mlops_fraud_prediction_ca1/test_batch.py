import functools
import http.server
import re
import threading
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
import pytest
import skops.io
from mlflow import MlflowClient

from . import batch, models, retrain
from .config import SettingError, load_settings
from .training import ORDER_FEATURE_COLUMNS

URLS = [f"https://example.test/{name}?sig=private" for name in ("c", "o", "l")]


@pytest.mark.parametrize(
    "args", [URLS[:2], [*URLS, URLS[0]], ["customers.csv", *URLS[1:]]]
)
def test_bad_arguments_stop_before_configuration(args, monkeypatch, capsys):
    monkeypatch.setattr(batch, "load_settings", lambda: pytest.fail("read config"))
    with pytest.raises(SystemExit) as exc:
        batch.main(args)
    assert exc.value.code == 2
    assert "private" not in capsys.readouterr().err


def test_missing_configuration(monkeypatch, capsys):
    def missing():
        raise SettingError("Set DB_HOST")

    monkeypatch.setattr(batch, "load_settings", missing)
    assert batch.main(URLS) == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert "Set DB_HOST" in output.err


@pytest.fixture
def mlflow_settings(tmp_path, monkeypatch):
    """Settings for a made-up database and a throwaway MLflow store in `tmp_path`."""
    monkeypatch.chdir(tmp_path)  # MLflow stores artifacts under the working folder.
    tracking_uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "pipeline-test")
    monkeypatch.delenv("MLFLOW_EXPERIMENT_ID", raising=False)
    monkeypatch.setattr(mlflow.tracking._tracking_service.utils, "_tracking_uri", None)
    monkeypatch.setattr(mlflow.tracking._model_registry.utils, "_registry_uri", None)
    monkeypatch.setattr(mlflow.tracking.fluent, "_active_experiment_id", None)
    settings = load_settings(
        {
            "DB_HOST": "db.example",
            "DB_NAME": "practice",
            "DB_USER": "private-db-user",
            "DB_PASSWORD": "private-db-password",
            "MLFLOW_TRACKING_URI": tracking_uri,
            "MLFLOW_EXPERIMENT_NAME": "pipeline-test",
            "MLFLOW_TRACKING_USERNAME": "private-mlflow-user",
            "MLFLOW_TRACKING_PASSWORD": "private-mlflow-password",
        }
    )
    monkeypatch.setattr(batch, "load_settings", lambda: settings)
    return settings


@pytest.fixture
def file_server(tmp_path):
    """A local web server for the files in `tmp_path`, standing in for the batch links."""
    handler = functools.partial(
        http.server.SimpleHTTPRequestHandler, directory=str(tmp_path)
    )
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()


def _practice_batch(write_batch):
    """60 orders: 40 in the training window and 20 in validation.

    Order o1's label is 2, which cleaning stores as missing, so 39 train the model.
    """
    customers = [
        f"MY,c{i},true,1.0,0.0,0.0,10.0,0.0,2022-10-01 10:00:00" for i in range(10)
    ]
    orders, labels = [], []
    for i in range(60):
        day = f"2023-01-{1 + i % 28:02d}" if i < 40 else f"2023-04-{1 + i % 10:02d}"
        fraud = int(i % 4 == 0)
        value = 100.0 if fraud else 10.0
        orders.append(f"MY,o{i},Delivery,Credit Card,{value},1.0,0.0,{day}")
        labels.append(f"MY,o{i},c{i % 10},{2 if i == 1 else fraud}")
    return write_batch("b1", customers, orders, labels)


def _without_times(text: str) -> list[str]:
    return re.sub(r"\d+\.\d s", "T s", text).splitlines()


def _register_champion(tracking_uri: str) -> str:
    """Register a tiny `fraud-candidate@champion` in the test MLflow store."""
    fit_rows = pd.DataFrame(
        {
            "payment_method": ["credit card", "balance"],
            "collect_type": ["delivery", "pickup"],
            "order_value": [10.0, 100.0],
            "num_items_ordered": [1.0, 2.0],
        }
    )[list(ORDER_FEATURE_COLUMNS)]
    champion = models.baseline_model().fit(fit_rows, [0, 1])
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)
    with mlflow.start_run(run_name="champion-seed") as run:
        mlflow.sklearn.log_model(
            champion,
            name="model",
            registered_model_name="fraud-candidate",
            skops_trusted_types=skops.io.get_untrusted_types(
                data=skops.io.dumps(champion)
            ),
        )
        run_id = run.info.run_id
    MlflowClient(
        tracking_uri=tracking_uri, registry_uri=tracking_uri
    ).set_registered_model_alias("fraud-candidate", "champion", "1")
    return run_id


def test_a_batch_runs_every_stage_within_its_budget_and_reports(
    database, write_batch, file_server, mlflow_settings, monkeypatch, capsys
):
    champion_run_id = _register_champion(mlflow_settings.mlflow_tracking_uri)
    monkeypatch.setattr(batch, "create_db_engine", lambda settings: database)
    files = _practice_batch(write_batch)
    urls = [f"{file_server}/{Path(path).name}" for path in files]

    assert batch.main(urls) == 0

    experiment = mlflow.get_experiment_by_name("pipeline-test").experiment_id
    runs = mlflow.search_runs([experiment], output_format="list")
    runs = {r.data.tags["model"]: r for r in runs if "model" in r.data.tags}
    assert set(runs) == {"boosted-trees-v2", "baseline-logistic"}
    etl_run_id = runs["boosted-trees-v2"].data.tags["etl_run_id"]
    assert all(r.data.tags["batch_id"] == "b1" for r in runs.values())
    assert all(r.data.tags["champion_run_id"] == champion_run_id for r in runs.values())
    assert all(r.data.metrics["train_orders"] == 39 for r in runs.values())
    lines = _without_times(capsys.readouterr().out)
    assert "MLflow runs logged: 2" in lines
    assert (
        f"MLflow runs: {mlflow_settings.mlflow_tracking_uri}/#/experiments/"
        f"{experiment}?searchFilter=tags.etl_run_id%20%3D%20%27{etl_run_id}%27"
    ) in lines
    assert lines[-1] == "Total: T s of 180 s"


def test_run_batch_wires_retrain_offline_and_isolates_candidate_failures(
    tmp_path, write_batch, file_server, mlflow_settings, monkeypatch, capsys
):
    champion_run_id = _register_champion(mlflow_settings.mlflow_tracking_uri)
    files = _practice_batch(write_batch)
    urls = [f"{file_server}/{Path(path).name}" for path in files]

    all_rows = pd.DataFrame(
        {
            "order_id": [f"o{i}" for i in range(60)],
            "country_code": "MY",
            "customer_id": [f"c{i % 10}" for i in range(60)],
            "order_date": [
                f"2023-01-{1 + i % 28:02d}" if i < 40 else f"2023-04-{1 + i % 10:02d}"
                for i in range(60)
            ],
            "batch_id": "b1",
            "collect_type": "delivery",
            "payment_method": "credit card",
            "order_value": [100.0 if i % 4 == 0 else 10.0 for i in range(60)],
            "num_items_ordered": 1,
            "is_fraud": [None if i == 1 else int(i % 4 == 0) for i in range(60)],
        }
    )
    train_rows = all_rows.iloc[:40].copy()
    val_rows = all_rows.iloc[40:].copy()

    class _FakeEngine:
        def dispose(self):
            return None

    monkeypatch.setattr(batch, "create_db_engine", lambda settings: _FakeEngine())
    monkeypatch.setattr(batch, "load_batch", lambda engine, parsed: None)
    monkeypatch.setattr(
        batch,
        "read_training_snapshot",
        lambda engine: (train_rows, {"excluded_date_rows": 0, "excluded_late_rows": 0}),
    )
    monkeypatch.setattr(
        batch, "read_labelled_orders", lambda engine, *, purpose: val_rows
    )
    counts_seq = iter(
        [
            {"batches": 0, "customers": 0, "orders": 0, "labels": 0},
            {"batches": 1, "customers": 10, "orders": 60, "labels": 60},
        ]
    )
    monkeypatch.setattr(batch, "_count_rows", lambda engine: next(counts_seq))

    assert batch.main(urls) == 0
    out_ok = _without_times(capsys.readouterr().out)
    assert "MLflow runs logged: 2" in out_ok
    assert "Report: T s of 15 s" in out_ok

    experiment = mlflow.get_experiment_by_name("pipeline-test").experiment_id
    runs = [
        r
        for r in mlflow.search_runs([experiment], output_format="list")
        if "model" in r.data.tags
    ]
    assert {r.data.tags["model"] for r in runs} == {
        "boosted-trees-v2",
        "baseline-logistic",
    }
    assert all(r.data.tags["champion_run_id"] == champion_run_id for r in runs)
    assert all(r.data.tags["champion_eligible"] == "False" for r in runs)

    # Candidate failure isolates one broken model, logs the survivor, and exits 1.
    def _broken_manifest(validation_rows, *, champion_run_id, champion_model, k=250):
        base = retrain.team_evaluation_inputs(
            validation_rows,
            champion_run_id=champion_run_id,
            champion_model=champion_model,
            k=k,
        )
        return retrain.EvaluationInputs(
            models=(
                base.models[0],
                retrain.ModelSpec(
                    "baseline-logistic",
                    "SpiralSky",
                    lambda: (_ for _ in ()).throw(RuntimeError("boom")),
                ),
            ),
            engineers=base.engineers,
            validation_rows=base.validation_rows,
            k=base.k,
            champion_run_id=base.champion_run_id,
            champion_model=base.champion_model,
            seed=base.seed,
        )

    monkeypatch.setattr(batch, "team_evaluation_inputs", _broken_manifest)
    monkeypatch.setattr(
        batch,
        "_count_rows",
        lambda engine: {"batches": 1, "customers": 10, "orders": 60, "labels": 60},
    )
    assert batch.main(urls) == 1
    captured = capsys.readouterr()
    assert "Train and log: failed after" in captured.out
    assert "Candidate retraining failed: baseline-logistic" in captured.err


def test_a_failed_download_stops_the_run_at_its_stage(
    file_server, mlflow_settings, capsys
):
    tables = ("customers", "orders", "labels")
    urls = [f"{file_server}/{table}_b1.csv" for table in tables]

    assert batch.main(urls) == 1

    output = capsys.readouterr()
    assert _without_times(output.out)[4:] == [
        "Fetch and validate: failed after T s of 25 s",
        "Total: T s of 180 s",
    ]
    assert "Batch stopped: Downloading customers_b1.csv failed with HTTP 404" in (
        output.err
    )
