import functools
import http.server
import re
import threading

import mlflow
import pytest

from . import batch
from .config import SettingError, load_settings

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
    tracking_uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "pipeline-test")
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
    """60 labelled orders: 40 in the training window and 20 in validation."""
    customers = [
        f"MY,c{i},true,1.0,0.0,0.0,10.0,0.0,2022-10-01 10:00:00" for i in range(10)
    ]
    orders, labels = [], []
    for i in range(60):
        day = f"2023-01-{1 + i % 28:02d}" if i < 40 else f"2023-04-{1 + i % 10:02d}"
        fraud = int(i % 4 == 0)
        value = 100.0 if fraud else 10.0
        orders.append(f"MY,o{i},Delivery,Credit Card,{value},1.0,0.0,{day}")
        labels.append(f"MY,o{i},c{i % 10},{fraud}")
    return write_batch("b1", customers, orders, labels)


def _without_times(text: str) -> list[str]:
    return re.sub(r"\d+\.\d s", "T s", text).splitlines()


def test_a_batch_runs_every_stage_within_its_budget_and_reports(
    database, write_batch, file_server, mlflow_settings, monkeypatch, capsys
):
    monkeypatch.setattr(batch, "create_db_engine", lambda settings: database)
    files = _practice_batch(write_batch)
    urls = [f"{file_server}/{path.rsplit('/', 1)[-1]}" for path in files]

    assert batch.main(urls) == 0

    experiment = mlflow.get_experiment_by_name("pipeline-test").experiment_id
    [run] = mlflow.search_runs([experiment], output_format="list")
    etl_run_id = run.data.tags["etl_run_id"]
    assert run.data.tags["batch_id"] == "b1"
    assert _without_times(capsys.readouterr().out) == [
        "Database host: db.example",
        "Database name: practice",
        "MLflow experiment: pipeline-test",
        f"ETL run ID: {etl_run_id}",
        "Fetch and validate: T s of 25 s",
        "Batch b1: 10 customers, 60 orders, 60 labels",
        "Clean and load: T s of 30 s",
        "Read training data: T s of 15 s",
        "Train and log: T s of 85 s",
        "Rows per table, before and after:",
        "  batches: 0 -> 1",
        "  customers: 0 -> 10",
        "  orders: 0 -> 60",
        "  labels: 0 -> 60",
        "MLflow runs logged: 1",
        (
            f"MLflow runs: {mlflow_settings.mlflow_tracking_uri}/#/experiments/"
            f"{experiment}?searchFilter=tags.etl_run_id%20%3D%20%27{etl_run_id}%27"
        ),
        "Report: T s of 15 s",
        "Total: T s of 180 s",
    ]


def test_a_failed_download_stops_the_run_at_its_stage(
    file_server, mlflow_settings, capsys
):
    tables = ("customers", "orders", "labels")
    urls = [f"{file_server}/{table}_b1.csv" for table in tables]

    assert batch.main(urls) == 1

    output = capsys.readouterr()
    assert _without_times(output.out)[4:] == ["Fetch and validate: failed after T s"]
    assert "Batch stopped: Downloading customers_b1.csv failed with HTTP 404" in (
        output.err
    )
