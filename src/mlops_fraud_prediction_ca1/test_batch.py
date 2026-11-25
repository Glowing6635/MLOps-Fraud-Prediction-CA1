import socket

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


def test_preflight_order_and_output(monkeypatch, capfd):
    settings = load_settings(
        {
            "DB_HOST": "db.example",
            "DB_NAME": "practice",
            "DB_USER": "private-db-user",
            "DB_PASSWORD": "private-db-password",
            "MLFLOW_TRACKING_URI": "https://private-mlflow.example",
            "MLFLOW_EXPERIMENT_NAME": "rehearsal",
            "MLFLOW_TRACKING_USERNAME": "private-mlflow-user",
            "MLFLOW_TRACKING_PASSWORD": "private-mlflow-password",
        }
    )
    monkeypatch.setattr(batch, "load_settings", lambda: settings)
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a: pytest.fail("DNS lookup"))
    monkeypatch.setattr(socket.socket, "connect", lambda *a: pytest.fail("connection"))
    monkeypatch.setattr(batch.sys, "stderr", batch.sys.stdout)
    assert batch.main(URLS) == 1
    assert capfd.readouterr().out.splitlines() == [
        "Database host: db.example",
        "Database name: practice",
        "MLflow experiment: rehearsal",
        "Batch stopped: fetch and validate stage (#19) is unavailable.",
    ]


def test_missing_configuration(monkeypatch, capsys):
    def missing():
        raise SettingError("Set DB_HOST")

    monkeypatch.setattr(batch, "load_settings", missing)
    assert batch.main(URLS) == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert "Set DB_HOST" in output.err
