"""Tests for the settings loader and the shared database engine. Nothing here connects."""

from functools import partial

import pytest
from dotenv import load_dotenv

from mlops_fraud_prediction_ca1 import config
from mlops_fraud_prediction_ca1.config import (
    SettingError,
    Settings,
    create_db_engine,
    load_settings,
)

ENVIRON = {
    "DB_HOST": "db-a.example.com",
    "DB_PORT": "3307",
    "DB_NAME": "fraud",
    "DB_USER": "loader",
    "DB_PASSWORD": "p@ss:w/rd",
    "MLFLOW_TRACKING_URI": "http://mlflow.example.com",
    "MLFLOW_EXPERIMENT_NAME": "fraud-models",
    "MLFLOW_TRACKING_USERNAME": "engineer",
    "MLFLOW_TRACKING_PASSWORD": "secret",
}


def test_every_setting_comes_from_the_environment():
    assert load_settings(ENVIRON) == Settings(
        db_host="db-a.example.com",
        db_port=3307,
        db_name="fraud",
        db_user="loader",
        db_password="p@ss:w/rd",
        mlflow_tracking_uri="http://mlflow.example.com",
        mlflow_experiment_name="fraud-models",
        mlflow_tracking_username="engineer",
        mlflow_tracking_password="secret",
    )


def test_changing_the_host_setting_moves_the_engine_with_no_code_edit():
    engine_a = create_db_engine(load_settings(ENVIRON))
    engine_b = create_db_engine(
        load_settings({**ENVIRON, "DB_HOST": "db-b.example.com"})
    )
    assert (engine_a.url.host, engine_b.url.host) == (
        "db-a.example.com",
        "db-b.example.com",
    )
    assert engine_a.url.password == "p@ss:w/rd"


def test_the_port_defaults_to_mysql():
    assert load_settings({**ENVIRON, "DB_PORT": ""}).db_port == 3306


def test_missing_settings_are_all_named():
    environ = {**ENVIRON, "DB_HOST": "", "MLFLOW_TRACKING_PASSWORD": ""}
    del environ["DB_PASSWORD"]
    with pytest.raises(SettingError) as error:
        load_settings(environ)
    assert str(error.value) == (
        "Set DB_HOST, DB_PASSWORD, MLFLOW_TRACKING_PASSWORD in the environment"
        " or in .env (see .env.example)."
    )


@pytest.mark.parametrize("port", ["33o6", "0", "-1", "65536"])
def test_a_port_that_is_not_valid_is_named(port):
    with pytest.raises(SettingError) as error:
        load_settings({**ENVIRON, "DB_PORT": port})
    assert str(error.value) == (
        "DB_PORT must be a whole number from 1 to 65535, such as 3306."
    )


def test_dotenv_fills_unset_variables_and_the_environment_wins(tmp_path, monkeypatch):
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "".join(f"{name}={value}\n" for name, value in ENVIRON.items()),
        encoding="utf-8",
    )
    # Use the temporary .env, never the developer's real one.
    monkeypatch.setattr(config, "load_dotenv", partial(load_dotenv, dotenv))
    for name in ENVIRON:
        monkeypatch.setenv(name, "")  # So undo removes what load_dotenv sets.
        monkeypatch.delenv(name)
    monkeypatch.setenv("DB_HOST", "db-from-environment.example.com")

    settings = load_settings()

    assert settings.db_host == "db-from-environment.example.com"
    assert settings.db_name == "fraud" and settings.db_port == 3307


def test_passwords_never_appear_in_repr():
    shown = repr(load_settings(ENVIRON))
    assert "p@ss:w/rd" not in shown and "secret" not in shown
