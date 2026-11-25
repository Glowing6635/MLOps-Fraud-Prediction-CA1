"""Resource settings for the database and MLflow, and the shared database connection.

Every setting comes from an environment variable, and `.env.example` lists them all.
A git-ignored `.env` file fills in any variable the environment does not set, so
pointing the code at another host is a configuration change, never a code edit.
python-dotenv finds `.env` by searching upward from this package's folder, so the
one at the repository root is used whatever the working directory; in a notebook
or a debugger it searches upward from the working directory instead. Only the
MySQL port has a default.

MLflow reads `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME` and
`MLFLOW_TRACKING_PASSWORD` from the environment itself, which is why
`load_settings()` loads `.env` into `os.environ` rather than only reading it.
"""

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import URL, Engine, create_engine

_DEFAULT_DB_PORT = "3306"
# PyMySQL waits forever on an unreachable host without this; seconds.
_CONNECT_TIMEOUT = 10
# RDS's certificates, from https://truststore.pki.rds.amazonaws.com/global/global-bundle.pem.
# The database refuses unencrypted connections, and this lets the driver check it is RDS.
RDS_CA_BUNDLE = Path(__file__).with_name("rds-global-bundle.pem")


class SettingError(Exception):
    """A required setting is not in the environment or `.env`, or is not valid."""


@dataclass(frozen=True)
class Settings:
    """The resource settings; passwords are left out of `repr` so they never reach a log."""

    db_host: str
    db_port: int
    db_name: str
    db_user: str
    db_password: str = field(repr=False)
    mlflow_tracking_uri: str
    mlflow_experiment_name: str
    mlflow_tracking_username: str
    mlflow_tracking_password: str = field(repr=False)


def load_settings(environ: Mapping[str, str] | None = None) -> Settings:
    """Read every setting from `environ`, or from the process environment plus `.env` when omitted.

    Raises `SettingError` naming every variable that is unset or empty, or a `DB_PORT`
    that is not a whole number from 1 to 65535.
    """
    if environ is None:
        load_dotenv()  # Does not override variables the environment already sets.
        environ = os.environ
    names = [
        "DB_HOST",
        "DB_NAME",
        "DB_USER",
        "DB_PASSWORD",
        "MLFLOW_TRACKING_URI",
        "MLFLOW_EXPERIMENT_NAME",
        "MLFLOW_TRACKING_USERNAME",
        "MLFLOW_TRACKING_PASSWORD",
    ]
    missing = [name for name in names if not environ.get(name)]
    if missing:
        raise SettingError(
            f"Set {', '.join(missing)} in the environment or in .env (see .env.example)."
        )
    try:
        db_port = int(environ.get("DB_PORT") or _DEFAULT_DB_PORT)
    except ValueError:
        db_port = 0  # Not a number: reported below, with out-of-range ports.
    if not 1 <= db_port <= 65535:
        raise SettingError(
            "DB_PORT must be a whole number from 1 to 65535, such as 3306."
        )
    return Settings(
        db_host=environ["DB_HOST"],
        db_port=db_port,
        db_name=environ["DB_NAME"],
        db_user=environ["DB_USER"],
        db_password=environ["DB_PASSWORD"],
        mlflow_tracking_uri=environ["MLFLOW_TRACKING_URI"],
        mlflow_experiment_name=environ["MLFLOW_EXPERIMENT_NAME"],
        mlflow_tracking_username=environ["MLFLOW_TRACKING_USERNAME"],
        mlflow_tracking_password=environ["MLFLOW_TRACKING_PASSWORD"],
    )


def create_db_engine(settings: Settings) -> Engine:
    """The SQLAlchemy engine for the configured MySQL database. It connects on first use."""
    url = URL.create(
        "mysql+pymysql",
        username=settings.db_user,
        password=settings.db_password,
        host=settings.db_host,
        port=settings.db_port,
        database=settings.db_name,
    )
    # hide_parameters keeps the values a failed statement sent, which are data
    # rows, out of the error text that commands print and logs keep.
    return create_engine(
        url,
        pool_pre_ping=True,
        hide_parameters=True,
        connect_args={
            "connect_timeout": _CONNECT_TIMEOUT,
            "ssl": {"ca": str(RDS_CA_BUNDLE)},
        },
    )
