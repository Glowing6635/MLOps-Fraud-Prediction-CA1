"""Time how long the team's services take to work again after a sandbox restart (issue #11).

Run `uv run restart-drill` as soon as the sandbox is started. It waits until RDS answers,
then until the MLflow server answers, then checks that the model saved by
`uv run redeploy-check record` loads and reproduces its predictions, and logs a run
with the times as metrics, tagged `purpose=restart-drill`. It prints the time from its
own start to each step, so the demo-day start time can be set from it. It gives up
after 30 minutes.

It needs every setting in `.env.example`, and `output/redeploy_check.json` from an
earlier `uv run redeploy-check record` on the same laptop.
"""

import os
import sys
import time
import urllib.request
from collections.abc import Callable

import mlflow
from mlflow.exceptions import MlflowException
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from mlops_fraud_prediction_ca1 import redeploy_check
from mlops_fraud_prediction_ca1.config import (
    SettingError,
    create_db_engine,
    load_settings,
)

TIMEOUT_SECONDS = 30 * 60
RETRY_SECONDS = 15


def wait_until(
    check: Callable[[], object],
    deadline: float,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], object] = time.sleep,
) -> None:
    """Call `check` until it stops failing to connect; past `deadline`, raise its last error."""
    while True:
        try:
            check()
            return
        # A service still starting refuses connections or answers with an error page.
        except (OSError, OperationalError):
            if clock() + RETRY_SECONDS > deadline:
                raise
            sleep(RETRY_SECONDS)


def log_times(times: dict[str, float]) -> str:
    """Log each step's time as a run in the active experiment, and return its run ID."""
    # The drill shares the model runs' experiment, so model queries filter this tag out.
    with mlflow.start_run(
        run_name="restart-drill", tags={"purpose": "restart-drill"}
    ) as run:
        mlflow.log_metrics({f"{step}_seconds": t for step, t in times.items()})
    return run.info.run_id


def minutes(seconds: float) -> str:
    return f"{int(seconds // 60)}:{int(seconds % 60):02d}"


def main() -> None:
    # MLflow prints an emoji when a run ends; Windows cannot encode it when output is piped.
    sys.stdout.reconfigure(errors="replace")
    start = time.monotonic()
    try:
        settings = load_settings()
    except SettingError as error:
        sys.exit(str(error))
    if not redeploy_check.RECORD.exists():
        sys.exit(
            f"{redeploy_check.RECORD} is missing. Run `uv run redeploy-check record`"
            " before the sandbox stops."
        )
    # Each attempt fails fast, so the drill retries instead of hanging on one request.
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "2")
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "10")
    engine = create_db_engine(settings)
    health_url = settings.mlflow_tracking_uri.rstrip("/") + "/health"
    times = {}

    def check_rds() -> None:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))

    def done(step: str, message: str) -> None:
        times[step] = time.monotonic() - start
        print(f"{minutes(times[step])}  {message}")

    try:
        wait_until(check_rds, start + TIMEOUT_SECONDS)
        done("rds", f"RDS answers at {settings.db_host}.")
        wait_until(
            lambda: urllib.request.urlopen(health_url, timeout=10).close(),
            start + TIMEOUT_SECONDS,
        )
        done("mlflow", f"MLflow answers at {settings.mlflow_tracking_uri}.")
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(settings.mlflow_experiment_name)
        redeploy_check.verify()
        done("model", "The model loads and reproduces its predictions.")
        run_id = log_times(times)
        done("total", f"Everything works. The times are logged as MLflow run {run_id}.")
    except (OSError, OperationalError, MlflowException) as error:
        sys.exit(
            f"Not working {minutes(time.monotonic() - start)} after the drill started."
            f" Details: {error}"
        )
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
