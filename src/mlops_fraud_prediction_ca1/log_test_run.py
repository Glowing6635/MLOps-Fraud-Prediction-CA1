"""Log one test run to the team's MLflow server, to show a laptop can reach it (issue #7).

`uv run log-test-run` reads MLFLOW_TRACKING_URI and MLFLOW_EXPERIMENT_NAME from the
environment or `.env`, logs a parameter, a metric and a small artifact under the
engineer's login name. MLflow prints the run's web address. It needs no database
settings, so it works before the database exists. Unless the environment says
otherwise, it gives up after about half a minute when the server does not answer.
"""

import getpass
import os
import sys

import mlflow
from dotenv import load_dotenv
from mlflow.exceptions import MlflowException


def main() -> None:
    load_dotenv()
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME")
    if not uri or not experiment_name:
        sys.exit(
            "Set MLFLOW_TRACKING_URI and MLFLOW_EXPERIMENT_NAME in .env (see .env.example)."
        )
    # MLflow's defaults (7 retries, 120 seconds each) make a wrong URL hang for minutes.
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "2")
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "10")
    engineer = getpass.getuser()
    mlflow.set_tracking_uri(uri)
    try:
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name=f"connection-test-{engineer}"):
            mlflow.log_param("engineer", engineer)
            mlflow.log_metric("connection_ok", 1)
            # The artifact proves uploads pass through the server, not only the tracking API.
            mlflow.log_text(
                "This artifact came from an engineer's laptop.", "connection_test.txt"
            )
    # Artifact uploads raise requests' errors, which are OSErrors.
    except (MlflowException, OSError) as error:
        sys.exit(
            f"Could not log the test run to {uri}. Check MLFLOW_TRACKING_URI in .env"
            f" and that the server is running. Details: {error}"
        )


if __name__ == "__main__":
    main()
