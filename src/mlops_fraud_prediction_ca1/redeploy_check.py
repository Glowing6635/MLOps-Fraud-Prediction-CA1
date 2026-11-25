"""Check that the MLflow server keeps runs, models and artifacts across a redeploy (issue #9).

Before a redeploy or the end of a sandbox session, `uv run redeploy-check record` logs a
run with a parameter and a metric, registers a small model, and saves the run ID, the
model version and the model's predictions for a fixed input to `output/redeploy_check.json`.
Afterwards, `uv run redeploy-check verify` reads the same run and model version back from
the server and fails unless every value and prediction is unchanged.
`uv run redeploy-check large` uploads a 100 MB artifact through the server and downloads
it again, which checks the upload limit and that laptops need no AWS keys.

It reads MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_USERNAME and
MLFLOW_TRACKING_PASSWORD from the environment or `.env`.
"""

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import mlflow
import numpy as np
from dotenv import load_dotenv
from mlflow.exceptions import MlflowException
from sklearn.linear_model import LogisticRegression

MODEL_NAME = "redeploy-check"
RECORD = Path("output/redeploy_check.json")
PARAMS = {"purpose": "redeploy-check", "c": "1.0"}
METRICS = {"check_metric": 0.125}
# A fixed, made-up input: the check must not depend on any real data.
X = np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 3.0], [3.0, 1.0], [4.0, 4.0], [5.0, 2.0]])
Y = np.array([0, 0, 1, 0, 1, 1])
LARGE_BYTES = 100 * 1024 * 1024


def record() -> None:
    model = LogisticRegression(C=float(PARAMS["c"])).fit(X, Y)
    with mlflow.start_run(run_name="redeploy-check") as run:
        mlflow.log_params(PARAMS)
        mlflow.log_metrics(METRICS)
        info = mlflow.sklearn.log_model(
            model, name="model", registered_model_name=MODEL_NAME
        )
    saved = {
        "run_id": run.info.run_id,
        "model_version": info.registered_model_version,
        "predictions": model.predict_proba(X)[:, 1].tolist(),
    }
    RECORD.parent.mkdir(exist_ok=True)
    RECORD.write_text(json.dumps(saved, indent=2), encoding="utf-8")
    print(
        f"Recorded run {saved['run_id']} and {MODEL_NAME} version {saved['model_version']}."
    )


def verify() -> None:
    if not RECORD.exists():
        sys.exit(
            f"Run `uv run redeploy-check record` before the redeploy; {RECORD} is missing."
        )
    saved = json.loads(RECORD.read_text(encoding="utf-8"))
    run = mlflow.get_run(saved["run_id"])
    problems = []
    if run.data.params != PARAMS:
        problems.append(f"parameters are {run.data.params}, expected {PARAMS}")
    if run.data.metrics != METRICS:
        problems.append(f"metrics are {run.data.metrics}, expected {METRICS}")
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{saved['model_version']}")
    predictions = model.predict_proba(X)[:, 1]
    if not np.allclose(predictions, saved["predictions"], rtol=0, atol=1e-12):
        problems.append(
            f"predictions are {predictions.tolist()}, expected {saved['predictions']}"
        )
    if problems:
        sys.exit("The server lost data: " + "; ".join(problems))
    print(
        f"Run {saved['run_id']} and {MODEL_NAME} version {saved['model_version']} are"
        " unchanged, and the model reproduces its predictions."
    )


def large() -> None:
    with tempfile.TemporaryDirectory() as folder:
        upload = Path(folder, "large_check.bin")
        upload.write_bytes(os.urandom(LARGE_BYTES))
        with mlflow.start_run(run_name="redeploy-check-large") as run:
            mlflow.log_artifact(str(upload))
        download = mlflow.artifacts.download_artifacts(
            run_id=run.info.run_id,
            artifact_path=upload.name,
            dst_path=str(Path(folder, "back")),
        )
        sent, received = (
            hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (upload, download)
        )
    if sent != received:
        sys.exit("The 100 MB artifact came back different from what was uploaded.")
    print(f"A 100 MB artifact went up and came back unchanged (run {run.info.run_id}).")


def main() -> None:
    # MLflow prints an emoji when a run ends; Windows cannot encode it when output is piped.
    sys.stdout.reconfigure(errors="replace")
    steps = {"record": record, "verify": verify, "large": large}
    if len(sys.argv) != 2 or sys.argv[1] not in steps:
        sys.exit("Usage: uv run redeploy-check record | verify | large")
    load_dotenv()
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME")
    if not uri or not experiment_name:
        sys.exit(
            "Set MLFLOW_TRACKING_URI and MLFLOW_EXPERIMENT_NAME in .env (see .env.example)."
        )
    # Fail fast on a wrong URL, but give a 100 MB upload time to finish.
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "2")
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "300")
    mlflow.set_tracking_uri(uri)
    try:
        mlflow.set_experiment(experiment_name)
        steps[sys.argv[1]]()
    # Artifact transfers raise requests' errors, which are OSErrors.
    except (MlflowException, OSError) as error:
        sys.exit(f"The check could not finish against {uri}. Details: {error}")


if __name__ == "__main__":
    main()
