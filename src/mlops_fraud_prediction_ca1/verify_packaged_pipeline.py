"""Fit, register and reload the order feature pipeline in a fresh local MLflow store."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
from mlflow import MlflowClient

from .feature_pipeline import FeatureGroup, build_pipeline
from .models import baseline_model
from .splitting import load_policy
from .top_k import K, evaluate_top_k
from .training import dataset_hashes, load_window, train

MODEL_NAME = "packaged-pipeline-reload"
FEATURE_GROUP = "order_log"
SEED = 0
ORDER_KEYS = ["country_code", "order_id"]


def verify(data_dir: Path, registry_dir: Path) -> dict:
    """Return validation parity evidence from a fresh, isolated MLflow registry."""
    policy = load_policy()
    if not policy.test_locked or policy.final_evaluation_enabled:
        raise ValueError(
            "Lock the test window and disable final evaluation before running."
        )

    input_hashes = dataset_hashes(data_dir)
    registry_dir = registry_dir.resolve()
    registry_dir.mkdir(mode=0o700, exist_ok=False)
    uri = f"sqlite:///{registry_dir / 'mlflow.db'}"
    artifact_root = registry_dir / "artifacts"
    artifact_root.mkdir()
    os.environ["MLFLOW_TRACKING_URI"] = uri
    os.environ["MLFLOW_EXPERIMENT_NAME"] = MODEL_NAME
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    client = MlflowClient(tracking_uri=uri, registry_uri=uri)
    experiment_id = client.create_experiment(MODEL_NAME, artifact_root.as_uri())

    run_id, result = train(
        build_pipeline(baseline_model(), FEATURE_GROUP),
        data_dir,
        k=K,
        seed=SEED,
        tags={"evidence_scope": "local_pipeline_reload", "champion_eligible": "False"},
    )
    keys, raw = load_window(data_dir, "validation", exploratory=False)
    if (
        keys.empty
        or keys[ORDER_KEYS].duplicated().any()
        or keys[ORDER_KEYS].isna().any().any()
        or len(keys) != result.validation_orders
    ):
        raise RuntimeError(
            "Expected the complete validation cohort with unique order keys."
        )
    before = result.model.predict_proba(raw)
    if (
        not np.isfinite(before).all()
        or evaluate_top_k(keys, before[:, 1], K) != result.report
    ):
        raise RuntimeError(
            "In-memory validation scores disagree with the training report."
        )

    run = client.get_run(run_id)
    logged = client.search_logged_models(
        [experiment_id], filter_string=f"source_run_id = '{run_id}'"
    )
    if (
        run.info.status != "FINISHED"
        or any(run.data.params.get(key) != value for key, value in input_hashes.items())
        or len(logged) != 1
        or logged[0].name != "model"
        or logged[0].status.name != "READY"
    ):
        raise RuntimeError(
            "Expected one READY packaged model from the finished frozen run."
        )
    model = logged[0]
    source = model.artifact_location
    client.create_registered_model(MODEL_NAME)
    created = client.create_model_version(MODEL_NAME, source, run_id=run_id)
    version = client.get_model_version(MODEL_NAME, created.version)
    if (
        version.status != "READY"
        or version.run_id != run_id
        or version.source != source
        or client.get_model_version_download_uri(MODEL_NAME, version.version) != source
    ):
        raise RuntimeError(
            "The registered version does not resolve to the logged artifact."
        )

    model_uri = f"models:/{MODEL_NAME}/{version.version}"
    loaded = mlflow.sklearn.load_model(model_uri)
    if (
        not isinstance(loaded.named_steps.get("features"), FeatureGroup)
        or loaded.named_steps["features"].group != FEATURE_GROUP
    ):
        raise RuntimeError("The registered model is missing the order feature group.")
    reloaded_keys, reloaded_raw = load_window(data_dir, "validation", exploratory=False)
    if (
        not reloaded_keys.equals(keys)
        or not reloaded_raw.equals(raw)
        or dataset_hashes(data_dir) != input_hashes
    ):
        raise RuntimeError("The ordered validation inputs changed during reload.")
    after = loaded.predict_proba(reloaded_raw)
    if not np.array_equal(after, before) or not np.array_equal(
        loaded.predict(reloaded_raw), result.model.predict(raw)
    ):
        raise RuntimeError(
            "Reloaded probabilities or predictions differ from the fitted pipeline."
        )

    order_digest = hashlib.sha256(
        keys[ORDER_KEYS].to_csv(index=False, lineterminator="\n").encode()
    ).hexdigest()
    return {
        "evidence_scope": "local_pipeline_reload",
        "tracking_uri": uri,
        "experiment_id": experiment_id,
        "source_run_id": run_id,
        "run_commit": run.data.tags["git_commit"],
        "run_git_dirty": run.data.tags["git_dirty"],
        "logged_model_uri": f"models:/{model.model_id}",
        "registered_model_uri": model_uri,
        "registered_source": source,
        "registered_status": version.status,
        "feature_group": FEATURE_GROUP,
        "train_start": policy.train_start.isoformat(),
        "validation_start": policy.validation_start.isoformat(),
        "validation_end_exclusive": policy.test_start.isoformat(),
        "test_locked": policy.test_locked,
        "final_evaluation_enabled": policy.final_evaluation_enabled,
        "seed": SEED,
        "k": K,
        "train_rows": result.train_orders,
        "validation_rows": result.validation_orders,
        "validation_order_sha256": order_digest,
        "ordered_validation_inputs_identical": True,
        "probabilities_identical": True,
        "predictions_identical": True,
        "max_probability_abs_delta": float(np.max(np.abs(after - before))),
        "fraud_caught_at_k": result.report.at_k.fraud_caught,
        "average_precision": result.report.average_precision,
        **input_hashes,
    }


def main() -> None:
    """Write the verification evidence as JSON or report an existing registry."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", required=True, type=Path, help="Approved base CSVs."
    )
    parser.add_argument(
        "--registry-dir", required=True, type=Path, help="A new private directory."
    )
    args = parser.parse_args()
    try:
        evidence = verify(args.data_dir, args.registry_dir)
    except FileExistsError:
        parser.exit(
            1,
            f"{parser.prog}: Registry directory already exists: {args.registry_dir}\n",
        )

    # JSON on stdout is the CLI's machine-readable output.
    print(json.dumps(evidence, indent=2))  # noqa: T201


if __name__ == "__main__":
    main()
