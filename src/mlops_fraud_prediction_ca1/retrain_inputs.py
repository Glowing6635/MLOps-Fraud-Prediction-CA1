"""Build retraining inputs from an approved validation export and registered model."""

import hashlib
import os
import re
from io import BytesIO
from pathlib import Path

import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from sklearn.utils.validation import check_is_fitted

from .retrain import EvaluationInputs
from .retrain_roster import load_roster
from .splitting import split_orders
from .training import ORDER_FEATURE_COLUMNS, K

_ORDER_KEYS = ["country_code", "order_id"]
_TEXT_COLUMNS = [*_ORDER_KEYS, "customer_id", "order_date"]
_REQUIRED_COLUMNS = [*_TEXT_COLUMNS, "is_fraud", *ORDER_FEATURE_COLUMNS]


def _required_setting(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ValueError(f"Set {name} for the retraining input factory.")
    return value


def _validation_export(path: Path, expected_sha256: str) -> pd.DataFrame:
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != expected_sha256:
        raise ValueError(
            "Frozen validation CSV does not match RETRAIN_VALIDATION_SHA256."
        )
    rows = pd.read_csv(
        BytesIO(content),
        dtype={
            **dict.fromkeys(
                [*_TEXT_COLUMNS, "collect_type", "payment_method"], "string"
            ),
            "order_value": "float64",
            "num_items_ordered": "float64",
        },
        keep_default_na=False,
        na_values={name: [""] for name in ORDER_FEATURE_COLUMNS},
    )
    missing = sorted(set(_REQUIRED_COLUMNS) - set(rows.columns))
    if missing:
        raise ValueError(f"Frozen validation CSV is missing columns: {missing}.")
    _, audit = split_orders(rows, purpose="validation")
    if (
        rows.empty
        or audit["excluded_rows"]
        or not rows["is_fraud"].isin([0, 1]).all()
        or rows[[*_ORDER_KEYS, "customer_id"]].eq("").any().any()
        or rows.duplicated(_ORDER_KEYS).any()
    ):
        raise ValueError(
            "Frozen validation CSV must contain unique labelled orders exclusively "
            "from the #6 validation window."
        )
    return rows


def build_inputs() -> EvaluationInputs:
    """Use the frozen roster, validation CSV and exact #15 registry version."""
    validation_path = Path(_required_setting("RETRAIN_VALIDATION_CSV"))
    expected_sha256 = _required_setting("RETRAIN_VALIDATION_SHA256").lower()
    model_name = _required_setting("RETRAIN_CHAMPION_MODEL_NAME")
    model_version = _required_setting("RETRAIN_CHAMPION_MODEL_VERSION")
    run_id = _required_setting("RETRAIN_CHAMPION_RUN_ID")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("RETRAIN_VALIDATION_SHA256 must be a SHA-256 hex digest.")
    if not re.fullmatch(r"[a-zA-Z0-9_.-]+", model_name) or not re.fullmatch(
        r"[1-9][0-9]*", model_version
    ):
        raise ValueError("Supply the champion's registered name and numeric version.")
    if not re.fullmatch(r"[0-9a-f]{32}", run_id):
        raise ValueError("RETRAIN_CHAMPION_RUN_ID must be an MLflow run ID.")

    engineers, models = load_roster()
    validation = _validation_export(validation_path, expected_sha256)
    version = MlflowClient().get_model_version(model_name, model_version)
    if version.run_id != run_id:
        raise ValueError("Pinned champion version belongs to a different MLflow run.")
    champion = mlflow.sklearn.load_model(f"models:/{model_name}/{model_version}")
    check_is_fitted(champion)
    return EvaluationInputs(models, engineers, validation, K, run_id, champion)
