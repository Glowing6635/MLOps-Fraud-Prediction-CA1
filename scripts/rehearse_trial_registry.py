#!/usr/bin/env python3
"""Rehearse #15 with synthetic packages in a dedicated local workspace.

Run with uv run python scripts/rehearse_trial_registry.py WORKSPACE.
Use one process per workspace. The database and fixtures persist for readback
and resumption. The script has no live registry mode.
"""

import argparse
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

# Disable MLflow usage reporting before initializing MLflow.
os.environ["MLFLOW_DISABLE_TELEMETRY"] = "true"
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

import mlflow.sklearn
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from mlflow.models import Model
from mlflow.store.model_registry.sqlalchemy_store import SqlAlchemyStore
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)

MODEL = "fraud-candidate"
ALIASES = {"champion": "1", "challenger": "2"}
PREVIOUS = {"1": "2", "2": "1"}
ATTEMPT = "synthetic_extra_trees_attempt"
STARTED = "attempt started; outcome unconfirmed"
HIST_TREE = "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor"
LOGGED_IDS = {
    "1": "m-34712ed5add845fe838835bd2107ec47",
    "2": "m-b550fcee4dff4a39a9e7f1000ce63446",
    "3": "m-2f7ebb90292f405c9482ef5e6e0bd90a",
}
CANDIDATES = {
    "1": ("9c3a59771cf8499d92972f5c7873e280", HistGradientBoostingClassifier),
    "2": ("9a40f8f5e35a48ddb39a97f9f052dcad", RandomForestClassifier),
    "3": ("8af9db9f95e543cfbfebdef82545322c", ExtraTreesClassifier),
}
X, Y = [[0], [1], [2], [3]], [0, 0, 1, 1]


def package(root, version):
    run_id, _ = CANDIDATES[version]
    path = root / run_id
    if path.is_symlink():
        raise ValueError(f"v{version} package path is a symlink")
    return path


def check_package(root, version):
    path = package(root, version)
    _, estimator = CANDIDATES[version]
    if any((path / name).is_symlink() for name in ("MLmodel", "model.skops")):
        raise ValueError(f"v{version} package has symlinked model files")
    manifest = Model.load(str(path / "MLmodel"))
    flavor = manifest.flavors.get("sklearn", {})
    if (
        manifest.metadata != fixture_metadata(version)
        or flavor.get("serialization_format") != "skops"
        or flavor.get("pickled_model") != "model.skops"
        or flavor.get("code")
        or flavor.get("skops_trusted_types") != trusted_types(version)
    ):
        raise ValueError(f"v{version} synthetic package manifest does not match")
    model = mlflow.sklearn.load_model(path.as_uri())
    if type(model) is not estimator or len(model.predict(X)) != len(Y):
        raise ValueError(f"v{version} synthetic package failed loadability check")


def fixture_metadata(number):
    return {"synthetic_run_reference": CANDIDATES[number][0]}


def trusted_types(number):
    # These types are allowed only for packages in the trusted synthetic workspace.
    return [HIST_TREE] if number == "1" else ["sklearn.tree._tree.Tree"]


def prepare_package(root, version):
    path = package(root, version)
    if not path.exists():
        _, estimator = CANDIDATES[version]
        with TemporaryDirectory(dir=root) as work:
            staged = Path(work) / "model"
            mlflow.sklearn.save_model(
                estimator(random_state=15).fit(X, Y),
                str(staged),
                serialization_format="skops",
                skops_trusted_types=trusted_types(version),
                metadata=fixture_metadata(version),
                pip_requirements=[],
            )
            staged.rename(path)
    check_package(root, version)


def tags_for(number):
    tags = {"synthetic_fixture": "true", "logged_model_id": LOGGED_IDS[number]}
    if number in PREVIOUS:
        tags["previous_champion_version"] = PREVIOUS[number]
    else:
        tags["trial_role"] = "approved_candidate"
    return tags


def check_version(client, root, version):
    number = str(version.version)
    run_id, _ = CANDIDATES[number]
    source = package(root, number).as_uri()
    if (
        version.run_id != run_id
        or version.status != "READY"
        or version.source != source
        or client.get_model_version_download_uri(MODEL, number) != source
        or any(
            version.tags.get(key) != value for key, value in tags_for(number).items()
        )
    ):
        raise ValueError(f"v{number} registry provenance or tags do not match")


def snapshot(client, root):
    try:
        registered = client.get_registered_model(MODEL)
    except MlflowException as exc:
        if exc.error_code != "RESOURCE_DOES_NOT_EXIST":
            raise
        return None, set(), {}

    rows = client.search_model_versions(f"name = '{MODEL}'", max_results=4)
    versions = {str(row.version) for row in rows}
    if versions not in (set(), {"1"}, {"1", "2"}, {"1", "2", "3"}):
        raise ValueError(f"Unexpected version set: {sorted(versions)}")
    for row in rows:
        check_version(client, root, row)

    aliases = {alias: str(number) for alias, number in registered.aliases.items()}
    if any(ALIASES.get(alias) != number for alias, number in aliases.items()):
        raise ValueError(f"Unexpected alias bindings: {aliases}")
    if not set(aliases.values()) <= versions or (
        "3" in versions and aliases != ALIASES
    ):
        raise ValueError("Alias bindings and existing versions are inconsistent")
    return registered, versions, aliases


def create_version(client, root, number):
    run_id, _ = CANDIDATES[number]
    row = client.create_model_version(
        MODEL, package(root, number).as_uri(), run_id=run_id, tags=tags_for(number)
    )
    if str(row.version) != number:
        raise ValueError(f"Expected v{number}, got v{row.version}")
    check_version(client, root, row)


def create_extra_once(client, root):
    # MLflow 3.18 defaults to three inserts on a SQLite integrity error.
    # Limit this store's retry budget only for the approved candidate.
    store = client._get_registry_client().store
    if not isinstance(store, SqlAlchemyStore):
        raise TypeError("Expected the MLflow SQLite registry store")
    retries = store.CREATE_MODEL_VERSION_RETRIES
    store.CREATE_MODEL_VERSION_RETRIES = 1
    try:
        create_version(client, root, "3")
    finally:
        store.CREATE_MODEL_VERSION_RETRIES = retries


def verify_baseline(client, root):
    for alias, number in ALIASES.items():
        row = client.get_model_version_by_alias(MODEL, alias)
        if str(row.version) != number:
            raise ValueError(f"{alias} did not resolve to v{number}")
        check_version(client, root, row)
        check_package(root, number)


def rehearse(root):
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    database = root / "registry.sqlite"
    if database.is_symlink() or "?" in str(database):
        raise ValueError("Expected a local SQLite path without URI options")
    uri = f"sqlite:///{database}"
    client = MlflowClient(tracking_uri=uri, registry_uri=uri)
    registered, versions, aliases = snapshot(client, root)

    for number in ALIASES.values():
        if number in versions:
            check_package(root, number)
        else:
            prepare_package(root, number)
    if registered is None:
        registered = client.create_registered_model(MODEL)
    for alias, number in ALIASES.items():
        if number not in versions:
            create_version(client, root, number)
        if alias not in aliases:
            client.set_registered_model_alias(MODEL, alias, number)
    verify_baseline(client, root)

    failure = registered.tags.get(ATTEMPT)
    if "3" not in versions and failure is None:
        client.set_registered_model_tag(MODEL, ATTEMPT, STARTED)
        try:
            prepare_package(root, "3")
            create_extra_once(client, root)
        except Exception as exc:  # noqa: BLE001
            # A failed create may have committed. Read back below before claiming
            # the fallback state, and keep this workspace at one attempt.
            detail = str(exc).partition("\n")[0][:500]
            failure = f"{type(exc).__name__}: {detail}"
            client.set_registered_model_tag(MODEL, ATTEMPT, failure)

    _, versions, aliases = snapshot(client, root)
    verify_baseline(client, root)
    if "3" in versions:
        check_package(root, "3")
        candidate = {"status": "registered_unaliased", "version": "3"}
    elif failure is not None:
        candidate = {"status": "fallback", "reason": failure}
    else:
        raise ValueError("Extra Trees attempt returned without a candidate or failure")
    return {
        "registry": str(database),
        "synthetic": True,
        "aliases": aliases,
        "previous_champion_version": PREVIOUS,
        "run_ids": {number: CANDIDATES[number][0] for number in sorted(versions)},
        "extra_trees": candidate,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace", type=Path)
    args = parser.parse_args()
    print(json.dumps(rehearse(args.workspace), sort_keys=True))  # noqa: T201


if __name__ == "__main__":
    main()
