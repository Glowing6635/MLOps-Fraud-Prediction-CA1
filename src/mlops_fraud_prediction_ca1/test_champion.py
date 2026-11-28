"""Champion selection uses eligible runs only and moves the alias with a recorded previous version."""

import runpy
import subprocess
import sys
from pathlib import Path
from shutil import which
from unittest.mock import Mock, call

import mlflow
import mlflow.sklearn
import numpy as np
import pytest
from mlflow import MlflowClient
from mlflow.entities import FileInfo
from mlflow.exceptions import MlflowException
from mlflow.store.entities.paged_list import PagedList
from mlflow.utils.file_utils import local_file_uri_to_path
from sklearn.dummy import DummyClassifier

from mlops_fraud_prediction_ca1 import champion
from mlops_fraud_prediction_ca1.champion import (
    CHALLENGER,
    CHAMPION,
    comparison_id,
    comparison_table,
    eligible_candidates,
    move_champion,
    register_top_two,
)

MODEL = "fraud-candidate"


PROVENANCE = {
    "dataset_sha256_customers": "a" * 64,
    "dataset_sha256_orders": "b" * 64,
    "dataset_sha256_labels": "c" * 64,
    "validation_start": "2025-01-01",
    "test_start": "2025-02-01",
    "cleaning_policy_version": "pbi12-v3",
}


def _log(
    name,
    caught,
    ap,
    *,
    eligible="True",
    exploratory="False",
    k="250",
    dirty="False",
    provenance=PROVENANCE,
    nested=False,
    model_size_bytes=None,
    legacy_manifest=True,
    total_fraud=1000,
    fraud_catchable=700,
):
    with mlflow.start_run(run_name=name, nested=nested) as run:
        mlflow.log_params({"k": k, "exploratory": exploratory, **provenance})
        mlflow.set_tags({"champion_eligible": eligible, "git_dirty": dirty})
        values = {
            "fraud_caught_at_k": caught,
            "average_precision": ap,
            "validation_orders": 2000,
            "total_fraud_at_k": total_fraud,
            "fraud_catchable_at_k": fraud_catchable,
        }
        mlflow.log_metrics(
            {key: value for key, value in values.items() if value is not None}
        )
        if model_size_bytes is not None:
            mlflow.log_metric("model_size_bytes", model_size_bytes)
        if legacy_manifest:
            mlflow.log_text("model", "model/MLmodel")
    return run.info.run_id


@pytest.fixture
def client(tmp_path, monkeypatch):
    uri = f"sqlite:///{tmp_path / 'm.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.setenv("MLFLOW_EXPERIMENT_ID", "")
    monkeypatch.setattr(mlflow.tracking.fluent, "_active_experiment_id", None)
    monkeypatch.setattr(mlflow.tracking._tracking_service.utils, "_tracking_uri", uri)
    client = MlflowClient(tracking_uri=uri, registry_uri=uri)
    client.create_experiment("e", artifact_location=(tmp_path / "artifacts").as_uri())
    mlflow.set_experiment("e")
    return client


def _experiment(client):
    return [client.get_experiment_by_name("e").experiment_id]


def _register(client, model_name, candidates):
    return register_top_two(
        client,
        model_name,
        candidates,
        approved_comparison=comparison_id(client, candidates),
    )


def test_exploratory_other_k_and_ineligible_runs_are_not_candidates(client):
    good_a = _log("a", 100, 0.3)
    good_b = _log("b", 90, 0.4)
    _log("exp", 500, 0.9, exploratory="True")
    _log("ineligible", 500, 0.9, eligible="False")
    _log("otherk", 400, 0.9, k="50", provenance={})
    _log("dirty", 600, 0.9, dirty="True", provenance={})
    candidates = eligible_candidates(client, _experiment(client), 250)
    assert [c.run_id for c in candidates] == [good_a, good_b]


def test_size_measurements_are_informational(client):
    oversized = _log("oversized", 200, 0.9, model_size_bytes=50_000_001)
    at_limit = _log("at limit", 100, 0.3, model_size_bytes=50_000_000)
    missing = _log("missing size", 90, 0.4)
    malformed = _log("malformed size", 80, 0.4, model_size_bytes=1.5)

    candidates = eligible_candidates(client, _experiment(client), 250)
    assert [c.run_id for c in candidates] == [oversized, at_limit, missing, malformed]
    assert [c.saved_model_size_bytes for c in candidates] == [
        50_000_001,
        50_000_000,
        None,
        None,
    ]
    table = comparison_table(candidates)
    assert "50,000,001 bytes (>50 MB)" in table
    assert "50,000,000 bytes |" in table
    assert table.count("| unknown | no |") == 2


def test_ties_on_fraud_caught_go_to_average_precision(client):
    _log("low", 100, 0.2)
    high = _log("high", 100, 0.5)
    candidates = eligible_candidates(client, _experiment(client), 250)
    assert candidates[0].run_id == high
    assert "average precision decides" in comparison_table(candidates)


def test_fewer_than_two_eligible_runs_is_refused(client):
    _log("only", 1, 0.1)
    with pytest.raises(ValueError):
        eligible_candidates(client, _experiment(client), 250)


def test_top_two_are_registered_with_aliases_and_artifact_tags(client):
    best = _log("best", 100, 0.3)
    second = _log("second", 90, 0.4)
    _log("third", 10, 0.1)
    candidates = eligible_candidates(client, _experiment(client), 250)
    versions = _register(client, MODEL, candidates)
    assert len(client.search_model_versions(f"name='{MODEL}'")) == 2
    champion = client.get_model_version_by_alias(MODEL, CHAMPION)
    assert champion.run_id == best
    assert champion.tags["artifact"] == f"runs:/{best}/model"
    assert client.get_model_version_by_alias(MODEL, CHALLENGER).run_id == second
    assert _register(client, MODEL, candidates) == versions
    assert len(client.search_model_versions(f"name='{MODEL}'")) == 2


def test_registry_gets_best_two_within_50_mb_and_ranking_keeps_the_rest(
    client, monkeypatch, capsys
):
    huge = _log("huge forest", 300, 0.9, model_size_bytes=187_000_000)
    real = champion._artifact_size
    monkeypatch.setattr(
        champion,
        "_artifact_size",
        lambda c, source: 187_000_000 if huge in source.location else real(c, source),
    )
    best = _log("best", 200, 0.3)
    second = _log("second", 100, 0.4)
    _log("third", 50, 0.1)
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "e")
    cli = _installed_cli()
    monkeypatch.setattr(sys, "argv", [cli, "--k", "250"])
    with pytest.raises(SystemExit) as status:
        runpy.run_path(cli, run_name="__main__")
    assert status.value.code in (None, 0)

    preview = capsys.readouterr().out
    assert f"| 1 | huge forest ({huge}) |" in preview
    assert "187,000,000 bytes (>50 MB)" in preview
    assert f"Champion: best ({best}) (rank 2)." in preview
    assert f"Champion: huge forest ({huge})" not in preview
    approval = preview.split("Comparison ID: ")[1].strip()

    monkeypatch.setattr(
        sys,
        "argv",
        [cli, "--k", "250", "--register", MODEL, "--approve-comparison", approval],
    )
    with pytest.raises(SystemExit) as status:
        runpy.run_path(cli, run_name="__main__")
    assert status.value.code in (None, 0)
    assert f"Champion: best ({best}) (rank 2)." in capsys.readouterr().out
    assert client.get_model_version_by_alias(MODEL, CHAMPION).run_id == best
    assert client.get_model_version_by_alias(MODEL, CHALLENGER).run_id == second
    assert len(client.search_model_versions(f"name='{MODEL}'")) == 2


def test_alias_moves_to_the_second_model_and_back(client):
    _log("best", 100, 0.3)
    _log("second", 90, 0.4)
    versions = _register(
        client, MODEL, eligible_candidates(client, _experiment(client), 250)
    )
    old = move_champion(client, MODEL, versions[CHALLENGER], "analyst preferred it")
    assert old == versions[CHAMPION]
    now = client.get_model_version_by_alias(MODEL, CHAMPION)
    assert str(now.version) == versions[CHALLENGER]
    assert now.tags["previous_champion_version"] == old
    assert now.tags["champion_change_reason"] == "analyst preferred it"
    back = move_champion(client, MODEL, old, "rollback")
    assert back == versions[CHALLENGER]
    assert str(client.get_model_version_by_alias(MODEL, CHAMPION).version) == old
    assert (
        client.get_model_version_by_alias(MODEL, CHAMPION).tags[
            "previous_champion_version"
        ]
        == versions[CHALLENGER]
    )


def test_moving_to_an_unknown_version_changes_nothing(client):
    _log("best", 100, 0.3)
    _log("second", 90, 0.4)
    versions = _register(
        client, MODEL, eligible_candidates(client, _experiment(client), 250)
    )
    with pytest.raises(MlflowException):
        move_champion(client, MODEL, "99", "typo")
    assert (
        str(client.get_model_version_by_alias(MODEL, CHAMPION).version)
        == versions[CHAMPION]
    )


def test_a_child_run_is_named_after_its_parent(client):
    with mlflow.start_run(run_name="Model: parent"):
        child_id = _log("child", 5, 0.1, nested=True)
    _log("other", 1, 0.1)
    names = {
        c.run_id: c.name for c in eligible_candidates(client, _experiment(client), 250)
    }
    assert names[child_id] == "Model: parent"


@pytest.mark.parametrize(
    ("provenance", "error"),
    [
        ({**PROVENANCE, "dataset_sha256_orders": "d" * 64}, "different"),
        ({**PROVENANCE, "validation_start": "2025-01-02"}, "different"),
        ({}, "missing or invalid"),
        ({**PROVENANCE, "dataset_sha256_labels": ""}, "missing or invalid"),
    ],
)
def test_incomparable_validation_runs_are_rejected(client, provenance, error):
    _log("first", 100, 0.3)
    _log("second", 90, 0.4)
    _log("incomparable", 200, 0.5, provenance=provenance)
    with pytest.raises(ValueError, match=error):
        eligible_candidates(client, _experiment(client), 250)


def test_matching_fingerprints_and_fraud_free_ap(client):
    provenance = {**PROVENANCE, "validation_rows_sha256": "d" * 64}
    runs = [
        _log(name, 0, None, total_fraud=0, fraud_catchable=0, provenance=provenance)
        for name in ("a", "b")
    ]
    candidates = eligible_candidates(client, _experiment(client), 250)
    assert [candidate.run_id for candidate in candidates] == sorted(runs)
    table = comparison_table(candidates)
    assert "| 0 | 0 | N/A |" in table
    assert "run ID breaks the tie" in table
    assert "Exact scored-row identity is unverified" not in table


def test_run_search_follows_page_tokens(client, monkeypatch):
    first = _log("first", 100, 0.3)
    best = _log("best", 200, 0.4)
    search = Mock(
        side_effect=[
            PagedList([client.get_run(first)], "next"),
            PagedList([client.get_run(best)], None),
        ]
    )
    monkeypatch.setattr(client, "search_runs", search)
    ranked = eligible_candidates(client, _experiment(client), 250)
    assert [candidate.run_id for candidate in ranked] == [best, first]
    assert [call.kwargs["page_token"] for call in search.call_args_list] == [
        None,
        "next",
    ]


def test_registry_version_search_follows_page_tokens(client, monkeypatch):
    _log("best", 100, 0.3)
    _log("second", 90, 0.4)
    candidates = eligible_candidates(client, _experiment(client), 250)
    expected = _register(client, MODEL, candidates)
    versions = list(client.search_model_versions(f"name='{MODEL}'"))
    search = Mock(
        side_effect=[
            PagedList(versions[:1], "next"),
            PagedList(versions[1:], None),
        ]
    )
    create = Mock()
    monkeypatch.setattr(client, "search_model_versions", search)
    monkeypatch.setattr(client, "create_model_version", create)
    assert _register(client, MODEL, candidates) == expected
    assert [call.kwargs["page_token"] for call in search.call_args_list] == [
        None,
        "next",
    ]
    create.assert_not_called()


@pytest.mark.parametrize(
    "error", [MlflowException("denied", "PERMISSION_DENIED"), TimeoutError("offline")]
)
def test_exceptional_alias_lookup_stops_registry_changes(client, monkeypatch, error):
    _log("best", 100, 0.3)
    _log("second", 90, 0.4)
    ranking = eligible_candidates(client, _experiment(client), 250)
    approval = comparison_id(client, ranking)
    lookup = Mock(side_effect=error)
    create = Mock()
    write_alias = Mock()
    monkeypatch.setattr(client, "get_registered_model", lookup)
    monkeypatch.setattr(client, "create_registered_model", create)
    monkeypatch.setattr(client, "set_registered_model_alias", write_alias)

    with pytest.raises(type(error)) as registration:
        register_top_two(client, MODEL, ranking, approved_comparison=approval)
    with pytest.raises(type(error)) as move:
        move_champion(client, MODEL, "2", "reason")

    assert registration.value is move.value is error
    assert lookup.call_args_list == [call(MODEL), call(MODEL)]
    create.assert_not_called()
    write_alias.assert_not_called()


def test_rollback_from_a_third_version_keeps_both_displaced_versions(client):
    _log("best", 100, 0.3)
    _log("second", 90, 0.4)
    third = _log("third", 80, 0.2)
    ranked = eligible_candidates(client, _experiment(client), 250)
    versions = _register(client, MODEL, ranked)
    target = str(
        client.create_model_version(MODEL, ranked[2].artifact_uri, run_id=third).version
    )
    assert move_champion(client, MODEL, target, "promote third") == versions[CHAMPION]
    assert {
        alias: str(version)
        for alias, version in client.get_registered_model(MODEL).aliases.items()
    } == {
        CHAMPION: target,
        CHALLENGER: versions[CHAMPION],
    }
    previous = client.get_model_version(MODEL, target).tags["previous_champion_version"]
    assert move_champion(client, MODEL, previous, "rollback") == target
    assert {
        alias: str(version)
        for alias, version in client.get_registered_model(MODEL).aliases.items()
    } == {
        CHAMPION: versions[CHAMPION],
        CHALLENGER: target,
    }
    assert len(client.search_model_versions(f"name='{MODEL}'")) == 3


def _installed_cli():
    cli = which("champion-pick", path=str(Path(sys.executable).parent))
    assert cli is not None, "champion-pick must be installed in the test environment"
    return cli


def _cli(*args, cwd):
    return subprocess.run(
        [_installed_cli(), "--k", "250", *args],
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )


def _preview(tmp_path):
    preview = _cli(cwd=tmp_path)
    assert preview.returncode == 0, preview.stderr
    return preview.stdout.split("Comparison ID: ")[1].strip()


def _cli_register(approval, tmp_path):
    return _cli("--register", MODEL, "--approve-comparison", approval, cwd=tmp_path)


def _classifier(value):
    return DummyClassifier(strategy="constant", constant=value).fit([[0], [1]], [0, 1])


def _log_logged_model(run_id, value):
    with mlflow.start_run(run_id=run_id):
        return mlflow.sklearn.log_model(_classifier(value), name="model")


def _log_legacy_model(client, tmp_path, run_id, value, path):
    local = tmp_path / path
    mlflow.sklearn.save_model(_classifier(value), local)
    client.log_artifacts(run_id, str(local), artifact_path=path)
    return f"runs:/{run_id}/{path}"


def _aliases(client):
    return {
        alias: str(version)
        for alias, version in client.get_registered_model(MODEL).aliases.items()
    }


def test_cli_registers_and_loads_only_exact_sources(client, monkeypatch, tmp_path):
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "e")
    winner = _log("winner", 200, 0.3, legacy_manifest=False)
    runner_up = _log("runner-up", 90, 0.4, legacy_manifest=False)
    logged = _log_logged_model(winner, 1)
    location = client.get_logged_model(logged.model_id).artifact_location
    nested = Path(local_file_uri_to_path(location)) / "extra"
    nested.mkdir()
    with (nested / "payload").open("wb") as payload:
        payload.truncate(1_000)
    alternate_source = _log_legacy_model(client, tmp_path, winner, 0, "alternate")
    legacy_source = _log_legacy_model(client, tmp_path, runner_up, 0, "model")

    preview = _cli(cwd=tmp_path)
    assert preview.returncode == 0, preview.stderr
    assert (
        f"| 1 | winner ({winner}) | 200 | 700 | 0.3000 | unknown | yes |"
        in preview.stdout
    )
    size = sum(
        path.stat().st_size for path in nested.parent.rglob("*") if path.is_file()
    )
    assert (
        f"champion (rank 1) artifact: {logged.model_uri} (artifact size: {size:,} bytes)"
        in preview.stdout
    )
    assert "Exact scored-row identity is unverified" in preview.stdout
    assert f"challenger (rank 2) artifact: {legacy_source}" in preview.stdout
    approval = preview.stdout.split("Comparison ID: ")[1].strip()
    assert not client.search_registered_models()
    client.create_registered_model(MODEL)
    alternate = client.create_model_version(
        MODEL, alternate_source, run_id=winner, tags={"artifact": logged.model_uri}
    )
    wrong_layout = client.create_model_version(
        MODEL, f"runs:/{winner}/model", run_id=winner
    )
    legacy = client.create_model_version(MODEL, legacy_source, run_id=runner_up)

    result = _cli_register(approval, tmp_path)
    assert result.returncode == 0, result.stderr
    champion = client.get_model_version_by_alias(MODEL, CHAMPION)
    challenger = client.get_model_version_by_alias(MODEL, CHALLENGER)
    assert champion.version not in {alternate.version, wrong_layout.version}
    assert champion.source == client.get_logged_model(logged.model_id).artifact_location
    assert (
        client.get_model_version_download_uri(MODEL, champion.version)
        == champion.source
    )
    assert challenger.version == legacy.version
    assert champion.tags["resolved_artifact"] == champion.source
    assert client.get_model_version(MODEL, alternate.version).tags == {
        "artifact": logged.model_uri
    }
    assert mlflow.sklearn.load_model(f"models:/{MODEL}@champion").predict(
        np.array([[0]])
    ).tolist() == [1]
    assert mlflow.sklearn.load_model(f"models:/{MODEL}@challenger").predict(
        np.array([[0]])
    ).tolist() == [0]
    assert mlflow.sklearn.load_model(f"models:/{MODEL}/{alternate.version}").predict(
        np.array([[0]])
    ).tolist() == [0]

    # Recording the separately saved copy's size is informational, including on retry.
    client.log_metric(winner, "model_size_bytes", 80_000_000)
    again = _cli_register(approval, tmp_path)
    assert again.returncode == 0, again.stderr
    assert _aliases(client) == {
        CHAMPION: str(champion.version),
        CHALLENGER: str(challenger.version),
    }
    assert len(client.search_model_versions(f"name='{MODEL}'")) == 4


@pytest.mark.parametrize(
    ("defect", "error"),
    [
        ("malformed-k", "invalid recorded K"),
        ("bad-hash", "invalid validation provenance or metrics"),
        ("mixed-hashes", "different validation provenance"),
        ("different-hashes", "different validation provenance"),
        ("fraud-free-ap", "invalid validation provenance or metrics"),
        ("missing-ap", "invalid validation provenance or metrics"),
        ("non-finite-ap", "invalid validation provenance or metrics"),
        ("different-catchable", "different fraud catchable"),
    ],
)
def test_cli_refuses_bad_comparison_before_registry_write(
    client, monkeypatch, tmp_path, defect, error
):
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "e")
    fingerprinted = {**PROVENANCE, "validation_rows_sha256": "d" * 64}
    baseline = (
        fingerprinted if defect in {"mixed-hashes", "different-hashes"} else PROVENANCE
    )
    options = {}
    ap = 0.3
    caught = 100
    if defect == "malformed-k":
        options["k"] = "NaN"
    elif defect == "bad-hash":
        options["provenance"] = {**PROVENANCE, "validation_rows_sha256": ""}
    elif defect == "different-hashes":
        options["provenance"] = {**PROVENANCE, "validation_rows_sha256": "e" * 64}
    elif defect == "fraud-free-ap":
        caught = 0
        ap = 1.0
        options.update(total_fraud=0, fraud_catchable=0)
    elif defect == "missing-ap":
        ap = None
    elif defect == "non-finite-ap":
        ap = float("inf")
    elif defect == "different-catchable":
        options["fraud_catchable"] = 701

    if defect == "fraud-free-ap":
        _log("first", 0, None, total_fraud=0, fraud_catchable=0)
        _log("second", 0, None, total_fraud=0, fraud_catchable=0)
    else:
        _log("winner", 120, 0.3, provenance=baseline)
        _log("runner-up", 110, 0.4, provenance=baseline)
    approval = _preview(tmp_path)
    _log("uncomparable", caught, ap, **options)

    preview = _cli(cwd=tmp_path)
    assert preview.returncode == 1
    assert error in preview.stderr
    registration = _cli_register(approval, tmp_path)
    assert registration.returncode == 1
    assert error in registration.stderr
    assert not client.search_registered_models()


def test_cli_approval_survives_optional_size_inventory_failure(
    client, monkeypatch, tmp_path, capsys
):
    class InventoryUnavailable(Exception):
        pass

    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "e")
    winner = _log("winner", 100, 0.3)
    _log("runner-up", 90, 0.4)
    approval = _preview(tmp_path)

    inventory = Mock(
        side_effect=[
            [FileInfo("MLmodel", False, 5), FileInfo("nested", True, None)],
            InventoryUnavailable("nested listing denied by artifact backend"),
            [FileInfo("MLmodel", False, 5)],
        ]
    )
    monkeypatch.setattr(
        champion,
        "get_artifact_repository",
        lambda *_args, **_kwargs: Mock(list_artifacts=inventory),
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", [_installed_cli(), "--k", "250"])
    with pytest.raises(SystemExit) as exit_status:
        runpy.run_path(_installed_cli(), run_name="__main__")

    assert exit_status.value.code in (None, 0)
    output = capsys.readouterr().out
    assert (
        f"champion (rank 1) artifact: runs:/{winner}/model (artifact size: unknown)"
        in output
    )
    assert f"Comparison ID: {approval}" in output
    assert inventory.call_args_list == [call(None), call("nested"), call(None)]
    assert not client.search_registered_models()


def test_cli_rejects_late_candidate_before_writing(client, monkeypatch, tmp_path):
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "e")
    _log("winner", 100, 0.3)
    _log("runner-up", 90, 0.4)
    approval = _preview(tmp_path)
    assert not client.search_registered_models()

    missing = _cli("--register", MODEL, cwd=tmp_path)
    assert missing.returncode == 2
    assert "--approve-comparison" in missing.stderr
    _log("late winner", 200, 0.6)
    stale = _cli_register(approval, tmp_path)
    assert stale.returncode == 1
    assert "Comparison changed" in stale.stderr
    assert not client.search_registered_models()


def test_cli_rejects_ambiguous_or_changed_elected_source(client, monkeypatch, tmp_path):
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "e")
    winner = _log("winner", 100, 0.3)
    _log("runner-up", 90, 0.4)
    approval = _preview(tmp_path)
    _log_logged_model(winner, 1)

    result = _cli_register(approval, tmp_path)
    assert result.returncode == 1
    assert "exactly one model artifact" in result.stderr
    assert not client.search_registered_models()

    legacy_uri = client.get_run(winner).info.artifact_uri
    (Path(local_file_uri_to_path(legacy_uri)) / "model" / "MLmodel").unlink()
    result = _cli_register(approval, tmp_path)
    assert result.returncode == 1
    assert "Comparison changed" in result.stderr
    assert not client.search_registered_models()


def test_cli_retry_completes_interrupted_registration(client, monkeypatch, tmp_path):
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "e")
    _log("winner", 100, 0.3)
    _log("runner-up", 90, 0.4)
    approval = _preview(tmp_path)
    original_write = MlflowClient.set_registered_model_alias

    def interrupt(client, name, alias, version):
        if alias == CHAMPION:
            raise KeyboardInterrupt
        original_write(client, name, alias, version)

    with monkeypatch.context() as injected:
        injected.setattr(MlflowClient, "set_registered_model_alias", interrupt)
        injected.setattr(
            sys,
            "argv",
            [
                _installed_cli(),
                "--k",
                "250",
                "--register",
                MODEL,
                "--approve-comparison",
                approval,
            ],
        )
        with pytest.raises(KeyboardInterrupt):
            runpy.run_path(_installed_cli(), run_name="__main__")

    before = _aliases(client)
    assert list(before) == [CHALLENGER]
    assert len(client.search_model_versions(f"name='{MODEL}'")) == 2

    retry = _cli_register(approval, tmp_path)
    assert retry.returncode == 0, retry.stderr
    after = _aliases(client)
    assert after[CHALLENGER] == before[CHALLENGER]
    assert after[CHAMPION] != after[CHALLENGER]
    assert len(client.search_model_versions(f"name='{MODEL}'")) == 2


@pytest.mark.parametrize("failure", ["interrupt", "before", "after", "restore"])
def test_move_and_rollback_reconcile_after_write_failure(client, monkeypatch, failure):
    _log("winner", 100, 0.3)
    _log("runner-up", 90, 0.4)
    versions = _register(
        client, MODEL, eligible_candidates(client, _experiment(client), 250)
    )
    first, second = versions[CHAMPION], versions[CHALLENGER]
    desired = {CHAMPION: second, CHALLENGER: first}
    write = client.set_registered_model_alias

    def fail(name, alias, version):
        if alias == CHAMPION:
            if failure == "interrupt":
                raise KeyboardInterrupt
            if failure == "after":
                write(name, alias, version)
            raise TimeoutError("champion write failed")
        if failure == "restore" and version == second:
            raise TimeoutError("restore failed")
        write(name, alias, version)

    with monkeypatch.context() as injected:
        injected.setattr(client, "set_registered_model_alias", fail)
        exception = KeyboardInterrupt if failure == "interrupt" else TimeoutError
        with pytest.raises(exception) as caught:
            move_champion(client, MODEL, second, "promote")
        if failure == "restore":
            assert any("restore failed" in note for note in caught.value.__notes__)

    expected = {
        "interrupt": {CHAMPION: first, CHALLENGER: first},
        "restore": {CHAMPION: first, CHALLENGER: first},
        "before": versions,
        "after": desired,
    }
    assert _aliases(client) == expected[failure]
    move_champion(client, MODEL, second, "promote")
    assert _aliases(client) == desired
    assert (
        client.get_model_version(MODEL, second).tags["previous_champion_version"]
        == first
    )

    assert move_champion(client, MODEL, first, "rollback") == second
    assert _aliases(client) == versions
    assert (
        client.get_model_version(MODEL, first).tags["previous_champion_version"]
        == second
    )


def test_retry_recovers_old_champion_first_partial_swap(client):
    _log("winner", 100, 0.3)
    _log("runner-up", 90, 0.4)
    versions = _register(
        client, MODEL, eligible_candidates(client, _experiment(client), 250)
    )
    first, second = versions[CHAMPION], versions[CHALLENGER]
    client.set_model_version_tag(MODEL, second, "previous_champion_version", first)
    client.set_model_version_tag(MODEL, second, "champion_change_reason", "promote")
    client.set_registered_model_alias(MODEL, CHAMPION, second)

    with pytest.raises(ValueError, match="ambiguous"):
        move_champion(client, MODEL, second, "different move")
    assert _aliases(client) == {CHAMPION: second, CHALLENGER: second}

    assert move_champion(client, MODEL, second, "promote") == first
    assert _aliases(client) == {CHAMPION: second, CHALLENGER: first}
