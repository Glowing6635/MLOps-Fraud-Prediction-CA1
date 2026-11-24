"""Champion selection uses eligible runs only and moves the alias with a recorded previous version."""

from unittest.mock import Mock, call

import mlflow
import pytest
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from mlflow.store.entities.paged_list import PagedList

from mlops_fraud_prediction_ca1.champion import (
    CHALLENGER,
    CHAMPION,
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
):
    with mlflow.start_run(run_name=name, nested=nested) as run:
        mlflow.log_params({"k": k, "exploratory": exploratory, **provenance})
        mlflow.set_tags({"champion_eligible": eligible, "git_dirty": dirty})
        mlflow.log_metrics(
            {
                "fraud_caught_at_k": caught,
                "average_precision": ap,
                "validation_orders": 2000,
                "total_fraud_at_k": 1000,
            }
        )
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


def test_exploratory_other_k_and_ineligible_runs_are_not_candidates(client):
    good_a = _log("a", 100, 0.3)
    good_b = _log("b", 90, 0.4)
    _log("exp", 500, 0.9, eligible="False", exploratory="True")
    _log("otherk", 400, 0.9, k="50", provenance={})
    _log("dirty", 600, 0.9, dirty="True", provenance={})
    candidates = eligible_candidates(client, _experiment(client), 250)
    assert [c.run_id for c in candidates] == [good_a, good_b]


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
    versions = register_top_two(client, MODEL, candidates)
    assert len(client.search_model_versions(f"name='{MODEL}'")) == 2
    champion = client.get_model_version_by_alias(MODEL, CHAMPION)
    assert champion.run_id == best
    assert champion.tags["artifact"] == f"runs:/{best}/model"
    assert client.get_model_version_by_alias(MODEL, CHALLENGER).run_id == second
    assert register_top_two(client, MODEL, candidates) == versions
    assert len(client.search_model_versions(f"name='{MODEL}'")) == 2


def test_alias_moves_to_the_second_model_and_back(client):
    _log("best", 100, 0.3)
    _log("second", 90, 0.4)
    versions = register_top_two(
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
    versions = register_top_two(
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
    expected = register_top_two(client, MODEL, candidates)
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
    assert register_top_two(client, MODEL, candidates) == expected
    assert [call.kwargs["page_token"] for call in search.call_args_list] == [
        None,
        "next",
    ]
    create.assert_not_called()


@pytest.mark.parametrize(
    "error", [MlflowException("denied", "PERMISSION_DENIED"), TimeoutError("offline")]
)
def test_exceptional_alias_lookup_stops_registry_changes(error):
    client = Mock()
    client.get_registered_model.side_effect = error
    for operation, args in (
        (register_top_two, ([],)),
        (move_champion, ("2", "reason")),
    ):
        client.reset_mock()
        with pytest.raises(type(error)) as caught:
            operation(client, MODEL, *args)
        assert caught.value is error
        assert client.method_calls == [call.get_registered_model(MODEL)]


def test_rollback_from_a_third_version_keeps_both_displaced_versions(client):
    _log("best", 100, 0.3)
    _log("second", 90, 0.4)
    third = _log("third", 80, 0.2)
    ranked = eligible_candidates(client, _experiment(client), 250)
    versions = register_top_two(client, MODEL, ranked)
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
