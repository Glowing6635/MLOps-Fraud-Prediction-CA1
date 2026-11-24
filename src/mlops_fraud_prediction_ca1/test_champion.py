"""Champion selection uses eligible runs only and moves the alias with a recorded previous version."""

import mlflow
import pytest
from mlflow import MlflowClient

from mlops_fraud_prediction_ca1.champion import (
    CHALLENGER,
    CHAMPION,
    comparison_table,
    eligible_candidates,
    move_champion,
    register_top_two,
)

MODEL = "fraud-candidate"


def _log(name, caught, ap, *, eligible="True", exploratory="False", k="250"):
    with mlflow.start_run(run_name=name) as run:
        mlflow.log_params({"k": k, "exploratory": exploratory, "model_class": name})
        mlflow.set_tag("champion_eligible", eligible)
        mlflow.log_metrics({"fraud_caught_at_k": caught, "average_precision": ap})
        mlflow.log_text("model", "model/MLmodel")
    return run.info.run_id


@pytest.fixture
def client(tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'm.db'}")
    mlflow.set_experiment("e")
    return MlflowClient()


def _experiment(client):
    return [client.get_experiment_by_name("e").experiment_id]


def test_exploratory_other_k_and_ineligible_runs_are_not_candidates(client):
    good_a = _log("a", 100, 0.3)
    good_b = _log("b", 90, 0.4)
    _log("exp", 500, 0.9, eligible="False", exploratory="True")
    _log("otherk", 400, 0.9, k="50")
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
    with pytest.raises(Exception):  # noqa: B017
        move_champion(client, MODEL, "99", "typo")
    assert (
        str(client.get_model_version_by_alias(MODEL, CHAMPION).version)
        == versions[CHAMPION]
    )


def test_a_child_run_is_named_after_its_parent(client):
    with (
        mlflow.start_run(run_name="Model: parent"),
        mlflow.start_run(run_name="child", nested=True) as child,
    ):
        mlflow.log_params({"k": "250", "exploratory": "False"})
        mlflow.set_tag("champion_eligible", "True")
        mlflow.log_metrics({"fraud_caught_at_k": 5, "average_precision": 0.1})
        mlflow.log_text("model", "model/MLmodel")
    _log("other", 1, 0.1)
    names = {
        c.run_id: c.name for c in eligible_candidates(client, _experiment(client), 250)
    }
    assert names[child.info.run_id] == "Model: parent"
