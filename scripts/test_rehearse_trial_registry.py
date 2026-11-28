import json

import pytest
import rehearse_trial_registry as trial
from mlflow.exceptions import MlflowException


def client_for(root):
    uri = f"sqlite:///{root / 'registry.sqlite'}"
    return trial.MlflowClient(tracking_uri=uri, registry_uri=uri)


def test_fresh_registry_and_idempotent_readback(tmp_path):
    result = trial.rehearse(tmp_path)
    assert result["aliases"] == {"champion": "1", "challenger": "2"}
    assert result["previous_champion_version"] == {"1": "2", "2": "1"}
    assert result["run_ids"] == {
        "1": "9c3a59771cf8499d92972f5c7873e280",
        "2": "9a40f8f5e35a48ddb39a97f9f052dcad",
        "3": "8af9db9f95e543cfbfebdef82545322c",
    }
    assert result["extra_trees"] == {"status": "registered_unaliased", "version": "3"}
    assert trial.rehearse(tmp_path) == result


def test_resumes_matching_partial_state(tmp_path):
    client = client_for(tmp_path)
    client.create_registered_model(trial.MODEL)
    trial.prepare_package(tmp_path, "1")
    trial.create_version(client, tmp_path, "1")
    client.set_registered_model_alias(trial.MODEL, "champion", "1")

    result = trial.rehearse(tmp_path)
    assert result["aliases"] == {"champion": "1", "challenger": "2"}
    assert result["extra_trees"]["version"] == "3"


@pytest.mark.parametrize("mismatch", ["source", "alias", "rollback_tag"])
def test_refuses_mismatched_state_before_registry_writes(tmp_path, mismatch):
    client = client_for(tmp_path)
    client.create_registered_model(trial.MODEL)
    trial.prepare_package(tmp_path, "1")
    source = trial.package(tmp_path, "1") if mismatch != "source" else tmp_path
    client.create_model_version(
        trial.MODEL,
        source.as_uri(),
        run_id=trial.CANDIDATES["1"][0],
        tags=trial.tags_for("1"),
    )
    if mismatch == "alias":
        client.set_registered_model_alias(trial.MODEL, "challenger", "1")
    if mismatch == "rollback_tag":
        client.set_model_version_tag(trial.MODEL, "1", "previous_champion_version", "9")
    before = client.get_registered_model(trial.MODEL)

    with pytest.raises(ValueError, match="provenance or tags|alias bindings"):
        trial.rehearse(tmp_path)

    after = client.get_registered_model(trial.MODEL)
    assert (after.aliases, after.tags) == (before.aliases, before.tags)
    assert [str(row.version) for row in client.search_model_versions()] == ["1"]


def test_extra_trees_failure_preserves_baseline_and_is_never_retried(
    tmp_path, monkeypatch, capsys
):
    create = trial.MlflowClient.create_model_version
    attempted = []

    def fail_extra(client, name, source, *, run_id, tags):
        if run_id == trial.CANDIDATES["3"][0]:
            attempted.append(run_id)
            store = client._get_registry_client().store
            assert store.CREATE_MODEL_VERSION_RETRIES == 1
            trial.verify_baseline(client, tmp_path)
            raise MlflowException("simulated Extra Trees registry failure")
        return create(client, name, source, run_id=run_id, tags=tags)

    monkeypatch.setattr(trial.MlflowClient, "create_model_version", fail_extra)
    monkeypatch.setattr("sys.argv", ["rehearse_trial_registry.py", str(tmp_path)])

    trial.main()
    result = json.loads(capsys.readouterr().out)
    assert result["aliases"] == {"champion": "1", "challenger": "2"}
    assert result["previous_champion_version"] == {"1": "2", "2": "1"}
    assert result["extra_trees"] == {
        "status": "fallback",
        "reason": "MlflowException: simulated Extra Trees registry failure",
    }
    assert set(result["run_ids"]) == {"1", "2"}
    assert trial.rehearse(tmp_path) == result
    assert attempted == ["8af9db9f95e543cfbfebdef82545322c"]
    with capsys.disabled():
        print(json.dumps(result, sort_keys=True))  # noqa: T201


def test_failed_response_with_committed_candidate_reads_back_success(
    tmp_path, monkeypatch
):
    create = trial.MlflowClient.create_model_version

    def fail_after_commit(client, name, source, *, run_id, tags):
        row = create(client, name, source, run_id=run_id, tags=tags)
        if run_id == trial.CANDIDATES["3"][0]:
            raise MlflowException("simulated lost registry response")
        return row

    monkeypatch.setattr(trial.MlflowClient, "create_model_version", fail_after_commit)
    result = trial.rehearse(tmp_path)
    assert result["aliases"] == {"champion": "1", "challenger": "2"}
    assert result["extra_trees"] == {"status": "registered_unaliased", "version": "3"}
