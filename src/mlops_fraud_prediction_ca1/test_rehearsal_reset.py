from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from .rehearsal_reset import (
    RehearsalRefused,
    check_rehearsal_target,
    clear_experiment,
    presign_batches,
    rebuild,
)


def _settings(db="rehearsal_db", exp="rehearsal-exp"):
    return SimpleNamespace(db_name=db, mlflow_experiment_name=exp)


@pytest.mark.parametrize(
    "settings, environ",
    [
        (_settings(db="fraud"), {}),
        (_settings(exp="fraud"), {}),
        (_settings(db="rehearsal_db"), {"TEAM_DB_NAME": "rehearsal_db"}),
        (
            _settings(exp="rehearsal-exp"),
            {"TEAM_MLFLOW_EXPERIMENT_NAME": "rehearsal-exp"},
        ),
    ],
)
def test_refuses_team_resources(settings, environ):
    with pytest.raises(RehearsalRefused):
        check_rehearsal_target(settings, environ)


def test_accepts_rehearsal_resources():
    check_rehearsal_target(_settings(), {"TEAM_DB_NAME": "fraud"})


def _touch_base(tmp_path):
    for name in ("customers", "orders", "labels"):
        (tmp_path / f"{name}_base.csv").write_text("x\n")


def test_refusal_touches_nothing(tmp_path):
    _touch_base(tmp_path)
    calls = MagicMock()
    with pytest.raises(RehearsalRefused):
        rebuild(
            _settings(db="fraud"),
            {},
            tmp_path,
            engine=calls.engine,
            client=calls.client,
            load_files=calls.load,
            create_tables=calls.create,
            drop_tables=calls.drop,
        )
    assert calls.mock_calls == []


def test_rebuild_order_drop_create_load_clear(tmp_path):
    _touch_base(tmp_path)
    calls = MagicMock()
    exp = SimpleNamespace(experiment_id="1")
    calls.client.get_experiment_by_name.return_value = exp
    run = SimpleNamespace(info=SimpleNamespace(run_id="r1"))
    calls.client.search_runs.side_effect = [[run], []]
    calls.load.return_value = SimpleNamespace(loaded={"orders": 3}, rejected={})
    result = rebuild(
        _settings(),
        {},
        tmp_path,
        engine=calls.engine,
        client=calls.client,
        load_files=calls.load,
        create_tables=calls.create,
        drop_tables=calls.drop,
    )
    names = [c[0] for c in calls.mock_calls if c[0] in {"drop", "create", "load"}]
    assert names == ["drop", "create", "load"]
    assert result["runs_deleted"] == 1
    calls.client.delete_run.assert_called_once_with("r1")


def test_missing_base_file_stops_before_drop(tmp_path):
    calls = MagicMock()
    with pytest.raises(FileNotFoundError):
        rebuild(
            _settings(),
            {},
            tmp_path,
            engine=calls.engine,
            client=calls.client,
            load_files=calls.load,
            create_tables=calls.create,
            drop_tables=calls.drop,
        )
    calls.drop.assert_not_called()


def test_missing_experiment_is_created():
    client = MagicMock()
    client.get_experiment_by_name.return_value = None
    assert clear_experiment(client, "rehearsal-x") == 0
    client.create_experiment.assert_called_once_with("rehearsal-x")


def test_presign_uploads_privately_and_signs(tmp_path):
    batch = tmp_path / "rehearsal_a"
    batch.mkdir()
    (batch / "orders_rehearsal_a.csv").write_text("x\n")
    s3 = MagicMock()
    s3.generate_presigned_url.return_value = "https://signed"
    urls = presign_batches(s3, "bkt", tmp_path, expires=60)
    assert urls == {"rehearsal_a": ["https://signed"]}
    s3.upload_file.assert_called_once()
    assert "ACL" not in str(s3.upload_file.call_args)
    s3.generate_presigned_url.assert_called_once_with(
        "get_object",
        Params={"Bucket": "bkt", "Key": "rehearsal/rehearsal_a/orders_rehearsal_a.csv"},
        ExpiresIn=60,
    )
