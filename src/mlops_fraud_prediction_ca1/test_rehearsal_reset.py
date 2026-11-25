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
        )
    assert calls.mock_calls == []


def test_missing_base_file_stops_before_drop(tmp_path):
    calls = MagicMock()
    with pytest.raises(FileNotFoundError):
        rebuild(_settings(), {}, tmp_path, engine=calls.engine, client=calls.client)
    assert calls.mock_calls == []


def test_rebuild_drops_recreates_and_loads_for_real(database, tmp_path):
    """Uses the real drop, create and load on a test MySQL (skipped without one)."""
    from sqlalchemy import text

    from .rehearsal import build_valid_batch, write_batch

    folder = write_batch(build_valid_batch("rehearsal_a"), tmp_path / "gen")
    for table in ("customers", "orders", "labels"):
        (folder / f"{table}_rehearsal_a.csv").rename(tmp_path / f"{table}_base.csv")
    with database.begin() as connection:
        connection.execute(text("INSERT INTO batches (batch_id) VALUES ('old')"))
    client = MagicMock()
    client.get_experiment_by_name.return_value = SimpleNamespace(experiment_id="1")
    run = SimpleNamespace(info=SimpleNamespace(run_id="r1"))
    client.search_runs.side_effect = [[run], []]
    result = rebuild(_settings(), {}, tmp_path, engine=database, client=client)
    with database.connect() as connection:
        old = connection.execute(
            text("SELECT COUNT(*) FROM batches WHERE batch_id='old'")
        )
        assert old.scalar() == 0  # tables were dropped and recreated
        assert connection.execute(text("SELECT COUNT(*) FROM orders")).scalar() == 120
    assert result["runs_deleted"] == 1
    client.delete_run.assert_called_once_with("r1")


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


def test_presign_against_mocked_s3_keeps_objects_private(tmp_path, monkeypatch):
    import urllib.parse

    import boto3
    from botocore.config import Config
    from moto import mock_aws

    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    (tmp_path / "rehearsal_a").mkdir()
    (tmp_path / "rehearsal_a" / "orders_rehearsal_a.csv").write_text("x\n")
    with mock_aws():
        s3 = boto3.client("s3", config=Config(signature_version="s3v4"))
        s3.create_bucket(Bucket="rehearsal-test-bucket")
        urls = presign_batches(s3, "rehearsal-test-bucket", tmp_path, expires=120)
        query = urllib.parse.parse_qs(
            urllib.parse.urlsplit(urls["rehearsal_a"][0]).query
        )
        assert query["X-Amz-Expires"] == ["120"]
        grants = s3.get_object_acl(
            Bucket="rehearsal-test-bucket",
            Key="rehearsal/rehearsal_a/orders_rehearsal_a.csv",
        )["Grants"]
        assert all(g["Grantee"].get("URI") is None for g in grants)


def _stub_path(tmp_path, bucket_exists, tagged):
    import os
    import stat

    calls = tmp_path / "calls.log"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    aws = bin_dir / "aws"
    aws.write_text(
        "#!/usr/bin/env bash\n"
        'echo "aws $*" >> "$CALLS"\n'
        'case "$*" in\n'
        f"  *head-bucket*) exit {0 if bucket_exists else 1};;\n"
        f"  *get-bucket-tagging*) echo {'rehearsal-live' if tagged else 'other'};;\n"
        "esac\n"
    )
    curl = bin_dir / "curl"
    curl.write_text("#!/usr/bin/env bash\necho 203.0.113.9\n")
    uv = bin_dir / "uv"
    uv.write_text('#!/usr/bin/env bash\necho "uv $*" >> "$CALLS"; echo 0\n')
    for tool in (aws, curl, uv):
        tool.chmod(tool.stat().st_mode | stat.S_IEXEC)
    return {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "CALLS": str(calls),
    }, calls


def test_live_script_refuses_an_existing_bucket(tmp_path):
    import subprocess
    from pathlib import Path

    env, calls = _stub_path(tmp_path, bucket_exists=True, tagged=False)
    script = Path(__file__).parents[2] / "scripts" / "rehearsal-live.sh"
    result = subprocess.run(  # noqa: PLW1510
        ["bash", str(script), "run"], env=env, capture_output=True, text=True
    )
    assert result.returncode == 1 and "refusing" in result.stderr
    log = calls.read_text() if calls.exists() else ""
    assert "authorize-security-group-ingress" not in log and "delete-bucket" not in log


def test_live_cleanup_leaves_a_bucket_it_did_not_tag(tmp_path):
    import subprocess
    from pathlib import Path

    env, calls = _stub_path(tmp_path, bucket_exists=True, tagged=False)
    script = Path(__file__).parents[2] / "scripts" / "rehearsal-live.sh"
    result = subprocess.run(  # noqa: PLW1510
        ["bash", str(script), "cleanup"],
        env={**env, "KEEP_SCHEMA": "1"},
        capture_output=True,
        text=True,
    )
    assert "left alone" in result.stdout
    assert "delete-bucket" not in calls.read_text() and "s3 rm" not in calls.read_text()
