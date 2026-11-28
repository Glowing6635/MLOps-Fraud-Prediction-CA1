import os
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "rehearsal-live.sh"


@pytest.fixture
def rehearsal(tmp_path):
    bucket = tmp_path / "bucket"
    team_rule = tmp_path / "sgr-b456"
    team_rule.touch()
    stubs = {
        "aws": """#!/bin/bash
printf '%s %s\n' "$1" "$2" >> "$CALLS"
case "$1 $2" in
  's3api get-bucket-tagging')
    if [ -n "$TAG_ERROR" ]; then
      printf '\nAn error occurred (%s) when calling the GetBucketTagging operation: stub diagnostic\n' "$TAG_ERROR" >&2
      exit 254
    fi
    if [ ! -f "$STATE/bucket" ]; then
      echo 'An error occurred (NoSuchBucket) when calling the GetBucketTagging operation: missing' >&2
      exit 254
    fi
    printf '%s\n' "$OWNER_TAG"
    ;;
  's3api head-bucket') exit 254 ;;
  'ec2 authorize-security-group-ingress')
    if [ -n "$INGRESS_ERROR" ]; then echo "$INGRESS_ERROR" >&2; exit 254; fi
    [ ! -f "$STATE/sgr-a123" ] || exit 98
    touch "$STATE/sgr-a123"
    echo sgr-a123
    ;;
  'ec2 revoke-security-group-ingress')
    [ "$5" = --security-group-rule-ids ] && [ "$6" = sgr-a123 ] || exit 98
    if [ -n "$REVOKE_ERROR" ]; then echo "$REVOKE_ERROR" >&2; exit 254; fi
    rm "$STATE/$6" && echo True
    ;;
  's3 rm'|'s3api put-bucket-tagging') ;;
  's3api delete-bucket') rm "$STATE/bucket" ;;
  's3api create-bucket') touch "$STATE/bucket" ;;
  's3api put-public-access-block') echo 'original run failure' >&2; exit 42 ;;
  *) exit 99 ;;
esac
""",
        "curl": '#!/bin/bash\necho curl >> "$CALLS"\necho 192.0.2.1\n',
        "uv": """#!/bin/bash
printf 'uv %s\n' "${*: -1}" >> "$CALLS"
case "${*: -1}" in
  'DROP DATABASE IF EXISTS rehearsal')
    [ -f "$STATE/sgr-a123" ] || exit 98
    if [ -n "$SQL_ERROR" ]; then echo "$SQL_ERROR" >&2; exit 42; fi
    rm "$STATE/schema"
    ;;
esac
""",
    }
    for name, content in stubs.items():
        stub = tmp_path / name
        stub.write_text(content)
        stub.chmod(0o755)

    def run(command, *, owner="rehearsal-live", error="", **settings):
        log = tmp_path / "calls"
        log.write_text("")
        result = subprocess.run(
            ["bash", str(SCRIPT), command],
            env={
                "PATH": f"{tmp_path}:{os.defpath}",
                "BUCKET": "rehearsal-test",
                "CALLS": str(log),
                "STATE": str(tmp_path),
                "OWNER_TAG": owner,
                "TAG_ERROR": error,
                **settings,
            },
            capture_output=True,
            check=False,
            text=True,
            timeout=10,
        )
        calls = log.read_text().splitlines()
        assert team_rule.exists()
        return result, calls

    bucket.touch()
    (tmp_path / "schema").touch()
    return run


@pytest.mark.parametrize(("command", "status"), [("cleanup", 1), ("run", 42)])
def test_denied_lookup_reports_cleanup_failure(rehearsal, command, status):
    result, calls = rehearsal(command, error="AccessDenied")
    assert result.returncode == status
    assert "s3 rm" not in calls
    assert "s3api delete-bucket" not in calls
    assert "Cannot verify ownership of bucket rehearsal-test" in result.stderr
    assert "AccessDenied" in result.stderr
    assert "stub diagnostic" in result.stderr
    assert "CLEANUP FAILED: delete-bucket" in result.stderr
    assert "CLEANUP INCOMPLETE" in result.stderr
    if command == "run":
        assert "original run failure" in result.stderr
        assert calls.count("ec2 revoke-security-group-ingress") == 1


@pytest.mark.parametrize("error", ["NoSuchBucket", "NoSuchTagSet"])
def test_absent_bucket_or_tags_is_successful_skip(rehearsal, error):
    result, calls = rehearsal("cleanup", error=error)
    assert result.returncode == 0
    assert "s3 rm" not in calls
    assert "s3api delete-bucket" not in calls
    assert "absent or has no tags: left alone" in result.stdout
    assert not result.stderr


def test_mismatched_owner_is_successful_skip(rehearsal):
    result, calls = rehearsal("cleanup", owner="someone-else")
    assert result.returncode == 0
    assert "s3 rm" not in calls
    assert "s3api delete-bucket" not in calls
    assert "not created by this script: left alone" in result.stdout
    assert not result.stderr


def test_owned_bucket_is_emptied_then_repeat_skips_without_ingress(rehearsal, tmp_path):
    result, calls = rehearsal("cleanup")
    assert result.returncode == 0
    assert calls == ["s3api get-bucket-tagging", "s3 rm", "s3api delete-bucket"]
    assert "deleted bucket rehearsal-test" in result.stdout
    assert not result.stderr
    assert not (tmp_path / "bucket").exists()
    assert (tmp_path / "schema").exists()

    result, calls = rehearsal("cleanup")
    assert result.returncode == 0
    assert calls == ["s3api get-bucket-tagging"]
    assert "absent or has no tags" in result.stdout
    assert not result.stderr


@pytest.mark.parametrize(
    "denial", ["UnauthorizedOperation", "InvalidPermission.Duplicate"]
)
def test_denied_sql_ingress_still_cleans_bucket(rehearsal, tmp_path, denial):
    result, calls = rehearsal("cleanup", KEEP_SCHEMA="0", INGRESS_ERROR=denial)
    assert result.returncode == 1
    assert denial in result.stderr
    assert "CLEANUP FAILED: drop-schema" in result.stderr
    assert "ec2 authorize-security-group-ingress" in calls
    assert "ec2 revoke-security-group-ingress" not in calls
    assert not (tmp_path / "bucket").exists()
    assert not (tmp_path / "sgr-a123").exists()
    assert (tmp_path / "schema").exists()


@pytest.mark.parametrize(
    ("settings", "failed_step", "bucket_remains", "schema_remains", "rule_remains"),
    [
        ({"KEEP_BUCKET": "1"}, "", True, False, False),
        (
            {"KEEP_BUCKET": "1", "SQL_ERROR": "SQL denied"},
            "drop-schema",
            True,
            True,
            False,
        ),
        (
            {"KEEP_BUCKET": "1", "REVOKE_ERROR": "EC2 denied"},
            "revoke-ingress",
            True,
            False,
            True,
        ),
        ({"error": "AccessDenied"}, "delete-bucket", True, False, False),
    ],
)
def test_explicit_schema_cleanup_reconciles_only_owned_rule(
    rehearsal,
    tmp_path,
    settings,
    failed_step,
    bucket_remains,
    schema_remains,
    rule_remains,
):
    result, calls = rehearsal("cleanup", KEEP_SCHEMA="0", **settings)
    assert result.returncode == bool(failed_step)
    if failed_step:
        assert f"CLEANUP FAILED: {failed_step}" in result.stderr
    if rule_remains:
        assert "Failed to confirm removal of ingress sgr-a123" in result.stderr
    if settings.get("KEEP_BUCKET"):
        assert calls == [
            "curl",
            "ec2 authorize-security-group-ingress",
            "uv DROP DATABASE IF EXISTS rehearsal",
            "ec2 revoke-security-group-ingress",
        ]
    assert (tmp_path / "bucket").exists() == bucket_remains
    assert (tmp_path / "schema").exists() == schema_remains
    assert (tmp_path / "sgr-a123").exists() == rule_remains


def test_cleanup_refuses_configured_team_schema(rehearsal, tmp_path):
    result, calls = rehearsal("cleanup", KEEP_SCHEMA="0", TEAM_DB_NAME="rehearsal")
    assert result.returncode == 1
    assert "Refusing to drop the team schema rehearsal" in result.stderr
    assert calls == ["s3api get-bucket-tagging", "s3 rm", "s3api delete-bucket"]
    assert (tmp_path / "schema").exists()
