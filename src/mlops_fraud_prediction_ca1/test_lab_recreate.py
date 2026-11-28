"""Offline entry-point and scoped AWS/MySQL contract checks."""

import hashlib
import io
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import pytest

ROOT = Path(__file__).resolve().parents[2] / "mlflow-server"
# Match the standalone script's sibling imports without leaving its directory
# on the default suite's import path.
with pytest.MonkeyPatch.context() as imports:
    imports.syspath_prepend(str(ROOT / "lab"))
    import common
    import infrastructure
    import recovery
    from common import AWS, GIB, TAG, Approval, Stop, json_bytes
    from infrastructure import DB_CLASS, ENGINE, IMAGE_TAG, PARAMETERS, Stack
    from recovery import KEYS, OMISSIONS, Recovery, portable_paths

ACCOUNT = "111111111111"
OTHER = "222222222222"
DB_SG = "sg-11111111"
EB_SG = "sg-22222222"
SUBNET = "subnet-12345678"
DIGEST = "sha256:" + "a" * 64
ARTIFACT = "artifacts/3/run/test.txt"
NOW = datetime(2026, 11, 27, 12, tzinfo=UTC)


@pytest.fixture
def private(tmp_path):
    path = tmp_path.resolve() / "private"
    path.mkdir(mode=0o700)
    password = path / "secret"
    password.write_text("Fake-Secret-For-Tests-And-Not-A-Password")
    password.chmod(0o600)
    return path


def approval(operation, instant=None):
    instant = instant or common.now()
    return {
        "reference": "synthetic-scope",
        "operation": operation,
        "start_utc": (instant - timedelta(minutes=1)).isoformat(),
        "work_end_utc": (instant + timedelta(hours=2)).isoformat(),
        "lab_end_utc": (instant + timedelta(hours=3)).isoformat(),
        "credential_expiry_utc": (instant + timedelta(hours=3)).isoformat(),
        "observation_minutes": 10,
        "cleanup_minutes": 30,
        "retained_hours": 4,
        "available_credit_usd": "34.8",
        "credit_limit_usd": "50",
        "unposted_and_concurrent_usd": "1",
        "trial_spent_and_committed_usd": "0.2",
        "trial_remaining_bound_usd": "5",
        "extra_cost_bound_usd": "0.7",
        "ce_calls_bound": 10,
    }


def config(private, operation="create-lab/foundation"):
    return {
        "profile": "dummy",
        "account": ACCOUNT,
        "trial": "trialone",
        "region": "us-east-1",
        "private_dir": str(private),
        "quota_and_rate_reviewed": True,
        "private_workspace_reviewed": True,
        "approval": approval(operation),
        "cname": "lab-recreate-test",
        "db_security_group": DB_SG,
        "eb_security_group": EB_SG,
        "db_subnet_group": "default",
        "eb_subnets": [SUBNET],
        "eb_http_hosts": ["192.0.2.1/32"],
        "db_client_hosts": [],
        "db_public": False,
        "rds_master_user": "fraud_admin",
        "rds_password_file": str(private / "secret"),
        "rds_wait_minutes": 15,
        "eb_wait_minutes": 15,
        "image_max_bytes": 512 * 1024**2,
        "image_build_minutes": 10,
        "docker_capacity_dir": str(private),
        "docker_socket": "/var/run/docker.sock",
        "image_digest": DIGEST,
        "runtime_inputs_verified": True,
        "iam_creation_prevented": True,
        "scope_and_quiescence_verified": True,
        "db_dump_max_bytes": 1024**2,
        "artifact_max_bytes": 1024**2,
        "max_objects": 10,
        "minimum_mib_per_second": 10,
        "local_cost_bound_usd": "0",
        "backup_name": "synthetic-backup",
        "source_db": "fraud-db",
        "source_bucket": "synthetic-artifacts",
        "data_user": "test_reader",
        "data_password_file": str(private / "secret"),
        "platform_arn": (
            "arn:aws:elasticbeanstalk:us-east-1::platform/"
            "Docker running on 64bit Amazon Linux 2023/4.14.3"
        ),
    }


def write_private(path, data):
    path.write_bytes(data)
    path.chmod(0o600)


def cli(private, settings, *args, env=None):
    path = private / "inputs.json"
    write_private(path, json_bytes(settings))
    return subprocess.run(
        ["bash", str(ROOT / "lab-recreate"), *args, "--config", str(path)],
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )


def test_offline_entry_points_do_not_call_aws_or_create_data(private, tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    sentinel = tmp_path / "aws-was-called"
    for name in ("aws", "docker"):
        exe = fake_bin / name
        exe.write_text(f"#!/bin/sh\ntouch '{sentinel}'\nexit 8\n")
        exe.chmod(0o700)
    env = dict(os.environ, PATH=f"{fake_bin}:{os.environ['PATH']}")
    settings = config(private)
    settings["runtime_parameter_prefix"] = "/p26-trialone/mlflow/"
    for phase in ("foundation", "image", "server"):
        settings["approval"] = approval(f"create-lab/{phase}")
        result = cli(
            private, settings, "create-lab", "--phase", phase, "--offline", env=env
        )
        assert result.returncode == 0, (result.stdout, result.stderr)
        assert ACCOUNT not in result.stdout and "Fake-Secret" not in result.stdout
    settings["approval"] = approval("backup-data")
    result = cli(private, settings, "backup-data", "--offline", env=env)
    assert result.returncode == 0, result.stderr
    assert {p.name for p in private.iterdir()} == {"secret", "inputs.json"}
    settings["approval"] = approval("verify/stack")
    assert cli(private, settings, "verify", "--offline", env=env).returncode == 0
    assert not sentinel.exists()


@pytest.mark.parametrize(
    "prefix", ["/p26-othertrial/mlflow/", "/mlflow/", "/p26-trialone/mlflow", None]
)
def test_offline_cli_rejects_runtime_prefix_outside_exact_trial(private, prefix):
    settings = config(private, "create-lab/server")
    settings["runtime_parameter_prefix"] = prefix
    result = cli(private, settings, "create-lab", "--phase", "server", "--offline")
    assert result.returncode == 1
    assert "runtime_parameter_prefix must be /p26-<trial>/mlflow/" in result.stderr


def test_cli_wrong_account_stops_before_mutation(private, tmp_path):
    exe = tmp_path / "aws"
    trace = tmp_path / "trace"
    exe.write_text(
        f"#!{sys.executable}\n"
        "import json, sys\n"
        "if 'configure' in sys.argv: sys.exit(1)\n"
        f"with open({str(trace)!r}, 'a') as f: f.write(' '.join(sys.argv[1:]) + '\\n')\n"
        f"print(json.dumps({{'Account': {OTHER!r}, 'Arn': 'wrong-account-sensitive-marker'}}))\n"
    )
    exe.chmod(0o700)
    env = dict(os.environ, PATH=f"{tmp_path}:{os.environ['PATH']}")
    result = cli(
        private,
        config(private),
        "create-lab",
        "--phase",
        "foundation",
        "--apply",
        env=env,
    )
    assert result.returncode == 1
    assert "account does not match" in result.stderr
    assert "wrong-account-sensitive-marker" not in (result.stdout + result.stderr)
    calls = trace.read_text().splitlines()
    assert len(calls) == 1 and "sts get-caller-identity" in calls[0]


def test_default_restore_checks_bundle_before_network(private):
    settings = destination_config(private, make_bundle(private))
    result = cli(private, settings, "restore-data", "--offline")
    assert result.returncode == 0, result.stderr
    assert "conditional object puts" in result.stdout
    assert ACCOUNT not in result.stdout and "synthetic-artifacts" not in result.stdout
    artifact = private / settings["backup_name"] / ARTIFACT
    artifact.write_bytes(b"FAIL")  # Same byte count as the approved object.
    result = cli(private, settings, "restore-data", "--offline")
    assert result.returncode == 1
    assert "size or checksum" in result.stderr


def test_verify_data_offline_and_wrong_source(private):
    settings = destination_config(private, make_bundle(private))
    settings["approval"] = approval("verify/data")
    result = cli(private, settings, "verify", "--data", "--offline")
    assert result.returncode == 0, result.stderr
    assert "verify restored rows" in result.stdout
    settings["account"] = OTHER
    result = cli(private, settings, "verify", "--data", "--offline")
    assert result.returncode == 1
    assert "restore to the source account" in result.stderr


@pytest.mark.parametrize(
    "keys",
    [
        ["artifacts/../secret"],
        ["artifacts/x", "artifacts/x/a"],
        ["artifacts/X/a", "artifacts/x/b"],
        ["artifacts/CON.txt"],
        ["artifacts/a/"],
    ],
)
def test_unsafe_artifact_paths_are_rejected(keys):
    with pytest.raises(Stop):
        portable_paths(keys)


class Guard:
    def __init__(self):
        self.work_end = common.now() + timedelta(hours=2)
        self.ce = 0

    def check(self, minutes=0):
        return None

    def estimate(self, **kwargs):
        return None

    def check_ce(self, aws):
        self.ce += 1


class FoundationAWS:
    account = ACCOUNT

    def __init__(self):
        self.writes = []
        self.calls = []
        self.rds = False
        self.params = False
        self.db_group = False
        self.repo = False
        self.buckets = {}
        self.failure = None

    def call(self, service, action, *args, **kwargs):
        self.calls.append((service, action))
        if kwargs.get("change"):
            return self.write(service, action, *args)
        if (service, action) == ("rds", "describe-db-parameter-groups"):
            return (
                {
                    "DBParameterGroups": [
                        {
                            "DBParameterGroupFamily": "mysql8.4",
                            "DBParameterGroupArn": "private-arn",
                        }
                    ]
                }
                if self.db_group
                else None
            )
        if action == "list-tags-for-resource":
            return {
                "TagList" if service == "rds" else "tags": [
                    {"Key": TAG, "Value": "trialone"}
                ]
            }
        if action == "describe-db-parameters":
            pairs = PARAMETERS if self.params else {}
            return {
                "Parameters": [
                    {"ParameterName": k, "ParameterValue": v} for k, v in pairs.items()
                ]
            }
        if action == "describe-db-instances":
            if not self.rds:
                return None
            return {
                "DBInstances": [
                    {
                        "DBInstanceArn": f"arn:aws:rds:us-east-1:{ACCOUNT}:db:p26-trialone-db",
                        "TagList": [{"Key": TAG, "Value": "trialone"}],
                        "Engine": "mysql",
                        "EngineVersion": ENGINE,
                        "DBInstanceClass": DB_CLASS,
                        "AllocatedStorage": 20,
                        "StorageType": "gp3",
                        "Iops": 3000,
                        "StorageThroughput": 125,
                        "StorageEncrypted": True,
                        "KmsKeyId": "aws-managed-key-fixture",
                        "MultiAZ": False,
                        "PubliclyAccessible": False,
                        "BackupRetentionPeriod": 1,
                        "DeletionProtection": True,
                        "AutoMinorVersionUpgrade": True,
                        "IAMDatabaseAuthenticationEnabled": False,
                        "MasterUsername": "fraud_admin",
                        "VpcSecurityGroups": [
                            {"VpcSecurityGroupId": DB_SG, "Status": "active"}
                        ],
                        "DBParameterGroups": [
                            {
                                "DBParameterGroupName": "p26-trialone-mysql84",
                                "ParameterApplyStatus": "in-sync",
                            }
                        ],
                        "DBSubnetGroup": {"DBSubnetGroupName": "default"},
                        "DBInstanceStatus": "available",
                        "Endpoint": {"Port": 3306, "Address": "private-endpoint"},
                    }
                ]
            }
        if (service, action) == ("kms", "describe-key"):
            return {"KeyMetadata": {"KeyManager": "AWS", "Enabled": True}}
        if action == "describe-account-attributes":
            return {
                "AccountQuotas": [
                    {"AccountQuotaName": k, "Max": n, "Used": 0}
                    for k, n in (
                        ("DBParameterGroups", 1),
                        ("DBInstances", 1),
                        ("AllocatedStorage", 30),
                    )
                ]
            }
        if action == "describe-orderable-db-instance-options":
            return {
                "OrderableDBInstanceOptions": [
                    {
                        "StorageType": "gp3",
                        "SupportsStorageEncryption": True,
                        "MinStorageSize": 20,
                        "MaxStorageSize": 100,
                        "Vpc": True,
                    }
                ]
            }
        if action == "describe-repositories":
            return (
                {
                    "repositories": [
                        {
                            "repositoryArn": "private-ecr",
                            "registryId": ACCOUNT,
                            "imageTagMutability": "IMMUTABLE",
                            "imageScanningConfiguration": {"scanOnPush": True},
                            "encryptionConfiguration": {"encryptionType": "AES256"},
                        }
                    ]
                }
                if self.repo
                else None
            )
        if service == "s3api":
            name = args[args.index("--bucket") + 1]
            if action == "head-bucket":
                return {} if name in self.buckets else None
            if action == "get-bucket-tagging":
                return (
                    {"TagSet": [{"Key": TAG, "Value": "trialone"}]}
                    if self.buckets[name]
                    else None
                )
            if action == "get-public-access-block":
                return {
                    "PublicAccessBlockConfiguration": dict.fromkeys(
                        (
                            "BlockPublicAcls",
                            "IgnorePublicAcls",
                            "BlockPublicPolicy",
                            "RestrictPublicBuckets",
                        ),
                        True,
                    )
                }
            if action == "get-bucket-ownership-controls":
                return {
                    "OwnershipControls": {
                        "Rules": [{"ObjectOwnership": "BucketOwnerEnforced"}]
                    }
                }
            if action == "get-bucket-encryption":
                return {
                    "ServerSideEncryptionConfiguration": {
                        "Rules": [
                            {
                                "ApplyServerSideEncryptionByDefault": {
                                    "SSEAlgorithm": "AES256"
                                }
                            }
                        ]
                    }
                }
            if action in {"get-bucket-location", "get-bucket-versioning"}:
                return {}
            if kwargs.get("absent"):
                return None
        raise AssertionError((service, action))

    def write(self, service, action, *args):
        self.writes.append((service, action))
        if action == self.failure:
            raise Stop("simulated partial service failure")
        if action == "create-db-parameter-group":
            self.db_group = True
        elif action == "modify-db-parameter-group":
            self.params = True
        elif action == "create-db-instance":
            content = json.loads(Path(args[-1].removeprefix("file://")).read_bytes())
            assert content["MasterUserPassword"].startswith("Fake-Secret")
            assert "Fake-Secret" not in " ".join(args)
            self.rds = True
        elif action == "create-bucket":
            self.buckets[args[1]] = False
        elif action == "put-bucket-tagging":
            self.buckets[args[1]] = True
        elif action == "create-repository":
            self.repo = True
        else:
            raise AssertionError((service, action))
        return {}


def foundation_stack(private, monkeypatch):
    fake = FoundationAWS()
    stack = Stack(config(private), fake)
    monkeypatch.setattr(stack, "network", lambda: None)
    monkeypatch.setattr(stack, "roles", lambda: None)
    monkeypatch.setattr(stack, "service_role", lambda name, service: None)
    return stack, fake


def test_foundation_dry_run_apply_and_repeat(private, monkeypatch):
    stack, aws = foundation_stack(private, monkeypatch)
    guard = Guard()
    stack.foundation(False, guard)
    assert aws.writes == [] and guard.ce == 0
    stack.foundation(True, guard)
    assert [action for _, action in aws.writes] == [
        "create-db-parameter-group",
        "modify-db-parameter-group",
        "create-db-instance",
        "create-bucket",
        "put-bucket-tagging",
        "create-bucket",
        "put-bucket-tagging",
        "create-repository",
    ]
    assert {p.name for p in private.iterdir()} == {"secret"}
    aws.writes.clear()
    stack.foundation(True, guard)
    assert aws.writes == [] and guard.ce == 1


def test_partial_create_fails_closed_and_does_not_repair_drift(private, monkeypatch):
    stack, aws = foundation_stack(private, monkeypatch)
    aws.failure = "modify-db-parameter-group"
    with pytest.raises(Stop, match="simulated partial"):
        stack.foundation(True, Guard())
    assert [action for _, action in aws.writes] == [
        "create-db-parameter-group",
        "modify-db-parameter-group",
    ]
    aws.writes.clear()
    with pytest.raises(Stop, match="incomplete previous create"):
        stack.foundation(True, Guard())
    assert aws.writes == []


def test_bundle_pins_destination_digest_and_preserves_existing_settings(private):
    stack = Stack(config(private), FoundationAWS())
    zip_bytes = stack.zip_bundle(DIGEST)
    assert zip_bytes == stack.zip_bundle(DIGEST)
    with ZipFile(io.BytesIO(zip_bytes)) as zip_file:
        assert set(zip_file.namelist()) == {
            *infrastructure.BUNDLE_FILES,
            "Dockerrun.aws.json",
        }
        dockerrun = json.loads(zip_file.read("Dockerrun.aws.json"))
        assert dockerrun["Image"]["Name"] == f"{stack.registry}/{stack.repo}@{DIGEST}"
    stack.vpc = "vpc-00000000"
    opts = {
        (i["Namespace"], i["OptionName"]): i["Value"] for i in stack.option_settings()
    }
    assert opts[infrastructure.ENVIRONMENT, "EnvironmentType"] == "SingleInstance"
    assert opts[infrastructure.LAUNCH, "DisableDefaultEC2SecurityGroup"] == "true"
    assert opts[infrastructure.LAUNCH, "SecurityGroups"] == EB_SG
    assert opts[infrastructure.LAUNCH, "IamInstanceProfile"] == "LabInstanceProfile"
    assert len([i for i in opts if "application:environment" in i[0]]) == 7


@pytest.mark.parametrize("prefix", ["/mlflow/", "/p26-trialone/mlflow/"])
def test_eb_metadata_and_options_use_the_same_exact_runtime_parameters(
    private, monkeypatch, prefix
):
    settings = config(private)
    if prefix != "/mlflow/":
        settings["runtime_parameter_prefix"] = prefix
    paths = {
        "MLFLOW_BACKEND_STORE_URI": f"{prefix}backend-store-uri",
        "MLFLOW_AUTH_DATABASE_URI": f"{prefix}auth-database-uri",
        "MLFLOW_AUTH_ADMIN_PASSWORD": f"{prefix}auth-admin-password",
        "MLFLOW_FLASK_SERVER_SECRET_KEY": f"{prefix}flask-secret-key",
    }
    metadata = [{"Name": path, "Type": "SecureString"} for path in paths.values()]

    def call(service, action, *args):
        if (service, action) == ("elasticbeanstalk", "describe-platform-version"):
            return {
                "PlatformDescription": {
                    "PlatformArn": settings["platform_arn"],
                    "PlatformStatus": "Ready",
                }
            }
        assert (service, action) == ("ssm", "describe-parameters")
        assert args == (
            "--parameter-filters",
            f"Key=Name,Option=BeginsWith,Values={prefix}",
        )
        return {"Parameters": metadata}

    stack = Stack(settings, SimpleNamespace(call=call))
    monkeypatch.setattr(stack, "roles", lambda: None)
    monkeypatch.setattr(stack, "service_role", lambda *_: None)
    stack.eb_prerequisites()
    options = {
        item["OptionName"]: item["Value"]
        for item in stack.option_settings()
        if item["Namespace"] == "aws:elasticbeanstalk:application:environmentsecrets"
    }
    assert options == {
        name: f"arn:aws:ssm:us-east-1:{ACCOUNT}:parameter{path}"
        for name, path in paths.items()
    }

    metadata[0]["Type"] = "String"
    with pytest.raises(Stop, match="missing or have the wrong type"):
        stack.eb_prerequisites()
    metadata[0]["Type"] = "SecureString"
    metadata[0]["Name"] += "-backup"
    with pytest.raises(Stop, match="missing or have the wrong type"):
        stack.eb_prerequisites()


def test_ecr_image_must_match_pinned_digest_and_platform(private):
    manifests = [
        {
            "digest": "sha256:" + "b" * 64,
            "platform": {"os": "linux", "architecture": "amd64"},
        }
    ]

    def call(service, action, *args, **kwargs):
        assert service == "ecr"
        if action == "batch-get-image":
            assert f"imageTag={IMAGE_TAG}" in args
            return {
                "images": [
                    {
                        "imageId": {"imageDigest": DIGEST},
                        "imageManifest": json.dumps({"manifests": manifests}),
                    }
                ]
            }
        return {"imageDetails": [{"imageSizeInBytes": 313510335}]}

    stack = Stack(config(private), SimpleNamespace(call=call))
    assert stack.image() == DIGEST
    stack.image_digest = manifests[0]["digest"]
    with pytest.raises(Stop, match="independently witnessed"):
        stack.image()
    manifests[0]["platform"]["architecture"] = "arm64"
    with pytest.raises(Stop, match="Linux AMD64"):
        stack.image(new=True)


def make_bundle(private):
    folder = private / "synthetic-backup"
    folder.mkdir(mode=0o700)
    (folder / "artifacts").mkdir(mode=0o700)
    (folder / ARTIFACT).parent.mkdir(mode=0o700, parents=True)
    # pathlib creates intermediate paths with its default mode (ignoring mode).
    (folder / "artifacts/3").chmod(0o700)
    payload = b"good"
    write_private(folder / ARTIFACT, payload)
    column_names = {table: list(keys) for table, keys in KEYS.items()}
    tables = {
        key: {"rows": 0, "bytes": 0, "sha256": hashlib.sha256(b"").hexdigest()}
        for key in KEYS
    }
    for key in KEYS:
        write_private(folder / f"{key}.jsonl", b"")
    manifest = {
        "version": 1,
        "source_account": OTHER,
        "source_region": "us-east-1",
        "source_bucket": "synthetic-artifacts",
        "source_db": "synthetic-source-db",
        "tables": tables,
        "columns": column_names,
        "signatures": dict.fromkeys(KEYS, "a" * 64),
        "artifacts": {
            ARTIFACT: {
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        },
        "omissions": OMISSIONS,
    }
    raw = json_bytes(manifest)
    write_private(folder / "manifest.json", raw)
    return hashlib.sha256(raw).hexdigest()


def destination_config(private, digest):
    return dict(
        config(private, "restore-data"),
        data_user="test_writer",
        manifest_sha256=digest,
        approved_source_account=OTHER,
    )


def test_restore_dry_run_safety_and_partial_rollback(private, monkeypatch):
    scoped = destination_config(private, make_bundle(private))
    events = []

    @contextmanager
    def connection(*args):
        yield SimpleNamespace(
            open=True,
            commit=lambda: events.append("commit"),
            rollback=lambda: events.append("rollback"),
        )

    fake = SimpleNamespace(
        identity=lambda: events.append("identity"),
        call=lambda *args, **kwargs: pytest.fail("Unexpected unmocked AWS API"),
    )
    plan = Recovery(scoped, fake, destination=True)
    manifest = plan.manifest()
    monkeypatch.setattr(recovery, "bucket_state", lambda *args, **kwargs: True)
    monkeypatch.setattr(plan, "host", lambda: "fake.us-east-1.rds.amazonaws.com")
    monkeypatch.setattr(plan, "db_capacity", lambda size: events.append("capacity"))
    monkeypatch.setattr(recovery, "connection", connection)
    monkeypatch.setattr(
        recovery,
        "schema_info",
        lambda conn: (manifest["signatures"], manifest["columns"], 0),
    )
    monkeypatch.setattr(
        recovery,
        "empty_tables",
        lambda conn, lock=False: events.append(("empty", lock)),
    )
    monkeypatch.setattr(
        recovery,
        "snapshot",
        lambda conn, restore=False: events.append(("transaction", restore)),
    )
    monkeypatch.setattr(recovery, "s3_inventory", lambda *args: {})
    monkeypatch.setattr(
        recovery, "table_import", lambda *args: pytest.fail("Unexpected import")
    )

    def partial(*args):
        events.append("conditional-upload")
        raise Stop("simulated incomplete S3 upload")

    monkeypatch.setattr(recovery, "upload_artifacts", partial)
    guard = Guard()
    plan.restore(guard, apply=False, offline=False)
    assert events == [("empty", False), "capacity"] and guard.ce == 0
    events.clear()
    with pytest.raises(Stop, match="simulated incomplete"):
        plan.restore(guard, apply=True, offline=False)
    assert events == [
        ("empty", False),
        "capacity",
        "identity",
        ("transaction", True),
        ("empty", True),
        "conditional-upload",
        "rollback",
    ]
    events.clear()
    monkeypatch.setattr(
        recovery, "s3_inventory", lambda *args: {ARTIFACT: {"bytes": 4}}
    )
    with pytest.raises(Stop, match="prefix is not empty"):
        plan.restore(guard, apply=True, offline=False)
    assert events == [("empty", False)]


def test_s3_restore_writes_conditionally_and_checks_remote_bytes(private):
    plan = Recovery(
        destination_config(private, make_bundle(private)), None, destination=True
    )
    records = plan.manifest()["artifacts"]
    calls = []

    def call(service, operation, *args, **kwargs):
        calls.append(operation)
        params = dict(zip(args[::2], args[1::2], strict=False))
        assert service == "s3api" and params["--expected-bucket-owner"] == ACCOUNT
        if operation == "put-object":
            assert kwargs["change"] is True and params["--if-none-match"] == "*"
            assert params["--checksum-sha256"] == common.b64(
                bytes.fromhex(records[ARTIFACT]["sha256"])
            )
            return {}
        if operation == "list-objects-v2":
            return {
                "IsTruncated": False,
                "Contents": [
                    {
                        "Key": ARTIFACT,
                        "Size": 4,
                        "ETag": '"' + "0" * 32 + '"',
                        "LastModified": "2026-11-26",
                        "StorageClass": "STANDARD",
                    },
                ],
            }
        return {
            "ChecksumSHA256": common.b64(bytes.fromhex(records[ARTIFACT]["sha256"]))
        }

    aws = SimpleNamespace(account=ACCOUNT, call=call)
    recovery.upload_artifacts(
        aws, Guard(), plan.bucket, records, plan.path / "artifacts", 10
    )
    assert calls == ["put-object", "list-objects-v2", "head-object"]


def test_ce_and_scope_fail_before_writes(monkeypatch):
    monkeypatch.setattr(common, "now", lambda: NOW)
    record = approval("backup-data", NOW)
    guard = Approval(record, "backup-data")
    guard.estimate(outbound_bytes=4 * GIB, local_usd=Decimal("0.10"), stack=True)
    with pytest.raises(Stop, match="exact operation"):
        Approval(record, "restore-data")
    record["trial_spent_and_committed_usd"] = "6"
    with pytest.raises(Stop, match="exceed the approved cap"):
        Approval(record, "backup-data").estimate(stack=True)
    response = {
        "ResultsByTime": [
            {
                "TimePeriod": {"Start": "2026-10-01"},
                "Total": {"UnblendedCost": {"Amount": "11", "Unit": "USD"}},
            },
            {
                "TimePeriod": {"Start": "2026-11-01"},
                "Total": {"UnblendedCost": {"Amount": "20", "Unit": "USD"}},
            },
        ]
    }
    with pytest.raises(Stop, match="Fresh CE"):
        guard.check_ce(SimpleNamespace(call=lambda *args: response))
    with pytest.raises(Stop, match="Internal guard"):
        AWS({"account": ACCOUNT, "profile": "dummy"}).write("ecr", "create-repository")


def test_backup_dry_run_and_apply_seal_only_complete_bundle(private, monkeypatch):
    plan = Recovery(
        config(private, "backup-data"), SimpleNamespace(identity=lambda: None)
    )
    calls = []

    @contextmanager
    def connection(*args):
        yield SimpleNamespace(rollback=lambda: calls.append("rollback"))

    columns = {table: list(keys) for table, keys in KEYS.items()}
    signatures = dict.fromkeys(KEYS, "a" * 64)
    source_inventory = {
        ARTIFACT: {"bytes": 4, "etag": '"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"'}
    }
    monkeypatch.setattr(plan, "host", lambda: "fake.us-east-1.rds.amazonaws.com")
    monkeypatch.setattr(recovery, "bucket_state", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        recovery, "s3_inventory", lambda *args, **kwargs: source_inventory
    )
    monkeypatch.setattr(recovery, "connection", connection)
    monkeypatch.setattr(
        recovery, "schema_info", lambda conn: (signatures, columns, 100)
    )
    monkeypatch.setattr(recovery, "snapshot", lambda conn: calls.append("snapshot"))

    def dump(conn, table, columns, remaining, guard, out):
        calls.append(table)
        payload = json_bytes(["synthetic-private-row"] * len(columns))
        out.write(payload)
        os.fchmod(out.fileno(), 0o600)  # Entry point sets umask 077.
        return {
            "bytes": len(payload),
            "rows": 1,
            "sha256": hashlib.sha256(payload).hexdigest(),
        }

    def fail_sync(*args):
        calls.append("sync")
        raise Stop("simulated interrupted object download")

    monkeypatch.setattr(recovery, "table_dump", dump)
    monkeypatch.setattr(recovery, "sync_artifacts", fail_sync)
    guard = Guard()
    plan.backup(guard, apply=False, offline=False)
    assert calls == [] and not plan.path.exists() and guard.ce == 0
    with pytest.raises(Stop, match="interrupted object download"):
        plan.backup(guard, apply=True, offline=False)
    assert calls == ["snapshot", *KEYS, "rollback", "sync"]
    assert plan.path.stat().st_mode & 0o777 == 0o700
    assert not (plan.path / "manifest.json").exists()
    with pytest.raises(Stop, match="already exists"):
        plan.backup(guard, apply=True, offline=False)

    def good_sync(aws, guard, bucket, inventory, root):
        root.mkdir(mode=0o700)
        return {}

    monkeypatch.setattr(recovery, "sync_artifacts", good_sync)
    monkeypatch.setattr(recovery, "s3_inventory", lambda *args, **kwargs: {})
    settings = dict(plan.config, backup_name="second-private-backup")
    plan = Recovery(settings, plan.aws)
    monkeypatch.setattr(plan, "host", lambda: "fake.us-east-1.rds.amazonaws.com")
    old = os.umask(0o077)
    try:
        plan.backup(guard, apply=True, offline=False)
    finally:
        os.umask(old)
    manifest = json.loads((plan.path / "manifest.json").read_bytes())
    assert manifest["omissions"] == OMISSIONS
    assert set(manifest["tables"]) == set(KEYS)
    assert all(entry["rows"] == 1 for entry in manifest["tables"].values())


def test_rds_service_role_preflight_denial_stops_before_foundation_writes(
    private, monkeypatch
):
    stack, aws = foundation_stack(private, monkeypatch)

    def missing(name, service):
        assert (name, service) == ("AWSServiceRoleForRDS", "rds")
        raise Stop("RDS service-linked role not present")

    monkeypatch.setattr(stack, "service_role", missing)
    with pytest.raises(Stop, match="role not present"):
        stack.foundation(True, Guard())
    assert aws.calls == [] and aws.writes == []
