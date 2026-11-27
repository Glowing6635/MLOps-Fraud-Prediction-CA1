"""Recreate and verify the settled SingleInstance lab resources."""

import hashlib
import io
import ipaddress
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from stat import S_ISSOCK
from urllib.error import URLError
from urllib.request import HTTPRedirectHandler, ProxyHandler, build_opener
from zipfile import ZipFile, ZipInfo

from common import (
    GIB,
    REGION,
    ROOT,
    TAG,
    Stop,
    b64,
    emit,
    json_bytes,
    matching,
    private_dir,
    private_file,
    require,
    run,
    whole,
)

ENGINE = "8.4.11"
DB_CLASS = "db.t3.micro"
DB_GIB = 20
EB_CLASS = "t3.small"
IMAGE_TAG = "3.18.0-r1"
DIGEST = r"sha256:[0-9a-f]{64}"
PARAMETERS = {"require_secure_transport": "1", "log_bin_trust_function_creators": "1"}
SECRETS = {
    "MLFLOW_BACKEND_STORE_URI": "backend-store-uri",
    "MLFLOW_AUTH_DATABASE_URI": "auth-database-uri",
    "MLFLOW_AUTH_ADMIN_PASSWORD": "auth-admin-password",
    "MLFLOW_FLASK_SERVER_SECRET_KEY": "flask-secret-key",
}
BUILD_FILES = ("Dockerfile", "requirements.txt", "start.sh")
BUNDLE_FILES = (
    ".ebextensions/health.config",
    ".platform/nginx/conf.d/upload_size.conf",
)
PLATFORM = r"arn:aws:elasticbeanstalk:us-east-1::platform/Docker running on 64bit Amazon Linux 2023/4\.14\.3"
ENVIRONMENT = "aws:elasticbeanstalk:environment"
LAUNCH = "aws:autoscaling:launchconfiguration"


def ipv4_hosts(values):
    require(isinstance(values, list), "Provide approved IPv4 host rules as a list.")
    for value in values:
        ip = ipaddress.IPv4Network(value)
        require(
            ip.prefixlen == 32 and str(ip) == value,
            "Only approved canonical IPv4 /32 rules are supported.",
        )
    require(len(values) == len(set(values)), "Duplicate approved network rules.")
    return set(values)


def owned(resource_tags, trial):
    return [entry["Value"] for entry in resource_tags if entry["Key"] == TAG] == [trial]


def bucket_name(value):
    return matching(
        value,
        r"(?!xn--|sthree-|amzn-s3-demo-)[a-z][a-z0-9-]{1,61}[a-z0-9]",
        "S3 bucket",
    )


def bucket_state(aws, name, *, trial=None, absent=False):
    owner = ("--expected-bucket-owner", aws.account)
    head = aws.call(
        "s3api",
        "head-bucket",
        "--bucket",
        name,
        *owner,
        absent=("404", "NotFound", "NoSuchBucket") if absent else (),
    )
    if head is None:
        return False

    def get(operation, **options):
        return aws.call("s3api", operation, "--bucket", name, *owner, **options)

    require(
        get("get-bucket-location").get("LocationConstraint") in (None, REGION),
        "Bucket is outside the approved region.",
    )
    block = get("get-public-access-block")["PublicAccessBlockConfiguration"]
    require(
        all(
            block.get(flag) is True
            for flag in (
                "BlockPublicAcls",
                "IgnorePublicAcls",
                "BlockPublicPolicy",
                "RestrictPublicBuckets",
            )
        ),
        "Bucket public access protection has drifted.",
    )
    policy = get("get-bucket-policy-status", absent=("NoSuchBucketPolicy",))
    require(
        policy is None or policy["PolicyStatus"]["IsPublic"] is False,
        "Bucket has a public policy.",
    )
    rules = get("get-bucket-ownership-controls")["OwnershipControls"]["Rules"]
    require(
        rules == [{"ObjectOwnership": "BucketOwnerEnforced"}],
        "Bucket ownership has drifted.",
    )
    rules = get("get-bucket-encryption")["ServerSideEncryptionConfiguration"]["Rules"]
    require(
        [x["ApplyServerSideEncryptionByDefault"]["SSEAlgorithm"] for x in rules]
        == ["AES256"],
        "Bucket encryption has drifted.",
    )
    if trial is not None:
        require(policy is None, "Trial bucket must have no bucket policy.")
        require(
            get(
                "get-bucket-replication",
                absent=("ReplicationConfigurationNotFoundError",),
            )
            is None
            and get(
                "get-bucket-lifecycle-configuration",
                absent=("NoSuchLifecycleConfiguration",),
            )
            is None,
            "Trial bucket has a replication or lifecycle configuration.",
        )
        tags = get("get-bucket-tagging", absent=("NoSuchTagSet",))
        require(
            tags and owned(tags["TagSet"], trial),
            "Existing bucket has no matching trial ownership.",
        )
        require(
            not get("get-bucket-versioning").get("Status"),
            "Trial bucket must be unversioned.",
        )
    return True


class Stack:
    def __init__(self, config, aws):
        self.aws = aws
        self.config = config
        self.trial = config["trial"]
        self.account = config["account"]
        stem = f"p26-{self.trial}"
        self.db = f"{stem}-db"
        self.group = f"{stem}-mysql84"
        self.app = f"{stem}-mlflow"
        self.env = self.app
        self.artifact = f"p26-{self.account}-{self.trial}-artifacts"
        self.bundle = f"p26-{self.account}-{self.trial}-bundle"
        self.repo = f"{stem}/mlflow-server"
        self.registry = f"{self.account}.dkr.ecr.{REGION}.amazonaws.com"
        self.cname = matching(
            config["cname"], r"[a-z][a-z0-9-]{2,59}[a-z0-9]", "EB CNAME prefix"
        )
        self.host = f"{self.cname}.{REGION}.elasticbeanstalk.com"
        self.db_sg = matching(
            config["db_security_group"], r"sg-[0-9a-f]{8,17}", "DB security group"
        )
        self.eb_sg = matching(
            config["eb_security_group"], r"sg-[0-9a-f]{8,17}", "EB security group"
        )
        require(self.eb_sg != self.db_sg, "Use distinct EB and DB security groups.")
        self.subnet_group = matching(
            config["db_subnet_group"], r"[a-z][a-z0-9.-]{0,62}", "DB subnet group"
        )
        self.subnets = config["eb_subnets"]
        require(
            isinstance(self.subnets, list) and 1 <= len(self.subnets) <= 6,
            "Specify one to six EB subnets.",
        )
        for subnet in self.subnets:
            matching(subnet, r"subnet-[0-9a-f]{8,17}", "EB subnet")
        require(len(set(self.subnets)) == len(self.subnets), "Duplicate EB subnets.")
        self.http_hosts = ipv4_hosts(config["eb_http_hosts"])
        self.db_hosts = ipv4_hosts(config["db_client_hosts"])
        require(
            self.http_hosts, "An existing approved EB operator /32 rule is required."
        )
        self.db_public = config["db_public"]
        require(
            type(self.db_public) is bool, "Specify whether RDS is publicly addressable."
        )
        self.admin = matching(
            config["rds_master_user"], r"[A-Za-z][A-Za-z0-9_]{0,15}", "RDS master user"
        )
        require(self.admin.lower() != "mlflow", "The MLflow server login is forbidden.")
        self.platform = matching(
            config["platform_arn"], PLATFORM, "pinned EB Docker platform"
        )
        self.image_digest = config.get("image_digest")
        if self.image_digest is not None:
            matching(self.image_digest, DIGEST, "destination image digest")
        self.image_bytes = whole(config["image_max_bytes"], "image size ceiling", 1)
        self.temp_dir = private_dir(config["private_dir"])
        self.tags = [{"Key": TAG, "Value": self.trial}]
        self.vpc = None

    def network(self):
        groups = self.aws.call(
            "ec2", "describe-security-groups", "--group-ids", self.db_sg, self.eb_sg
        )["SecurityGroups"]
        require(
            {g["GroupId"] for g in groups} == {self.db_sg, self.eb_sg}
            and all(g["OwnerId"] == self.account for g in groups)
            and len({g["VpcId"] for g in groups}) == 1,
            "Security groups are outside the approved account/VPC.",
        )
        self.vpc = groups[0]["VpcId"]
        expected = {
            self.db_sg: {(3306, "group", self.eb_sg)}
            | {(3306, "host", host) for host in self.db_hosts},
            self.eb_sg: {(80, "host", host) for host in self.http_hosts},
        }
        for group in groups:
            observed = set()
            for permission in group["IpPermissions"]:
                port = permission.get("FromPort")
                require(
                    permission["IpProtocol"] == "tcp"
                    and port == permission.get("ToPort")
                    and not permission.get("Ipv6Ranges")
                    and not permission.get("PrefixListIds"),
                    "An unapproved ingress rule exists.",
                )
                observed.update(
                    (port, "host", entry["CidrIp"]) for entry in permission["IpRanges"]
                )
                for entry in permission["UserIdGroupPairs"]:
                    require(
                        entry.get("UserId") == self.account
                        and not entry.get("VpcPeeringConnectionId"),
                        "An unapproved peer group rule exists.",
                    )
                    observed.add((port, "group", entry["GroupId"]))
            require(
                observed == expected[group["GroupId"]],
                "Existing ingress differs from approved rules. Network owner must resolve it separately.",
            )
        db_subnet = self.aws.call(
            "rds",
            "describe-db-subnet-groups",
            "--db-subnet-group-name",
            self.subnet_group,
        )["DBSubnetGroups"]
        require(
            len(db_subnet) == 1
            and db_subnet[0]["VpcId"] == self.vpc
            and db_subnet[0]["SubnetGroupStatus"] == "Complete"
            and len(
                {
                    s["SubnetAvailabilityZone"]["Name"]
                    for s in db_subnet[0]["Subnets"]
                    if s["SubnetStatus"] == "Active"
                }
            )
            >= 2,
            "DB subnet group is outside the VPC or lacks two active zones.",
        )
        subnets = self.aws.call(
            "ec2", "describe-subnets", "--subnet-ids", *self.subnets
        )["Subnets"]
        require(
            {s["SubnetId"] for s in subnets} == set(self.subnets)
            and all(
                s["VpcId"] == self.vpc
                and s["OwnerId"] == self.account
                and s["State"] == "available"
                and s["AvailableIpAddressCount"] >= 2
                for s in subnets
            ),
            "EB subnets do not match the approved account/VPC or lack IP capacity.",
        )

        self.eb_zones = {s["AvailabilityZone"] for s in subnets}

    def roles(self):
        profile = self.aws.call(
            "iam",
            "get-instance-profile",
            "--instance-profile-name",
            "LabInstanceProfile",
        )["InstanceProfile"]
        require(
            [r["Arn"] for r in profile["Roles"]]
            == [f"arn:aws:iam::{self.account}:role/LabRole"],
            "The existing instance profile does not attach the expected LabRole.",
        )
        role = self.aws.call("iam", "get-role", "--role-name", "LabRole")["Role"]
        trusted = set()
        for statement in role["AssumeRolePolicyDocument"]["Statement"]:
            actions = statement.get("Action", [])
            if isinstance(actions, str):
                actions = [actions]
            if statement["Effect"] == "Allow" and "sts:AssumeRole" in actions:
                services = statement.get("Principal", {}).get("Service", [])
                trusted.update([services] if isinstance(services, str) else services)
        require(
            {"ec2.amazonaws.com", "elasticbeanstalk.amazonaws.com"} <= trusted,
            "Existing LabRole lacks the recorded EC2/EB trust.",
        )

    def no_new_iam(self):
        require(
            self.config["iam_creation_prevented"] is True,
            "Confirm that the approved lab scope prevents IAM creation.",
        )

    def service_role(self, role_name, service):
        role = self.aws.call("iam", "get-role", "--role-name", role_name)["Role"]
        path = f"/aws-service-role/{service}.amazonaws.com/"
        require(
            role["Path"] == path
            and role["Arn"] == f"arn:aws:iam::{self.account}:role{path}{role_name}",
            "Required existing service-linked role is missing.",
        )

    def db_parameter_group(self):
        result = self.aws.call(
            "rds",
            "describe-db-parameter-groups",
            "--db-parameter-group-name",
            self.group,
            absent=("DBParameterGroupNotFound", "DBParameterGroupNotFoundFault"),
        )
        if result is None:
            return None
        (group,) = result["DBParameterGroups"]
        tags = self.aws.call(
            "rds",
            "list-tags-for-resource",
            "--resource-name",
            group["DBParameterGroupArn"],
        )["TagList"]
        require(
            group["DBParameterGroupFamily"] == "mysql8.4" and owned(tags, self.trial),
            "Existing DB parameter group has ownership or family drift.",
        )
        entries = self.aws.call(
            "rds",
            "describe-db-parameters",
            "--db-parameter-group-name",
            self.group,
            "--source",
            "user",
        )["Parameters"]
        values = {entry["ParameterName"]: entry["ParameterValue"] for entry in entries}
        require(
            values == PARAMETERS,
            "DB parameter group has drift or an incomplete previous create.",
        )
        return group

    def db_instance(self):
        response = self.aws.call(
            "rds",
            "describe-db-instances",
            "--db-instance-identifier",
            self.db,
            absent=("DBInstanceNotFound", "DBInstanceNotFoundFault"),
        )
        if response is None:
            return None
        (db,) = response["DBInstances"]
        settings = {
            "Engine": "mysql",
            "EngineVersion": ENGINE,
            "DBInstanceClass": DB_CLASS,
            "AllocatedStorage": DB_GIB,
            "StorageType": "gp3",
            "Iops": 3000,
            "StorageThroughput": 125,
            "StorageEncrypted": True,
            "MultiAZ": False,
            "PubliclyAccessible": self.db_public,
            "BackupRetentionPeriod": 1,
            "DeletionProtection": True,
            "AutoMinorVersionUpgrade": True,
            "IAMDatabaseAuthenticationEnabled": False,
            "MasterUsername": self.admin,
        }
        require(
            db["DBInstanceArn"] == f"arn:aws:rds:{REGION}:{self.account}:db:{self.db}"
            and owned(db.get("TagList", []), self.trial)
            and all(db.get(k) == v for k, v in settings.items())
            and not db.get("MaxAllocatedStorage")
            and not db.get("PendingModifiedValues")
            and [g["VpcSecurityGroupId"] for g in db["VpcSecurityGroups"]]
            == [self.db_sg]
            and [p["DBParameterGroupName"] for p in db["DBParameterGroups"]]
            == [self.group]
            and db["DBSubnetGroup"]["DBSubnetGroupName"] == self.subnet_group,
            "Existing RDS instance differs from the approved trial shape.",
        )
        key = self.aws.call("kms", "describe-key", "--key-id", db["KmsKeyId"])[
            "KeyMetadata"
        ]
        require(
            key["KeyManager"] == "AWS" and key["Enabled"] is True,
            "RDS encryption differs from the recorded enabled AWS-managed key.",
        )
        return db

    def db_ready(self, db):
        require(
            db
            and db["DBInstanceStatus"] == "available"
            and db.get("Endpoint", {}).get("Port") == 3306
            and all(
                p["ParameterApplyStatus"] == "in-sync" for p in db["DBParameterGroups"]
            )
            and all(g["Status"] == "active" for g in db["VpcSecurityGroups"]),
            "RDS is not available with the approved groups in sync. Inspect status before continuing.",
        )
        return db["Endpoint"]["Address"]

    def db_capacity(self, group, db):
        quotas = self.aws.call("rds", "describe-account-attributes")["AccountQuotas"]
        free = {row["AccountQuotaName"]: row["Max"] - row["Used"] for row in quotas}
        if group is None:
            require(
                free.get("DBParameterGroups", 0) >= 1,
                "Insufficient RDS parameter-group quota.",
            )
        if db is None:
            require(
                free.get("DBInstances", 0) >= 1
                and free.get("AllocatedStorage", 0) >= DB_GIB,
                "Insufficient RDS instance/storage quota.",
            )
            options = self.aws.call(
                "rds",
                "describe-orderable-db-instance-options",
                "--engine",
                "mysql",
                "--engine-version",
                ENGINE,
                "--db-instance-class",
                DB_CLASS,
            )["OrderableDBInstanceOptions"]
            require(
                any(
                    o["StorageType"] == "gp3"
                    and o["SupportsStorageEncryption"]
                    and o["MinStorageSize"] <= DB_GIB <= o["MaxStorageSize"]
                    and o["Vpc"]
                    for o in options
                ),
                "The pinned RDS engine, instance and encrypted gp3 combination is unavailable.",
            )

    def repository(self):
        response = self.aws.call(
            "ecr",
            "describe-repositories",
            "--registry-id",
            self.account,
            "--repository-names",
            self.repo,
            absent=("RepositoryNotFoundException",),
        )
        if response is None:
            return None
        (repo,) = response["repositories"]
        tags = self.aws.call(
            "ecr", "list-tags-for-resource", "--resource-arn", repo["repositoryArn"]
        )["tags"]
        require(
            owned([{"Key": t["Key"], "Value": t["Value"]} for t in tags], self.trial)
            and repo["registryId"] == self.account
            and repo["imageTagMutability"] == "IMMUTABLE"
            and repo["imageScanningConfiguration"]["scanOnPush"] is True
            and repo["encryptionConfiguration"]["encryptionType"] == "AES256",
            "ECR ownership or settings drift.",
        )
        return repo

    def image(self, *, new=False):
        response = self.aws.call(
            "ecr",
            "batch-get-image",
            "--registry-id",
            self.account,
            "--repository-name",
            self.repo,
            "--image-ids",
            f"imageTag={IMAGE_TAG}",
        )
        if not response["images"]:
            require(
                [f["failureCode"] for f in response["failures"]] == ["ImageNotFound"],
                "ECR could not confirm whether the trial image exists.",
            )
            return None
        (image,) = response["images"]
        digest = matching(image["imageId"]["imageDigest"], DIGEST, "ECR image digest")
        require(
            new or digest == self.image_digest,
            "Pin the independently witnessed destination digest.",
        )
        manifest = json.loads(image["imageManifest"])
        amd64 = [
            m["digest"]
            for m in manifest.get("manifests", [])
            if m.get("platform", {}).get("architecture") == "amd64"
            and m["platform"].get("os") == "linux"
        ]
        require(
            len(amd64) == 1, "Expected an image index with one Linux AMD64 manifest."
        )
        detail = self.aws.call(
            "ecr",
            "describe-images",
            "--registry-id",
            self.account,
            "--repository-name",
            self.repo,
            "--image-ids",
            f"imageDigest={amd64[0]}",
        )["imageDetails"]
        require(
            len(detail) == 1 and 0 < detail[0]["imageSizeInBytes"] <= self.image_bytes,
            "ECR image exceeds the approved compressed size or its AMD64 manifest is unavailable.",
        )
        return digest

    def bundle_version(self, digest):
        # Hash the complete deterministic bundle, including destination registry and image.
        return f"p26-{hashlib.sha256(self.zip_bundle(digest)).hexdigest()[:32]}"

    def zip_bundle(self, digest):
        out = io.BytesIO()
        with ZipFile(out, "w") as bundle:
            files = {name: (ROOT / name).read_bytes() for name in BUNDLE_FILES}
            require(
                all(
                    hashlib.sha256(raw).hexdigest() == SOURCE_SHA256[name]
                    for name, raw in files.items()
                ),
                "An EB bundle input differs from the reviewed source.",
            )
            files["Dockerrun.aws.json"] = json_bytes(
                {
                    "AWSEBDockerrunVersion": "1",
                    "Image": {
                        "Name": f"{self.registry}/{self.repo}@{digest}",
                        "Update": "true",
                    },
                    "Ports": [{"ContainerPort": 5000}],
                }
            )
            for name, content in files.items():
                info = ZipInfo(name)
                info.create_system = 3
                bundle.writestr(info, content)
        return out.getvalue()

    def foundation(self, apply, guard):
        self.no_new_iam()
        self.network()
        self.roles()
        self.service_role("AWSServiceRoleForRDS", "rds")
        group, db, repo = (
            self.db_parameter_group(),
            self.db_instance(),
            self.repository(),
        )
        buckets = {
            which: bucket_state(self.aws, name, trial=self.trial, absent=True)
            for which, name in (("artifact", self.artifact), ("bundle", self.bundle))
        }
        self.db_capacity(group, db)
        if db:
            require(group is not None, "RDS parameter group is missing.")
            self.db_ready(db)
        if not db:
            password = self.bootstrap_password()
            guard.check(minutes=whole(self.config["rds_wait_minutes"], "RDS wait", 15))
        guard.estimate(stack=True, image_bytes=self.image_bytes)
        emit(
            "Foundation: "
            + ", ".join(
                f"{name} {'reuse verified' if state else 'create'}"
                for name, state in {
                    "DB parameters": group,
                    "RDS": db,
                    "ECR": repo,
                    **buckets,
                }.items()
            )
        )
        if not apply or (group and db and repo and all(buckets.values())):
            return
        guard.check_ce(self.aws)
        if group is None:
            self.aws.write(
                "rds",
                "create-db-parameter-group",
                "--db-parameter-group-name",
                self.group,
                "--db-parameter-group-family",
                "mysql8.4",
                "--description",
                "P26 pinned MySQL settings",
                "--tags",
                json.dumps(self.tags),
            )
            params = [
                {
                    "ParameterName": key,
                    "ParameterValue": value,
                    "ApplyMethod": "immediate",
                }
                for key, value in PARAMETERS.items()
            ]
            self.aws.write(
                "rds",
                "modify-db-parameter-group",
                "--db-parameter-group-name",
                self.group,
                "--parameters",
                json.dumps(params),
            )
        if db is None:
            self.create_database(password, guard)
        for which, name in (("artifact", self.artifact), ("bundle", self.bundle)):
            if not buckets[which]:
                self.aws.write(
                    "s3api",
                    "create-bucket",
                    "--bucket",
                    name,
                )
                self.aws.write(
                    "s3api",
                    "put-bucket-tagging",
                    "--bucket",
                    name,
                    "--expected-bucket-owner",
                    self.account,
                    "--tagging",
                    json.dumps({"TagSet": self.tags}),
                )
                bucket_state(self.aws, name, trial=self.trial)
        if repo is None:
            self.aws.write(
                "ecr",
                "create-repository",
                "--repository-name",
                self.repo,
                "--image-tag-mutability",
                "IMMUTABLE",
                "--image-scanning-configuration",
                "scanOnPush=true",
                "--encryption-configuration",
                "encryptionType=AES256",
                "--tags",
                json.dumps(self.tags),
            )
        self.db_ready(self.db_instance())
        require(
            self.db_parameter_group() and self.repository(),
            "Foundation inventory is incomplete.",
        )

    def bootstrap_password(self):
        password = (
            private_file(self.config["rds_password_file"], 512)
            .read_text()
            .removesuffix("\n")
        )
        require(
            16 <= len(password) <= 41
            and all(33 <= ord(c) <= 126 and c not in '/@"' for c in password),
            "RDS bootstrap password does not meet supported RDS input constraints.",
        )
        return password

    def create_database(self, password, guard):
        deadline = time.monotonic() + 60 * self.config["rds_wait_minutes"]
        request = {
            "DBInstanceIdentifier": self.db,
            "DBInstanceClass": DB_CLASS,
            "Engine": "mysql",
            "EngineVersion": ENGINE,
            "MasterUsername": self.admin,
            "MasterUserPassword": password,
            "Port": 3306,
            "DBSubnetGroupName": self.subnet_group,
            "VpcSecurityGroupIds": [self.db_sg],
            "DBParameterGroupName": self.group,
            "PubliclyAccessible": self.db_public,
            "MultiAZ": False,
            "AllocatedStorage": DB_GIB,
            "StorageType": "gp3",
            "StorageEncrypted": True,
            "BackupRetentionPeriod": 1,
            "DeletionProtection": True,
            "AutoMinorVersionUpgrade": True,
            "EnablePerformanceInsights": False,
            "EnableIAMDatabaseAuthentication": False,
            "EngineLifecycleSupport": "open-source-rds-extended-support-disabled",
            "Tags": self.tags,
        }
        with tempfile.TemporaryDirectory(dir=self.temp_dir) as work:
            data = Path(work) / "rds.json"
            data.write_bytes(json_bytes(request))
            self.aws.write(
                "rds", "create-db-instance", "--cli-input-json", f"file://{data}"
            )
        emit("RDS submitted; waiting for available status and in-sync settings.")
        while time.monotonic() < deadline:
            guard.check()
            result = self.aws.call(
                "rds",
                "describe-db-instances",
                "--db-instance-identifier",
                self.db,
                absent=("DBInstanceNotFound", "DBInstanceNotFoundFault"),
            )
            if result is None:
                time.sleep(20)
                continue
            (db,) = result["DBInstances"]
            status = db["DBInstanceStatus"]
            if status == "available" and all(
                p["ParameterApplyStatus"] == "in-sync" for p in db["DBParameterGroups"]
            ):
                self.db_ready(self.db_instance())
                return
            require(
                status
                in {
                    "creating",
                    "available",
                    "backing-up",
                    "configuring-enhanced-monitoring",
                    "modifying",
                },
                "RDS entered an unexpected state. Stop and inspect the trial.",
            )
            time.sleep(20)
        raise Stop(
            "RDS wait expired; RDS may still be provisioning. Inspect before retrying."
        )

    def image_build_preflight(self, guard, *, offline=False):
        minutes = whole(self.config["image_build_minutes"], "image build time", 1)
        require(
            self.image_digest is None,
            "A pinned image is missing. Investigate before publishing anything.",
        )
        guard.check(minutes=minutes)
        for filename in BUILD_FILES:
            require(
                hashlib.sha256((ROOT / filename).read_bytes()).hexdigest()
                == SOURCE_SHA256[filename],
                "An image build input differs from the reviewed source.",
            )
        required = 2 * GIB + 4 * self.image_bytes
        docker_disk = private_dir(self.config["docker_capacity_dir"])
        require(
            shutil.disk_usage(docker_disk).free >= required,
            "Docker build storage lacks estimated room.",
        )
        docker_socket = Path(self.config["docker_socket"])
        require(docker_socket.is_absolute(), "Specify an absolute Docker Unix socket.")
        if not offline:
            require(
                S_ISSOCK(docker_socket.stat().st_mode),
                "Use only the approved local Docker Unix socket.",
            )
        emit(f"Image build estimate: {required:,} local bytes and {minutes} minutes.")
        return minutes

    def publish(self, apply, guard):
        require(self.repository(), "Create and verify the trial ECR repository first.")
        digest = self.image()
        if not digest:
            minutes = self.image_build_preflight(guard)
        emit(
            "Image: reuse verified immutable Linux AMD64 image."
            if digest
            else "Image: build Linux AMD64 from the pinned base, hashed dependencies and checked source."
        )
        guard.estimate(stack=True, image_bytes=self.image_bytes)
        if not apply or digest:
            return
        socket = Path(self.config["docker_socket"])
        guard.check_ce(self.aws)
        with tempfile.TemporaryDirectory(dir=self.temp_dir) as work:
            env = {
                k: v
                for k, v in os.environ.items()
                if not k.startswith(("DOCKER_", "BUILDKIT_", "BUILDX_"))
            }
            env.update(DOCKER_CONFIG=work, DOCKER_HOST=f"unix://{socket}")
            password = self.aws.call("ecr", "get-login-password", raw=True).strip()
            require(bool(password), "ECR returned no Docker credential.")
            (Path(work) / "config.json").write_bytes(
                json_bytes(
                    {
                        "auths": {self.registry: {"auth": b64(b"AWS:" + password)}},
                    }
                )
            )
            with tempfile.TemporaryDirectory(dir=self.temp_dir) as context:
                for filename in BUILD_FILES:
                    (Path(context) / filename).write_bytes(
                        (ROOT / filename).read_bytes()
                    )
                guard.check(minutes=minutes)
                self.aws.identity()
                result = run(
                    [
                        "docker",
                        "buildx",
                        "build",
                        "--platform",
                        "linux/amd64",
                        "--provenance=mode=min",
                        "--push",
                        "--tag",
                        f"{self.registry}/{self.repo}:{IMAGE_TAG}",
                        context,
                    ],
                    env=env,
                    timeout=60 * minutes,
                )
                require(
                    result.returncode == 0,
                    "Image build/push failed. Check for an immutable tag before retrying.",
                )
        digest = self.image(new=True)
        require(digest, "Published image could not be verified.")
        emit(
            f"Destination image digest {digest}; pin it in private config before continuing."
        )

    def runtime_approval(self):
        self.no_new_iam()
        require(
            self.config["runtime_inputs_verified"] is True,
            "Confirm verified destination runtime inputs.",
        )

    def eb_prerequisites(self):
        self.runtime_approval()
        self.roles()
        # EB uses LabRole explicitly. The underlying ASG may otherwise
        # create this role even with the service role set.
        self.service_role("AWSServiceRoleForAutoScaling", "autoscaling")
        self.service_role("AWSServiceRoleForRDS", "rds")
        platform = self.aws.call(
            "elasticbeanstalk",
            "describe-platform-version",
            "--platform-arn",
            self.platform,
        )["PlatformDescription"]
        require(
            platform["PlatformArn"] == self.platform
            and platform["PlatformStatus"] == "Ready",
            "Pinned Docker AL2023 platform is unavailable.",
        )
        parameters = self.aws.call(
            "ssm",
            "describe-parameters",
            "--parameter-filters",
            "Key=Name,Option=BeginsWith,Values=/mlflow/",
        )["Parameters"]
        actual = {p["Name"]: p["Type"] for p in parameters}
        require(
            all(
                actual.get(f"/mlflow/{name}") == "SecureString"
                for name in SECRETS.values()
            ),
            "Approved SecureString names are missing or have the wrong type.",
        )

    def option_settings(self):
        required = {
            ENVIRONMENT: {
                "EnvironmentType": "SingleInstance",
                "ServiceRole": "LabRole",
            },
            LAUNCH: {
                "IamInstanceProfile": "LabInstanceProfile",
                "DisableIMDSv1": "true",
                "DisableDefaultEC2SecurityGroup": "true",
                "SecurityGroups": self.eb_sg,
                "RootVolumeSize": "8",
                "RootVolumeType": "gp3",
            },
            "aws:ec2:instances": {
                "InstanceTypes": EB_CLASS,
                "SupportedArchitectures": "x86_64",
                "EnableSpot": "false",
            },
            "aws:autoscaling:asg": {"MinSize": "1", "MaxSize": "1"},
            "aws:ec2:vpc": {
                "VPCId": self.vpc,
                "Subnets": ",".join(sorted(self.subnets)),
            },
            "aws:elasticbeanstalk:managedactions": {"ManagedActionsEnabled": "false"},
            "aws:elasticbeanstalk:cloudwatch:logs": {"StreamLogs": "false"},
            "aws:elasticbeanstalk:application:environment": {
                "MLFLOW_ARTIFACTS_DESTINATION": f"s3://{self.artifact}/artifacts",
                "MLFLOW_SERVER_ALLOWED_HOSTS": self.host,
                "MLFLOW_SERVER_CORS_ALLOWED_ORIGINS": f"http://{self.host}",
            },
            "aws:elasticbeanstalk:application:environmentsecrets": {
                name: f"arn:aws:ssm:{REGION}:{self.account}:parameter/mlflow/{suffix}"
                for name, suffix in SECRETS.items()
            },
        }
        return [
            {"Namespace": namespace, "OptionName": option, "Value": value}
            for namespace, values in required.items()
            for option, value in values.items()
        ]

    def eb_tags(self, arn):
        tags = self.aws.call(
            "elasticbeanstalk", "list-tags-for-resource", "--resource-arn", arn
        )["ResourceTags"]
        require(owned(tags, self.trial), "EB resource has no matching trial ownership.")

    def eb_state(self, digest):
        apps = self.aws.call("elasticbeanstalk", "describe-applications")[
            "Applications"
        ]
        envs = self.aws.call("elasticbeanstalk", "describe-environments")[
            "Environments"
        ]
        app = next((a for a in apps if a["ApplicationName"] == self.app), None)
        env = next((e for e in envs if e["EnvironmentName"] == self.env), None)
        if app:
            self.eb_tags(app["ApplicationArn"])
        if env:
            self.eb_tags(env["EnvironmentArn"])
            require(
                app
                and env["ApplicationName"] == self.app
                and env["CNAME"] == self.host
                and env["PlatformArn"] == self.platform
                and env["VersionLabel"] == self.bundle_version(digest),
                "Existing EB environment has account, CNAME, version or platform drift.",
            )
        else:
            available = self.aws.call(
                "elasticbeanstalk",
                "check-dns-availability",
                "--cname-prefix",
                self.cname,
            )
            require(
                available["Available"] is True
                and available["FullyQualifiedCNAME"] == self.host,
                "The expected EB CNAME is unavailable.",
            )
        caps = self.aws.call("elasticbeanstalk", "describe-account-attributes")[
            "ResourceQuotas"
        ]
        if app is None:
            require(
                len(apps) < caps["ApplicationQuota"]["Maximum"],
                "No remaining EB application quota.",
            )
        if env is None:
            require(
                len(envs) < caps["EnvironmentQuota"]["Maximum"],
                "No remaining EB environment quota.",
            )
        return app, env, caps

    def ec2_capacity(self):
        attrs = self.aws.call(
            "ec2", "describe-account-attributes", "--attribute-names", "max-elastic-ips"
        )
        quotas = [
            int(v["AttributeValue"])
            for row in attrs["AccountAttributes"]
            if row["AttributeName"] == "max-elastic-ips"
            for v in row["AttributeValues"]
        ]
        addresses = self.aws.call("ec2", "describe-addresses")["Addresses"]
        require(
            len(quotas) == 1 and len(addresses) < quotas[0],
            "The SingleInstance trial needs one free Elastic IP allocation.",
        )
        offerings = self.aws.call(
            "ec2",
            "describe-instance-type-offerings",
            "--location-type",
            "availability-zone",
            "--filters",
            f"Name=instance-type,Values={EB_CLASS}",
        )["InstanceTypeOfferings"]
        require(
            self.eb_zones <= {i["Location"] for i in offerings},
            "An approved EB subnet is in a zone without the pinned instance offering.",
        )

    def bundle_state(self, digest, *, absent=False):
        version = self.bundle_version(digest)
        key = f"versions/{version}.zip"
        bundle = self.zip_bundle(digest)
        expected = hashlib.sha256(bundle).digest()
        head = self.aws.call(
            "s3api",
            "head-object",
            "--bucket",
            self.bundle,
            "--key",
            key,
            "--expected-bucket-owner",
            self.account,
            "--checksum-mode",
            "ENABLED",
            absent=("404", "NotFound", "NoSuchKey") if absent else (),
        )
        if head:
            require(
                head["ContentLength"] == len(bundle)
                and head.get("ChecksumSHA256") == b64(expected),
                "Existing EB source bundle failed its checksum.",
            )
        return key, bundle, b64(expected), bool(head)

    def eb_version(self, digest, key):
        versions = self.aws.call(
            "elasticbeanstalk",
            "describe-application-versions",
            "--application-name",
            self.app,
            "--version-labels",
            self.bundle_version(digest),
        )["ApplicationVersions"]
        if not versions:
            return None
        (version,) = versions
        self.eb_tags(version["ApplicationVersionArn"])
        require(
            version["SourceBundle"] == {"S3Bucket": self.bundle, "S3Key": key},
            "EB version points to a different source bundle.",
        )
        return version

    def verify_environment(self, env):
        require(
            env["Status"] == "Ready" and env["Health"] == "Green",
            "EB is not Ready/Green.",
        )
        settings = self.aws.call(
            "elasticbeanstalk",
            "describe-configuration-settings",
            "--application-name",
            self.app,
            "--environment-name",
            self.env,
        )["ConfigurationSettings"]
        deployed = [s for s in settings if s.get("DeploymentStatus") == "deployed"]
        require(len(deployed) == 1, "No unique deployed EB configuration.")
        actual = deployed[0]["OptionSettings"]
        for option in self.option_settings():
            values = [
                s["Value"]
                for s in actual
                if (s["Namespace"], s["OptionName"])
                == (option["Namespace"], option["OptionName"])
            ]
            if option["OptionName"] == "Subnets" and len(values) == 1:
                values = [",".join(sorted(values[0].split(",")))]
            require(
                values == [option["Value"]],
                f"Deployed EB option differs: {option['OptionName']}.",
            )
        resources = self.aws.call(
            "elasticbeanstalk",
            "describe-environment-resources",
            "--environment-id",
            env["EnvironmentId"],
        )["EnvironmentResources"]
        require(
            not resources.get("LoadBalancers")
            and len(resources["Instances"]) == 1
            and len(resources["AutoScalingGroups"]) == 1,
            "Expected one EB instance, one ASG and no LB.",
        )
        instances = self.aws.call(
            "ec2",
            "describe-instances",
            "--instance-ids",
            resources["Instances"][0]["Id"],
        )
        (instance,) = [i for row in instances["Reservations"] for i in row["Instances"]]
        require(
            instance["State"]["Name"] == "running"
            and instance["InstanceType"] == EB_CLASS
            and instance["Architecture"] == "x86_64"
            and instance["VpcId"] == self.vpc
            and instance["SubnetId"] in self.subnets
            and [s["GroupId"] for s in instance["SecurityGroups"]] == [self.eb_sg],
            "Running EB instance differs from the approved network or capacity.",
        )
        root = [
            device["Ebs"]["VolumeId"]
            for device in instance["BlockDeviceMappings"]
            if device["DeviceName"] == instance["RootDeviceName"]
        ]
        require(len(root) == 1, "EB instance does not have one root volume.")
        (volume,) = self.aws.call("ec2", "describe-volumes", "--volume-ids", root[0])[
            "Volumes"
        ]
        require(
            volume["Size"] == 8 and volume["VolumeType"] == "gp3",
            "EB root volume differs.",
        )
        # This settled endpoint serves unauthenticated health over HTTP.
        # Never send auth or follow a redirect from it.
        opener = build_opener(ProxyHandler({}), NoRedirect())
        try:
            with opener.open(f"http://{self.host}/health", timeout=10) as response:
                require(response.status == 200, "EB /health did not return HTTP 200.")
        except (URLError, TimeoutError):
            raise Stop("Unauthenticated EB /health failed.") from None
        emit(
            "Verified: Ready/Green, one t3.small, 8 GiB gp3, no LB, seven runtime settings and /health HTTP 200."
        )

    def server(self, apply, guard, *, verify=False):
        self.network()
        self.eb_prerequisites()
        require(
            self.db_parameter_group() and self.repository(),
            "Create the trial foundation first.",
        )
        self.db_ready(self.db_instance())
        bucket_state(self.aws, self.artifact, trial=self.trial)
        bucket_state(self.aws, self.bundle, trial=self.trial)
        digest = self.image()
        require(digest, "Publish and pin the destination image first.")
        app, env, caps = self.eb_state(digest)
        key, payload, checksum, present = self.bundle_state(digest, absent=not verify)
        version = self.eb_version(digest, key) if app else None
        require(
            not version or (present and version["Status"] == "PROCESSED"),
            "EB application version is incomplete or non-processed. Inspect before continuing.",
        )
        if env:
            require(version, "EB environment has no verified application version.")
            self.verify_environment(env)
            return
        require(not verify, "Expected EB environment is absent.")
        self.ec2_capacity()
        if not version:
            all_versions = self.aws.call(
                "elasticbeanstalk", "describe-application-versions"
            )["ApplicationVersions"]
            require(
                len(all_versions) < caps["ApplicationVersionQuota"]["Maximum"],
                "No remaining EB version quota.",
            )
        guard.estimate(
            stack=True, image_bytes=self.image_bytes, artifact_bytes=len(payload)
        )
        emit(
            "Server: create verified image bundle and EB SingleInstance; use existing scoped SG and roles."
        )
        minutes = whole(self.config["eb_wait_minutes"], "EB readiness time", 15)
        guard.check(minutes=minutes)
        if not apply:
            return
        guard.check_ce(self.aws)
        deadline = time.monotonic() + minutes * 60
        if not app:
            self.aws.write(
                "elasticbeanstalk",
                "create-application",
                "--application-name",
                self.app,
                "--tags",
                json.dumps(self.tags),
            )
        if not present:
            with tempfile.TemporaryDirectory(dir=self.temp_dir) as temp:
                path = Path(temp) / "release.zip"
                path.write_bytes(payload)
                self.aws.write(
                    "s3api",
                    "put-object",
                    "--bucket",
                    self.bundle,
                    "--key",
                    key,
                    "--expected-bucket-owner",
                    self.account,
                    "--body",
                    path,
                    "--if-none-match",
                    "*",
                    "--checksum-sha256",
                    checksum,
                    "--server-side-encryption",
                    "AES256",
                )
            self.bundle_state(digest)
        if not version:
            self.aws.write(
                "elasticbeanstalk",
                "create-application-version",
                "--application-name",
                self.app,
                "--version-label",
                self.bundle_version(digest),
                "--source-bundle",
                f"S3Bucket={self.bundle},S3Key={key}",
                "--process",
                "--tags",
                json.dumps(self.tags),
            )
            self.wait_until(
                deadline,
                guard,
                lambda: self.processed_version(digest, key),
                "application version",
            )
        self.network()
        environment = self.aws.write(
            "elasticbeanstalk",
            "create-environment",
            "--application-name",
            self.app,
            "--environment-name",
            self.env,
            "--version-label",
            self.bundle_version(digest),
            "--cname-prefix",
            self.cname,
            "--platform-arn",
            self.platform,
            "--option-settings",
            json.dumps(self.option_settings()),
            "--tags",
            json.dumps(self.tags),
        )
        emit("EB submitted; waiting for Ready/Green.")
        self.wait_until(
            deadline,
            guard,
            lambda: self.ready_environment(environment["EnvironmentId"]),
            "EB environment",
        )
        self.server(False, guard, verify=True)

    def processed_version(self, digest, key):
        version = self.eb_version(digest, key)
        require(
            version and version["Status"] in {"PROCESSING", "BUILDING", "PROCESSED"},
            "EB application bundle processing failed.",
        )
        return version["Status"] == "PROCESSED"

    def ready_environment(self, env_id):
        (env,) = self.aws.call(
            "elasticbeanstalk", "describe-environments", "--environment-ids", env_id
        )["Environments"]
        require(
            env["Status"] in {"Launching", "Updating", "Ready"},
            "EB entered an unexpected state.",
        )
        return env["Status"] == "Ready" and env["Health"] == "Green"

    def wait_until(self, deadline, guard, probe, resource):
        while time.monotonic() < deadline:
            guard.check()
            if probe():
                return
            time.sleep(20)
        raise Stop(
            f"Wait expired for {resource}. Check partial resources before retrying."
        )


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


SOURCE_SHA256 = {
    "Dockerfile": "72dcdccd31e99fea3ec5976ce306ad6ba7a9d8193bd1ef2b6253bcb286b62e35",
    "requirements.txt": "7e7b49122d646f898fbc7964a5f57d83aa0bea630570e39155e68e7cf89d4b04",
    "start.sh": "6f9122c822fa4ac0b86759d8babe3161dc4479eec876b6a9a7dd042d5bd52be6",
    ".ebextensions/health.config": "6eeded90aeb6cfcc4228c5d063c5dadc2e3823ff735228d71eb68e80188ed5ad",
    ".platform/nginx/conf.d/upload_size.conf": "611ec2d303b5e498ce5e1fd0dd71f0016f88427c39d1cf0e92a541e9af0bf3e9",
}
