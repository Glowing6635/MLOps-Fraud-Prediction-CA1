"""Scoped fraud data and artifact recovery with no SQL dump execution."""

import hashlib
import json
import os
import re
import shutil
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from decimal import Decimal

from common import (
    GIB,
    REGION,
    ROOT,
    Stop,
    b64,
    emit,
    json_bytes,
    matching,
    now,
    number,
    private_dir,
    private_file,
    require,
    sha256,
    whole,
)
from infrastructure import bucket_name, bucket_state, owned

SCHEMA = "fraud"
KEYS = {
    "batches": ("batch_id",),
    "customers": ("customer_id", "batch_id"),
    "orders": ("country_code", "order_id"),
    "labels": ("country_code", "order_id"),
}
PREFIX = "artifacts/"
HEX = r"[0-9a-f]{64}"
ETAG = r'"[0-9a-fA-F]{32}(?:-[1-9][0-9]{0,6})?"'
MAX_OBJECT = 5_000_000_000  # Single conditional PutObject only.
MAX_LINE = 64 * 1024
OMISSIONS = [
    "MLflow tracking and registry records",
    "MLflow authentication, users and grants",
    "SSM values and credentials",
    "S3 ACLs, tags, user metadata, object versions and keys outside artifacts/",
    "RDS snapshots, MySQL DDL, triggers, routines and views",
]


def artifact_path(key):
    require(
        isinstance(key, str) and key.startswith(PREFIX) and len(key) <= 1024,
        "Artifact key lies outside the supported prefix.",
    )
    parts = key.removeprefix(PREFIX).split("/")
    for part in parts:
        matching(part, r"[A-Za-z0-9_.=+-]{1,150}", "portable artifact key component")
        require(
            not part.endswith(".")
            and part not in (".", "..")
            and not re.fullmatch(
                r"(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])",
                part.split(".")[0],
                re.IGNORECASE,
            ),
            "Artifact key has a special path component.",
        )
    return "/".join(parts)


def portable_paths(keys):
    prefixes = {}
    paths = set()
    for key in keys:
        parts = artifact_path(key).split("/")
        for i in range(1, len(parts) + 1):
            original = "/".join(parts[:i])
            normalized = original.casefold()
            require(
                prefixes.setdefault(normalized, original) == original,
                "Artifact keys have conflicting case on a portable filesystem.",
            )
            if i == len(parts):
                require(
                    normalized not in paths,
                    "Artifact inventory contains duplicate keys.",
                )
                paths.add(normalized)
    for path in paths:
        parts = path.split("/")
        require(
            not any("/".join(parts[:i]) in paths for i in range(1, len(parts))),
            "Artifact keys collide as both file and directory.",
        )


def s3_inventory(aws, bucket, limit):
    result = {}
    token = None
    while True:
        paging = ("--continuation-token", token) if token else ()
        page = aws.call(
            "s3api",
            "list-objects-v2",
            "--bucket",
            bucket,
            "--prefix",
            PREFIX,
            "--expected-bucket-owner",
            aws.account,
            "--max-keys",
            str(min(1000, limit + 1)),
            "--no-paginate",
            *paging,
        )
        for item in page.get("Contents", []):
            key = item["Key"]
            size = whole(item["Size"], "artifact bytes")
            require(
                key not in result
                and size <= MAX_OBJECT
                and item["StorageClass"] == "STANDARD",
                "Artifact inventory has a duplicate, archived or oversized object.",
            )
            result[key] = {
                "bytes": size,
                "etag": matching(item["ETag"], ETAG, "S3 ETag"),
                "modified": item["LastModified"],
            }
            require(len(result) <= limit, "Artifact count exceeds scope.")
        if not page["IsTruncated"]:
            break
        previous, token = token, page.get("NextContinuationToken")
        require(token and token != previous, "S3 pagination did not advance.")
    portable_paths(result)
    return result


def sync_artifacts(aws, guard, bucket, inventory, root):
    entries = {}
    root.mkdir(mode=0o700)
    for key, expected in inventory.items():
        guard.check()
        path = root / artifact_path(key)
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        response = aws.call(
            "s3api",
            "get-object",
            "--bucket",
            bucket,
            "--key",
            key,
            "--if-match",
            expected["etag"],
            "--expected-bucket-owner",
            aws.account,
            "--checksum-mode",
            "ENABLED",
            path,
            timeout=transfer_timeout(guard),
        )
        path = private_file(path)
        require(
            path.stat().st_size == expected["bytes"] == response["ContentLength"]
            and response["ETag"] == expected["etag"],
            "Downloaded artifact changed or is incomplete.",
        )
        entries[key] = {"bytes": expected["bytes"], "sha256": sha256(path)}
    require(
        s3_inventory(aws, bucket, len(inventory)) == inventory,
        "S3 inventory changed during backup. Keep the partial output private.",
    )
    return entries


def transfer_timeout(guard):
    guard.check()
    return min(600, (guard.work_end - now()).total_seconds())


def upload_artifacts(aws, guard, bucket, entries, root, limit):
    for key, expected in entries.items():
        path = private_file(root / artifact_path(key))
        aws.call(
            "s3api",
            "put-object",
            "--bucket",
            bucket,
            "--key",
            key,
            "--body",
            path,
            "--expected-bucket-owner",
            aws.account,
            "--if-none-match",
            "*",
            "--server-side-encryption",
            "AES256",
            "--checksum-sha256",
            b64(bytes.fromhex(expected["sha256"])),
            change=True,
            timeout=transfer_timeout(guard),
        )
    check_artifacts(aws, guard, bucket, entries, limit)


def check_artifacts(aws, guard, bucket, expected, limit):
    actual = s3_inventory(aws, bucket, limit)
    require(
        {k: v["bytes"] for k, v in actual.items()}
        == {k: v["bytes"] for k, v in expected.items()},
        "Destination artifact count, keys or sizes differ.",
    )
    for key, entry in expected.items():
        guard.check()
        head = aws.call(
            "s3api",
            "head-object",
            "--bucket",
            bucket,
            "--key",
            key,
            "--expected-bucket-owner",
            aws.account,
            "--checksum-mode",
            "ENABLED",
        )
        require(
            head.get("ChecksumSHA256") == b64(bytes.fromhex(entry["sha256"])),
            "Destination artifact checksum differs.",
        )


def dump_value(value):
    if isinstance(value, datetime):
        return value.isoformat(sep=" ")
    if isinstance(value, (date, Decimal)):
        return str(value)
    require(
        type(value) in (str, int, float, type(None)),
        "Unsupported column value in fraud data.",
    )
    return value


@contextmanager
def connection(host, user, password_file):
    try:
        import pymysql
    except ImportError:
        raise Stop(
            "Use the uv environment with the locked PyMySQL dependency."
        ) from None

    password = private_file(password_file, 512).read_text().removesuffix("\n")
    require(
        password and "\n" not in password and "\r" not in password,
        "Invalid private SQL credential file.",
    )
    conn = None
    try:
        conn = pymysql.connect(
            host=host,
            port=3306,
            database=SCHEMA,
            user=user,
            password=password,
            ssl_ca=str(
                ROOT.parent / "src/mlops_fraud_prediction_ca1/rds-global-bundle.pem"
            ),
            ssl_verify_cert=True,
            ssl_verify_identity=True,
            charset="utf8mb4",
            connect_timeout=10,
            read_timeout=60,
            write_timeout=60,
            local_infile=False,
            autocommit=True,
        )
        yield conn
    except pymysql.MySQLError:
        raise Stop(
            "Fraud SQL operation failed. Check scoped grants, TLS/network and transaction state."
        ) from None
    finally:
        if conn and conn.open:
            conn.close()


def schema_info(conn):
    signatures = {}
    columns = {}
    with conn.cursor() as cursor:
        cursor.execute(
            "SELECT TABLE_NAME, ENGINE, DATA_LENGTH FROM information_schema.TABLES WHERE TABLE_SCHEMA=%s",
            (SCHEMA,),
        )
        tables = cursor.fetchall()
        require(
            {t[0] for t in tables} == set(KEYS)
            and all(t[1] == "InnoDB" for t in tables),
            "The scope requires exactly the four established InnoDB fraud tables.",
        )
        cursor.execute(
            "SELECT COUNT(*) FROM information_schema.TRIGGERS WHERE EVENT_OBJECT_SCHEMA=%s",
            (SCHEMA,),
        )
        require(cursor.fetchone()[0] == 0, "Fraud schema has visible triggers. Stop.")
        for table, keys in KEYS.items():
            cursor.execute(f"SHOW CREATE TABLE `{table}`")
            ddl = re.sub(r" AUTO_INCREMENT=\d+(?=\s|$)", "", cursor.fetchone()[1])
            signatures[table] = hashlib.sha256(ddl.encode()).hexdigest()
            cursor.execute(f"SHOW INDEX FROM `{table}` WHERE Key_name = 'PRIMARY'")
            key_columns = tuple(
                row[4] for row in sorted(cursor.fetchall(), key=lambda row: row[3])
            )
            require(
                key_columns == keys,
                "Fraud table primary key differs from approved schema.",
            )
            cursor.execute(f"SHOW COLUMNS FROM `{table}`")
            col = [row[0] for row in cursor.fetchall()]
            require(
                set(keys) <= set(col),
                "Expected fraud primary key columns are missing.",
            )
            for name in col:
                matching(name, r"[A-Za-z_][A-Za-z0-9_]{0,63}", "fraud column")
            columns[table] = col
    return signatures, columns, sum(t[2] for t in tables)


def snapshot(conn, *, restore=False):
    with conn.cursor() as cursor:
        cursor.execute("SET SESSION TRANSACTION ISOLATION LEVEL REPEATABLE READ")
        if restore:
            cursor.execute(
                "SET SESSION sql_mode='STRICT_ALL_TABLES,NO_ENGINE_SUBSTITUTION'"
            )
            cursor.execute("START TRANSACTION READ WRITE")
        else:
            cursor.execute("START TRANSACTION WITH CONSISTENT SNAPSHOT, READ ONLY")


def empty_tables(conn, *, lock=False):
    with conn.cursor() as cursor:
        for table in KEYS:
            suffix = " FOR UPDATE" if lock else ""
            cursor.execute(f"SELECT 1 FROM `{table}` LIMIT 1{suffix}")
            require(
                cursor.fetchone() is None,
                "Destination fraud data already exists. No rows will be overwritten.",
            )


def table_dump(conn, table, columns, maximum, guard, out=None):
    from pymysql.cursors import SSCursor

    checksum = hashlib.sha256()
    bytes_written = 0
    rows = 0
    selection = ", ".join(f"`{name}`" for name in columns)
    order = ", ".join(f"`{name}`" for name in KEYS[table])
    cursor = conn.cursor(SSCursor)
    try:
        cursor.execute(f"SELECT {selection} FROM `{table}` ORDER BY {order}")
        for row in cursor:
            if rows % 256 == 0:
                guard.check()
            content = json_bytes([dump_value(value) for value in row])
            bytes_written += len(content)
            require(
                len(content) <= MAX_LINE and bytes_written <= maximum,
                "Fraud dump exceeded approved bytes or supported row size.",
            )
            if out is not None:
                out.write(content)
            checksum.update(content)
            rows += 1
    except BaseException:
        # Avoid SSCursor.close() draining an unbounded server result on failure.
        # The server rolls back any open transaction when the socket closes.
        cursor.connection = None
        if conn.open:
            conn.close()
        raise
    cursor.close()
    return {"bytes": bytes_written, "rows": rows, "sha256": checksum.hexdigest()}


def table_import(conn, table, columns, path, expected, guard):
    selection = ", ".join(f"`{name}`" for name in columns)
    placeholders = ", ".join(["%s"] * len(columns))
    query = f"INSERT INTO `{table}` ({selection}) VALUES ({placeholders})"
    checksum = hashlib.sha256()
    rows = 0
    with private_file(path).open("rb") as stream, conn.cursor() as cursor:
        while True:
            guard.check()
            chunk = []
            for _ in range(256):
                line = stream.readline(MAX_LINE + 1)
                if not line:
                    break
                require(
                    len(line) <= MAX_LINE and line.endswith(b"\n"),
                    "Invalid fraud row length.",
                )
                row = json.loads(line)
                require(
                    isinstance(row, list)
                    and len(row) == len(columns)
                    and all(type(v) in (str, int, float, type(None)) for v in row),
                    "Invalid typed fraud row.",
                )
                chunk.append(row)
                checksum.update(line)
                rows += 1
            if not chunk:
                break
            cursor.executemany(query, chunk)
    require(
        rows == expected["rows"] and checksum.hexdigest() == expected["sha256"],
        "Fraud dump row count or checksum changed during import.",
    )


class Recovery:
    def __init__(self, config, aws, *, destination=False):
        self.config = config
        self.aws = aws
        self.destination = destination
        self.trial = config["trial"]
        self.account = config["account"]
        self.max_db = whole(config["db_dump_max_bytes"], "DB dump capacity", 1)
        self.max_artifact = whole(config["artifact_max_bytes"], "artifact capacity", 1)
        self.max_objects = whole(config["max_objects"], "artifact count ceiling", 1)
        require(self.max_objects <= 10_000, "At most 10,000 artifacts are supported.")
        self.local_usd = number(
            config["local_cost_bound_usd"], "local staging and transfer cost"
        )
        self.source_account = (
            matching(
                config["approved_source_account"],
                r"[0-9]{12}",
                "approved source account",
            )
            if destination
            else self.account
        )
        self.mib_per_second = number(
            config["minimum_mib_per_second"],
            "minimum aggregate throughput",
            Decimal("0.1"),
        )
        parent = private_dir(config["private_dir"])
        name = matching(
            config["backup_name"],
            r"[a-zA-Z0-9][a-zA-Z0-9_-]{1,63}",
            "private backup name",
        )
        self.path = parent / name
        user_suffix = "writer" if destination else "reader"
        self.user = matching(
            config["data_user"],
            rf"[a-zA-Z][a-zA-Z0-9_]{{0,23}}_{user_suffix}",
            "approved fraud credential user",
        )
        require(
            not self.user.lower().startswith("mlflow"),
            "Never use an MLflow server credential.",
        )
        self.password = private_file(config["data_password_file"], 512)
        if destination:
            self.digest = matching(
                config["manifest_sha256"],
                HEX,
                "independently recorded manifest SHA-256",
            )
            self.db = f"p26-{self.trial}-db"
            self.bucket = f"p26-{self.account}-{self.trial}-artifacts"
        else:
            self.db = matching(
                config["source_db"],
                r"[a-z][a-z0-9-]{1,61}[a-z0-9]",
                "source RDS identifier",
            )
            self.bucket = bucket_name(config["source_bucket"])
        require(
            config["scope_and_quiescence_verified"] is True,
            "Confirm exact approved scopes, portable artifact keys and quiesced writers before continuing.",
        )

    def estimate(
        self,
        guard,
        *,
        db_bytes,
        artifact_bytes,
        count,
        local_write=False,
        offline=False,
    ):
        require(
            db_bytes <= self.max_db
            and artifact_bytes <= self.max_artifact
            and count <= self.max_objects,
            "Measured sizes exceed approved data capacity.",
        )
        if local_write:
            margin = max(GIB, (db_bytes + artifact_bytes) // 5)
            free = shutil.disk_usage(self.path.parent).free
            require(
                free >= db_bytes + artifact_bytes + margin,
                "Private disk lacks copying capacity and reserve.",
            )
            emit(
                f"Local disk available {free:,} bytes; write allowance {db_bytes + artifact_bytes + margin:,} bytes."
            )
        else:
            emit(
                f"Local input storage {db_bytes + artifact_bytes:,} bytes; no second copy planned."
            )
        # Includes checksum re-read, import/read-back and per-object request delay.
        estimated_seconds = (3 * db_bytes + 2 * artifact_bytes) / (
            self.mib_per_second * 1024**2
        )
        minutes = int((estimated_seconds + 3 * count) / 60) + 5
        guard.check(minutes=minutes)
        emit(
            f"Capacity: fraud bytes <= {db_bytes:,}, artifacts {artifact_bytes:,} bytes, "
            f"objects <= {count:,}; allow {minutes} min at assumed throughput."
        )
        outbound_bytes = db_bytes + artifact_bytes if local_write else db_bytes
        # Destination SQL writes also generate RDS undo/log data; preserve
        # room separately and include the allowed lifetime of the lab stack.
        guard.estimate(
            outbound_bytes=outbound_bytes,
            artifact_bytes=artifact_bytes,
            keys=count,
            stack=True,
            local_usd=self.local_usd,
        )
        if offline:
            emit(
                "Offline estimates use approved ceilings or the validated private manifest."
            )

    def host(self):
        (db,) = self.aws.call(
            "rds", "describe-db-instances", "--db-instance-identifier", self.db
        )["DBInstances"]
        require(
            db["DBInstanceArn"] == f"arn:aws:rds:{REGION}:{self.account}:db:{self.db}"
            and db["DBInstanceStatus"] == "available"
            and db["StorageEncrypted"] is True
            and db["Engine"] == "mysql"
            and db["EngineVersion"].startswith("8.4.")
            and db["Endpoint"]["Port"] == 3306,
            "RDS endpoint is outside the approved available MySQL 8.4 scope.",
        )
        require(
            not self.destination or owned(db.get("TagList", []), self.trial),
            "Destination RDS must be tagged for the exact approved trial.",
        )
        return matching(
            db["Endpoint"]["Address"],
            r"[a-z0-9.-]+\.us-east-1\.rds\.amazonaws\.com",
            "RDS endpoint",
        )

    def backup(self, guard, *, apply, offline):
        require(
            not os.path.lexists(self.path),
            "Backup output already exists. Choose a new private path.",
        )
        if offline:
            self.estimate(
                guard,
                db_bytes=self.max_db,
                artifact_bytes=self.max_artifact,
                count=self.max_objects,
                local_write=True,
                offline=True,
            )
            emit(
                "Plan: inspect approved source RDS and S3; snapshot fraud and sync objects; write private checksums."
            )
            return
        bucket_state(self.aws, self.bucket)
        src = s3_inventory(self.aws, self.bucket, self.max_objects)
        artifact_bytes = sum(v["bytes"] for v in src.values())
        host = self.host()
        with connection(host, self.user, self.password) as conn:
            sig, columns, data_length = schema_info(conn)
        require(
            2 * data_length <= self.max_db,
            "Twice measured InnoDB data length exceeds the dump ceiling.",
        )
        emit(
            f"Source RDS reports {data_length:,} approximate InnoDB data bytes before export."
        )
        self.estimate(
            guard,
            db_bytes=self.max_db,
            artifact_bytes=artifact_bytes,
            count=len(src),
            local_write=True,
        )
        emit(
            "Plan: read one consistent fraud snapshot, sync scoped objects and seal a new private backup."
        )
        if not apply:
            return
        guard.check_ce(self.aws)
        self.aws.identity()
        self.path.mkdir(mode=0o700)
        emit(
            "Backup started. A directory without the final manifest represents an incomplete backup."
        )
        with connection(host, self.user, self.password) as conn:
            snapshot(conn)
            require(
                schema_info(conn)[:2] == (sig, columns),
                "Source fraud schema changed after sizing.",
            )
            remaining = self.max_db
            tables = {}
            for table in KEYS:
                with (self.path / f"{table}.jsonl").open("xb") as out:
                    tables[table] = table_dump(
                        conn, table, columns[table], remaining, guard, out
                    )
                remaining -= tables[table]["bytes"]
            conn.rollback()
        artifacts = sync_artifacts(
            self.aws, guard, self.bucket, src, self.path / "artifacts"
        )
        guard.check()
        manifest = {
            "version": 1,
            "source_account": self.account,
            "source_region": REGION,
            "source_db": self.db,
            "source_bucket": self.bucket,
            "signatures": sig,
            "columns": columns,
            "tables": tables,
            "artifacts": artifacts,
            "omissions": OMISSIONS,
        }
        pending = self.path / ".manifest.pending"
        with pending.open("xb") as out:
            out.write(json_bytes(manifest))
        require(
            pending.stat().st_size <= 16 * 1024 * 1024,
            "Generated manifest exceeds the supported size.",
        )
        pending.rename(self.path / "manifest.json")
        self.evidence(manifest, sha256(self.path / "manifest.json"))

    def manifest(self):
        private_dir(self.path)
        path = private_file(self.path / "manifest.json", 16 * 1024 * 1024)
        raw = path.read_bytes()
        require(
            hashlib.sha256(raw).hexdigest() == self.digest,
            "Manifest hash differs from the approved value.",
        )
        manifest = json.loads(raw)
        require(
            manifest["version"] == 1
            and manifest["source_region"] == REGION
            and manifest["source_account"] == self.source_account
            and self.source_account != self.account
            and manifest["omissions"] == OMISSIONS,
            "Unsupported manifest or restore to the source account.",
        )
        bucket_name(manifest["source_bucket"])
        require(
            set(manifest["tables"])
            == set(manifest["columns"])
            == set(manifest["signatures"])
            == set(KEYS),
            "Manifest schema scope differs from the four supported fraud tables.",
        )
        totals = (
            sum(
                whole(v["bytes"], "manifest file size")
                for v in manifest["tables"].values()
            ),
            sum(
                whole(v["bytes"], "manifest file size")
                for v in manifest["artifacts"].values()
            ),
        )
        require(
            totals[0] <= self.max_db and totals[1] <= self.max_artifact,
            "Manifest exceeds approved local data capacity.",
        )
        files = {"manifest.json"}
        for table, record in manifest["tables"].items():
            files.add(f"{table}.jsonl")
            columns = manifest["columns"][table]
            require(
                isinstance(columns, list)
                and set(KEYS[table]) <= set(columns)
                and len(columns) == len(set(columns)),
                "Manifest has unsupported fraud columns.",
            )
            for name in columns:
                matching(name, r"[A-Za-z_][A-Za-z0-9_]{0,63}", "manifest column")
            whole(record["rows"], "table row count")
            matching(manifest["signatures"][table], HEX, "source schema signature")
            validate_file(self.path / f"{table}.jsonl", record)
        artifacts = manifest["artifacts"]
        require(
            len(artifacts) <= self.max_objects, "Manifest artifact count exceeds scope."
        )
        portable_paths(artifacts)
        private_dir(self.path / "artifacts")
        for key, record in artifacts.items():
            relative = f"artifacts/{artifact_path(key)}"
            files.add(relative)
            require(
                whole(record["bytes"], "artifact bytes") <= MAX_OBJECT,
                "Oversized artifact in manifest.",
            )
            validate_file(self.path / relative, record)
        actual = set()
        for entry in self.path.rglob("*"):
            if entry.is_dir():
                private_dir(entry)
            else:
                private_file(entry)
                actual.add(entry.relative_to(self.path).as_posix())
        require(actual == files, "Bundle contains unexpected or missing files.")
        return manifest

    def evidence(self, manifest, digest):
        counts = ", ".join(f"{k} {v['rows']:,}" for k, v in manifest["tables"].items())
        emit(
            f"Integrity: manifest SHA-256 {digest}; rows {counts}; objects {len(manifest['artifacts']):,}."
        )
        emit("Omissions: " + "; ".join(manifest["omissions"]) + ".")

    def restore(self, guard, *, apply, offline, verify=False):
        manifest = self.manifest()
        db_bytes = sum(v["bytes"] for v in manifest["tables"].values())
        artifact_bytes = sum(v["bytes"] for v in manifest["artifacts"].values())
        self.estimate(
            guard,
            db_bytes=db_bytes,
            artifact_bytes=artifact_bytes,
            count=len(manifest["artifacts"]),
            offline=offline,
        )
        self.evidence(manifest, self.digest)
        if offline:
            emit(
                "Plan: "
                + (
                    "verify restored rows and object checksums."
                    if verify
                    else "require empty fraud tables and prefix; conditional object puts, then transactional import."
                )
            )
            return
        bucket_state(self.aws, self.bucket, trial=self.trial)
        host = self.host()
        with connection(host, self.user, self.password) as conn:
            require(
                schema_info(conn)[:2] == (manifest["signatures"], manifest["columns"]),
                "Destination fraud schema does not match the source signature.",
            )
            if verify:
                snapshot(conn)
                self.verify_rows(conn, manifest, guard)
                conn.rollback()
            else:
                empty_tables(conn)
        if verify:
            check_artifacts(
                self.aws, guard, self.bucket, manifest["artifacts"], self.max_objects
            )
            emit(
                "Verified destination fraud and S3 bytes against the approved manifest."
            )
            return
        require(
            not s3_inventory(self.aws, self.bucket, self.max_objects),
            "Destination artifact prefix is not empty.",
        )
        if not manifest["artifacts"]:
            emit("No artifact objects were selected by the approved inventory.")
        self.db_capacity(db_bytes)
        emit(
            "Plan: restore only into a quiesced empty fraud schema and prefix. No existing data is overwritten."
        )
        if not apply:
            return
        guard.check_ce(self.aws)
        self.aws.identity()
        with connection(host, self.user, self.password) as conn:
            require(
                schema_info(conn)[:2] == (manifest["signatures"], manifest["columns"]),
                "Destination schema changed after preflight.",
            )
            snapshot(conn, restore=True)
            empty_tables(conn, lock=True)
            try:
                upload_artifacts(
                    self.aws,
                    guard,
                    self.bucket,
                    manifest["artifacts"],
                    self.path / "artifacts",
                    self.max_objects,
                )
                self.aws.identity()
                for table in KEYS:
                    table_import(
                        conn,
                        table,
                        manifest["columns"][table],
                        self.path / f"{table}.jsonl",
                        manifest["tables"][table],
                        guard,
                    )
                self.verify_rows(conn, manifest, guard)
                guard.check()
                conn.commit()
            except BaseException:
                if conn.open:
                    conn.rollback()
                raise
        emit(
            "Restore committed and verified. Keep cutover disabled until the scoped operator validates recovery."
        )

    def db_capacity(self, db_bytes):
        until = now()
        response = self.aws.call(
            "cloudwatch",
            "get-metric-statistics",
            "--namespace",
            "AWS/RDS",
            "--metric-name",
            "FreeStorageSpace",
            "--dimensions",
            f"Name=DBInstanceIdentifier,Value={self.db}",
            "--start-time",
            (until - timedelta(minutes=20)).isoformat(),
            "--end-time",
            until.isoformat(),
            "--period",
            "300",
            "--statistics",
            "Minimum",
        )["Datapoints"]
        require(
            response and min(p["Minimum"] for p in response) >= 3 * db_bytes + GIB,
            "Recent RDS FreeStorageSpace does not support import, undo/log and safety margin.",
        )

    def verify_rows(self, conn, manifest, guard):
        for table in KEYS:
            expected = manifest["tables"][table]
            actual = table_dump(
                conn, table, manifest["columns"][table], expected["bytes"], guard
            )
            require(actual == expected, "Destination fraud rows differ from manifest.")


def validate_file(path, record):
    matching(record["sha256"], HEX, "content SHA-256")
    require(
        private_file(path).stat().st_size
        == whole(record["bytes"], "manifest file size")
        and sha256(path) == record["sha256"],
        "Backup file size or checksum differs.",
    )
