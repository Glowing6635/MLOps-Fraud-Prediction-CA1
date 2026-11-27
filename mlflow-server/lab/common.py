"""Private inputs, account checks and time/budget limits for the lab commands."""

import base64
import fcntl
import hashlib
import json
import os
import re
import stat
import subprocess
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

REGION = "us-east-1"
TAG = "P26Trial"
ROOT = Path(__file__).resolve().parents[1]
GIB = 1024**3
MONTH = Decimal(730)


class Stop(Exception):
    """An actionable message containing no raw service output."""


def emit(message):
    # Keep evidence on stdout and errors on stderr through the CLI logger.
    print(message, flush=True)  # noqa: T201


def require(condition, message):
    if not condition:
        raise Stop(message)


def now():
    return datetime.now(UTC)


def timestamp(value):
    try:
        result = datetime.fromisoformat(value)
        require(result.utcoffset() == timedelta(0), "Use UTC times.")
        return result
    except (AttributeError, TypeError, ValueError):
        raise Stop("Use ISO 8601 UTC times.") from None


def number(value, label, minimum=0):
    try:
        require(type(value) in (str, int, float, Decimal), f"Invalid {label}.")
        result = Decimal(str(value))
        require(result.is_finite() and result >= minimum, f"Invalid {label}.")
        return result
    except ArithmeticError:
        raise Stop(f"Invalid {label}.") from None


def whole(value, label, minimum=0):
    require(type(value) is int and value >= minimum, f"Invalid {label}.")
    return value


def matches(value, pattern):
    return isinstance(value, str) and re.fullmatch(pattern, value) is not None


def matching(value, pattern, label):
    require(matches(value, pattern), f"Invalid {label}.")
    return value


def b64(content):
    return base64.b64encode(content).decode("ascii")


def sha256(path):
    with path.open("rb") as incoming:
        return hashlib.file_digest(incoming, "sha256").hexdigest()


def json_bytes(value):
    return (
        json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n"
    ).encode()


def private_dir(path):
    path = Path(path)
    require(
        path.is_absolute() and path.resolve() == path,
        "Use an absolute, canonical directory.",
    )
    info = path.lstat()
    require(
        stat.S_ISDIR(info.st_mode)
        and info.st_uid == os.getuid()
        and stat.S_IMODE(info.st_mode) == 0o700,
        "Use a real directory owned by you with mode 0700.",
    )
    return path


@contextmanager
def exclusive_workspace(path):
    fd = os.open(private_dir(path), os.O_RDONLY | os.O_DIRECTORY)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise Stop(
                "Another approved operation is using this private workspace."
            ) from None
        yield
    finally:
        os.close(fd)


def private_file(path, maximum=None):
    path = Path(path)
    private_dir(path.parent)
    info = path.lstat()
    require(
        stat.S_ISREG(info.st_mode)
        and info.st_nlink == 1
        and info.st_uid == os.getuid()
        and stat.S_IMODE(info.st_mode) == 0o600,
        "Use a regular single-link private file with mode 0600.",
    )
    require(
        maximum is None or info.st_size <= maximum,
        "Private file exceeds its supported size.",
    )
    return path


def load_json(path):
    return json.loads(private_file(path, 1024 * 1024).read_bytes())


def run(args, *, env=None, payload=None, timeout=90):
    # Never print the arguments or forward third-party stdout/stderr.
    try:
        return subprocess.run(
            args,
            env=env,
            input=payload,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError:
        raise Stop(f"Required executable is missing: {args[0]}.") from None
    except subprocess.TimeoutExpired:
        raise Stop(
            "External command timed out. Check for a partial operation before retrying."
        ) from None


class AWS:
    def __init__(self, config, guard=None):
        self.account = config["account"]
        self.profile = config["profile"]
        self.guard = guard
        self.env = {
            k: v
            for k, v in os.environ.items()
            if not k.startswith(
                ("AWS_ENDPOINT_URL", "AWS_ACCESS_KEY", "AWS_SECRET_KEY")
            )
            and k
            not in {
                "AWS_SESSION_TOKEN",
                "AWS_SECURITY_TOKEN",
                "AWS_CA_BUNDLE",
                "AWS_WEB_IDENTITY_TOKEN_FILE",
                "AWS_ROLE_ARN",
                "AWS_CONTAINER_CREDENTIALS_FULL_URI",
                "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
                "BOTO_CONFIG",
            }
        }
        self.env.update(
            AWS_CLI_AUTO_PROMPT="off",
            AWS_PAGER="",
            AWS_IGNORE_CONFIGURED_ENDPOINT_URLS="true",
            AWS_EC2_METADATA_DISABLED="true",
            AWS_RETRY_MODE="standard",
            AWS_MAX_ATTEMPTS="1",
        )

    def private_cli(self):
        for profile in {self.profile, "default"}:
            state = run(
                ["aws", "--profile", profile, "configure", "get", "cli_history"],
                env=self.env,
            )
            unset = state.returncode == 1 and not state.stdout and not state.stderr
            disabled = (
                state.returncode == 0 and state.stdout.strip().lower() == b"disabled"
            )
            missing_default = (
                profile == "default"
                and self.profile != "default"
                and not state.stdout
                and b"The config profile (default) could not be found" in state.stderr
            )
            require(
                unset or disabled or missing_default,
                "AWS CLI history must be disabled in the selected and default profiles.",
            )

    def call(
        self, service, operation, *args, absent=(), raw=False, timeout=90, change=False
    ):
        if change:
            require(self.guard is not None, "Internal guard refused a write.")
            self.guard.check()
            self.identity()
        result = run(
            [
                "aws",
                "--profile",
                self.profile,
                "--region",
                REGION,
                "--no-cli-pager",
                "--output",
                "json",
                "--cli-connect-timeout",
                "10",
                "--cli-read-timeout",
                "30",
                service,
                operation,
                *map(str, args),
            ],
            env=self.env,
            timeout=timeout,
        )
        if result.returncode:
            match = re.search(
                rb"An error occurred \(([A-Za-z0-9.]+)\) when calling", result.stderr
            )
            if match and match[1].decode() in absent:
                return None
            code = f" ({match[1].decode()})" if match else ""
            raise Stop(
                f"AWS {service} {operation} failed{code}. Check approved access and partial state."
            )
        if raw:
            return result.stdout
        return json.loads(result.stdout) if result.stdout.strip() else {}

    def identity(self):
        who = self.call("sts", "get-caller-identity")
        require(
            who.get("Account") == self.account,
            "AWS account does not match the private configuration.",
        )
        require(
            matches(
                who.get("Arn"),
                rf"arn:aws:sts::{self.account}:assumed-role/voclabs/[^/]+",
            ),
            "AWS caller is not the expected Learner Lab role.",
        )

    def write(self, service, operation, *args):
        return self.call(service, operation, *args, change=True)


class Approval:
    """Window and the owner's conservative, cumulative budget for one operation."""

    def __init__(self, record, operation):
        require(
            record["operation"] == operation,
            "Approval does not name this exact operation and phase.",
        )
        require(
            isinstance(record["reference"], str) and bool(record["reference"].strip()),
            "A scoped approval reference is required.",
        )
        self.start = timestamp(record["start_utc"])
        self.end = timestamp(record["lab_end_utc"])
        self.expiry = timestamp(record["credential_expiry_utc"])
        self.work_end = timestamp(record["work_end_utc"])
        self.remaining = number(record["available_credit_usd"], "available credit")
        self.credit_limit = number(record["credit_limit_usd"], "credit limit", 15)
        require(
            self.remaining <= self.credit_limit,
            "Available credit exceeds the verified credit limit.",
        )
        self.unposted = number(
            record["unposted_and_concurrent_usd"], "unposted/concurrent allowance"
        )
        self.spent = number(record["trial_spent_and_committed_usd"], "trial spend")
        self.future = number(
            record["trial_remaining_bound_usd"],
            "remaining trial bound",
            Decimal("0.01"),
        )
        self.other = number(
            record["extra_cost_bound_usd"], "extra cost bound", Decimal("0.01")
        )
        self.request_count = whole(record["ce_calls_bound"], "CE request count", 1)
        self.retained_hours = number(record["retained_hours"], "retained hours", 1)
        self.observation_minutes = whole(
            record["observation_minutes"], "observation minutes"
        )
        self.cleanup_minutes = whole(record["cleanup_minutes"], "cleanup minutes", 15)
        require(
            self.start < self.work_end
            and self.work_end
            + timedelta(minutes=self.observation_minutes + self.cleanup_minutes)
            <= min(self.end, self.expiry),
            "The UTC window leaves insufficient time for observation and cleanup.",
        )
        require(
            self.retained_hours
            >= Decimal(str((self.end - self.start).total_seconds())) / 3600,
            "Retained-hour bound must cover the approved lab window.",
        )
        self.check()

    def check(self, minutes=0):
        instant = now()
        require(
            self.start <= instant < self.work_end
            and instant + timedelta(minutes=minutes) <= self.work_end,
            "The approved work window is closed or too short. Keep the observation and cleanup reserve.",
        )

    def estimate(
        self,
        *,
        outbound_bytes=0,
        artifact_bytes=0,
        keys=0,
        image_bytes=0,
        stack=False,
        local_usd=Decimal(0),
    ):
        # Conservative fallback rates for the supported N. Virginia trial shape.
        # The owner independently confirms rates and accounts for variable charges.
        out_gib = Decimal(outbound_bytes) / GIB
        s3_gib = Decimal(artifact_bytes) / GIB
        ecr_gib = Decimal(image_bytes) / GIB
        costs = {
            "compute/IPv4": Decimal("0.06") * self.retained_hours
            if stack
            else Decimal(0),
            "block/DB storage and backup": Decimal(6) * self.retained_hours / MONTH
            if stack
            else Decimal(0),
            "storage": (s3_gib * Decimal("0.03") + ecr_gib * Decimal("0.10"))
            * self.retained_hours
            / MONTH,
            "outbound transfer": Decimal("0.12") * out_gib,
            "requests": Decimal("0.01") * self.request_count
            + Decimal("0.00002") * keys,
            "other charges": self.other,
            "local charges": local_usd,
        }
        total = sum(costs.values())
        emit(
            "Cost bound (USD, estimates): "
            + ", ".join(f"{k} {v:.4f}" for k, v in costs.items())
        )
        emit(
            f"Operation bound {total:.4f}; cumulative trial ceiling 10; credit reserve 15."
        )
        require(
            total <= self.future and self.spent + self.future <= 10,
            "The estimated operation and other committed trial charges exceed the approved cap.",
        )
        require(
            self.remaining - self.unposted - self.future >= 15,
            "Owner-reported credit would fall below the USD 15 reserve.",
        )

    def check_ce(self, aws):
        today = now().date()
        # CE end is exclusive. Query through yesterday, including the
        # prior calendar month to avoid a balance reset on the first day.
        month = today.replace(day=1)
        period_start = (month - timedelta(days=1)).replace(day=1)
        starts = (
            [str(period_start)] if today == month else [str(period_start), str(month)]
        )
        grouped = aws.call(
            "ce",
            "get-cost-and-usage",
            "--time-period",
            f"Start={period_start},End={today}",
            "--granularity",
            "MONTHLY",
            "--metrics",
            "UnblendedCost",
            "--no-paginate",
        )
        require("NextPageToken" not in grouped, "CE response is incomplete.")
        periods = grouped.get("ResultsByTime", [])
        require(
            [part["TimePeriod"]["Start"] for part in periods] == starts,
            "CE returned incomplete billing periods.",
        )
        amounts = [part["Total"]["UnblendedCost"] for part in periods]
        require(
            all(item["Unit"] == "USD" for item in amounts),
            "CE has no complete USD estimate.",
        )
        posted = sum(
            number(item["Amount"], "CE total", -self.credit_limit) for item in amounts
        )
        credit = min(self.remaining, self.credit_limit - max(0, posted))
        emit(
            f"Fresh CE estimate through prior UTC day (lag possible): USD {posted:.2f}."
        )
        require(
            credit - self.unposted - self.future >= 15,
            "Fresh CE and the unposted allowance leave insufficient credit.",
        )
        self.check()


def config_base(config):
    matching(config["account"], r"[0-9]{12}", "account")
    matching(config["profile"], r"[A-Za-z0-9_.-]{1,64}", "explicit AWS profile")
    matching(config["trial"], r"[a-z][a-z0-9]{2,14}", "trial label")
    require(config["region"] == REGION, "Only us-east-1 is supported.")
