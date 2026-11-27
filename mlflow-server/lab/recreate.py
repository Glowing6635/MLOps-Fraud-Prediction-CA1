"""Command dispatcher and sanitized plans for Learner Lab recreation."""

import argparse
import sys
from contextlib import nullcontext

from common import (
    AWS,
    Approval,
    Stop,
    config_base,
    emit,
    exclusive_workspace,
    load_json,
    require,
    whole,
)
from infrastructure import Stack
from recovery import Recovery


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    actions = cli.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("create-lab", "create missing trial resources in one approved phase"),
        ("backup-data", "make a private fraud and artifact backup"),
        ("restore-data", "restore an approved backup into a new empty trial"),
        ("verify", "check the trial stack or restored data"),
    ):
        command = actions.add_parser(name, help=help_text)
        command.add_argument(
            "--config",
            required=True,
            help="absolute path to the private JSON configuration",
        )
        mode = command.add_mutually_exclusive_group()
        mode.add_argument(
            "--dry-run", action="store_true", help="read validation only (default)"
        )
        mode.add_argument(
            "--offline",
            action="store_true",
            help="validate local inputs and describe actions without AWS/DB",
        )
        if name != "verify":
            mode.add_argument(
                "--apply",
                action="store_true",
                help="execute the separately approved operation",
            )
        if name == "create-lab":
            command.add_argument(
                "--phase", choices=("foundation", "image", "server"), required=True
            )
        if name == "verify":
            command.add_argument(
                "--data",
                action="store_true",
                help="verify the restored data against the private manifest",
            )
    return cli


def offline_stack(stack, phase, guard):
    if phase == "foundation":
        stack.no_new_iam()
        stack.bootstrap_password()
        guard.check(minutes=whole(stack.config["rds_wait_minutes"], "RDS wait", 15))
        emit(
            "Plan: verify roles, existing restricted rules and regional limits; create missing RDS, S3 and ECR."
        )
    elif phase == "image":
        if not stack.image_digest:
            stack.image_build_preflight(guard, offline=True)
        emit(
            "Plan: build pinned Linux AMD64 source and publish one immutable trial tag when absent."
        )
    else:
        require(stack.image_digest, "Pin the expected image digest before continuing.")
        stack.runtime_approval()
        # Construct the exact account-derived bundle locally without writing it.
        stack.zip_bundle(stack.image_digest)
        if phase == "server":
            guard.check(minutes=whole(stack.config["eb_wait_minutes"], "EB wait", 15))
            emit(
                "Plan: verify foundation, approved SSM inputs and IAM controls; publish bundle and launch SingleInstance."
            )
        else:
            emit(
                "Plan: check RDS, both private buckets, ECR, SSM metadata, EB configuration, capacity and /health."
            )
    guard.estimate(stack=True, image_bytes=stack.image_bytes)


def preflight(config):
    require(
        config["quota_and_rate_reviewed"] is True,
        "Confirm AWS account quota headroom and current regional rates with the owner.",
    )
    require(
        config["private_workspace_reviewed"] is True,
        "Confirm approved private storage and isolated operator access.",
    )
    config_base(config)


def execute(args):
    config = load_json(args.config)
    preflight(config)
    apply = getattr(args, "apply", False)
    phase = getattr(args, "phase", None)
    data_verify = getattr(args, "data", False)
    if phase:
        operation = f"create-lab/{phase}"
    elif args.command == "verify":
        operation = "verify/data" if data_verify else "verify/stack"
    else:
        operation = args.command
    guard = Approval(config["approval"], operation)
    emit(
        f"{operation}: "
        + (
            "apply approved operation"
            if apply
            else "offline plan"
            if args.offline
            else "read-only validation"
        )
    )
    aws = AWS(config, guard if apply else None)
    if not args.offline:
        aws.private_cli()
    with exclusive_workspace(config["private_dir"]) if apply else nullcontext():
        dispatch(args, config, guard, aws, apply, phase, data_verify)


def dispatch(args, config, guard, aws, apply, phase, data_verify):
    if phase or (args.command == "verify" and not data_verify):
        stack = Stack(config, aws)
        if args.offline:
            return offline_stack(stack, phase, guard)
        aws.identity()
        if phase == "foundation":
            stack.foundation(apply, guard)
        elif phase == "image":
            stack.publish(apply, guard)
        else:
            stack.server(apply, guard, verify=not phase)
    else:
        restore = args.command == "restore-data" or data_verify
        recovery = Recovery(config, aws, destination=restore)
        if not args.offline:
            aws.identity()
        if restore:
            recovery.restore(
                guard, apply=apply, offline=args.offline, verify=data_verify
            )
        else:
            recovery.backup(guard, apply=apply, offline=args.offline)
    if not apply and not args.offline:
        emit(
            "Read-only validation complete. Execute only with separately approved scope."
        )


def main():
    args = parser().parse_args()
    try:
        execute(args)
    except Stop as exc:
        sys.stderr.write(f"Stopped: {exc}\n")
    except (
        KeyError,
        ValueError,
        TypeError,
        OSError,
        ArithmeticError,
        AttributeError,
        IndexError,
    ) as exc:
        sys.stderr.write(
            f"Stopped: invalid or unavailable private input or service response ({type(exc).__name__}).\n"
        )
    except KeyboardInterrupt:
        sys.stderr.write(
            "Stopped: interrupted. Check for already-submitted AWS work or partial data.\n"
        )
    else:
        return 0
    if getattr(args, "apply", False):
        sys.stderr.write(
            "Review scoped partial resources/data before retrying. No automatic cleanup was attempted.\n"
        )
    return 1


if __name__ == "__main__":
    sys.exit(main())
