# Recreate the Learner Lab

Use `mlflow-server/lab-recreate` from the repository root on an approved,
headless Linux or macOS operator machine. It recreates the #26 deployment
shape in a separate account scope and can transfer the approved fraud data
and artifact objects. The normal deployment and its seven settings are
documented in [Deploy](DEPLOY.md) and [Settings](README.md#settings). See
[decision D26-02](../docs/decision-log.md#d26-02--scope-learner-lab-recreation-and-data-recovery)
for what can be recovered.

## Set up the operator machine

- Use Bash, Python 3.11 or newer, AWS CLI v2, and a private, encrypted local
  filesystem. For RDS data commands, use the repository's Python 3.12 and
  locked PyMySQL through uv. Set up with `uv sync --locked` before the approved
  window. For image publication, prepare the
  [local image engine and buildx](#prepare-the-image-engine) before the window.
- Use an explicit AWS CLI profile for the approved account's short-lived
  `voclabs` role. Refresh credentials privately. Keep CLI history disabled
  in the selected and default profiles and use standard AWS service endpoints
  and certificate verification. The tool accepts only `us-east-1` and checks
  the exact account and caller before opening SQL or making changes.
- Create one operator-owned directory outside the repository with mode
  `0700`, on the approved encrypted filesystem. Store JSON configuration
  and password-only files there with mode `0600`. Keep the directory out
  of backups to unapproved storage, shell history, Git, and shared logs.
  AWS and Docker output is captured privately.
  Redacted output includes sizes, state, counts and digests. The manifest and
  archive remain private.
- Confirm exclusive trial scope across machines and coordinate the source and
  destination quiescence windows. Use the same private directory for phases
  on one machine so its apply lock can prevent concurrent local execution.
  The local lock cannot protect an account against another operator or service.

### Prepare private config paths

From the repository root, set `P26_PRIVATE_DIR` to the existing, approved
operator directory. Set `P26_ENGINE_STORAGE_DIR` to an operator-owned
directory on the filesystem that actually backs the image engine. Run this
in Bash before creating the private copies. `cd -P` and `pwd -P` work on
Linux and macOS and resolve directory symlinks without `realpath` options.

```bash
P26_PRIVATE_DIR=$(CDPATH= cd -P -- "${P26_PRIVATE_DIR:?Set the approved private trial directory}" && pwd -P) || exit 1
P26_ENGINE_STORAGE_DIR=$(CDPATH= cd -P -- "${P26_ENGINE_STORAGE_DIR:?Set the approved engine storage directory}" && pwd -P) || exit 1
FOUNDATION_CONFIG="$P26_PRIVATE_DIR/foundation.json"
IMAGE_CONFIG="$P26_PRIVATE_DIR/image.json"
SERVER_CONFIG="$P26_PRIVATE_DIR/server.json"
VERIFY_STACK_CONFIG="$P26_PRIVATE_DIR/verify-stack.json"
RDS_PASSWORD_FILE="$P26_PRIVATE_DIR/rds-bootstrap-password"
DATA_PASSWORD_FILE="$P26_PRIVATE_DIR/data-password"
(
  umask 077
  set -eC
  for config in "$FOUNDATION_CONFIG" "$IMAGE_CONFIG" "$SERVER_CONFIG" "$VERIFY_STACK_CONFIG"; do
    cat mlflow-server/lab/config.example.json > "$config"
  done
)
```

Use the same path pattern for `--config` in backup, restore and data verify,
in the approved source or destination directory for each. Set each JSON
path field to the literal resulting absolute path on the machine running
the command. Bash variables and `~` are not expanded in JSON.

| Path | Value and validation |
| --- | --- |
| Every `--config` | A mode `0600` regular single-link JSON file in a mode `0700` canonical directory owned by the operator |
| `private_dir` | The canonical workspace (`P26_PRIVATE_DIR`), mode `0700` |
| `rds_password_file` | The approved bootstrap password file (`RDS_PASSWORD_FILE`) for foundation |
| `data_password_file` | The approved reader or writer password file (`DATA_PASSWORD_FILE`) for backup, restore and data verify |
| `docker_capacity_dir` | The canonical engine storage check directory (`P26_ENGINE_STORAGE_DIR`), mode `0700`, for a new image build |
| `docker_socket` | The approved absolute local Unix socket path used by the image runner. Set a filesystem path, without the `unix://` prefix. |

Password files follow the same ownership, mode and parent directory rules
as the configuration. For a recovery-only machine, omit the engine
storage assignment and use only the relevant recovery paths.

Replace the placeholders from [the example](lab/config.example.json).
For the scoped trial route, set
`runtime_parameter_prefix` to `/p26-<trial>/mlflow/` using the exact `trial`
label and keep it consistent across the trial's stack configs. For an approved
legacy deployment in a clean account, omit the field to use `/mlflow/`.
An explicit prefix must match the configured trial, including the trailing
slash. Each invocation uses an approval record for the exact operation and
scope. Set `approval.operation` to
`create-lab/foundation`, `create-lab/image`, `create-lab/server`, `backup-data`,
`restore-data`, `verify/stack` or `verify/data` as applicable. Use an explicit
`source_db` and `source_bucket` for backup. Restore and verify-data need
`approved_source_account`, the independently received `manifest_sha256`, and
the destination user's password file. Keep source and destination
configurations separate and use different AWS account IDs. Do not reuse source network IDs,
RDS endpoints, Docker registry addresses, or secret values.

```bash
./mlflow-server/lab-recreate --help
./mlflow-server/lab-recreate create-lab --phase foundation --config "$FOUNDATION_CONFIG" --offline
./mlflow-server/lab-recreate create-lab --phase foundation --config "$FOUNDATION_CONFIG" --dry-run
```

`--offline` checks private inputs, sizes, time and estimates and makes no
network calls or files. `--dry-run` is the default. It validates against live
AWS and, for data operations, reads SQL metadata or the approved verification
scope. It creates no files or resources and does no transfer or restore.
AWS requests and reads may have charges. `--apply` is an explicit action
for a separately authorized operation, account, window and trial. The tool
fetches Cost Explorer (CE) before its first change in each apply invocation
that has work to do. CE is a billable API. Treat local dry-run and the live
read preflight as different evidence.

### Prepare the image engine

The image phase invokes `docker buildx build`, even when the engine is
Podman. Use the Docker CLI with a buildx plugin discoverable under a fresh
`DOCKER_CONFIG` (for example, a system-installed plugin). The engine must
support a Linux AMD64 build and publishing an OCI index with provenance.
Check AMD64 emulation on ARM hosts in advance. The runner clears
`DOCKER_*`, `BUILDKIT_*` and `BUILDX_*` settings and sets `DOCKER_HOST`
from `docker_socket`, so an interactive Docker context or custom builder
selection does not configure the runner.

For rootless Podman, use an already approved local Podman API socket or,
with approval, run the service in a separate headless process. Set
`P26_DOCKER_SOCKET` to the chosen absolute path in a private directory and
keep the service running for the image phase.

```bash
podman system service --time=0 "unix://${P26_DOCKER_SOCKET:?Set the approved Podman socket path}"
```

The ordinary Docker fallback is the Docker CLI plus buildx pointed at an
approved local Docker Engine socket (for example, `/var/run/docker.sock`
when that is the approved socket). Use the actual socket in the image
config for either engine. Check connectivity and plugin availability
from the same operator account before starting the phase.

```bash
(
  set -e
  unset DOCKER_CONTEXT DOCKER_API_VERSION
  : "${P26_DOCKER_SOCKET:?Set the approved image engine socket}"
  DOCKER_CONFIG=$(mktemp -d "${P26_PRIVATE_DIR:?Set the approved private trial directory}/docker-check.XXXXXXXX")
  trap 'rmdir -- "$DOCKER_CONFIG"' EXIT
  export DOCKER_CONFIG
  export DOCKER_HOST="unix://$P26_DOCKER_SOCKET"
  docker version
  docker buildx version
)
```

These local checks establish socket access and plugin discovery. If the
Podman engine cannot perform the pinned build, use an approved Docker
Engine and point the config and capacity check at that engine. Check for
an existing immutable tag before retrying a failed publication.

## Exact preflight

Confirm the following with the coordinator for the current lab. The JSON
attestations refer to checks already performed by the approved operator
or observer.

| Check | Operator evidence and command behavior |
| --- | --- |
| Account, trial and window | Bind the profile, account ID, region and unique trial name to the authorization. Record credential expiry separately from lab expiry. Calculate a work cutoff that preserves stopped observation and teardown. Check every resource with the same profile. A wrong account, expired window or disallowed caller stops work. |
| Ownership and limits | Reserve the derived trial names with the account owner before launch. Precheck IAM access and account limits for S3 buckets, ECR repositories, EC2 On-Demand vCPU, and any lab-specific restrictions. The tool also checks RDS instance, storage and parameter-group quotas and engine availability, EB application/environment/version limits, EC2 elastic IP availability and instance offering, and subnet free IPs. A quota result cannot reserve capacity. |
| Network groups and rules | Use the two trial-owned groups and exact ingress described below. Verify egress, routes, subnet choice and reachability for EB, RDS, ECR, S3 and SSM. The tool compares the approved ingress with the observed rules. |
| Roles and inputs | Check existing `LabRole`, `LabInstanceProfile`, RDS and Auto Scaling service-linked roles and operator ability to pass the existing roles. Confirm applicable lab controls prevent IAM creation and collect the no-IAM audit separately. A caller's ability to read a role does not prove the service can use it. Confirm the owner has installed only approved destination SQL and SSM inputs and that SSM and S3 access via the instance profile is allowed. |
| Billing and storage | Verify available credit, CE posting lag, current rates, paid requests/transfer, local disk and throughput estimate, other builds or lab users, and retention through teardown. Confirm the operator's exact data-read/write and cleanup scope. Stop for unavailable CE, an unreliable upper bound, over USD 10 cumulative trial cost, or less than USD 15 projected remaining credit. |

### Order network preparation before foundation

1. Confirm the target account/VPC, trial label, approved SQL client and
   routes with the network owner. Prefer `db_public: false` and an
   authorized reachable client, such as an approved host inside the VPC or
   an approved private access path. Resolve the client source IPv4 that
   RDS will see after any routing or forwarding. HOLD before foundation if
   there is no approved bootstrap route.
2. Inventory regional EIP quota and existing allocations with the owner
   before creating RDS. Public RDS consumes the lab's EIP capacity.
   Account for any public RDS addresses when assessing headroom, including
   service-managed use that the visible EC2 inventory might miss. Plan
   for one free EIP for the new SingleInstance EB environment **after**
   RDS provisioning, then recheck just before the server phase. If a
   public RDS bootstrap exception is authorized, agree the exposure-revert
   owner and timing first. When private addressing is required to free
   capacity, the owner must confirm the DB is private and available, the
   planned ingress is in effect, and the EIP headroom has returned before
   server apply. Use the observed network and `db_public` values in the
   subsequent configs. HOLD if capacity or the approved route is missing.
3. Use two distinct groups in the target VPC owned by the trial account.
   Select existing groups tagged `P26Trial=<trial>` only after checking
   that this trial owns them and uses their exact rules. If they do not
   exist, the network owner creates and tags them, or explicitly authorizes
   a named actor to do so, before foundation. Record the two IDs privately
   in `db_security_group` and `eb_security_group`. Restrict EB TCP 80 to
   the approved `eb_http_hosts` IPv4 /32s. Restrict DB TCP 3306 to the EB
   group peer and the approved SQL client IPv4 /32s only. Preserve shared
   or existing rules outside this scope.
4. Set `db_client_hosts` to the exact SQL client /32 list on the DB group,
   even when `db_public` is false. Use `[]` when the EB peer is the only
   admitted source. Separately have the SQL owner configure the MySQL
   `user@host` accounts and grants for the actual authorized client
   paths, including the EB runtime user. Security group ingress and SQL
   grants are independent. If the approved client, address, or rules
   change between phases, have the owners confirm all three still agree
   and update the private configs before proceeding.

The underlying network and SSM preparations belong to the approved owner.
The tool creates no IAM resource, load balancer, VPC, SQL grant, login, or SSM
value. It leaves existing ingress and shared resources untouched. The
private config must not stand in for actual authorization. Missing permission
or uncertainty about who owns an existing resource stops the procedure.

## Create a replacement lab

The trial name derives RDS instance `p26-<trial>-db`, a `mysql8.4` parameter
group `p26-<trial>-mysql84`, EB application/environment `p26-<trial>-mlflow`,
ECR repository `p26-<trial>/mlflow-server`, and two
`p26-<account>-<trial>-{artifacts,bundle}` buckets. Reuse is limited to matching
resources tagged `P26Trial=<trial>`. Reserve an available EB CNAME separately.

1. Use `create-lab --phase foundation` with the approved private RDS bootstrap
   input and existing network. The foundation makes the parameter group
   first, then encrypted Single-AZ MySQL 8.4.11 on db.t3.micro with 20 GiB
   gp3, baseline 3,000 IOPS/125 MiB/s, AWS-managed encryption, deletion protection
   and a one-day retention period. Certificate-verified TLS remains required. An
   available instance needs its parameter group in sync and its security
   group active. Allow at least 15 minutes, plus headroom for slower
   provisioning. It verifies the two account-owned private SSE-S3 buckets,
   with public access blocked and BucketOwnerEnforced ownership, and an
   immutable, AES256, scan-on-push ECR repository. A reused resource must
   match the expected settings, owner and trial marker.
2. The permitted account owners bootstrap the established schemas and
   service access separately. The fraud data path requires the exact four
   `fraud` tables from [the database runbook](../docs/runbook.md#rds-database)
   and separately authorized source and destination users. For MLflow,
   the owner can attest that the current runtime inputs were provisioned
   through an authorized process. For a scoped trial, the owner must confirm
   the four names are unused before provisioning and use destination-specific
   values. Append `backend-store-uri`, `auth-database-uri`, `auth-admin-password`
   and `flask-secret-key` to the selected prefix. The tool checks the exact
   names and SecureString types via SSM metadata and passes their ARNs to EB
   for the same four [runtime settings](README.md#settings). The owner checks
   value correctness and existing instance-profile access, including any
   required decryption permission, before setting `runtime_inputs_verified`.
   Stop if provisioning or access would cross the approved trial boundary.
   Complete the [SQL and SSM handoff](#sql-and-ssm-owner-handoff) before
   server apply.
3. Use `create-lab --phase image`. The Dockerfile base digest, dependencies
   with hashes, CA checksum and startup script are pinned. The tool checks
   source input hashes, creates a minimal Docker context, and publishes
   the immutable `3.18.0-r1` tag into the trial repository with a Linux AMD64
   manifest. Save the returned destination digest in private `image_digest`
   and verify it before use. A rebuild from pinned inputs need not match
   the source account's byte-level image digest. An existing tag requires an
   independently witnessed matching destination digest. The trial path has
   no registry-to-registry copy. Allow about 10–20 minutes for a cold build
   as a planning assumption and at least `2 GiB + 4 × image_max_bytes`
   available on the checked Docker filesystem. Cap upload bandwidth, build
   time and registry/storage budget in the authorization. The size of a new
   image is confirmed after its first push.
4. Use `create-lab --phase server`. This applies the reviewed input checks
   and all seven runtime options at launch. It assembles a deterministic
   ZIP with the reviewed health and nginx settings, generating Dockerrun
   with the destination ECR registry and pinned digest. The checksum-bound
   application version uses the separate bundle bucket. EB runs the pinned
   Docker AL2023 platform, SingleInstance t3.small x86-64, 8 GiB gp3, with
   Spot, managed actions and the public default group disabled. Allow at
   least 15 minutes and extra cleanup time. If the exact pinned RDS engine,
   Docker platform, or roles are unavailable, stop. An approved replacement
   version needs a separate compatibility decision.
5. Use `verify` for the stack. It checks the exact version and bundle checksum,
   existing roles, the four SSM names, RDS readiness and groups, expected EB
   options, actual instance and volume, absence of an LB and unauthenticated
   `/health` status. It does not send credentials or follow redirects.
   Complete the [Quickstart](#quickstart) for authenticated verification.
   Record the non-author result, IAM audit, stopped-window bill and cleanup
   evidence separately. HTTP 200 alone does not prove authenticated access
   or persistence.

Enable apply for an individual phase only after its dry-run and scope review.

```bash
./mlflow-server/lab-recreate create-lab --phase image --config "$IMAGE_CONFIG" --dry-run
./mlflow-server/lab-recreate create-lab --phase server --config "$SERVER_CONFIG" --offline
./mlflow-server/lab-recreate create-lab --phase server --config "$SERVER_CONFIG" --dry-run
./mlflow-server/lab-recreate verify --config "$VERIFY_STACK_CONFIG"
```

### SQL and SSM owner handoff

The coordinator names the authorized SQL/SSM owner and agrees how the
operator will receive the handoff before foundation. The owner confirms
the exact trial account and DB instance is available on port 3306,
then connects from the approved route using its verified RDS DNS name
in `P26_DB_HOST`. Run from the repository root. The bare `--password`
option prompts without putting the credential on argv. Use an unrecorded
terminal with tracing disabled.

Oracle MySQL client:

```bash
MYSQL_HISTFILE=/dev/null mysql --no-defaults --protocol=TCP \
  --host="${P26_DB_HOST:?Set the verified RDS DNS name}" --port=3306 \
  --ssl-mode=VERIFY_IDENTITY \
  --ssl-ca="$PWD/src/mlops_fraud_prediction_ca1/rds-global-bundle.pem" \
  --user="${P26_DB_USER:?Set the approved SQL user}" --password
```

MariaDB client (also check `mysql --version` if the installed `mysql` is
a MariaDB alias):

```bash
MYSQL_HISTFILE=/dev/null mariadb --no-defaults --protocol=TCP \
  --host="${P26_DB_HOST:?Set the verified RDS DNS name}" --port=3306 \
  --ssl --ssl-verify-server-cert \
  --ssl-ca="$PWD/src/mlops_fraud_prediction_ca1/rds-global-bundle.pem" \
  --user="${P26_DB_USER:?Set the approved SQL user}" --password
```

Follow [the MLflow schema and grant instructions](DEPLOY.md#set-up-rds-and-s3-once)
with the exact scoped SQL principals agreed in preflight. Privately
check `CURRENT_USER()`, a nonempty `Ssl_cipher` from
`SHOW SESSION STATUS LIKE 'Ssl_cipher'`, and successful schema selection
with the approved runtime principal from its allowed route. HOLD if
hostname verification, reachability, or the grant check fails.

The authorized SSM owner checks that the four scoped names are unused.
Store each value in its own approved mode `0600` private file as raw text,
with no trailing newline. A newline in `auth-admin-password`
would become part of the SecureString and would not match the Quickstart
password. For an interactive entry, use an approved unrecorded terminal.
This example creates the file without showing the value or overwriting
anything. For headless preparation, have the owner supply that same
raw file over the approved private channel.

```bash
(
  set +x
  set -eC
  umask 077
  IFS= read -r -s -p 'Approved trial admin password: ' p26_password
  printf '\n' >&2
  test -n "$p26_password"
  printf '%s' "$p26_password" > "${P26_PRIVATE_DIR:?Set the private trial directory}/auth-admin-password"
)
```

Use the equivalent private file for each other approved value. For example,
after confirming the exact account, region and prefix and receiving
authorization to create this parameter, the owner can pass the file
reference to AWS CLI v2. Set `P26_PROFILE` and `P26_RUNTIME_PREFIX` from
the verified trial config. Keep AWS CLI history and debug logging off.
The default create-only behavior stops on a name collision.

```bash
aws --profile "${P26_PROFILE:?Set the approved AWS profile}" --region us-east-1 \
  ssm put-parameter --name "${P26_RUNTIME_PREFIX:?Set the exact prefix including the final slash}auth-admin-password" \
  --type SecureString --value "file://${P26_PRIVATE_DIR:?Set the private trial directory}/auth-admin-password"
```

Check the four names and types and confirm value correctness and authorized
runtime access privately before attesting `runtime_inputs_verified`. The
lab tool checks SecureString metadata and the binding, so the owner's
access check is a separate responsibility.

## Verified trial on 2026-11-28

The retained #26 trial started from a clean checkout of
`45569024565300c6d6c5c1962f3bbff9aac867bd` (tree
`afde2388a6cd5ae47fc636ba61bc96a63d2ec360`). The operator ran the
`lab-recreate` foundation, image and server phases with a fresh trial namespace.
Network preparation and destination SQL and SSM bootstrap were separately
authorized. The first operator route to private RDS timed out. The approved
trial retry temporarily enabled public RDS addressing with a scoped host /32
rule and passed authenticated TLS SQL
before submitting Elastic Beanstalk at 03:37Z. Confirm the authorized route
for each new trial and follow its exposure-revert handoff.

The deployed configuration used the existing `LabRole` as the EB service
role and `LabInstanceProfile` for the EC2 instance. Preflight verified both
and the existing RDS and Auto Scaling service-linked roles. Stack verification
checked the deployed role/profile settings and all seven runtime options,
including the four scoped SecureString bindings. Successful startup after the
scoped refresh demonstrated that the configured instance could load its runtime
inputs. The runner provisions no IAM resources. The separate no-IAM audit
listed in [Exact preflight](#exact-preflight) is outside this record.

- Elastic Beanstalk reached Ready/Green, but the initial application health
  request returned HTTP 502. The operator corrected the two trial SQL URI
  SecureStrings and restarted the application server to refresh them.
- The pinned stack check passed after the refresh. It verified Ready/Green,
  SingleInstance with one running x86-64 `t3.small`, 8 GiB gp3, no load
  balancer, and unauthenticated `/health` returning HTTP 200.
- At 03:44:53Z the encrypted Session Manager route returned HTTP 200 for health
  and HTTP 401 for anonymous discovery. Authenticated experiment discovery
  returned one experiment and authenticated registry search succeeded with
  zero registered models. These were discovery and registry read checks.
  The separate teammate quickstart is still pending.

The raw-log SHA-256 is
`9bda33bbc55513273c1e3c674812445b5e155b9e4a6b89a522234298a967d6a7`.
From the authorized private evidence directory, reproduce it in this order.

```bash
cat public-host-{route,bootstrap,eb,discovery}.log | sha256sum
```

Steward holds the sanitized foundation trace in
`evidence/agile-mlops/p26-fresh-start-1128/phase-trace.log` (lines 1–37,
SHA-256 `d521f4492dc68bb23f57480812af7e3a6a9e3b0826818d6cc39080f8f0170425`)
and the bootstrap evidence reference in
`evidence/agile-mlops/p26-server-public-bootstrap/receipt-reference.md`.

## Retained deployment and cost audit on 2026-11-28

The read-only audit used the retained trial in `us-east-1`. The auditor
reproduced the deployed Elastic Beanstalk version label from the pinned
bundle inputs and image digest and matched it to the live environment.
At 13:28Z EB was Ready/Yellow and still used SingleInstance with the
supplied `LabRole` and `LabInstanceProfile`. EB and Auto Scaling identified
a replacement instance in the existing deployment.

The retained RDS instance was available and private. The temporary host
ingress rule from the bootstrap was absent. The approved EB peer still
allowed TCP/3306 to RDS. The audit confirmed the exposure revert and found
no further exposure remediation needed.

At 15:43Z the coordinator's scoped readback found both the retained trial
and the separate AC3a witness trial Ready/Green. Each had its own available,
private RDS instance and an Online Systems Manager instance. Both returned
HTTP 200 for anonymous health and HTTP 401 for anonymous API discovery.
The coordinator holds the exact bindings and route checks in the private
Steward directory `evidence/agile-mlops/manager-1542/`.

CloudTrail recorded successful SSM `GetParameter` with decryption and KMS
`Decrypt` activity for the retained parameter scope in EB instance sessions
using `LabRole`, including the current replacement. The supplied role and
profile and the observed RDS and Auto Scaling service-linked roles predated
deployment. The 02:30–03:50Z CloudTrail resource-name and IAM-source searches,
including a filter tied to the deployment actor, returned no matching IAM
mutations. That result covers the queried event history and filters. AC1
still needs independent evidence covering IAM creation across the deployed
scope, including service-mediated calls.

The audit observed three UTC EC2 stop/replacement transition bounds.

| Stop request | Replacement-related event |
| --- | --- |
| 04:12:54Z | 04:14:34Z, Auto Scaling terminated the prior instance |
| 08:32:16Z | 08:34:04Z, Auto Scaling terminated the prior instance |
| 13:12:05Z | 13:13:46Z, Auto Scaling terminated the prior instance and began launching the replacement |

The stop requests recorded instances entering `stopping`. The later
termination or launch events show Auto Scaling replacing instances.
These events give neither the full stack's stop and resume times nor the
billing state of RDS, storage, and other retained resources. The audit
confirmed no stopped-stack interval.

At 15:43Z Cost Explorer showed **USD 17.119804569** in estimated
month-to-date account unblended cost against the inclusive USD 35 cap.
During the earlier audit, hourly and resource-level requests returned `AccessDeniedException` with payer opt-in
messages. An account total includes other usage and lacks the time and
resource attribution needed for AC2. Actual stopped-window cost remains
unmeasured. The coordinator needs authoritative AWS or lab UTC records of
the exact stack entering and leaving the stopped state, plus posted usage
and cost line items from an authorized payer observer. The line items need
resource identifiers, usage times, currency and coverage for those bounds,
including retained resources that can accrue charges during the stop.

In Steward, see `results/agile-mlops/p26-retained-audit-fresh-1325.json`
and the scoped logs in `evidence/agile-mlops/p26-retained-audit-fresh-1325/`.
From that evidence directory, reproduce the aggregate SHA-256
`621a7528c5b4137fb8abb8899b1b7bbff3809e7f9c5906aec43ae569ba794b87`.

```bash
export LC_ALL=C
sha256sum -- *.log | sha256sum
```

## AC3 fresh-operator witness on 2026-11-28

SpiralSky accepted a fresh, context-free agent session as a practical,
assisted non-author witness for AC3 on the separate AC3a trial. The operator
used the clean pinned checkout, its runbook and ordinary trial inputs. The
coordinator supplied the designated SQL and SSM preparation. The witnessed
source remains pinned at commit
`1c22d2c0abb635081daaae2c7b70d0bb9e3cde9b`, tree
`0b72f92b01f15645fe694cdecd0ad111d412c625`.

- On 2026-11-28 (UTC), the operator applied foundation, image and server
  phases from 05:24:15Z to 05:48:09Z. Stack verification returned
  Ready/Green and unauthenticated `/health` HTTP 200.
- At 05:50:45Z the Quickstart verified the synthetic metric and downloaded
  the expected artifact for run `009a5efaee064827a041aa6812aa0853`.
  The report also records one object in the trial artifact bucket.
- The SHA-256 of the private raw `AC3-REPORT.md` is
  `4e4b4495bc7e0f93191d696e7d860456fe48f966bf29d92ff835e786776a0b36`.

The witness applies to the pinned source. The follow-up instructions have
local checks only. The report shows runtime startup and artifact access.
A direct decrypted read under `LabInstanceProfile` was outside the
witness. The literal unaided-deployment criterion remains open. SpiralSky
owns obtaining the marker's disposition or arranging an unaided witness.

| Open criterion | Remaining evidence and owner |
| --- | --- |
| AC1 | SpiralSky arranges independent evidence covering IAM creation across the deployed scope, including service-mediated calls. |
| AC2 | The coordinator obtains authoritative stopped-state bounds and payer-posted usage and cost for the retained SingleInstance trial. See the [retained audit](#retained-deployment-and-cost-audit-on-2026-11-28). |
| AC3 | SpiralSky obtains the marker's disposition on the assisted witness or arranges an unaided deployment witness. |
| AC4 | Glowing6635 independently follows the [teammate Quickstart handoff](#teammate-quickstart-handoff) and supplies a distinct private run and artifact receipt. |

See [decision D26-04](../docs/decision-log.md#d26-04--bind-the-fresh-operator-witness-and-fix-the-runbook).

## Teammate Quickstart handoff

Glowing6635 is authorized to follow the [Quickstart](#quickstart) on the
separate AC3a witness trial, using only the `ac3a-synthetic` experiment and
the coordinator's approved tracking URI, secure transport and scoped login.
Confirm the AC3a target and the active access window with the coordinator
before starting. The current overnight authority expires at 00:00Z on
2026-11-29. The retained deployment is reserved for its own evidence.

Use a clean checkout of this runbook's revision with the locked environment.
After the check, return a private receipt to the coordinator with the
checkout pin, UTC time, trial and experiment binding, new run ID, metric
verification, downloaded artifact verification and any assistance received.
The earlier operator run remains AC3 evidence. The teammate run and its
distinct receipt are pending.

Preserve both trial stacks and their evidence while the external teardown
dependencies are open. The coordinator owns retention, spend monitoring
under the inclusive USD 35 cap, and cleanup authorization. The Quickstart
handoff covers the synthetic check on AC3a.

## Quickstart

Run this authenticated check from the repository root after stack
verification. For a deployment trial, first [create the replacement lab](#create-a-replacement-lab).
For the teammate handoff, use the existing AC3a trial. Use the approved transport, exact trial tracking URI, and
an existing trial login authorized to create runs and read artifacts in the
approved synthetic experiment. Confirm the target and secure transport with
the owner before entering credentials. Stop if either is unavailable. Use a
trial OS account whose username can be logged as synthetic operator metadata
by `log-test-run`. This check sends one synthetic run and a small text artifact.

1. Confirm the stack verification passed, including the recorded deployed
   version, instance type, and unauthenticated `/health` response. Set up
   the locked environment with `uv sync --locked` before the approved window.
   Use a dedicated Bash session and the existing canonical `P26_PRIVATE_DIR`.
   The tracking URI must contain no credentials. Keep the trial inputs and
   command output private. Use one of the following input methods.

   For an unrecorded terminal, enter the values interactively.

   ```bash
   set +x
   set -e
   IFS= read -r -p 'Approved trial tracking URI: ' MLFLOW_TRACKING_URI
   IFS= read -r -p 'Approved synthetic experiment: ' MLFLOW_EXPERIMENT_NAME
   IFS= read -r -p 'Trial login: ' MLFLOW_TRACKING_USERNAME
   IFS= read -r -s -p 'Trial password: ' MLFLOW_TRACKING_PASSWORD
   printf '\n'
   ```

   For headless use, have the owner supply four operator-owned mode `0600`
   files in `P26_PRIVATE_DIR`, one raw nonempty value per file with no line
   terminator. Use the same approved password value that the owner installed
   if it is the initial admin login. Run these lines in Bash with no TTY.

   ```bash
   set +x
   set -e
   p26_read_value() {
     local value
     test -f "$1" || return 1
     if IFS= read -r -d '' value < "$1"; then
       return 1
     fi
     case "$value" in
       ''|*$'\r'*|*$'\n'*) return 1 ;;
     esac
     printf '%s' "$value"
   }
   : "${P26_PRIVATE_DIR:?Set the canonical private trial directory}"
   MLFLOW_TRACKING_URI=$(p26_read_value "$P26_PRIVATE_DIR/tracking-uri")
   MLFLOW_EXPERIMENT_NAME=$(p26_read_value "$P26_PRIVATE_DIR/experiment-name")
   MLFLOW_TRACKING_USERNAME=$(p26_read_value "$P26_PRIVATE_DIR/trial-login")
   MLFLOW_TRACKING_PASSWORD=$(p26_read_value "$P26_PRIVATE_DIR/trial-password")
   ```

2. In the same shell, run the synthetic client. Keep the output in a new
   private log. Take the run ID after `/runs/` in the printed `View run`
   line, with any query string removed. The command below accepts a single
   matching run ID and stops if it is missing or ambiguous. Inspect the
   private output on failure before any retry.

   ```bash
   : "${P26_PRIVATE_DIR:?Set the canonical private trial directory}"
   export MLFLOW_TRACKING_URI MLFLOW_EXPERIMENT_NAME
   export MLFLOW_TRACKING_USERNAME MLFLOW_TRACKING_PASSWORD
   export PYTHON_DOTENV_DISABLED=1
   P26_QUICKSTART_LOG=$(mktemp "$P26_PRIVATE_DIR/quickstart.XXXXXXXX")
   uv run --locked --no-env-file log-test-run > "$P26_QUICKSTART_LOG" 2>&1
   P26_RUN_ID=$(sed -nE 's@^.*View run .*/runs/([0-9a-f]{32})([?].*)?$@\1@p' "$P26_QUICKSTART_LOG")
   [[ "$P26_RUN_ID" =~ ^[0-9a-f]{32}$ ]] || exit 1
   ```

3. In the same shell, retrieve the artifact by run ID through MLflow.
   This command requires the run to belong to the selected experiment,
   checks `connection_ok`, and compares the downloaded file with the
   exact synthetic contents. Keep the private trial directory as the
   destination for the temporary download. Keep client output private.

   ```bash
   export P26_PRIVATE_DIR P26_RUN_ID
   uv run --locked --no-env-file python - >> "$P26_QUICKSTART_LOG" 2>&1 <<'PY'
   import os
   from pathlib import Path
   from tempfile import TemporaryDirectory

   import mlflow
   from mlflow import MlflowClient

   run_id = os.environ["P26_RUN_ID"]
   client = MlflowClient(tracking_uri=os.environ["MLFLOW_TRACKING_URI"])
   run = client.get_run(run_id)
   experiment = client.get_experiment(run.info.experiment_id)
   if (
       experiment.name != os.environ["MLFLOW_EXPERIMENT_NAME"]
       or run.info.status != "FINISHED"
       or run.data.metrics.get("connection_ok") != 1
   ):
       raise SystemExit("Synthetic run verification failed")
   with TemporaryDirectory(dir=os.environ["P26_PRIVATE_DIR"]) as directory:
       downloaded = mlflow.artifacts.download_artifacts(
           run_id=run_id,
           artifact_path="connection_test.txt",
           dst_path=directory,
           tracking_uri=os.environ["MLFLOW_TRACKING_URI"],
       )
       if Path(downloaded).read_bytes() != b"This artifact came from an engineer's laptop.":
           raise SystemExit("Synthetic artifact contents did not match")
   print("Synthetic metric and artifact verified")
   PY
   ```

4. Record the successful run ID, UTC time, metric and artifact retrieval
   result privately with the independent operator's evidence. On failure,
   record the failing step and stop. Exit the dedicated Bash session to
   clear the trial variables. Stack health and a synthetic run leave stopped-window cost,
   IAM audit, and trial cleanup as separate observations.

### Source handoff

Start with the clean checkout supplied for the trial. From the repository
root, run `git status --porcelain`, `git rev-parse HEAD` and
`git rev-parse 'HEAD^{tree}'`. Require empty status and compare both hashes
with the coordinator's approved source receipt before continuing. Record
the verified hashes with the trial evidence. This runbook and its local links
contain the instructions for the trial.

The synthetic verification follows the separately reviewed #26 Quickstart
at `27ca5139622f7354f98a6ee859cf6d5d3fe53043`. For this trial, the
[recreation phases](#create-a-replacement-lab) supply the deployment using
existing roles and approved security groups. Confirm account-specific
inputs against the trial account before any live action.

## Back up approved data

Use a future, separately authorized operation against the explicit source
account and artifact bucket after confirming the runtime actually used it.
Stop all writers to the selected fraud tables and artifact prefix and any
DDL or MLflow changes that would break a coherent checkpoint. Confirm
permission to copy both scopes and the absence of auth material in the
approved artifacts. The data commands only accept the existing fraud
scope and four InnoDB base tables. An approved `*_reader` user needs access
to read the four tables and their metadata. An owner must confirm no
triggers or DDL changes, including triggers hidden from this reader.
The tool always connects with certificate and hostname verification.

```bash
uv run --locked --no-env-file bash mlflow-server/lab-recreate backup-data --config "$SOURCE_CONFIG" --dry-run
```

Before any copy, the tool lists the exact S3 prefix, measures object count
and bytes, reads approximate InnoDB lengths, checks the approved upper
bounds and private disk, and estimates duration and cost. Use a new private
`backup_name`. It dumps data only from a single repeatable-read, read-only
InnoDB snapshot in primary-key order. No executable SQL, grants or schema
DDL are exported. S3 sync uses owner-checked and ETag-conditional reads
under exactly `artifacts/`, and verifies the same inventory before and
after. Rows and artifacts get byte counts and SHA-256. The manifest is
published last and its digest is printed. Record that digest independently
with the source-account alias, date, scope, approvals and diligence record.
The database snapshot and S3 listing do not share a transaction. Keep
writers quiesced throughout the export.

The artifact scope covers ordinary STANDARD objects of at most 5 GB each.
Directory markers, archived or unsupported keys, traversal and file/dir
or case collisions stop the run. It preserves payload bytes and relative
object keys. The manifest records omissions, including S3 metadata,
versions, ACLs and tags. An incomplete directory has no final manifest and
cannot be used for restore. Keep it private until the owner approves
disposal. Never commit the archive or use the manifest as a public report.

## Restore data and verify

Separate approval is required for the exact destination trial and the
independently received source manifest SHA-256. Use the authorized fraud
`*_writer` with existing SELECT/INSERT and metadata access on only the
required scope. Confirm empty tables initialized to the same DDL, no
triggers, and an empty verified trial artifact prefix. Keep all writers
and the application stopped on that destination scope until recovery is
verified. The read-only default checks the manifest, source/destination
accounts, per-file hashes, SQL schema, empty target and recent RDS free
storage. Import requires at least `3 × dump bytes + 1 GiB` of RDS headroom
for data, undo and logs.

```bash
uv run --locked --no-env-file bash mlflow-server/lab-recreate restore-data --config "$RESTORE_CONFIG" --offline
uv run --locked --no-env-file bash mlflow-server/lab-recreate restore-data --config "$RESTORE_CONFIG" --dry-run
uv run --locked --no-env-file bash mlflow-server/lab-recreate verify --data --config "$VERIFY_DATA_CONFIG"
```

On apply, S3 uses conditional create-only uploads with server-validated
SHA-256, and reads back the checksum and expected inventory. The DB side
locks empty tables and imports bound values in one InnoDB transaction.
It re-reads and hashes all restored rows before committing. There is no
cross-service transaction. After an interrupted restore, matching S3
objects may remain while SQL rolled back, or a lost DB response can leave
commit state uncertain. Do not rerun over populated data. With the
coordinator, compare exact partial object checksums and DB state, then use
approved targeted cleanup or a fresh trial. Run `verify --data` while
quiesced for an independent read of the restored bytes and counts.

### Account-specific references and recovery gaps

The deployment derives the new ECR registry, account-owned buckets and SSM
ARNs. The owner binds the private destination backend URI to the new RDS
endpoint and assigns the approved CNAME in clients' tracking configuration
when authorized. Restoring artifact objects alone does not make old
experiments, registered model versions, aliases or stored artifact URIs
appear in the new registry. Old references to source S3 locations or the
source server will remain source-specific. The tool does not copy tracking
metadata or rewrite URI values inside any artifact. The owner must decide
whether to keep the old account reachable, recreate runs/models from
approved inputs and map/re-register the approved artifacts through a
supported application workflow, or arrange a separate approved migration
with integrity evidence. Do not reuse an old account registry reference
as if it named the rebuilt image.

The source inventory found automated snapshots encrypted with the default
AWS-managed RDS key. They do not supply a shareable restore point for
another account. There is no current approved recovery for MLflow auth
tables, grants, passwords, SSM values, tracking or model-registry metadata.
The tooling does not enumerate accounts or logins, touch login tables,
query them from any principal, or use the MLflow server DB login. A
foreign or modified fraud schema, missing approved users, unresolved
artifact ownership, or unavailable source is a stop. [AWS snapshot sharing
restrictions](https://docs.aws.amazon.com/AmazonRDS/latest/UserGuide/USER_ShareSnapshot.html)

## Estimates and stop conditions

Use a per-phase work cutoff and preserve an observation and cleanup
reserve. Record `approval.credential_expiry_utc` from the independently
verified credential issuance or lab session information. A credentials
file may omit the expiry. Obtain it from the authorized issuer if it is
absent. Record `approval.lab_end_utc` separately. The work cutoff plus
observation and cleanup minutes must fit before the earlier expiry.

Use these USD bounds from the approval record:

- Before live billing reads, require `available_credit_usd - unposted_and_concurrent_usd - trial_remaining_bound_usd >= 15`.
- At live apply, first use the lower of `available_credit_usd` and
  `credit_limit_usd - max(0, posted_ce_usd)`. Subtract the same unposted
  allowance and remaining trial bound and require at least USD 15. For
  example, a USD 50 limit, USD 31 posted, USD 21 reported available, USD 1
  unposted allowance and USD 3 remaining trial bound leave USD 15.
- Separately, require the operation estimate to fit within
  `trial_remaining_bound_usd` and `trial_spent_and_committed_usd + trial_remaining_bound_usd <= 10`.

The tool meters conservative planning
rates for compute and two public IPv4 addresses (USD 0.06/hour together),
provisioned DB/EBS and backup (USD 6/730 hours), artifact and image storage
(USD 0.03 and USD 0.10/GiB-month), outbound (USD 0.12/GiB), request/CE charges
and approved local and other costs. For example, four retained hours of
compute and block storage give approximately USD 0.27 before requests,
transfer, local costs and contingency. Count the full resource lifetime
through stopped observation and the end of teardown, including any
retained copies. Unknown variable costs include T3 CPU credits, excess
automated/final snapshots, repeated calls, RDS logs, CloudWatch, ECR
scanning, data paths, and concurrent usage. Bound them in
`extra_cost_bound_usd`. CE costs USD 0.01 per primary-view request and
estimated account-wide charges lag. A fresh query through the prior UTC
day and the owner's measured credit cap plus unposted/concurrent allowance
are both required before mutation. Include current-month carryover and
other previously incurred spend in the owner-stated credit balance. Recheck
rates and the cumulative ledger for each approved phase. These projections
are distinct from observed stopped-window spend. [CE pricing](https://aws.amazon.com/aws-cost-management/aws-cost-explorer/pricing/)
· [RDS regional rate catalog](https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonRDS/current/us-east-1/index.json)
· [EC2 T3 pricing](https://aws.amazon.com/ec2/instance-types/t3/)
· [EBS pricing](https://aws.amazon.com/ebs/pricing/)
· [VPC pricing](https://aws.amazon.com/vpc/pricing/)
· [S3 pricing](https://aws.amazon.com/s3/pricing/)
· [ECR pricing](https://aws.amazon.com/ecr/pricing/)

The redacted source inventory measured approximately 3.54 GiB occupied RDS
filesystem (not dump size), 0.49 GiB STANDARD artifacts, 50 objects and
0.29 GiB compressed image. As a planning example, an 8 GiB logical dump
ceiling plus 1 GiB artifacts needs about 10.8 GiB free on the approved local
filesystem. At 20 MiB/s aggregate effective throughput and 50 objects,
allow about 30 minutes for data transfer and verification, plus the
operator's restore and observation reserves. The command bases backup
disk checks on the full approved dump ceiling, exact artifact bytes and
the larger of 1 GiB or 20% margin. Restore uses the sealed sizes and needs
no second local copy. Include local encryption/storage costs and bandwidth
if billed. Sizing is provisional until measured data and re-read timings
replace these assumptions. Pause before copying if live speed, storage,
credential lifetime, approved scope or projected spend could exceed the
window.

## Partial failure and cleanup

1. Stop new actions on a mismatch, access denial, unexpected service status or
   timeout. Reconfirm account and trial and inventory what actually exists.
   An RDS or EB request can continue after the local timeout. A provisioning
   dependency may need time before it is safe to retry. A completed
   matching resource is reused. A drifted option, unmarked bucket or
   incomplete parameter group needs the owner's disposition before retry.
2. Preserve the exact trial's approved resources during the stopped
   observation. Record actual state, elapsed time, IAM evidence and billing
   as-of time. Price projections and lab expiry cannot supply AC2.
3. After approved observation, terminate only the trial EB environment and
   wait for termination. Confirm its ASG, instance and EIP are gone or
   explicitly accounted for. Delete the named trial application versions
   and bundle objects, then its application. Recheck the account, marker
   and references before deleting the named ECR tag/repository or empty
   trial bundle bucket. Avoid force-deleting a repo or recursive deleting
   an unreviewed S3 scope.
4. Before clearing RDS deletion protection or deleting the trial DB, have
   the owner decide retention, final snapshot and automated backup handling
   and account for their costs. Wait for DB deletion before removing its
   trial parameter group. Preserve pre-existing network groups and rules,
   subnet groups, roles and parameters. After EB termination, the owner
   handles only the four parameters created for this trial according to the
   approved cleanup scope. Delete or retain the private artifact files and
   the exact trial artifact bucket according to the signed retention scope.
   Record any retained assets and charges.
5. Re-list the exact trial resources and any dependent IDs from the
   preflight inventory, checking both absence and retained state. Record
   cleanup evidence and any residual account or data action. Never recycle
   a partial artifact or manifest as evidence of successful migration.

## Offline checks

```bash
bash -n mlflow-server/lab-recreate
uv run --offline --locked --no-env-file pytest -q src/mlops_fraud_prediction_ca1/test_lab_recreate.py
uv run --offline --locked --no-env-file pytest
uv run --offline --locked --no-env-file ruff check
uv run --offline --locked --no-env-file ruff format --check
git diff --check
```

The focused tests use synthetic config and provider mocks to check offline
and read-only paths, wrong account, repeat create, partial failure, budget
gates, artifact integrity and restore behavior. They cannot prove AWS
permissions, MySQL data transfer under the real grants, ECR image availability,
EB creation or billing evidence. A later authorized run is required for each.
