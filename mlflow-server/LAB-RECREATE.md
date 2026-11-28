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
  window. For image publication, use a local Docker engine and buildx that can
  build Linux AMD64 and publish an OCI index with provenance. On an ARM host,
  check AMD64 emulation in advance. Use the explicitly approved local Unix
  socket. Reserve space on the actual Docker engine filesystem.
- Use an explicit AWS CLI profile for the approved account's short-lived
  `voclabs` role. Refresh credentials privately. Keep CLI history disabled
  in the selected and default profiles and use standard AWS service endpoints
  and certificate verification. The tool accepts only `us-east-1` and checks
  the exact account and caller before opening SQL or making changes.
- Create one operator-owned directory outside the repository with mode
  `0700`, on the approved encrypted filesystem. Store JSON configuration
  and password-only files there with mode `0600`. Use absolute, canonical
  paths. Keep the directory out of backups to unapproved storage, shell
  history, Git, and shared logs. AWS and Docker output is captured privately.
  Redacted output includes sizes, state, counts and digests. The manifest and
  archive remain private.
- Confirm exclusive trial scope across machines and coordinate the source and
  destination quiescence windows. Use the same private directory for phases
  on one machine so its apply lock can prevent concurrent local execution.
  The local lock cannot protect an account against another operator or service.

Copy [the example](lab/config.example.json) into the private directory and
replace the example values. For the scoped trial route, set
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

## Exact preflight

Confirm the following with the coordinator for the current lab. The JSON
attestations refer to checks already performed by the approved operator
or observer.

| Check | Operator evidence and command behavior |
| --- | --- |
| Account, trial and window | Bind the profile, account ID, region and unique trial name to the authorization. Record credential expiry separately from lab expiry. Calculate a work cutoff that preserves stopped observation and teardown. Check every resource with the same profile. A wrong account, expired window or disallowed caller stops work. |
| Ownership and limits | Reserve the derived trial names with the account owner before launch. Precheck IAM access and account limits for S3 buckets, ECR repositories, EC2 On-Demand vCPU, and any lab-specific restrictions. The tool also checks RDS instance, storage and parameter-group quotas and engine availability, EB application/environment/version limits, EC2 elastic IP availability and instance offering, and subnet free IPs. A quota result cannot reserve capacity. |
| Network groups and rules | Use distinct existing trial-approved groups in the target VPC. The DB group must already admit TCP 3306 from the EB group and only the explicitly approved client /32s. The EB group must admit TCP 80 from the named approved /32s only. Verify egress, routes, subnet choice and reachability for EB, RDS, ECR, S3 and SSM. The tool rejects broader or different ingress. It does not create a rule, add a client or use the public default EB group. If the exact rules do not exist, stop and take the gap to the network owner. |
| Roles and inputs | Check existing `LabRole`, `LabInstanceProfile`, RDS and Auto Scaling service-linked roles and operator ability to pass the existing roles. Confirm applicable lab controls prevent IAM creation and collect the no-IAM audit separately. A caller's ability to read a role does not prove the service can use it. Confirm the owner has installed only approved destination SQL and SSM inputs and that SSM and S3 access via the instance profile is allowed. |
| Billing and storage | Verify available credit, CE posting lag, current rates, paid requests/transfer, local disk and throughput estimate, other builds or lab users, and retention through teardown. Confirm the operator's exact data-read/write and cleanup scope. Stop for unavailable CE, an unreliable upper bound, over USD 10 cumulative trial cost, or less than USD 15 projected remaining credit. |

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
The trial's actual stopped-sandbox charges remain open for the independently
tracked attributable-window billing measurement. Account-wide Cost Explorer
readings and preflight estimates cannot establish that charge. The AC3
coordinator trial uses separate resources, and AC4 still needs teammate receipts.

## Quickstart

Run this authenticated check from the repository root after
[creating the replacement lab](#create-a-replacement-lab) and passing stack
verification. Use the approved transport, exact trial tracking URI, and
an existing trial login authorized to create runs and read artifacts in the
approved synthetic experiment. Confirm the target and secure transport with
the owner before entering credentials. Stop if either is unavailable. Use a
trial OS account whose username can be logged as synthetic operator metadata
by `log-test-run`. This check sends one synthetic run and a small text artifact.

1. Confirm the stack verification passed, including the recorded deployed
   version, instance type, and unauthenticated `/health` response. Prepare
   the repository's locked environment with `uv sync --locked` before the
   approved window. Start a Bash subshell with tracing disabled, and set
   the tracking values interactively. The URI must contain no credentials.
   Keep the login and password out of Git, command arguments, and the
   evidence record.

   ```bash
   bash
   set +x
   read -r -p 'Approved trial tracking URI: ' MLFLOW_TRACKING_URI
   read -r -p 'Approved synthetic experiment: ' MLFLOW_EXPERIMENT_NAME
   read -r -p 'Trial login: ' MLFLOW_TRACKING_USERNAME
   read -r -s -p 'Trial password: ' MLFLOW_TRACKING_PASSWORD
   printf '\n'
   export MLFLOW_TRACKING_URI MLFLOW_EXPERIMENT_NAME
   export MLFLOW_TRACKING_USERNAME MLFLOW_TRACKING_PASSWORD
   export PYTHON_DOTENV_DISABLED=1
   ```

2. Run the reviewed synthetic client. Stop on failure. Open the resulting
   run in the approved tracking UI and copy the run ID from its run details.

   ```bash
   uv run --locked --no-env-file log-test-run
   ```

3. In the same subshell, retrieve the artifact by run ID through MLflow.
   This command requires the run to belong to the selected experiment,
   checks `connection_ok`, and compares the downloaded file with the
   exact synthetic contents. Keep the private trial directory as the
   destination for the temporary download.

   ```bash
   read -r -p 'Private trial directory: ' P26_PRIVATE_DIR
   read -r -p 'Synthetic run ID: ' P26_RUN_ID
   export P26_PRIVATE_DIR P26_RUN_ID
   uv run --locked --no-env-file python - <<'PY'
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
   record the failing step and stop. Exit the subshell to clear the trial
   variables. Stack health and a synthetic run leave stopped-window cost,
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
reserve. The hard limits in the tool are USD 10 cumulative trial spend plus
commitments and USD 15 residual credit. It meters conservative planning
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
