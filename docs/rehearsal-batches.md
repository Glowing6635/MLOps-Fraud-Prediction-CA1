# Synthetic rehearsal batches (PBI #24, practice-batch item)

`mlops_fraud_prediction_ca1.rehearsal` generates fully synthetic practice
batches in the base file format. Values come from seeded random generators, so
no real order, customer or label is read or copied. Nothing is committed: the
CSVs are written on demand to a directory you choose.

```bash
uv run python -m mlops_fraud_prediction_ca1.rehearsal --out output/rehearsal
```

Each batch is a directory whose three files share the suffix that #20 uses as
`batch_id`, with the base-file columns:

| Batch | Files | Purpose |
| --- | --- | --- |
| `rehearsal_a`, `rehearsal_b` | `customers_<id>.csv`, `orders_<id>.csv`, `labels_<id>.csv` | Valid, distinct (disjoint IDs and values), 40 customers and 120 labelled orders each |
| `rehearsal_edge` | same | Small: one unseen payment method on an otherwise valid row, and rows flagged for rejection |

Edge batch cases (identifiers all start with `SYN-EDGE-`):

- **Unseen category:** the first order has payment method
  `synthetic unseen wallet`. Per #19 it must not stop the batch; it goes to
  cleaning (unknown).
- **Label whose order is missing:** `SYN-EDGE-ORPHAN-ORDER-01` and `-02` have
  labels but no orders row. #20: rejected and counted, no abort.
- **Label whose customer is missing:** order `SYN-EDGE-O-NOCUST` is labelled
  to `SYN-EDGE-C-ABSENT`, which has no customers row. #20 names this case too.

These flags record what #20 says should happen. The loader (`uv run load-batch`)
is part of this branch and the live rehearsal loaded these batches with it; the unit
tests do not assert its rejection counts, and this change does not define its policy.

## Rebuilding the rehearsal environment (reset command)

`uv run rehearsal-reset --data-dir data` drops the rehearsal database's four
tables, recreates them (`rds_schema.create_tables`), loads the three base CSV files
with the #20 loader and marks the runs in the rehearsal MLflow experiment as deleted (MLflow `delete_run` only
hides them: their files stay in the artifact bucket until the MLflow server's garbage collection).

- **Selection:** Set `DB_NAME`, `MLFLOW_TRACKING_URI` and
  `MLFLOW_EXPERIMENT_NAME` in the environment or your git-ignored `.env`
  using [the #2 settings template](../.env.example). Use the `READERS` option
  below to give existing read-only users rehearsal access. Keep their existing
  passwords and team access when setting up rehearsal.
- **Refusal:** before touching anything, both names must start with `rehearsal`, and
  neither may equal `TEAM_DB_NAME` / `TEAM_MLFLOW_EXPERIMENT_NAME` when set. Set those
  two to the team names in the operator's `.env` for a second line of defence.
- **Practice batches:** `uv run rehearsal-reset --presign <bucket>` generates the
  synthetic batches, uploads them as private objects under `rehearsal/`, and prints
  presigned GET URLs (default 1 hour). Signing uses boto3's normal credential chain in
  the sandbox session (environment or instance profile). No credential, bucket policy
  or public ACL is stored in the repository.

### One-command live run

`scripts/rehearsal-live.sh run` (environment variables listed at the top of the script)
adds this machine's `/32` to the RDS security group, creates the `rehearsal` schema, the
tables, optional reader grants (`READERS="name1 name2"`: `GRANT SELECT` only, the `<name>_reader` users must already exist), runs the reset, creates a private
bucket (tagged `created-by=rehearsal-live`) and prints presigned links with their expiry.
It removes only the security-group rule it added itself, names any cleanup step that failed and exits non-zero in that case. `BUCKET` must start with `rehearsal`. On success the bucket, objects and schema
stay so the links keep working; on failure it deletes only what that run created. After
the rehearsal, `scripts/rehearsal-live.sh cleanup` deletes the bucket only if it carries
the tag. An absent bucket, an absent tag set, or a different owner tag is a successful
skip. Other tag lookup errors report the bucket and AWS error without deleting any
bucket objects. Standalone cleanup exits 1 if a cleanup step fails. If cleanup fails
after a failed `run`, the script also reports the cleanup failure and preserves the
original exit status. Default cleanup uses only S3. Set `KEEP_SCHEMA=0` to also
drop the `rehearsal` schema, or run
`KEEP_BUCKET=1 KEEP_SCHEMA=0 scripts/rehearsal-live.sh cleanup` for schema-only
cleanup. Schema cleanup requires authorized temporary ingress and revokes the
rule by its returned ID. A denied authorization or failed revocation makes
cleanup fail even if the bucket was removed. Bucket cleanup errors also leave
the requested schema cleanup enabled. The `run` command refuses to start if
the bucket already exists.
`--expires` / `PRESIGN_EXPIRES` sets the link lifetime in seconds (default 3600, enough for
a 3-minute batch run; use more for a longer demo). Links are valid only while the session
credentials that signed them are, so generate them in the demo session.

## Acceptance evidence for #24

### AC1 · Rehearsal configuration and reader access

The 26 November 2026 reset receipt records `DB_NAME=rehearsal` and
`MLFLOW_EXPERIMENT_NAME=rehearsal` through #2. A separate authenticated
MLflow readback at 13:23 UTC identified experiment `rehearsal`, ID `3`,
on the `sp-buy-mlflow` deployment. It was active and had zero active runs
(live receipt 4, `60e3df4`). The 27 November cleanup receipt records a
successful reset under the same database and experiment names, with zero
runs deleted and no remaining rehearsal buckets (`d81a156`).

| Reader | Redacted evidence from 28 November 2026 |
| --- | --- |
| SpiralSky | The 07:34 UTC receipt for draft head `4602406e` records a personal read-only connection to `rehearsal`, visibility of `batches`, `customers`, `labels` and `orders`, and a successful read from `batches`. |
| Glowing6635 | In the receipt relayed by SpiralSky at 15:01 UTC, Glowing6635 reported a successful `uv run rds-schema check` and SELECT using a separate read-only account on `rehearsal`. The same four tables were visible. The reported counts were 1 batch and 949,889 orders. The relay gives no source commit, and the check has not been independently rerun. |

The receipts support the two personal database reads. `rds-schema check`
checks SQL table visibility and permissions. It loads the tracking settings
without contacting MLflow. The earlier reset and readback establish
historical tracking selection. Both reader receipts leave the effective
tracking target unverified. For AC1 review, SpiralSky can supply a dated,
redacted record that ties the effective #2 settings to an authenticated
lookup of the rehearsal experiment name and ID. An existing record from
the authorized rehearsal context can provide this evidence without
repeating the teammate's SQL check.

### AC2–AC4 · Reset, synthetic batches and private delivery

- **AC2.** The 26 November live `uv run rehearsal-reset --data-dir data`
  completed against the rehearsal database and experiment. It rebuilt the four
  tables from three base CSVs and loaded 901,312 customers, 949,889 orders and
  949,872 labels (live receipt 5, `8f09a48`). Draft `4602406e` has
  regression tests for refusal of both team targets before database or
  MLflow operations.
- **AC3.** The 26 November live loader accepted 40 customers, 120 orders and
  120 labels from each of the two regular synthetic batches. The edge batch
  included the unseen payment category and loaded 4 customers, 11 orders and
  10 labels. It rejected one label for a missing customer and two for
  missing orders.
- **AC4.** The 26 November rehearsal generated and consumed all nine CSV
  links in the same sandbox session and removed the temporary bucket afterward.
  The separate 27 November verification (receipt `da6b9ad8`, source
  `29d27a0`) confirmed all four S3 public-access blocks, nine signed GETs
  returning HTTP 200 and all nine unsigned GETs returning HTTP 403. The signed
  links had a five-minute TTL, and cleanup verified bucket removal and
  unchanged ingress. Boto3 used the sandbox's temporary AWS session credentials.
  These are historical results. The live check used the recorded source, and
  the later draft has offline coverage for the revised cleanup logic.
