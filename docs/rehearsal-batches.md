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

These flags record what #20 says should happen. They are not verified loader
behaviour: the loader does not exist yet, and this change does not define its
rejection policy. Verify against the real command once #20 lands.

## Rebuilding the rehearsal environment (reset command)

`uv run rehearsal-reset --data-dir data` drops the rehearsal database's four
tables, recreates them (`rds_schema.create_tables`), loads the three base CSV files
with the #20 loader and deletes the runs in the rehearsal MLflow experiment.

- **Selection:** `.env` (#2 settings) points `DB_NAME` and `MLFLOW_EXPERIMENT_NAME`
  at the rehearsal resources. Each engineer's read-only user is created on that
  database with `uv run rds-schema reader ...` (see `docs/runbook.md`).
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
tables, optional reader users (`READERS="name1 name2"`), runs the reset, creates a private
bucket and prints presigned links. Its exit trap (and `scripts/rehearsal-live.sh cleanup`)
removes the security-group rule, deletes the bucket and drops the `rehearsal` schema
(`KEEP_SCHEMA=1` keeps it). `--expires` / `PRESIGN_EXPIRES` sets the link lifetime in
seconds; the default 3600 s covers a 3-minute batch run, but links generated before a
longer demo should use a larger value (the S3 limit with a session token is the token's
remaining life, so generate links in the demo session).
