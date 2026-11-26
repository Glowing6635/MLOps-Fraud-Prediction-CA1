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

- **Selection:** `.env` (#2 settings) points `DB_NAME` and `MLFLOW_EXPERIMENT_NAME`
  at the rehearsal resources. Each engineer's existing read-only user
  (`<name>_reader`, created once with `uv run rds-schema reader <name>`, see
  `docs/runbook.md`) gets access to this database with a plain
  `GRANT SELECT ON rehearsal.* TO '<name>_reader'@'%'`. Do not rerun
  `rds-schema reader` for this: MySQL users are server-wide, so it would reset that
  user's password on the team database too.
- **Refusal:** before touching anything, both names must start with `rehearsal`, and
  neither may equal `TEAM_DB_NAME` / `TEAM_MLFLOW_EXPERIMENT_NAME` when set. Set those
  two to the team names in the operator's `.env` for a second line of defence.
- **Practice batches:** `uv run rehearsal-reset --presign <bucket>` generates the
  synthetic batches, uploads them as private objects under `rehearsal/`, and prints
  presigned GET URLs (default 1 hour). Signing uses boto3's normal credential chain in
  the sandbox session (environment or instance profile). No credential, bucket policy
  or public ACL is stored in the repository.

### Reader access evidence (AC1, owner request)

`uv run rehearsal-grant <name> [<name> ...]` prints `GRANT SELECT ON \`rehearsal\`.* TO
'<name>_reader'@'%'` for each engineer; `--apply` runs them with the admin login in `.env`.
It is safe to re-run, only grants SELECT, never creates users or changes passwords, and
refuses a database that does not start with `rehearsal` or equals `TEAM_DB_NAME`. The
`<name>_reader` users must already exist.

**Request for the owner (needs the admin and each reader's credentials, so it is not
done in the repository).** Nothing here has been run against AWS.

1. As admin, with `DB_NAME=rehearsal` in `.env` and the schema created:
   `uv run rehearsal-grant richard spiralsky <other engineers> --apply`
2. Each engineer, with `DB_USER=<name>_reader`, their own password and `DB_NAME=rehearsal`:
   `uv run rds-schema check`
3. Expected output per engineer: `Connected to rehearsal on <host> as <name>_reader.`,
   `Tables:` listing the four tables, and a grant line
   `GRANT SELECT ON \`rehearsal\`.* TO \`<name>_reader\`@\`%\``, with no other privilege.
4. Paste the three lines per engineer (no passwords) as a comment on #24 and in the
   evidence table of the PR that resolves it. Do not paste `.env`.

### One-command live run

`scripts/rehearsal-live.sh run` (environment variables listed at the top of the script)
adds this machine's `/32` to the RDS security group, creates the `rehearsal` schema, the
tables, optional reader grants (`READERS="name1 name2"`: `GRANT SELECT` only, the `<name>_reader` users must already exist), runs the reset, creates a private
bucket (tagged `created-by=rehearsal-live`) and prints presigned links with their expiry.
It removes only the security-group rule it added itself, names any cleanup step that failed and exits non-zero in that case. `BUCKET` must start with `rehearsal`. On success the bucket, objects and schema
stay so the links keep working; on failure it deletes only what that run created. After
the rehearsal, `scripts/rehearsal-live.sh cleanup` deletes the bucket only if it carries
the tag (never a bucket someone else made) and drops the `rehearsal` schema only when
`KEEP_SCHEMA=0` is set (default keeps it so readers can check it; `cleanup` also opens and closes its own `/32` rule). It refuses to start if the bucket already exists.
`--expires` / `PRESIGN_EXPIRES` sets the link lifetime in seconds (default 3600, enough for
a 3-minute batch run; use more for a longer demo). Links are valid only while the session
credentials that signed them are, so generate them in the demo session.
