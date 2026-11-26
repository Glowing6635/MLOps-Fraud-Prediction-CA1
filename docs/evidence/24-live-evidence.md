# PBI #24 live evidence (2026-11-26, Learner Lab session)

No credential, token or password appears here. The AWS session used the Learner Lab
profile read from the operator's `~/.config/ca1-aws/` files (role `voclabs`, account
516375498949, us-east-1); that path is the only reference kept.

## Run receipt 1: presigned practice-batch links (AC3 files, AC4) , DONE

Command (run from this branch; the presign path never opens the database, so the DB and
MLflow variables were placeholders and `DB_NAME`/`MLFLOW_EXPERIMENT_NAME` were `rehearsal`):

    uv run rehearsal-reset --presign rehearsal-516375498949 --expires 300

| Time (UTC) | Observation |
|---|---|
| 09:39:57 | Bucket `rehearsal-516375498949` created, tagged `created-by=rehearsal-live`, all four public-access blocks on (`True True True True`). `head-bucket` had returned 404 first. |
| 09:40:06 | Command exit 0. Printed 9 links: `rehearsal_a`, `rehearsal_b` (synthetic batches) and `rehearsal_edge` (unseen category and rows that must be rejected), each with customers, labels and orders CSV (sizes 2468/3764/7960, 2475/3764/7925, 438/548/863 bytes). |
| 09:40 | Signed GET of three links: HTTP 200. Same URL without the signature query: HTTP 403. Object ACL grantee type is only `CanonicalUser` (the owner); no public grant. |
| 09:40 | Links carry `X-Amz-Expires=300`. |
| 09:45:16 | First link fetched after expiry: HTTP 403, `AccessDenied: Request has expired` (expiry 09:45:09Z). |
| after | Objects removed and bucket deleted; `aws s3 ls` shows no `rehearsal*` bucket. |

Where the signing credentials come from: boto3's default chain in the sandbox session
(the Learner Lab session credentials in the operator's AWS profile). Not stored in the repo.

Link lifetime (request 20261126-2 default): links are valid only while the run is live
and for at most `--expires` seconds; this receipt used a temporary bucket deleted
afterwards, so the links stop working at deletion as well as at expiry.

## Run receipt 2: live reset, reader grants, reader check , NOT DONE

`scripts/rehearsal-live.sh run` was not run. The RDS instance `fraud-db` (db.t3.micro,
20 GB) is in state `stopped`; starting it (about $0.02 per hour) was blocked by the session's
permission classifier, so it was left stopped and nothing was started or changed.
Because the instance is stopped the run also could not (a) create the `rehearsal` schema
and load the base CSVs, (b) `GRANT SELECT ON rehearsal.*`, or (c) run `rds-schema check` as
any `<name>_reader`. The engineers' reader passwords are held by each engineer, so (c) needs
them in any case. The security group `sg-075357648cbf564b9` was not changed.

## Acceptance criteria map for #24

| # | Criterion | Evidence | Status |
|---|---|---|---|
| 1 | Rehearsal DB and MLflow experiment selected via #2 config; each engineer's read-only user can read the database | Selection and refusal: `docs/rehearsal-batches.md`, unit tests. Live grant/read check: receipt 2 | **Gap**: needs RDS started, admin `rehearsal-grant --apply`, and each engineer's `rds-schema check` |
| 2 | One command rebuilds the rehearsal DB from the three base CSVs and refuses the team DB/experiment | Code and unit tests (`rehearsal_reset`, D24-01). Live rebuild: receipt 2 | Code done; **live rebuild gap** (RDS stopped) |
| 3 | Practice batches in the base format: two synthetic batches plus one small edge batch | Receipt 1: three batches, nine files uploaded with the sizes above; base-format and rejection checks are in the unit tests | Done for generation and serving; loader rejection counts on live DB are **gap** |
| 4 | Presigned links from the same sandbox session, not a public bucket or the repo; credential source recorded | Receipt 1: 200 signed, 403 unsigned, 403 after expiry, no public grants, bucket deleted; credential source stated above | **Done** |

#24 should stay open until the criterion 1 live reader check (and, ideally, the live rebuild) are captured.
