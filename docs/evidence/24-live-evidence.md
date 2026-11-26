# PBI #24 live evidence (2026-11-26, Learner Lab session)

No credential, token or password appears here. The AWS session used the Learner Lab
profile read from the operator's `~/.config/ca1-aws/` files (role `voclabs`, account
516375498949, us-east-1); that path is the only reference kept.

## Run receipt 1: presigned practice-batch links (AC3 files; AC4 partial) , DONE

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

## Run receipt 2: live grants, schema check, DB reset, and session end , PARTIAL

Run on 2026-11-26 (11:22–12:21 UTC) against `fraud-db.chzc6n97cwib.us-east-1.rds.amazonaws.com`
(`db.t3.micro`, MySQL 8.4), which was `available` at session start. No credential, token or
password appears here; DB admin settings came from `~/.config/ca1-aws/db.env` and MLflow
auth from AWS SSM (`/mlflow/auth-admin-password`, `admin`).

| Time (UTC) | Command / Check | Observation |
|---|---|---|
| 11:22 | `aws ec2 authorize-security-group-ingress` on `sg-075357648cbf564b9` | Added `/32` rule `sgr-01037dd68af1c3798` on tcp/3306 for the runner IP. |
| 11:23 | `MlflowClient().search_experiments()` | Confirmed experiment `('3', 'rehearsal', 'active')` on `sp-buy-mlflow.us-east-1.elasticbeanstalk.com`. |
| 11:23 | `uv run rehearsal-grant richard spiralsky --apply` | Exit 0: `Granted SELECT on rehearsal to 2 user(s).` |
| 11:23 | `SHOW GRANTS` for `richard_reader`, `spiralsky_reader` | Both show `GRANT SELECT ON \`rehearsal\`.* TO \`<name>_reader\`@\`%\`` (and `fraud`), no write/DDL privileges. |
| 11:23 | `uv run rds-schema check` (`DB_NAME=rehearsal`, as `fraud_admin`) | Exit 0: `Connected to rehearsal on fraud-db.chzc6n97cwib.us-east-1.rds.amazonaws.com as fraud_admin.` and `Tables: batches, customers, labels, orders`. |
| 11:24–12:20 | `uv run rehearsal-reset --data-dir .../data` (`DB_NAME=rehearsal`, `MLFLOW_EXPERIMENT_NAME=rehearsal`, `TEAM_DB_NAME=fraud`, `TEAM_MLFLOW_EXPERIMENT_NAME=fraud`) | **DB phase succeeded and committed**: dropped and recreated the 4 tables in `rehearsal`, loaded all 3 base CSVs (`batches`: 1 row `'base'` at `2026-11-26 11:24:36`, `customers`: 901,312, `orders`: 949,889, `labels`: 949,872). **MLflow phase timed out** (`clear_experiment("rehearsal")`, `ConnectTimeoutError` after 55m49s) because the Learner Lab session ended and `sp-buy-mlflow` was terminated (`describe-environments` returned `[]`). |
| 12:20 | `uv run rehearsal-reset` with `DB_NAME=fraud` | Refused immediately before touching anything: `Refused: The database 'fraud' is the team's (TEAM_DB_NAME).` |
| 12:21 | `revoke-security-group-ingress` (`sgr-01037dd68af1c3798`) & `aws rds stop-db-instance` | Denied by `arn:aws:iam::516375498949:policy/voc-cancel-cred` (Learner Lab session ended). |

Remaining items from receipt 2:
- **Per-engineer `rds-schema check`:** `richard_reader` and `spiralsky_reader` passwords are held by each engineer (not in `~/.config/ca1-aws/`), and `rds-schema reader` was not re-run because MySQL users are server-wide and it would reset their passwords on `fraud`.
- **DB-dependent `rehearsal-reset` vs MLflow-only `clear_experiment` ordering:** `rehearsal-reset` rebuilds the RDS `rehearsal` tables before clearing MLflow, so a full `rehearsal-reset` requires `fraud-db` to be `available` and SG ingress open (run it *before* stopping `fraud-db` and revoking the SG rule). By contrast, `clear_experiment("rehearsal")` only calls the MLflow tracking server and does not need RDS or `sg-075357648cbf564b9`, so it can run *after* RDS/SG cleanup whenever `sp-buy-mlflow` is serving.

## Run receipt 3: RDS stop, SG rule revoke, and MLflow EB check (2026-11-26 ~13:01 UTC) , DONE for RDS/SG; MLflow pending

| Time (UTC) | Command / Check | Observation |
|---|---|---|
| 13:01 | `aws rds describe-db-instances --db-instance-identifier fraud-db --region us-east-1` | `DBInstanceStatus` was `available`. |
| 13:01 | `aws rds stop-db-instance --db-instance-identifier fraud-db --region us-east-1` & `describe-db-instances` | `DBInstanceStatus` transitioned to `stopping` and verified `stopping`. |
| 13:01 | `aws ec2 describe-security-group-rules` (`sg-075357648cbf564b9`), `aws ec2 revoke-security-group-ingress` (`sgr-01037dd68af1c3798`), & re-check | Rule `sgr-01037dd68af1c3798` was still present; revoked (`Return: true`) and verified absent from `sg-075357648cbf564b9`. |
| 13:01 | `aws elasticbeanstalk describe-environments --region us-east-1` & `curl` to `http://sp-buy-mlflow.us-east-1.elasticbeanstalk.com/` | Environment `mlflow` reported `Status: Ready`, `Health: Grey` (instance `i-07e26819a94dc93fc` `running`, no load balancers), but HTTP port 80 refused connections (`curl` exit 7, HTTP 000). EB was not recreated; `clear_experiment("rehearsal")` remains pending until the MLflow service is up. |

## Acceptance criteria map for #24

| # | Criterion | Evidence | Status |
|---|---|---|---|
| 1 | Rehearsal DB and MLflow experiment selected via #2 config; each engineer's read-only user can read the database | Selection and refusal: `docs/rehearsal-batches.md`, unit tests. Live: `rehearsal-grant richard spiralsky --apply` succeeded, `SHOW GRANTS` confirmed `SELECT` on `rehearsal.*` for both readers, `rds-schema check` succeeded as `fraud_admin` | **Partial**: grants applied & verified; per-engineer `rds-schema check` as `<name>_reader` still needs each engineer's password |
| 2 | One command rebuilds the rehearsal DB from the three base CSVs and refuses the team DB/experiment | Code and unit tests (`rehearsal_reset`, D24-01). Live: `DB_NAME=fraud` refused immediately; `rehearsal` schema dropped, recreated, and loaded (`901,312` customers, `949,889` orders, `949,872` labels, `1` batch `'base'`) | **Partial**: DB rebuild and team-DB refusal verified live; MLflow `clear_experiment` timed out when Learner Lab session ended |
| 3 | Practice batches in the base format: two synthetic batches plus one small edge batch | Receipt 1: three batches, nine files uploaded with the sizes above; base-format and rejection checks are in the unit tests | Done for generation and serving; loader rejection counts on live DB are **gap** |
| 4 | Presigned links from the same sandbox session, not a public bucket or the repo; credential source recorded | Receipt 1: 200 signed, 403 unsigned, 403 after expiry, no public grants, bucket deleted; credential source stated above. But receipt 1 and receipt 2 are separate sessions, so links were not generated in the session that rehearsed the batches | **Partial / pending**: needs one live rehearsal run where the presigned links are generated in the same sandbox session as the rehearsal |

#24 should stay open until: (AC1) each engineer runs `rds-schema check` as `<name>_reader`; (AC2) `clear_experiment("rehearsal")` completes against a live MLflow server; (AC3) live `load-batch` rejection counts are verified on the live DB; and (AC4) one live rehearsal run generates the presigned links in the same sandbox session as the rehearsal.
