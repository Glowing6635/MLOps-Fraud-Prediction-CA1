# Handover

This is the handover pack for the release on 30 Nov 2026, 09:00 (SGT). It lets a reviewer outside the team find and check the fraud-prediction system without asking the team. Reviewers read: they get read-only logins, and every run is shown by the team at the stakeholder demo, with the rehearsal records as proof that it works. The release checklist below lists what the pack must answer; each row's answer links to the file, commit, MLflow run or screenshot that proves it. Logins and addresses for reviewers are never in git: they are added only to the copy delivered in the release archive.

## Release checklist

A row whose proof says "pending" isn't done yet, and names the issue that finishes it.

| # | Item | Answer | Proof |
| --- | --- | --- | --- |
| 1 | How to install the code and run the checks | `uv sync`, then `uv run pytest` | [README](../README.md), "Setup" and "Checks" |
| 2 | How to run one batch end to end within 3 minutes | `uv run run-batch` with the batch's three file URLs | [README](../README.md), "Batch command"; [runbook](runbook.md), "Running a batch"; timed runs pending (#27, #29) |
| 3 | How the database is built, and what each table and column holds | Tables from `uv run rds-schema tables`; one read-only login per engineer from `uv run rds-schema reader <name>` | [runbook](runbook.md), "RDS database"; [data dictionary](data-dictionary.md); `src/mlops_fraud_prediction_ca1/schema.sql` |
| 4 | Where MLflow keeps its runs and files, and why | Runs in RDS, files in S3, behind one Elastic Beanstalk server | [MLflow server design](mlflow-design.md); [decision log](decision-log.md), D8-01; research transcript (#8): https://claude.ai/share/02998146-af10-472d-a21f-05e215dc7623 |
| 5 | How to deploy the MLflow server, and which bundle is live | `eb deploy` from `mlflow-server/` | [deploy steps](../mlflow-server/DEPLOY.md); the live bundle in the release archive (see "Release archive") |
| 6 | Which models were trained, and which one is the champion | Pending | Champion and model card pending (#15, #35); model versions pending (#22) |
| 7 | How the model retrains on each new batch | Pending | Per-batch retraining pending (#25) |
| 8 | How the champion scored on the held-back test window | Pending | Final test pending (#23) |
| 9 | Which orders analysts review first each day | `uv run review-queue` | [runbook](runbook.md), "Daily review queue" |
| 10 | How the services come back after the sandbox stops, and when to start on demo day | Pending | Restart drill and demo-day steps pending (#11) |
| 11 | How the whole system did in rehearsal, and what failed | Pending | Rehearsals pending (#29); failure log pending (#31) |
| 12 | Why the system works the way it does | One dated entry per decision, with its owner and reason | [decision log](decision-log.md) |
| 13 | How a reviewer reaches RDS, MLflow and GitHub | In the delivered copy only | "Access for reviewers" below |
| 14 | What each engineer delivered alone | One private repository per engineer | "Individual packs" below |

## Release archive

The archive is one zip file holding:

| Path in the archive | What it is |
| --- | --- |
| `handover.md` | This document, with "Access for reviewers" filled in. Use this copy, not the blank one under `code/docs/` |
| `code/` | The pipeline code at the release tag, `v1.0.0` on `main` |
| `code/docs/runbook.md` | The runbook |
| `mlflow-server-bundle/<version label>.zip` | The exact Elastic Beanstalk bundle the MLflow server runs |

The release tag follows the contributor guide's `v<MAJOR>.<MINOR>.<RELEASE>` format and is made when the release merges into `main`. To build the archive, in Git Bash from an empty folder next to the repository, with the sandbox started:

```bash
git -C ../MLOps-Fraud-Prediction-CA1 archive --format=tar --prefix=code/ v1.0.0 | tar -x
aws elasticbeanstalk describe-environments --application-name mlflow-server --environment-names mlflow --region us-east-1 --query 'Environments[0].VersionLabel' --output text
aws elasticbeanstalk describe-application-versions --application-name mlflow-server --version-labels <version label> --region us-east-1 --query 'ApplicationVersions[0].SourceBundle.[S3Bucket,S3Key]' --output text
aws s3 cp s3://<S3Bucket>/<S3Key> mlflow-server-bundle/<version label>.zip
```

Then copy in the filled-in `handover.md` and zip `handover.md`, `code/` and `mlflow-server-bundle/` together.

## Access for reviewers

The delivered copy fills in this table. Every login only reads.

| Service | What a reviewer gets |
| --- | --- |
| RDS | Host, port, database name and a read-only login |
| MLflow | Server address and a login that can read every experiment |
| GitHub | Read access to each engineer's private repository. This repository is public |

## Checks before the release

Both are done at least 24 hours before the release, that is by 29 Nov 2026, 09:00.

| Check | Done on | From | By | Result |
| --- | --- | --- | --- | --- |
| Every login above works from a laptop outside the team | Pending | | | |
| The #8 research transcript link opens in a browser that isn't signed in | Pending | | | |

## Individual packs

Each engineer owns their own pack, kept in their private repository, apart from this repository.

| Engineer | Holds | Reviewer can open the private repository |
| --- | --- | --- |
| Glowing6635 | Fraud-pattern analyses (#21), the local analysis flow (#28), Prefect on AWS (#32) and the cloud analysis flow (#36) | Pending |
| SpiralSky | SpiralSky's own work on the same four issues | Pending |

## Team check

Every teammate checks the group pack before it is delivered on 29 Nov 2026.

| Teammate | Checked on | Result |
| --- | --- | --- |
| Glowing6635 | Pending | |
| SpiralSky | Pending | |
