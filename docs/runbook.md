# Runbook

How to bring the team's cloud services back after the sandbox stops, and the checks to repeat before the demo. The MLflow server's setup is in `mlflow-server/README.md`.

## Who runs the services

| Item | Record |
| --- | --- |
| Sandbox account hosting RDS, MLflow and S3 | TO RECORD |
| Who can start it | TO RECORD |
| Demo machine | TO RECORD |
| Demo-site network | TO RECORD |
| Backup hotspot | TO RECORD |

If only one person can start the sandbox, they are a single point of failure on demo day. Risk and backup plan: TO RECORD.

Every service uses only the sandbox's own `LabRole` and `LabInstanceProfile`. Nothing creates an IAM user, role or policy.

## Before the sandbox stops

Run `uv run redeploy-check record` once, from any laptop. It saves the model's predictions to `output/redeploy_check.json` on that laptop, so run the drill from the same laptop.

## Restart drill

1. Start the sandbox.
2. Straight away, run `uv run restart-drill`. It waits until RDS answers, then MLflow, then checks that the recorded model loads and reproduces its predictions, and logs a run named `restart-drill` with the times. It prints the time from its own start to each step, and gives up after 30 minutes.
3. Add the times to the table below.

| Date | Machine and network | RDS | MLflow | Model | Total |
| --- | --- | --- | --- | --- | --- |
| | | | | | |

Demo-day start time: TO RECORD, from the slowest total above plus a margin of TO RECORD minutes.

## Reaching RDS from each laptop

Each engineer runs `uv run restart-drill` once on the demo-site network and once on the backup hotspot.

| Engineer | Demo-site network | Backup hotspot |
| --- | --- | --- |
| Richard Go | | |
| Tung Shi An | | |
| Johnson Lim | | |

Security group rule for port 3306: TO RECORD, with the reason.

## Sandbox budget

At every sprint review, open the sandbox's start page, note the budget used and remaining, and add a row here. If the remaining budget would not last until the release, stop the services between work sessions.

| Sprint review | Used | Remaining |
| --- | --- | --- |
| | | |
