# Decision log

Major decisions, each with the date it was settled, its owner and the reason (#30).

## D8-01: The MLflow server keeps its data in RDS and S3

- Settled: 2026-11-18. Owner: Glowing6635. Affected PBIs: #7, #8, #9, #11.
- Decision: runs, models and logins live in RDS, and artifacts in S3 through the server. The server runs one pinned image from ECR, reads its secrets from Parameter Store, and reaches S3 through `LabInstanceProfile`, which attaches `LabRole`.
- Reason: the sandbox replaces the server's instance on every redeploy and at the start of a new session, so anything stored on it is lost.
- Source: the #8 research conversation, https://claude.ai/share/02998146-af10-472d-a21f-05e215dc7623.
- Details, the limits found and where we didn't follow the advice: [MLflow server design](mlflow-design.md).
