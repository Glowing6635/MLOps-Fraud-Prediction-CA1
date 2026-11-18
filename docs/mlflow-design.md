# MLflow server design

Why the team's MLflow server is built the way it is (issues #8 and #9). Before changing it, the team researched the options in one conversation with a web chat assistant, kept to fewer than 10 questions and limited to services the sandbox allows: https://claude.ai/share/02998146-af10-472d-a21f-05e215dc7623. This page records the limits that conversation found, the design we chose, and where we didn't follow its advice. `mlflow-server/README.md` and `mlflow-server/DEPLOY.md` say how to run it.

## Limits of the first setup

The first server followed the standard tutorial: runs in a SQLite file and artifacts in a folder, both inside the container.

- Both are lost whenever Elastic Beanstalk replaces the instance. That happens on every redeploy, and on 17 Nov we saw it happen at the start of a new sandbox session too.
- SQLite lets only one process write at a time, so two engineers logging runs together get "database is locked" errors.
- The instance's small disk fills quickly with 100 MB artifacts.
- Anyone with the address can read, change or delete runs and models, because there is no login.
- Beanstalk's nginx refuses uploads over 1 MB, which blocks logging most models.
- Nothing keeps passwords out of the image.

## The design

Nothing is stored on the instance, so replacing it loses nothing.

| Service | Its part |
| --- | --- |
| RDS MySQL (`fraud-db`, the team database) | The `mlflow` database holds runs and the model registry, and `mlflow_auth` holds the logins. The server has its own `mlflow` user for both. RDS refuses unencrypted connections, and port 3306 is open only to the server's security group and each engineer's own address. |
| S3 | Holds the artifacts. Laptops upload them through the server, so they need no AWS keys. |
| Elastic Beanstalk | One instance, with no load balancer. It runs the image from ECR, waits up to 15 minutes for RDS, and requires a login. The health check uses `/health`, which needs no login, and 401 responses don't count against the environment's health. Uploads up to 200 MB are allowed. |
| ECR | Holds the server image. The base image is pinned to one build and every package to a hashed lock, and tags can't be overwritten, so every new instance runs the same server. |
| Parameter Store | Holds the four secrets as encrypted parameters, so they never show on the Beanstalk settings page. Settings that aren't secret stay as plain environment properties. |
| Laptops | Need only the server's address and their own MLflow login to log runs. |

The server reaches S3 through `LabInstanceProfile`, set as the environment's instance profile. It attaches the sandbox's `LabRole` to the instance, and MLflow picks up that role's temporary credentials by itself. The same role lets the instance pull the image from ECR and read the parameters. Beanstalk's service role is also `LabRole`, so nothing creates an IAM user, role or policy.

## Where we didn't follow the advice

- New logins can read every experiment, where the conversation suggested they see nothing until given access. Only the engineers and the reviewers named in the decision log get logins, and the reviewers need to read everything.
- The web port stays open to every address, behind the login. The conversation suggested allowing only our home addresses, but reviewers and the demo audience connect from addresses we can't list in advance.
- `uv run restart-drill` and `uv run redeploy-check` replace the suggested smoke test. They already check that the server answers, that the login is enforced, and that runs, models and a 100 MB artifact come back unchanged.
- Each engineer's database user can only read. Loading data needs a user that can write, which issue #20 adds, so the admin user isn't used from laptops.

## Limits that remain

- The site is plain HTTP, so logins cross the network only lightly encoded. Never log passwords, keys or source data to it.
- Pressing Reset in the sandbox deletes everything. `docs/runbook.md` says when to back up.
- RDS keeps spending the sandbox budget while the sandbox is stopped. `docs/runbook.md` says when to stop it.
