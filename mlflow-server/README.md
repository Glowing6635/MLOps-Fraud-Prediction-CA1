# MLflow server

The Docker bundle that Elastic Beanstalk builds into the team's MLflow server. Runs, registered models and logins are kept in the team's RDS MySQL instance and artifacts in S3 (issue #9), so a redeploy or a new sandbox session keeps them. Laptops reach artifacts through the server, so they need no AWS keys. The server requires a login, but traffic is plain HTTP, so never log source data, passwords or keys to it.

## Files

- `Dockerfile`: installs MLflow with what it needs for RDS and S3.
- `start.sh`: checks every setting is present, sets up the login database and starts the server. A missing setting stops it, so it never falls back to storage inside the container.
- `.ebextensions/health.config`: points the health check at `/health`, the one page that needs no login.
- `.platform/`: raises nginx's upload limit from 1 MB to 200 MB.

## Settings

Elastic Beanstalk passes each one as an environment property; none is in the image or in git. In a database address, write `@`, `:`, `/` or `%` in the password as `%40`, `%3A`, `%2F` or `%25`.

| Property | Value |
| --- | --- |
| `MLFLOW_BACKEND_STORE_URI` | `mysql+pymysql://mlflow:<password>@<rds-endpoint>:3306/mlflow` |
| `MLFLOW_AUTH_DATABASE_URI` | the same, ending `/mlflow_auth` |
| `MLFLOW_ARTIFACTS_DESTINATION` | `s3://<bucket>/artifacts` |
| `MLFLOW_AUTH_ADMIN_PASSWORD` | the first admin password, used only while the login database is empty |
| `MLFLOW_FLASK_SERVER_SECRET_KEY` | a long random string; changing it signs everyone out |
| `MLFLOW_SERVER_ALLOWED_HOSTS` | `<prefix>.us-east-1.elasticbeanstalk.com` |
| `MLFLOW_SERVER_CORS_ALLOWED_ORIGINS` | the same with `http://`, or the web UI cannot change anything |

## Set up RDS and S3, once

1. MLflow's database setup creates triggers, which RDS refuses while binary logging is on. Attach a parameter group with `log_bin_trust_function_creators` set to `1`, and reboot the instance.
2. As the instance's admin user, run `CREATE DATABASE mlflow; CREATE DATABASE mlflow_auth;`, create a `mlflow` user, and grant it `ALL` on both.
3. Let the instance's security group accept port 3306 from the Beanstalk instance's security group.
4. Create an S3 bucket in `us-east-1` with public access blocked. `LabInstanceProfile` gives the server access.

## Deploy

You need the EB CLI (`uv tool install --python 3.12 awsebcli`) and the sandbox credentials in `~/.aws/credentials`, pasted fresh each session. From this folder:

```
eb init mlflow-server --platform docker --region us-east-1
eb create mlflow --single --instance_type t3.small --service-role LabRole --instance_profile LabInstanceProfile --cname <prefix>
eb setenv <each setting above>
```

`LabRole` and `LabInstanceProfile` are the sandbox's own, so nothing creates an IAM resource. The server starts once `eb setenv` has run; `eb status` then shows `Health: Green`. Later changes go out with `eb deploy`, and everything stored stays.

## Logins

Every login can read every experiment. Make each engineer's login an admin so they can log runs; leave reviewers' logins as they are. Passwords need at least 12 characters. With the admin login in `.env`:

```
uv run --with Flask-WTF==1.3.0 python -c "from mlflow.server import get_app_client as c; a = c('basic-auth', tracking_uri='<server URL>'); a.create_user('<name>', '<password>'); a.update_user_admin('<name>', True)"
```

`a.update_user_password('admin', '<new>', '<current>')` changes the admin password; update `MLFLOW_AUTH_ADMIN_PASSWORD` to match.

## Test locally with Docker

Run a MySQL 8.4 container with `--log-bin-trust-function-creators=1` in place of RDS, set it up as in step 2, then run this image on the same Docker network with `-m 2g` (the t3.small's memory; the server uses about 600 MB), the settings above, and `MLFLOW_ARTIFACTS_DESTINATION=/artifacts` on a volume in place of S3. In Git Bash, `export MSYS_NO_PATHCONV=1` first, or it rewrites `/artifacts` into a Windows path.

## Check from each laptop

Set the server URL, `MLFLOW_EXPERIMENT_NAME` and your own login in `.env`, then:

- `uv run log-test-run` logs a run with a small artifact.
- `uv run redeploy-check record` logs a run and a small registered model before a redeploy. After it, `uv run redeploy-check verify` fails unless both are unchanged.
- `uv run redeploy-check large` sends a 100 MB artifact through the server and back.
