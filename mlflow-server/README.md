# MLflow server

The team's MLflow server: a Docker image kept in ECR, and the bundle that tells Elastic Beanstalk to run it. Runs, registered models and logins are kept in the team's RDS MySQL instance and artifacts in S3 (issue #9), so a redeploy or a new sandbox session keeps them. Laptops reach artifacts through the server, so they need no AWS keys. The server requires a login, but traffic is plain HTTP, so never log source data, passwords or keys to it.

## Files

- `Dockerfile`: builds the image from a base image pinned to one build, the packages in `requirements.txt`, and RDS's certificates.
- `requirements.in`: the four packages chosen by hand. `requirements.txt`: every package they pull in, locked with hashes.
- `Dockerrun.aws.json`: the ECR image Beanstalk runs, so every new instance runs the same packages.
- `.ebignore`: keeps the files that build the image out of the bundle.
- `start.sh`: checks every setting is set, waits up to 15 minutes for RDS, sets up the login database and starts the server. A missing setting stops it.
- `.ebextensions/health.config`: points the health check at `/health`, the one page that needs no login.
- `.platform/`: raises nginx's upload limit from 1 MB to 200 MB.
- `DEPLOY.md`: setting up RDS and S3, building and pushing the image, and deploying it.

## Settings

None is in the image or in git. The four secrets are encrypted `SecureString` parameters in Parameter Store, which Beanstalk reads into environment variables when an instance starts, so they never show on the Beanstalk settings page. The rest are plain environment properties. In a database address, write `@`, `:`, `/` or `%` in the password as `%40`, `%3A`, `%2F` or `%25`.

| Variable | Kept in | Value |
| --- | --- | --- |
| `MLFLOW_BACKEND_STORE_URI` | parameter `/mlflow/backend-store-uri` | `mysql+pymysql://mlflow:<password>@<rds-endpoint>:3306/mlflow?ssl_ca=/etc/ssl/rds-global-bundle.pem` |
| `MLFLOW_AUTH_DATABASE_URI` | parameter `/mlflow/auth-database-uri` | the same, with `/mlflow_auth` in place of `/mlflow` |
| `MLFLOW_AUTH_ADMIN_PASSWORD` | parameter `/mlflow/auth-admin-password` | the first admin password, used only while the login database is empty |
| `MLFLOW_FLASK_SERVER_SECRET_KEY` | parameter `/mlflow/flask-secret-key` | a long random string; changing it signs everyone out |
| `MLFLOW_ARTIFACTS_DESTINATION` | environment property | `s3://<bucket>/artifacts` |
| `MLFLOW_SERVER_ALLOWED_HOSTS` | environment property | `<prefix>.us-east-1.elasticbeanstalk.com` |
| `MLFLOW_SERVER_CORS_ALLOWED_ORIGINS` | environment property | the same with `http://`, or the web UI cannot change anything |

`ssl_ca` makes the server encrypt its database connections and check the RDS certificate. RDS refuses unencrypted connections (`require_secure_transport` in the `fraud-mysql84` parameter group).

## Logins

Every login can read every experiment. Make each engineer's login an admin so they can log runs; leave reviewers' logins as they are. Passwords need at least 12 characters. With the admin login in `.env`:

```
uv run --with Flask-WTF==1.3.0 python -c "from mlflow.server import get_app_client as c; a = c('basic-auth', tracking_uri='<server URL>'); a.create_user('<name>', '<password>'); a.update_user_admin('<name>', True)"
```

`a.update_user_password('admin', '<new>', '<current>')` changes the admin password; update the `/mlflow/auth-admin-password` parameter to match.

## Test locally with Docker

Run a MySQL 8.4 container with `--log-bin-trust-function-creators=1` in place of RDS, set it up as in step 2, then run this image on the same Docker network with `-m 2g` (the t3.small's memory; the server uses about 600 MB), the settings above without `?ssl_ca=...`, and `MLFLOW_ARTIFACTS_DESTINATION=/artifacts` on a volume in place of S3. In Git Bash, `export MSYS_NO_PATHCONV=1` first, or it rewrites `/artifacts` into a Windows path.

## Check from each laptop

Set the server URL, `MLFLOW_EXPERIMENT_NAME` and your own login in `.env`, then:

- `uv run log-test-run` logs a run with a small artifact.
- `uv run redeploy-check record` logs a run and a small registered model before a redeploy. After it, `uv run redeploy-check verify` fails unless both are unchanged.
- `uv run redeploy-check large` sends a 100 MB artifact through the server and back.
