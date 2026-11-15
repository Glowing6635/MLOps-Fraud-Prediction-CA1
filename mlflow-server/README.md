# MLflow server

The Docker bundle that Elastic Beanstalk builds into the team's MLflow server (issue #7). This basic design keeps runs in SQLite and artifacts in a folder inside the container, so a redeploy or a replaced instance loses them. Issue #9 moves them to RDS and S3.

Until issue #9 adds a login, anyone who knows the server's address can read and change everything on it, and traffic is plain HTTP that others on the network can read. Never log course data, passwords or keys to it.

## Files

- `Dockerfile`: installs MLflow and starts its server on port 5000.
- `.platform/`: raises the upload limit of Beanstalk's nginx web proxy from 1 MB to 200 MB, so models can be logged.

## Deploy

You need the EB CLI (`uv tool install --python 3.12 awsebcli`) and the sandbox credentials in `~/.aws/credentials`. Learner Lab credentials expire when the session ends, so paste fresh ones from "AWS Details" each session.

Run these from this folder, replacing `<prefix>` with a name not yet taken, such as `sp-buy-mlflow`. This folder has no `.git` of its own, so the EB CLI uploads the folder as it is on disk, uncommitted edits included.

```
eb init mlflow-server --platform docker --region us-east-1
eb create mlflow --single --instance_type t3.small --service-role LabRole --instance_profile LabInstanceProfile --cname <prefix> --envvars MLFLOW_SERVER_ALLOWED_HOSTS=<prefix>.us-east-1.elasticbeanstalk.com,MLFLOW_SERVER_CORS_ALLOWED_ORIGINS=http://<prefix>.us-east-1.elasticbeanstalk.com
eb status
```

- `--single` runs one instance with no load balancer.
- `LabRole` and `LabInstanceProfile` are the sandbox account's own service role and instance profile, so nothing creates an IAM resource.
- `MLFLOW_SERVER_ALLOWED_HOSTS` must name the environment's URL: MLflow refuses requests for any other host name, except `/health`.
- `MLFLOW_SERVER_CORS_ALLOWED_ORIGINS` must name the same URL with `http://`: otherwise MLflow refuses every change and search made in the web UI, such as listing, renaming or deleting runs, because the browser sends that address as the request's origin.
- `eb status` shows `Health: Green` once the environment is healthy.

To deploy changes later, run `eb deploy` from this folder.

## Test locally with Docker

Before you deploy a change, you can run the same image on your laptop. You need Docker Desktop (on Windows it uses WSL 2, so run `wsl --update` first if Docker's engine fails to start). From this folder:

```
docker build -t mlflow-server .
docker run --rm -m 2g -p 5000:5000 mlflow-server
```

- `-m 2g` gives the container the same 2 GB of memory as the t3.small, so a change that would use too much memory fails on your laptop first.
- Open `http://localhost:5000/health` to check it answers OK, and `http://localhost:5000` for the web UI. The allowed-hosts and browser-origin settings above are not needed here, because MLflow accepts `localhost` by default.
- Press Ctrl+C to stop it. Runs logged to it disappear when it stops.

## Check from each laptop

Set `MLFLOW_TRACKING_URI=http://<prefix>.us-east-1.elasticbeanstalk.com` and `MLFLOW_EXPERIMENT_NAME` in `.env`, then run `uv run log-test-run`. It logs a run under your login name with a small artifact, and MLflow prints the run's address. Open it to see the run in the web UI.
