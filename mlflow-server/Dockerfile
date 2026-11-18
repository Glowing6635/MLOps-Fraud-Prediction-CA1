# The team's MLflow tracking server (issue #7), built by Elastic Beanstalk's Docker platform.
# Runs and artifacts live inside the container, so a redeploy loses them; issue #9 moves them to RDS and S3.
FROM python:3.12-slim

# Keep this version equal to the client's in ../pyproject.toml; CI checks that they match.
RUN pip install --no-cache-dir mlflow==3.18.0

# The job runner starts one MLflow process per server job type (none of which the team
# uses); together they fill the t3.small's 2 GB and freeze it.
ENV MLFLOW_SERVER_ENABLE_JOB_EXECUTION=false
# Stop MLflow sending usage data to its developers.
ENV MLFLOW_DISABLE_TELEMETRY=true

WORKDIR /mlflow
EXPOSE 5000

# MLflow refuses Host headers other than localhost and private IPs unless the
# MLFLOW_SERVER_ALLOWED_HOSTS environment property names the Beanstalk URL, and
# refuses changes made in the web UI unless MLFLOW_SERVER_CORS_ALLOWED_ORIGINS does.
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000", "--workers", "2", \
     "--backend-store-uri", "sqlite:////mlflow/mlflow.db", \
     "--artifacts-destination", "/mlflow/artifacts"]
