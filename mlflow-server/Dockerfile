# The team's MLflow tracking server. It is built once, pushed to ECR, and every Beanstalk
# instance runs that same image (see README.md). Runs and logins live in RDS and artifacts in
# S3 (issue #9), so a new instance keeps them.
# The base image is pinned to one build, so a rebuild starts from the same files.
FROM python:3.12-slim@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b

# Every package, including the ones MLflow pulls in, comes from the locked and hashed list.
# requirements.in holds the four chosen versions; README.md says how to relock.
COPY requirements.txt .
RUN pip install --no-cache-dir --require-hashes -r requirements.txt

# RDS's certificates, so the server checks it is talking to RDS over an encrypted link.
ADD --checksum=sha256:90a6f1a7978a1b1b096dca6c57511ca48d3b8cf052f8e704baf358f34ea2f21b     https://truststore.pki.rds.amazonaws.com/global/global-bundle.pem /etc/ssl/rds-global-bundle.pem

# MLflow's first login-database migration gives two foreign keys the same name, which MySQL
# refuses (error 1826). Rename the second; the build fails if that file has changed.
RUN f=/usr/local/lib/python3.12/site-packages/mlflow/server/auth/db/migrations/versions/8606fa83a998_initial_migration.py  && test "$(grep -c 'name="fk_user_id"' "$f")" = 2  && sed -i '0,/name="fk_user_id"/! s/name="fk_user_id"/name="fk_registered_model_user_id"/' "$f"  && test "$(grep -c 'name="fk_user_id"' "$f")" = 1

# The job runner starts one MLflow process per server job type (none of which the team
# uses); together they fill the t3.small's 2 GB and freeze it.
ENV MLFLOW_SERVER_ENABLE_JOB_EXECUTION=false
# Stop MLflow sending usage data to its developers.
ENV MLFLOW_DISABLE_TELEMETRY=true

WORKDIR /mlflow
COPY start.sh .
EXPOSE 5000

CMD ["sh", "start.sh"]
