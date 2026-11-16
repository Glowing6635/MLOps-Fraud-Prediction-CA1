# The team's MLflow tracking server, built by Elastic Beanstalk's Docker platform.
# Runs and logins live in RDS and artifacts in S3 (issue #9), so a redeploy keeps them.
FROM python:3.12-slim

# Keep this version equal to the client's in ../pyproject.toml; CI checks that they match.
# Flask-WTF is the login app's extra; PyMySQL reaches RDS; boto3 reaches S3.
RUN pip install --no-cache-dir mlflow==3.18.0 Flask-WTF==1.3.0 "pymysql[rsa]==1.2.4" boto3==1.44.3

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
