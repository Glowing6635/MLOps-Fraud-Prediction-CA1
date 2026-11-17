# Starts the MLflow server with a login, from Elastic Beanstalk environment properties.
#
# MLflow itself reads MLFLOW_BACKEND_STORE_URI (runs, in RDS), MLFLOW_ARTIFACTS_DESTINATION
# (artifacts, in S3), MLFLOW_AUTH_ADMIN_PASSWORD (the admin password, used only when the
# login database is empty) and MLFLOW_FLASK_SERVER_SECRET_KEY (signs the web UI's forms;
# changing it signs everyone out). The login database's address can only come from a config file,
# so this writes one from MLFLOW_AUTH_DATABASE_URI. A missing setting stops the server
# instead of letting MLflow fall back to storage inside the container, which a redeploy loses.
set -eu
: "${MLFLOW_BACKEND_STORE_URI:?is not set}" "${MLFLOW_ARTIFACTS_DESTINATION:?is not set}"
: "${MLFLOW_AUTH_DATABASE_URI:?is not set}" "${MLFLOW_AUTH_ADMIN_PASSWORD:?is not set}"
: "${MLFLOW_FLASK_SERVER_SECRET_KEY:?is not set}"

# The config parser treats % as special, and URL-encoded passwords contain it.
auth_uri=$(printf '%s' "$MLFLOW_AUTH_DATABASE_URI" | sed 's/%/%%/g')
cat > /tmp/basic_auth.ini <<INI
[mlflow]
default_permission = READ
database_uri = $auth_uri
admin_username = admin
authorization_function = mlflow.server.auth:authenticate_request_basic_auth
INI
export MLFLOW_AUTH_CONFIG_PATH=/tmp/basic_auth.ini

# A new sandbox session can start this instance before a stopped RDS is ready, so wait up
# to 15 minutes for it rather than crash.
python - <<'PY'
import os, sys, time
import sqlalchemy

engine = sqlalchemy.create_engine(os.environ["MLFLOW_AUTH_DATABASE_URI"])
for attempt in range(90):
    try:
        engine.connect().close()
        sys.exit(0)
    except sqlalchemy.exc.OperationalError as error:
        print(f"Waiting for RDS ({attempt + 1}/90): {error.orig}", flush=True)
        time.sleep(10)
sys.exit("RDS did not answer within 15 minutes.")
PY

# Set up or upgrade the login database first: when the server does it on MySQL, two of its
# connections wait on each other's table lock and it never starts.
python -m mlflow.server.auth db upgrade --url "$MLFLOW_AUTH_DATABASE_URI"

# MLflow refuses Host headers other than localhost and private IPs unless the
# MLFLOW_SERVER_ALLOWED_HOSTS environment property names the Beanstalk URL, and
# refuses changes made in the web UI unless MLFLOW_SERVER_CORS_ALLOWED_ORIGINS does.
exec mlflow server --app-name basic-auth --host 0.0.0.0 --port 5000 --workers 2
