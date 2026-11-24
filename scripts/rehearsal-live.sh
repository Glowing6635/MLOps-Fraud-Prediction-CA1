#!/usr/bin/env bash
# One-command live rehearsal run (PBI #24). Needs, in the environment only:
#   DB_HOST DB_PORT DB_USER DB_PASSWORD   (admin login valid on the rehearsal schema)
#   MLFLOW_TRACKING_URI MLFLOW_TRACKING_USERNAME MLFLOW_TRACKING_PASSWORD
#   AWS_SHARED_CREDENTIALS_FILE AWS_CONFIG_FILE AWS_DEFAULT_REGION=us-east-1
# Optional: SG_ID (default sg-075357648cbf564b9), BUCKET, READERS="name1 name2" (each asks
# for a password), KEEP_SCHEMA=1 to keep the schema, PRESIGN_EXPIRES seconds (default 3600).
# Usage: scripts/rehearsal-live.sh run | cleanup
set -euo pipefail
SG_ID=${SG_ID:-sg-075357648cbf564b9}
BUCKET=${BUCKET:-fraud-rehearsal-516375498949}
export DB_NAME=rehearsal MLFLOW_EXPERIMENT_NAME=rehearsal
export TEAM_DB_NAME=fraud TEAM_MLFLOW_EXPERIMENT_NAME=${TEAM_MLFLOW_EXPERIMENT_NAME:-fraud}
MYIP=$(curl -s https://checkip.amazonaws.com)/32

cleanup() {
  aws ec2 revoke-security-group-ingress --group-id "$SG_ID" --protocol tcp --port 3306 --cidr "$MYIP" || true
  aws s3 rm "s3://$BUCKET" --recursive --quiet || true
  aws s3api delete-bucket --bucket "$BUCKET" || true
  if [ -z "${KEEP_SCHEMA:-}" ]; then
    mysql_cmd="DROP DATABASE IF EXISTS rehearsal"
    uv run python -c "import os,sqlalchemy as s;from mlops_fraud_prediction_ca1.config import load_settings,create_db_engine as c;e=c(load_settings());e2=s.create_engine(e.url.set(database=None));e2.connect().execute(s.text('$mysql_cmd'))" || true
  fi
  echo "cleanup done: rule $MYIP removed, bucket deleted${KEEP_SCHEMA:+ (schema kept)}"
}
[ "${1:-}" = cleanup ] && { cleanup; exit 0; }

aws ec2 authorize-security-group-ingress --group-id "$SG_ID" --protocol tcp --port 3306 --cidr "$MYIP"
echo "added ingress $MYIP tcp/3306 on $SG_ID"
trap cleanup EXIT
uv run python -c "import sqlalchemy as s;from mlops_fraud_prediction_ca1.config import load_settings,create_db_engine as c;e=s.create_engine(c(load_settings()).url.set(database=None));e.connect().execute(s.text('CREATE DATABASE IF NOT EXISTS rehearsal'))"
uv run rds-schema tables
for name in ${READERS:-}; do uv run rds-schema reader "$name"; done
time uv run rehearsal-reset --data-dir data
aws s3api create-bucket --bucket "$BUCKET"
aws s3api put-public-access-block --bucket "$BUCKET" --public-access-block-configuration BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
uv run rehearsal-reset --presign "$BUCKET" --expires "${PRESIGN_EXPIRES:-3600}"
