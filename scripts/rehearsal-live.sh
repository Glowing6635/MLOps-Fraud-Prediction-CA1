#!/usr/bin/env bash
# One-command live rehearsal run (PBI #24). Needs, in the environment only:
#   DB_HOST DB_PORT DB_USER DB_PASSWORD   (admin login valid on the rehearsal schema)
#   MLFLOW_TRACKING_URI MLFLOW_TRACKING_USERNAME MLFLOW_TRACKING_PASSWORD
#   AWS_SHARED_CREDENTIALS_FILE AWS_CONFIG_FILE AWS_DEFAULT_REGION=us-east-1
# Optional: SG_ID (default sg-075357648cbf564b9), BUCKET, READERS="name1 name2" (each must
# already have <name>_reader from `rds-schema reader`; the script only GRANTs SELECT on rehearsal), PRESIGN_EXPIRES seconds (default 3600).
# Usage: scripts/rehearsal-live.sh run | cleanup
#
# `run` removes the security-group rule it added when it ends, and deletes what it
# created only if it fails. On success the bucket, its objects and the schema stay, so the
# printed links keep working until they expire (the expiry is printed); delete them
# afterwards with `cleanup`. `cleanup` deletes only a bucket tagged by this script
# (created-by=rehearsal-live), so it can never remove a bucket someone else made.
set -euo pipefail
SG_ID=${SG_ID:-sg-075357648cbf564b9}
BUCKET=${BUCKET:-fraud-rehearsal-516375498949}
case "$BUCKET" in fraud-rehearsal-*) ;; *) echo "BUCKET must start with fraud-rehearsal-" >&2; exit 1;; esac
TAG='TagSet=[{Key=created-by,Value=rehearsal-live}]'
export DB_NAME=rehearsal MLFLOW_EXPERIMENT_NAME=rehearsal
export TEAM_DB_NAME=fraud TEAM_MLFLOW_EXPERIMENT_NAME=${TEAM_MLFLOW_EXPERIMENT_NAME:-fraud}
SQL='import sys,sqlalchemy as s;from mlops_fraud_prediction_ca1.config import load_settings,create_db_engine as c;e=s.create_engine(c(load_settings()).url.set(database=None));r=e.connect().execute(s.text(sys.argv[1]));print(r.scalar() if r.returns_rows else "")'
sql() { uv run python -c "$SQL" "$1"; }

owned_bucket() {  # true only if the bucket exists and carries this script's tag
  aws s3api get-bucket-tagging --bucket "$BUCKET" --query "TagSet[?Key=='created-by'].Value" --output text 2>/dev/null | grep -qx rehearsal-live
}
delete_bucket() {
  if owned_bucket; then
    aws s3 rm "s3://$BUCKET" --recursive --quiet
    aws s3api delete-bucket --bucket "$BUCKET"
    echo "deleted bucket $BUCKET"
  else
    echo "bucket $BUCKET not created by this script: left alone"
  fi
}

MYIP=$(curl -s https://checkip.amazonaws.com)/32
RULE_ADDED=; CREATED_BUCKET=; CREATED_SCHEMA=; OK=
FAILED=
step() {  # run a cleanup step; remember its name if it fails
  local name=$1; shift
  "$@" || { FAILED="$FAILED $name"; echo "CLEANUP FAILED: $name" >&2; }
}
cleanup_run() {
  local status=$?
  if [ -n "$RULE_ADDED" ]; then
    # only the rule this run added (RULE_ADDED is set after authorize succeeded)
    step revoke-ingress aws ec2 revoke-security-group-ingress --group-id "$SG_ID" --protocol tcp --port 3306 --cidr "$MYIP"
    [ -z "$FAILED" ] && echo "removed ingress $MYIP"
  fi
  if [ -z "$OK" ]; then  # failure: remove only what this run created
    [ -n "$CREATED_BUCKET" ] && step delete-bucket delete_bucket
    [ -n "$CREATED_SCHEMA" ] && step drop-schema sql "DROP DATABASE rehearsal"
  fi
  if [ -n "$FAILED" ]; then
    echo "CLEANUP INCOMPLETE, fix by hand:$FAILED (port 3306 on $SG_ID may still be open to $MYIP)" >&2
    exit 1
  fi
  exit "$status"
}
trap cleanup_run EXIT
open_port() {
  aws ec2 authorize-security-group-ingress --group-id "$SG_ID" --protocol tcp --port 3306 --cidr "$MYIP"
  RULE_ADDED=1
  echo "added ingress $MYIP tcp/3306 on $SG_ID"
}

if [ "${1:-}" = cleanup ]; then
  OK=1  # nothing to undo beyond the ingress rule
  open_port
  step delete-bucket delete_bucket
  if [ -z "${KEEP_SCHEMA:-}" ]; then step drop-schema sql "DROP DATABASE IF EXISTS rehearsal"; fi
  exit 0
fi

# Refuse to reuse a bucket this script did not create.
if aws s3api head-bucket --bucket "$BUCKET" 2>/dev/null; then
  echo "bucket $BUCKET already exists; refusing to use or delete it" >&2; exit 1
fi
open_port
[ "$(sql "SELECT COUNT(*) FROM information_schema.schemata WHERE schema_name='rehearsal'")" = 0 ] && CREATED_SCHEMA=1
sql "CREATE DATABASE IF NOT EXISTS rehearsal"
uv run rds-schema tables
for name in ${READERS:-}; do  # SELECT only; never touches the user's password
  case "$name" in *[!a-z0-9_]*|"") echo "bad reader name $name" >&2; exit 1;; esac
  sql "GRANT SELECT ON rehearsal.* TO '${name}_reader'@'%'"
done
time uv run rehearsal-reset --data-dir data
aws s3api create-bucket --bucket "$BUCKET"; CREATED_BUCKET=1
aws s3api put-bucket-tagging --bucket "$BUCKET" --tagging "$TAG"
aws s3api put-public-access-block --bucket "$BUCKET" --public-access-block-configuration BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
EXP=${PRESIGN_EXPIRES:-3600}
uv run rehearsal-reset --presign "$BUCKET" --expires "$EXP"
OK=1
echo "links expire in ${EXP}s (at $(date -u -d "+${EXP} seconds" +%H:%M:%SZ)); run 'scripts/rehearsal-live.sh cleanup' afterwards"
