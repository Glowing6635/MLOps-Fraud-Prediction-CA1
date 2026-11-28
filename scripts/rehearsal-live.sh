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
# afterwards with `cleanup`. `cleanup` keeps the schema (so readers can check it) unless KEEP_SCHEMA=0, and deletes only a bucket tagged by this script
# (created-by=rehearsal-live), so it can never remove a bucket someone else made.
# Set KEEP_BUCKET=1 with KEEP_SCHEMA=0 to drop only the rehearsal schema.
set -euo pipefail
SG_ID=${SG_ID:-sg-075357648cbf564b9}
BUCKET=${BUCKET:-rehearsal-516375498949}
case "$BUCKET" in rehearsal*) ;; *) echo "BUCKET must start with rehearsal" >&2; exit 1;; esac
TAG='TagSet=[{Key=created-by,Value=rehearsal-live}]'
export DB_NAME=rehearsal MLFLOW_EXPERIMENT_NAME=rehearsal
export TEAM_DB_NAME=${TEAM_DB_NAME:-fraud} TEAM_MLFLOW_EXPERIMENT_NAME=${TEAM_MLFLOW_EXPERIMENT_NAME:-fraud}
SQL='import sys,sqlalchemy as s;from mlops_fraud_prediction_ca1.config import load_settings,create_db_engine as c;e=s.create_engine(c(load_settings()).url.set(database=None));r=e.connect().execute(s.text(sys.argv[1]));print(r.scalar() if r.returns_rows else "")'
sql() { uv run python -c "$SQL" "$1"; }

delete_bucket() {  # "force": skip the tag check for a bucket this run just created
  local owner
  if [ "${1:-}" != force ]; then
    if ! owner=$(aws s3api get-bucket-tagging --bucket "$BUCKET" --query "TagSet[?Key=='created-by'].Value" --output text 2>&1); then
      if grep -qE '^An error occurred \((NoSuchBucket|NoSuchTagSet)\) when calling the GetBucketTagging operation:' <<< "$owner"; then
        echo "bucket $BUCKET is absent or has no tags: left alone"
        return 0
      fi
      printf 'Cannot verify ownership of bucket %s (get-bucket-tagging): %s\n' "$BUCKET" "$owner" >&2
      return 1
    fi
    if [ "$owner" != rehearsal-live ]; then
      echo "bucket $BUCKET not created by this script: left alone"
      return 0
    fi
  fi
  # step() calls this inside ||, where errexit is off, so chain every command
  aws s3 rm "s3://$BUCKET" --recursive --quiet &&
    aws s3api delete-bucket --bucket "$BUCKET" &&
    echo "deleted bucket $BUCKET"
}

RULE_ID=; CREATED_BUCKET=; TAGGED=; CREATED_SCHEMA=; OK=
FAILED=
step() {  # run a cleanup step; remember its name if it fails
  local name=$1; shift
  "$@" || { FAILED="$FAILED $name"; echo "CLEANUP FAILED: $name" >&2; }
}
cleanup_run() {
  local status=$?
  if [ -z "$OK" ]; then  # failure: remove only what this run created
    if [ -n "$CREATED_BUCKET" ]; then
      # created by this run (head-bucket proved it did not exist); if tagging never landed, skip the tag check
      if [ -n "$TAGGED" ]; then step delete-bucket delete_bucket; else step delete-bucket delete_bucket force; fi
    fi
    [ -n "$CREATED_SCHEMA" ] && step drop-schema sql "DROP DATABASE rehearsal"
  fi
  if [ -n "$RULE_ID" ]; then
    step revoke-ingress revoke_port
  fi
  if [ -n "$FAILED" ]; then
    echo "CLEANUP INCOMPLETE, fix by hand:$FAILED" >&2
    [ "$status" -ne 0 ] || status=1
  fi
  exit "$status"
}
trap cleanup_run EXIT
open_port() {
  local ip rule_id
  ip=$(curl --fail --silent --show-error https://checkip.amazonaws.com) || return 1
  rule_id=$(aws ec2 authorize-security-group-ingress --group-id "$SG_ID" --protocol tcp \
    --port 3306 --cidr "$ip/32" --query 'SecurityGroupRules[0].SecurityGroupRuleId' \
    --output text) || return 1
  if [[ ! "$rule_id" =~ ^sgr-[0-9a-f]+$ ]]; then
    echo "Ingress authorization on $SG_ID for $ip/32 returned no usable rule ID. Verify the rule in EC2." >&2
    return 1
  fi
  RULE_ID=$rule_id
  echo "added ingress $RULE_ID for $ip/32 tcp/3306 on $SG_ID"
}

revoke_port() {
  local revoked
  if revoked=$(aws ec2 revoke-security-group-ingress --group-id "$SG_ID" \
    --security-group-rule-ids "$RULE_ID" --query Return --output text) &&
    [ "$revoked" = True ]; then
    echo "removed ingress $RULE_ID"
    return 0
  fi
  echo "Failed to confirm removal of ingress $RULE_ID on $SG_ID" >&2
  return 1
}

drop_rehearsal_schema() {
  if [ "$TEAM_DB_NAME" = "$DB_NAME" ]; then
    echo "Refusing to drop the team schema $TEAM_DB_NAME" >&2
    return 1
  fi
  open_port && sql "DROP DATABASE IF EXISTS rehearsal"
}

if [ "${1:-}" = cleanup ]; then
  OK=1  # Only reconcile ingress acquired by the optional schema cleanup.
  if [ "${KEEP_BUCKET:-0}" != 1 ]; then step delete-bucket delete_bucket; fi
  if [ "${KEEP_SCHEMA:-1}" = 0 ]; then step drop-schema drop_rehearsal_schema; fi
  exit 0
fi

if [ "$TEAM_DB_NAME" = "$DB_NAME" ]; then
  echo "Refusing to use the team schema $TEAM_DB_NAME" >&2
  exit 1
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
aws s3api put-bucket-tagging --bucket "$BUCKET" --tagging "$TAG"; TAGGED=1
aws s3api put-public-access-block --bucket "$BUCKET" --public-access-block-configuration BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
EXP=${PRESIGN_EXPIRES:-3600}
uv run rehearsal-reset --presign "$BUCKET" --expires "$EXP"
OK=1
echo "links expire in ${EXP}s (at $(date -u -d "+${EXP} seconds" +%H:%M:%SZ)); run 'scripts/rehearsal-live.sh cleanup' afterwards"
