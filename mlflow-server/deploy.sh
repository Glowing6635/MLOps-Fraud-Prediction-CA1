#!/usr/bin/env bash
# Deploy the MLflow server to Elastic Beanstalk (issue #26).
# Usage: ./deploy.sh <prefix>      e.g. ./deploy.sh sp-buy-mlflow
# Needs the EB CLI and sandbox credentials already configured (see docs/runbooks/mlflow-deploy.md).
# Creates no IAM resource: it uses the sandbox's own LabRole and LabInstanceProfile.
set -euo pipefail

prefix="${1:?usage: ./deploy.sh <cname-prefix>}"
region="${AWS_DEFAULT_REGION:-us-east-1}"
env_name="${EB_ENV_NAME:-mlflow}"
host="${prefix}.${region}.elasticbeanstalk.com"

cd "$(dirname "$0")"
command -v eb >/dev/null || { echo "eb not found: uv tool install --python 3.12 awsebcli" >&2; exit 1; }
aws sts get-caller-identity >/dev/null || { echo "AWS credentials missing or expired" >&2; exit 1; }

[ -d .elasticbeanstalk ] || eb init mlflow-server --platform docker --region "$region"

environments="$(eb list)"
if awk -v name="$env_name" '
  { sub(/^[*[:space:]]*/, "") }
  $0 == name { found = 1 }
  END { exit !found }
' <<< "$environments"; then
  status="$(eb status "$env_name")"
  live_host="$(awk '$1 == "CNAME:" { print $2; exit }' <<< "$status")"
  if [[ "$live_host" != "$host" ]]; then
    printf 'Environment %s has CNAME %s; expected %s. Redeploy cancelled.\n' \
      "$env_name" "${live_host:-unavailable}" "$host" >&2
    exit 1
  fi

  echo "Environment ${env_name} exists: redeploying."
  eb deploy "$env_name"
else
  eb create "$env_name" --single --instance_type t3.small \
    --service-role LabRole --instance_profile LabInstanceProfile \
    --cname "$prefix" \
    --envvars "MLFLOW_SERVER_ALLOWED_HOSTS=${host},MLFLOW_SERVER_CORS_ALLOWED_ORIGINS=http://${host}"
fi

eb status "$env_name"
echo "Health check: curl -fsS http://${host}/health"
curl -fsS "http://${host}/health" && echo
