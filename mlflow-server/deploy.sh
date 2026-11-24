#!/usr/bin/env bash
# Deploy the MLflow server to Elastic Beanstalk (issue #26).
# Usage: ./deploy.sh <prefix>      e.g. ./deploy.sh sp-buy-mlflow
# Needs the EB CLI and sandbox credentials already configured (see DEPLOY.md).
# Creates no IAM resource: it uses the sandbox's own LabRole and LabInstanceProfile.
set -euo pipefail

prefix="${1:?usage: ./deploy.sh <cname-prefix>}"
region="${AWS_DEFAULT_REGION:-us-east-1}"
env_name="${EB_ENV_NAME:-mlflow}"
host="${prefix}.${region}.elasticbeanstalk.com"

cd "$(dirname "$0")"
if [[ ! -f Dockerrun.aws.json || ! -f start.sh ]]; then
  echo "The #9 ECR deployment bundle is required. See DEPLOY.md." >&2
  exit 1
fi

command -v aws >/dev/null || { echo "AWS CLI not found" >&2; exit 1; }
command -v eb >/dev/null || { echo "eb not found: uv tool install --python 3.12 awsebcli" >&2; exit 1; }
account="$(aws sts get-caller-identity --region "$region" --query Account --output text)" || { echo "AWS credentials missing or expired" >&2; exit 1; }

if ! grep -Fq "\"$account.dkr.ecr.$region.amazonaws.com/mlflow-server:" Dockerrun.aws.json; then
  echo "Set Dockerrun.aws.json to the published mlflow-server image in $account/$region." >&2
  exit 1
fi

[ -f .elasticbeanstalk/config.yml ] || eb init mlflow-server --platform docker --region "$region"

environments="$(eb list --region "$region")"
if awk -v name="$env_name" '
  { sub(/^[*[:space:]]*/, "") }
  $0 == name { found = 1 }
  END { exit !found }
' <<< "$environments"; then
  status="$(eb status "$env_name" --region "$region")"
  live_host="$(awk '$1 == "CNAME:" { print $2; exit }' <<< "$status")"
  if [[ "$live_host" != "$host" ]]; then
    printf 'Environment %s has CNAME %s; expected %s. Redeploy cancelled.\n' \
      "$env_name" "${live_host:-unavailable}" "$host" >&2
    exit 1
  fi

  echo "Environment ${env_name} exists: redeploying."
  eb deploy "$env_name" --region "$region"
else
  : "${MLFLOW_ARTIFACTS_DESTINATION:?set to s3://<bucket>/artifacts for a new environment}"

  # Apply the settings at creation so the image has them on its first boot.
  config_dir="$(mktemp -d .elasticbeanstalk/mlflow-deploy.XXXXXX)"
  trap 'rm -rf -- "$config_dir"' EXIT
  config="$config_dir/$env_name-mlflow.cfg.yml"
  parameter_arn="arn:aws:ssm:$region:$account:parameter/mlflow"
  cat > "$config" <<EOF
AWSConfigurationTemplateVersion: 1.1.0.0
OptionSettings:
  aws:elasticbeanstalk:application:environment:
    MLFLOW_ARTIFACTS_DESTINATION: "$MLFLOW_ARTIFACTS_DESTINATION"
    MLFLOW_SERVER_ALLOWED_HOSTS: "$host"
    MLFLOW_SERVER_CORS_ALLOWED_ORIGINS: "http://$host"
  aws:elasticbeanstalk:application:environmentsecrets:
    MLFLOW_BACKEND_STORE_URI: "$parameter_arn/backend-store-uri"
    MLFLOW_AUTH_DATABASE_URI: "$parameter_arn/auth-database-uri"
    MLFLOW_AUTH_ADMIN_PASSWORD: "$parameter_arn/auth-admin-password"
    MLFLOW_FLASK_SERVER_SECRET_KEY: "$parameter_arn/flask-secret-key"
EOF

  eb create "$env_name" --single --instance_type t3.small \
    --service-role LabRole --instance_profile LabInstanceProfile \
    --cname "$prefix" --region "$region" --cfg "./$config"
fi

eb status "$env_name" --region "$region"
echo "Health check: curl -fsS http://${host}/health"
curl -fsS "http://${host}/health" && echo
