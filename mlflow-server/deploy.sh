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
command -v eb >/dev/null || { echo "eb not found: uv tool install --python 3.12 awsebcli==3.27.5" >&2; exit 1; }
account="$(aws sts get-caller-identity --region "$region" --query Account --output text)" || { echo "AWS credentials missing or expired" >&2; exit 1; }

if ! grep -Fq "\"$account.dkr.ecr.$region.amazonaws.com/mlflow-server:" Dockerrun.aws.json; then
  echo "Set Dockerrun.aws.json to the published mlflow-server image in $account/$region." >&2
  exit 1
fi

parameter_arn="arn:aws:ssm:$region:$account:parameter/mlflow"
environment_namespace="aws:elasticbeanstalk:application:environment"
secrets_namespace="aws:elasticbeanstalk:application:environmentsecrets"

check_redeploy_settings() {
  local query configured setting
  local missing=()
  # Inspect the active configuration and return only names of matching settings.
  query="ConfigurationSettings[?DeploymentStatus == 'deployed'].OptionSettings[] |
    [? (Namespace == '$environment_namespace' && (
      (OptionName == 'MLFLOW_ARTIFACTS_DESTINATION' &&
        starts_with(Value || '', 's3://') && ends_with(Value || '', '/artifacts') &&
        !starts_with(Value || '', 's3:///') &&
        length(Value || '') > length('s3:///artifacts')) ||
      (OptionName == 'MLFLOW_SERVER_ALLOWED_HOSTS' && Value == '$host') ||
      (OptionName == 'MLFLOW_SERVER_CORS_ALLOWED_ORIGINS' && Value == 'http://$host')
    )) || (Namespace == '$secrets_namespace' && (
      (OptionName == 'MLFLOW_BACKEND_STORE_URI' && Value == '$parameter_arn/backend-store-uri') ||
      (OptionName == 'MLFLOW_AUTH_DATABASE_URI' && Value == '$parameter_arn/auth-database-uri') ||
      (OptionName == 'MLFLOW_AUTH_ADMIN_PASSWORD' && Value == '$parameter_arn/auth-admin-password') ||
      (OptionName == 'MLFLOW_FLASK_SERVER_SECRET_KEY' && Value == '$parameter_arn/flask-secret-key')
    ))].[OptionName]"

  if ! configured="$(aws elasticbeanstalk describe-configuration-settings \
      --application-name mlflow-server --environment-name "$env_name" --region "$region" \
      --query "$query" --output text 2>/dev/null)"; then
    echo "Could not verify the deployed MLflow settings for $env_name. Redeploy cancelled." >&2
    return 1
  fi

  for setting in MLFLOW_ARTIFACTS_DESTINATION MLFLOW_SERVER_ALLOWED_HOSTS \
      MLFLOW_SERVER_CORS_ALLOWED_ORIGINS MLFLOW_BACKEND_STORE_URI MLFLOW_AUTH_DATABASE_URI \
      MLFLOW_AUTH_ADMIN_PASSWORD MLFLOW_FLASK_SERVER_SECRET_KEY; do
    if ! grep -Fxq "$setting" <<< "$configured"; then
      missing+=("$setting")
    fi
  done
  if (( ${#missing[@]} )); then
    printf 'Missing or invalid deployed MLflow settings: %s. See DEPLOY.md. Redeploy cancelled.\n' \
      "${missing[*]}" >&2
    return 1
  fi
}

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

  check_redeploy_settings
  echo "Environment ${env_name} exists: redeploying."
  eb deploy "$env_name" --region "$region"
else
  if [[ ${MLFLOW_ARTIFACTS_DESTINATION:-} != s3://?*/artifacts ||
        ${MLFLOW_ARTIFACTS_DESTINATION:-} == s3:///* ]]; then
    echo "Set MLFLOW_ARTIFACTS_DESTINATION to s3://<bucket>/artifacts or s3://<bucket>/<prefix>/artifacts." >&2
    exit 1
  fi

  # Apply the settings at creation so the image has them on its first boot.
  # EB CLI uploads a saved config under its basename and sends --cfg verbatim.
  mkdir -p .elasticbeanstalk/saved_configs
  config_dir="$(mktemp -d .elasticbeanstalk/saved_configs/mlflow-deploy-XXXXXX)"
  template_name="${config_dir##*/}"
  config="$config_dir.cfg.yml"
  trap 'rm -f -- "$config"; rmdir -- "$config_dir"' EXIT
  (umask 077; cat > "$config") <<EOF
AWSConfigurationTemplateVersion: 1.1.0.0
OptionSettings:
  $environment_namespace:
    MLFLOW_ARTIFACTS_DESTINATION: "$MLFLOW_ARTIFACTS_DESTINATION"
    MLFLOW_SERVER_ALLOWED_HOSTS: "$host"
    MLFLOW_SERVER_CORS_ALLOWED_ORIGINS: "http://$host"
  $secrets_namespace:
    MLFLOW_BACKEND_STORE_URI: "$parameter_arn/backend-store-uri"
    MLFLOW_AUTH_DATABASE_URI: "$parameter_arn/auth-database-uri"
    MLFLOW_AUTH_ADMIN_PASSWORD: "$parameter_arn/auth-admin-password"
    MLFLOW_FLASK_SERVER_SECRET_KEY: "$parameter_arn/flask-secret-key"
EOF

  eb create "$env_name" --single --instance_type t3.small \
    --service-role LabRole --instance_profile LabInstanceProfile \
    --cname "$prefix" --region "$region" --cfg "$template_name"
fi

eb status "$env_name" --region "$region"
echo "Health check: curl -fsS http://${host}/health"
curl -fsS "http://${host}/health" && echo
