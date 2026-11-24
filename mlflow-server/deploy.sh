#!/usr/bin/env bash
# Deploy the MLflow server to Elastic Beanstalk (issue #26).
# Usage: ./deploy.sh <prefix>      e.g. ./deploy.sh sp-buy-mlflow
# Needs the EB CLI and sandbox credentials already configured (see DEPLOY.md).
# Creates no IAM resource: it uses the sandbox's own LabRole and LabInstanceProfile.
set -euo pipefail

prefix="${1:?usage: ./deploy.sh <cname-prefix>}"
region="${AWS_DEFAULT_REGION:-us-east-1}"
env_name="${EB_ENV_NAME:-mlflow}"
if ! [[ $prefix =~ ^[a-zA-Z0-9][a-zA-Z0-9-]*$ ]]; then
  echo "Use a Beanstalk CNAME prefix with letters, digits and hyphens." >&2
  exit 1
fi
if ! [[ $region =~ ^[a-z0-9][a-z0-9-]*$ && $env_name =~ ^[a-zA-Z0-9][a-zA-Z0-9-]*$ ]]; then
  echo "Check AWS_DEFAULT_REGION and EB_ENV_NAME for invalid characters." >&2
  exit 1
fi
host="${prefix}.${region}.elasticbeanstalk.com"

cd "$(dirname "$0")"
if [[ ! -f Dockerrun.aws.json || ! -f start.sh ]]; then
  echo "The #9 ECR deployment bundle is required. See DEPLOY.md." >&2
  exit 1
fi

command -v aws >/dev/null || { echo "AWS CLI not found" >&2; exit 1; }
command -v eb >/dev/null || { echo "eb not found: uv tool install --python 3.12 awsebcli==3.27.5" >&2; exit 1; }
account="$(aws sts get-caller-identity --region "$region" --query Account --output text)" || { echo "AWS credentials missing or expired" >&2; exit 1; }

if ! [[ $account =~ ^[0-9]{12}$ ]]; then
  echo "Could not verify the AWS account." >&2
  exit 1
fi

if ! grep -Fq "\"$account.dkr.ecr.$region.amazonaws.com/mlflow-server:" Dockerrun.aws.json; then
  echo "Set Dockerrun.aws.json to the published mlflow-server image in $account/$region." >&2
  exit 1
fi

parameter_arn="arn:aws:ssm:$region:$account:parameter/mlflow"
environment_namespace="aws:elasticbeanstalk:application:environment"
secrets_namespace="aws:elasticbeanstalk:application:environmentsecrets"

valid_artifact_uri() {
  local uri="$1" bucket LC_ALL=C
  # MLflow parses raw query and fragment delimiters before choosing the S3 key.
  [[ $uri == s3://*/artifacts && $uri != *'?'* && $uri != *'#'* ]] || return 1

  # Keep control characters and YAML line breaks out of the saved configuration.
  case "$uri" in
    *[[:cntrl:]]*|*$'\xc2'[$'\x80'-$'\x9f']*|*$'\xe2\x80'[$'\xa8\xa9']*|*$'\xef\xbf'[$'\xbe\xbf']*)
      return 1 ;;
  esac

  bucket="${uri#s3://}"
  bucket="${bucket%%/*}"
  [[ $bucket =~ ^[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$ &&
     $bucket != *..* && ! $bucket =~ ^([0-9]{1,3}\.){3}[0-9]{1,3}$ ]] || return 1
  case "$bucket" in
    xn--*|sthree-*|amzn-s3-demo-*|*-s3alias|*--ol-s3|*.mrap|*--x-s3|*--table-s3)
      return 1 ;;
    *-an)
      # Account-regional names include the 12-digit account and full region code.
      [[ $bucket =~ ^[a-z0-9][a-z0-9.-]*-[0-9]{12}-[a-z]+(-[a-z0-9]+)+-[0-9]+-an$ ]] || return 1 ;;
  esac
}

check_redeploy_settings() {
  local query configured names artifact setting
  local missing=()
  # Keep the raw artifact value private. Count scoped rows before taking its value.
  query="ConfigurationSettings[?DeploymentStatus == 'deployed'].OptionSettings[] | {
    artifact: [?Namespace == '$environment_namespace' && OptionName == 'MLFLOW_ARTIFACTS_DESTINATION'],
    names: [? (Namespace == '$environment_namespace' && (
      (OptionName == 'MLFLOW_SERVER_ALLOWED_HOSTS' && Value == '$host') ||
      (OptionName == 'MLFLOW_SERVER_CORS_ALLOWED_ORIGINS' && Value == 'http://$host')
    )) || (Namespace == '$secrets_namespace' && (
      (OptionName == 'MLFLOW_BACKEND_STORE_URI' && Value == '$parameter_arn/backend-store-uri') ||
      (OptionName == 'MLFLOW_AUTH_DATABASE_URI' && Value == '$parameter_arn/auth-database-uri') ||
      (OptionName == 'MLFLOW_AUTH_ADMIN_PASSWORD' && Value == '$parameter_arn/auth-admin-password') ||
      (OptionName == 'MLFLOW_FLASK_SERVER_SECRET_KEY' && Value == '$parameter_arn/flask-secret-key')
    ))].OptionName
  } | join('|', [join(' ', names),
    (length(artifact) == \`1\` && type(artifact[0].Value) == 'string' && artifact[0].Value) || '', ''])"

  # Frame the artifact so shell capture preserves any trailing newline in the value.
  # Replace NUL with a rejected control byte before Bash reads the response.
  if ! configured="$(aws elasticbeanstalk describe-configuration-settings \
      --application-name mlflow-server --environment-name "$env_name" --region "$region" \
      --query "$query" --output text 2>/dev/null | LC_ALL=C tr '\000' '\001')" ||
      [[ $configured != *'|'*'|' ]]; then
    echo "Could not verify the deployed MLflow settings for $env_name. Redeploy cancelled." >&2
    return 1
  fi

  names="${configured%%|*}"
  artifact="${configured#*|}"
  if ! valid_artifact_uri "${artifact%|}"; then
    missing+=(MLFLOW_ARTIFACTS_DESTINATION)
  fi
  for setting in MLFLOW_SERVER_ALLOWED_HOSTS MLFLOW_SERVER_CORS_ALLOWED_ORIGINS \
      MLFLOW_BACKEND_STORE_URI MLFLOW_AUTH_DATABASE_URI MLFLOW_AUTH_ADMIN_PASSWORD \
      MLFLOW_FLASK_SERVER_SECRET_KEY; do
    if [[ " $names " != *" $setting "* ]]; then
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
    printf 'Environment %s has CNAME %q; expected %s. Redeploy cancelled.\n' \
      "$env_name" "${live_host:-unavailable}" "$host" >&2
    exit 1
  fi

  check_redeploy_settings
  echo "Environment ${env_name} exists: redeploying."
  eb deploy "$env_name" --region "$region"
else
  if ! valid_artifact_uri "${MLFLOW_ARTIFACTS_DESTINATION:-}"; then
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
  # Escape single quotes in the YAML scalar without changing the S3 key prefix.
  artifact_yaml="${MLFLOW_ARTIFACTS_DESTINATION//\'/\'\'}"
  (umask 077; cat > "$config") <<EOF
AWSConfigurationTemplateVersion: 1.1.0.0
OptionSettings:
  $environment_namespace:
    MLFLOW_ARTIFACTS_DESTINATION: '$artifact_yaml'
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
