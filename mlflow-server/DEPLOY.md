# Deploying the MLflow server

How to set up the MLflow server's AWS services, build its image and deploy it. `README.md` lists the files and settings.

## Set up RDS and S3, once

1. MLflow's database setup creates triggers, which RDS refuses while binary logging is on. Attach a parameter group, `fraud-mysql84`, with `log_bin_trust_function_creators` and `require_secure_transport` both set to `1`. The second makes RDS refuse unencrypted connections. Attach the group when you create the instance, as the RDS steps in `docs/runbook.md` (added by #10) do, and no reboot is needed. Reboot the instance only when you attach the group to one that already exists.
2. As the instance's admin user, run `CREATE DATABASE mlflow; CREATE DATABASE mlflow_auth;`, create a `mlflow` user, and grant it `ALL` on both.
3. Let the instance's security group accept port 3306 from the Beanstalk instance's security group.
4. Create an S3 bucket in `us-east-1` with public access blocked. `LabInstanceProfile` gives the server access.

Laptops check the RDS certificate against `src/mlops_fraud_prediction_ca1/rds-global-bundle.pem`, AWS's public certificate bundle for RDS. It was committed to the repository on 18 Nov 2026, from https://truststore.pki.rds.amazonaws.com/global/global-bundle.pem, the same address the `Dockerfile` uses. To check it, compare its SHA-256 with the checksum in the `Dockerfile`'s `ADD` line.

## Build and push the image

You need Docker, and the sandbox credentials in `~/.aws/credentials`, pasted fresh each session. Tags in the `mlflow-server` ECR repository can't be overwritten, so each new image gets the next tag: `<MLflow version>-r<n>`. Then put that tag in `Dockerrun.aws.json`. From this folder:

```
aws ecr create-repository --repository-name mlflow-server --image-tag-mutability IMMUTABLE --image-scanning-configuration scanOnPush=true
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account>.dkr.ecr.us-east-1.amazonaws.com
docker buildx build --platform linux/amd64 --load -t <account>.dkr.ecr.us-east-1.amazonaws.com/mlflow-server:<tag> .
docker push <account>.dkr.ecr.us-east-1.amazonaws.com/mlflow-server:<tag>
```

The first command is needed only once. `--platform linux/amd64` matches the t3 instance, even when the laptop has an ARM processor. After changing `requirements.in`, relock with `uv pip compile requirements.in --python-version 3.12 --python-platform x86_64-manylinux_2_28 --generate-hashes --no-header -o requirements.txt`.

## Deploy

You need the EB CLI (`uv tool install --python 3.12 awsebcli`). Store each secret once with `aws ssm put-parameter --name <parameter> --type SecureString --value <value>`. Then, from this folder:

```
eb init mlflow-server --platform docker --region us-east-1
eb create mlflow --single --instance_type t3.small --service-role LabRole --instance_profile LabInstanceProfile --cname <prefix>
eb setenv <each environment property in README.md>
```

Point each secret's variable at its parameter with `aws elasticbeanstalk update-environment --environment-name mlflow --option-settings Namespace=aws:elasticbeanstalk:application:environmentsecrets,OptionName=<variable>,Value=<parameter ARN>`. `LabRole` and `LabInstanceProfile` are the sandbox's own, so nothing creates an IAM resource. `LabInstanceProfile` also lets the instance pull the image from ECR and read the parameters. `eb status` then shows `Health: Green`. Later changes go out with `eb deploy`, and everything stored stays.
