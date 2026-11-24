# Runbook

How to create the team's cloud services in the AWS sandbox account and set them up. Every service uses only the sandbox's own `LabRole` and `LabInstanceProfile`. Nothing creates an IAM user, role or policy.

## RDS database

Run these once, with the AWS CLI and the sandbox credentials in `~/.aws/credentials`, which you paste in fresh each session. In Git Bash, run `export MSYS_NO_PATHCONV=1` first.

1. Create a parameter group. The MLflow server's database setup creates triggers, and RDS refuses them unless `log_bin_trust_function_creators` is on. `require_secure_transport` makes RDS refuse unencrypted connections.

   ```
   aws rds create-db-parameter-group --db-parameter-group-name fraud-mysql84 --db-parameter-group-family mysql8.4 --description "Allow MLflow's triggers"
   aws rds modify-db-parameter-group --db-parameter-group-name fraud-mysql84 --parameters "ParameterName=log_bin_trust_function_creators,ParameterValue=1,ApplyMethod=immediate"
   aws rds modify-db-parameter-group --db-parameter-group-name fraud-mysql84 --parameters "ParameterName=require_secure_transport,ParameterValue=1,ApplyMethod=immediate"
   ```

2. Choose the admin password, at least 12 characters, and keep it out of the command history:

   ```
   read -s DB_ADMIN_PASSWORD
   ```

3. Create the instance, and wait until it is ready, which takes about 10 minutes:

   ```
   aws rds create-db-instance --db-instance-identifier fraud-db --engine mysql --engine-version 8.4 --db-instance-class db.t3.micro --allocated-storage 20 --storage-type gp3 --storage-encrypted --deletion-protection --db-name fraud --master-username fraud_admin --master-user-password "$DB_ADMIN_PASSWORD" --db-parameter-group-name fraud-mysql84 --publicly-accessible --backup-retention-period 1 --no-multi-az
   aws rds wait db-instance-available --db-instance-identifier fraud-db
   aws rds describe-db-instances --db-instance-identifier fraud-db --query "DBInstances[0].[Endpoint.Address,VpcSecurityGroups[0].VpcSecurityGroupId]" --output text
   ```

   The last command prints the instance's address and its security group. Deletion protection stops anyone deleting the instance by mistake. To delete it on purpose, first turn it off with `aws rds modify-db-instance --db-instance-identifier fraud-db --no-deletion-protection`.

4. Let your laptop reach port 3306. Find your public address with `curl -s https://checkip.amazonaws.com`, then:

   ```
   aws ec2 authorize-security-group-ingress --group-id <security group> --protocol tcp --port 3306 --cidr <your address>/32
   ```

5. In `.env`, set `DB_HOST` to the instance's address, `DB_NAME=fraud`, `DB_USER=fraud_admin` and `DB_PASSWORD` to the admin password. Then create the tables, and a read-only user for each engineer. Each command asks for that user's password, which you then give to the engineer privately:

   ```
   uv run rds-schema tables
   uv run rds-schema reader richard
   uv run rds-schema reader spiralsky
   ```

   All three are safe to re-run. Running `reader` again for the same name changes that user's password.

6. Each engineer sets `DB_USER=<name>_reader` and their own password in their `.env`, and runs `uv run rds-schema check` from their own laptop. It lists the four tables and shows the user can only read (`GRANT SELECT`).

7. Make the user that loads batches. With the admin settings still in `.env`, run this; it asks for the new user's password:

   ```
   uv run rds-schema writer richard
   ```

   It is safe to re-run, and running it again changes the password.

## Loading a batch

The base files and every later batch load with the same command. Set `DB_USER=<name>_writer` and its password in `.env`, then give the three files as URLs or paths:

```
uv run load-batch data/customers_base.csv data/orders_base.csv data/labels_base.csv
```

The files must be named `customers_<batch>.csv`, `orders_<batch>.csv` and `labels_<batch>.csv`, and `<batch>` becomes the batch ID. Before touching the database, the command checks the files. It stops, and changes nothing, if a download fails after 3 retries, a file is not CSV, a column is missing or renamed, or a file has no rows. An extra column is only a warning, and it is not loaded.

Load `base` first, then each batch in the order it arrived, because the database treats the batch loaded last as the newest.

It then prints the rows added, already stored and rejected per table. Loading the same batch again is safe and adds nothing. For the base files, the counts should match the kickoff data analysis (issue #3): 901,312 customers, 949,889 orders and 949,872 labels added, with 444 labels rejected for "no matching order" and 2 for "no matching customer". On a laptop the first load took about 11 minutes, and a rerun about 2. "Loading rules" in `docs/data-dictionary.md` lists what is rejected and why.

`docs/data-dictionary.md` explains every table and column. The schema is frozen before data loading (issue #20) starts: after that, any change to `schema.sql` needs the team's agreement first.
