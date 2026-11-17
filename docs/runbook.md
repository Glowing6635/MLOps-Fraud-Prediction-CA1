# Runbook

How to create the team's cloud services in the AWS sandbox account, bring them back after the sandbox stops, and the checks to repeat before the demo. The MLflow server's setup is in `mlflow-server/README.md`.

Every service uses only the sandbox's own `LabRole` and `LabInstanceProfile`. Nothing creates an IAM user, role or policy.

## Who runs the services

| Item | Record |
| --- | --- |
| Sandbox account hosting RDS, MLflow and S3 | Richard Go's AWS Academy sandbox, region `us-east-1` |
| Who can start it | Richard Go only |
| Demo machine | TO RECORD |
| Demo-site network | TO RECORD |
| Backup hotspot | TO RECORD |

Only Richard can start the sandbox, so he is a single point of failure on demo day. Backup plan: TO RECORD.

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

5. In `.env`, set `DB_HOST` to the instance's address, `DB_NAME=fraud`, `DB_USER=fraud_admin` and `DB_PASSWORD` to the admin password. Then create the tables, and a read-only user for each engineer. Each `reader` command asks for that user's password, which you then give to the engineer privately:

   ```
   uv run rds-schema tables
   uv run rds-schema reader richard
   uv run rds-schema reader spiralsky
   ```

   All three are safe to re-run. Running `reader` again for the same name changes that user's password.

6. Each engineer sets `DB_USER=<name>_reader` and their own password in their `.env`, and runs `uv run rds-schema check` from their own laptop. It lists the five tables and shows the user can only read (`GRANT SELECT`).

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

It then prints the rows added, already stored and rejected per table. Loading the same batch again is safe and adds nothing. Run one load at a time: while a long load runs, a second load stops with MySQL's "Lock wait timeout exceeded" error, or with a duplicate-key error if both add the same row, and changes nothing, so run it again after the first finishes. For the base files, the counts should match the kickoff data analysis (issue #3): 901,312 customers, 949,889 orders and 949,872 labels added, with 444 labels rejected for "no matching order" and 2 for "no matching customer". On a laptop the first load took about 11 minutes, and a rerun about 2. "Loading rules" in `docs/data-dictionary.md` lists what is rejected and why.

`docs/data-dictionary.md` explains every table and column. The schema is frozen before data loading (issue #20) starts: after that, any change to `schema.sql` needs the team's agreement first.

## Running a batch

`uv run run-batch` loads a batch from three URLs, retrains and reports, as the README's "Batch command" says.

The command first prints the target database host, database name and MLflow experiment, and the run's `etl_run_id`, which tags every MLflow run it logs. It then runs five stages and prints each one's time against its budget (D27-01 in `docs/decision-log.md`), with `OVER BUDGET` after any that overran:

1. Fetch and validate: downloads and checks the three files, then prints the batch ID and the rows in each file, which is the batch size for timing.
2. Clean and load: cleans the batch and loads it as `uv run load-batch` does ("Loading a batch" above).
3. Read training data: reads the labelled training and validation windows from the database.
4. Train and log: trains the baseline model on them and logs the run to MLflow. Per-batch retraining of every candidate (#25) replaces this once it reaches `dev`.
5. Report: prints each table's rows before and after, the number of MLflow runs logged with this `etl_run_id`, and a link that lists them on the MLflow server.

It ends with the total time against the 180-second limit. That total starts after the settings are read, so it leaves out start-up. A failed stage prints `failed after` with its time against its budget, the total follows, and the command exits with status 1. A failure before the load ends leaves the database as it was. Running the command again for a loaded batch adds no rows and logs new runs.

To time a run as the three-minute check (#27) asks, rebuild the rehearsal database with #24's reset command (pull request #160 until it merges), point `.env` at it, and time the whole command in PowerShell. The exit status it prints last must be 0:

```powershell
Measure-Command { uv run run-batch $env:CUSTOMERS_URL $env:ORDERS_URL $env:LABELS_URL | Out-Default }; $LASTEXITCODE
```

## Daily review queue

After a batch is loaded, this ranks its orders with the champion: the version of the registered model `fraud-candidate` that its alias `champion` names in MLflow. Nothing is retrained. Use the same `.env` as for loading a batch, with the `<name>_writer` user.

```
uv run review-queue <batch>
```

The first time, `uv run rds-schema tables` with the admin settings must add the `review_queue` table. It leaves the other tables alone.

The command prints the champion's version, then:

1. The orders ranked, the orders the holdout setting withholds, and the orders without a label. The holdout setting in `config/time_split.toml` withholds orders in the locked test window, orders dated before the training data starts, and orders without a date. The champion is scored on the `orders` columns it was trained on, which scikit-learn records with the model, so an order without a label is scored like the rest. A model that doesn't record its input columns as plain column names stops the command with status 1.
2. The queue: each day's top K orders (K = 250, D5-01 in `docs/decision-log.md`), or all of them on a day with fewer, stored in `review_queue` with the model version. A rerun for the same batch and model version writes nothing and logs the stored queue.
3. Precision at K: the share of fraud among queued orders with a valid label. The labels are joined only after the queue is stored. Queued orders without a valid label are counted but left out of the share.

It logs one MLflow run, `review-queue-<batch>-v<version>`, holding the queue as a CSV (date, country, order ID, the customer ID from the labels file, and score), the counts and precision at K. If the batch is not loaded, or the holdout setting withholds all its orders, it stops with status 1 and writes nothing. If two runs for one batch and model version overlap, the second may stop with status 1 on a duplicate key, and the first one's queue stands either way. If MLflow fails after the queue is stored, run the command again: it writes nothing and finishes the logging. Precision from a rehearsal only shows the command works; never quote it as a result.

### Who can reach port 3306

The instance has a public address, because the engineers connect from home and campus networks. Its security group lets in only:

- one rule per engineer's current public address, each a single address (`/32`), added with step 4;
- the MLflow server's Beanstalk instances, by naming their security group as the source, so the rule still holds when Beanstalk replaces an instance.

We chose this over opening port 3306 to every address, which would leave the database's login exposed to the whole internet with only a password in the way. A VPN or a jump host would hide the database completely, but the sandbox gives no way to keep one running between sessions.

When an engineer's address changes, add a rule for the new one and delete the old one with `aws ec2 revoke-security-group-ingress`, using the same options as step 4. `aws ec2 describe-security-groups --group-ids <security group>` lists the rules.

The database also refuses unencrypted connections, because `require_secure_transport` is on in the `fraud-mysql84` parameter group. The shared connection code encrypts and checks the RDS certificate, so nothing else needs setting up.

## Between sessions

RDS keeps running, and spending the sandbox budget, after the sandbox stops. When the next session is a day or more away, stop it with `aws rds stop-db-instance --db-instance-identifier fraud-db`. RDS starts it again by itself after 7 days, so stop it again then. At the start of the next session, start it with `aws rds start-db-instance` before anything else, since it takes several minutes. The MLflow server waits up to 15 minutes for it.

Before each deadline, back up the `fraud`, `mlflow` and `mlflow_auth` databases with `mysqldump`, and the artifact bucket with `aws s3 sync s3://<bucket> <folder>`. Pressing Reset in the sandbox deletes everything in it.

## Before the sandbox stops

Run `uv run redeploy-check record` once, from any laptop. It saves the model's predictions to `output/redeploy_check.json` on that laptop, so run the drill from the same laptop.

## Restart drill

1. Start the sandbox.
2. Straight away, run `uv run restart-drill`. It waits until RDS answers, then MLflow, then checks that the recorded model loads and reproduces its predictions, and logs a run named `restart-drill` with the times. It prints the time from its own start to each step, and gives up after 30 minutes.
3. Add the times to the table below.

| Date | Machine and network | RDS | MLflow | Model | Total |
| --- | --- | --- | --- | --- | --- |
| 17 Nov 2026 | Richard Go's laptop, network TO RECORD | 0:06 | 0:47 | 0:58 | 1:04 |

On 17 Nov, ending the sandbox terminated the MLflow server's machine, and Beanstalk launched a new one when the sandbox started, about a minute before the drill. The runs and the model were still there, because they live in RDS and S3. RDS was not stopped: it logged no events. A sandbox stopped overnight may also stop RDS, which takes longer to start, so repeat the drill on such a morning before setting the demo-day start time.

Demo-day start time: TO RECORD, from the slowest total above plus a margin of TO RECORD minutes.

## Demo day

- The day before, end and start the sandbox, run the drill, and change nothing afterwards. Note the image tag in `mlflow-server/Dockerrun.aws.json`, so the version that worked can be deployed again. Keep RDS running from then until the demo is over.
- Start the sandbox at least an hour early, and run `uv run restart-drill` from both laptops.
- Just before presenting, click Start Lab again, which resets the session timer, so the session can't end during the demo.
- To query RDS live, add the venue's address to port 3306, and remove it afterwards.
- Have a fallback ready: a recording of the MLflow web UI, and a copy of the champion model on the demo laptop.

## Reaching RDS from each laptop

Each engineer runs `uv run restart-drill` once on the demo-site network and once on the backup hotspot.

| Engineer | Demo-site network | Backup hotspot |
| --- | --- | --- |
| Richard Go | | |
| Tung Shi An | | |

## Sandbox budget

At every sprint review, open the sandbox's start page, note the budget used and remaining, and add a row here. If the remaining budget would not last until the release, stop the services between work sessions.

| Sprint review | Used | Remaining |
| --- | --- | --- |
| | | |
