# Runbook

How to create the team's cloud services in the AWS sandbox account, bring them back after the sandbox stops, and the checks to repeat before the demo. The MLflow server's set-up steps are in `mlflow-server/DEPLOY.md`.

Every service uses only the sandbox's own `LabRole` and `LabInstanceProfile`. Nothing creates an IAM user, role or policy. Every command is for Bash: on Windows, run them in Git Bash, not PowerShell.

## Who runs the services

| Item | Record |
| --- | --- |
| Sandbox account hosting RDS, MLflow and S3 | Richard Go's AWS Academy sandbox, region `us-east-1` |
| Who can start it | Richard Go only |
| Demo machine | Richard Go's second computer, which he has tested with the services |
| Demo-site network | `SPStudent` |
| Backup hotspot | Richard Go's phone hotspot, `Samsung Galaxy Note20 Ultra` |

Only Richard can start the sandbox, so he is a single point of failure on demo day. The backup plan covers that. If Richard can't start his sandbox, because he is absent or can't sign in, AWS Academy is down, or the lab's budget or session time is used up, Tung Shi An signs in to his own AWS Academy sandbox and rebuilds the services there:

1. Create RDS by the "RDS database" steps below, which take about 10 minutes.
2. Load the three base files with `uv run load-batch`, which takes about 11 minutes.
3. Deploy MLflow with his deploy script from #26. That script is in his draft pull request #158, which isn't merged yet.

Logged runs and registered models don't carry over to his sandbox unless they're exported from Richard's first.

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

`docs/data-dictionary.md` explains every table and column. The schema is frozen before data loading (issue #20) starts: after that, any change to `schema.sql` needs the team's agreement first.

### Who can reach port 3306

The instance has a public address, because the engineers connect from home and campus networks. Its security group lets in only:

- one rule per engineer's current public address, each a single address (`/32`), added with step 4;
- the MLflow server's Beanstalk instances, by naming their security group as the source, so the rule still holds when Beanstalk replaces an instance.

We chose this over opening port 3306 to every address, which would leave the database's login exposed to the whole internet with only a password in the way. A VPN or a jump host would hide the database completely, but the sandbox gives no way to keep one running between sessions.

When an engineer's address changes, add a rule for the new one and delete the old one with `aws ec2 revoke-security-group-ingress`, using the same options as step 4. `aws ec2 describe-security-groups --group-ids <security group>` lists the rules.

The database also refuses unencrypted connections, because `require_secure_transport` is on in the `fraud-mysql84` parameter group. The shared connection code encrypts and checks the RDS certificate, so nothing else needs setting up.

## Between sessions

Ending the sandbox session doesn't stop RDS. On 17 Nov it kept running and logged no events, while Beanstalk's instance was terminated and replaced. So RDS keeps spending the sandbox budget until someone stops it. When the next session is a day or more away, stop it with `aws rds stop-db-instance --db-instance-identifier fraud-db`. A stopped RDS starts again by itself after 7 days, so stop it again then. At the start of the next session, start it before anything else, as the restart drill's step 2 shows, since it takes several minutes. The MLflow server waits up to 15 minutes for it.

Before each deadline, back up the `fraud`, `mlflow` and `mlflow_auth` databases with `mysqldump`, and the artifact bucket with `aws s3 sync s3://<bucket> <folder>`. Pressing Reset in the sandbox deletes everything in it.

## Before the sandbox stops

Each machine that will run the drill, which is each engineer's laptop and the demo machine, runs `uv run redeploy-check record` once, because the drill reads the `output/redeploy_check.json` it saves there. Each time, it logs a run and registers a new version of the `redeploy-check` model, so a second engineer's run changes nothing the first relies on. Reset deletes the recorded run and model, so after a Reset, run it again before the next drill.

## Restart drill

1. Start the sandbox.
2. Check whether RDS was stopped:

   ```
   aws rds describe-db-instances --db-instance-identifier fraud-db --query "DBInstances[0].DBInstanceStatus" --output text
   ```

   If it prints `stopped`, start it and time how long it takes to become available, since the drill's times start after it:

   ```
   aws rds start-db-instance --db-instance-identifier fraud-db
   time aws rds wait db-instance-available --db-instance-identifier fraud-db
   ```

   The MLflow server gives up if RDS isn't ready within about 15 minutes of its start, and may then stay down. If the MLflow URL still fails once RDS is `available`, restart the server, which starts it again and waits for RDS again:

   ```
   aws elasticbeanstalk restart-app-server --environment-name mlflow
   ```
3. Straight away, run `uv run restart-drill`. It waits until RDS answers, then MLflow, then checks that the recorded model loads and reproduces its predictions, and logs a run named `restart-drill` with the times, tagged `purpose=restart-drill` so queries for model runs can skip it. It prints the time from its own start to each step, and gives up after 30 minutes.
4. Add a row to "Drill results" below, with the RDS start-up wait from step 2 as its own stage, not added to the other times.

### Drill results

Times are minutes and seconds from the start of `uv run restart-drill`, so 1:04 is one minute and four seconds. The RDS start-up wait is the time `time aws rds wait db-instance-available` printed in step 2, before the drill started. These results move to the team's failure log once #31 adds it.

| Date | RDS stopped for | RDS start-up wait | RDS answers | MLflow answers | Model check passes | Total | MLflow run |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 17 Nov 2026 | Not stopped | None | 0:06 | 0:47 | 0:58 | 1:04 | `39e1099e1d374b048c8a1d8ee4bafec9` |

On 17 Nov, ending the session terminated the MLflow server's Beanstalk instance, and Beanstalk launched a new one about a minute before the drill. That minute isn't in the times. The drill ran from Richard Go's laptop, and the network wasn't recorded.

Demo-day start time: TO RECORD, from the slowest total of a drill after RDS was stopped, plus its RDS start-up wait, plus a margin of TO RECORD minutes.

## Demo day

- The day before, end and start the sandbox, run the drill, and change nothing afterwards. Note the image tag in `mlflow-server/Dockerrun.aws.json`, so the version that worked can be deployed again. Keep RDS running from then until the demo is over.
- Before any drill or live query on the demo-site network or the backup hotspot, let that network reach port 3306. Without the rule, the drill waits 30 minutes and then fails. Find the address with `curl -s https://checkip.amazonaws.com` and add it with the `aws ec2 authorize-security-group-ingress` command from step 4 of "RDS database". The hotspot's address changes between sessions and can change during one, so add its rule each time, and if the drill hangs at the RDS step, check the address again and add its rule. Remove each rule afterwards with `aws ec2 revoke-security-group-ingress`, as "Who can reach port 3306" says.
- Start the sandbox at least an hour early, and run `uv run restart-drill` from the demo machine and from Richard Go's laptop, the spare.
- Just before presenting, click Start Lab again, which resets the session timer, so the session can't end during the demo.
- Have a fallback ready: a recording of the MLflow web UI, and a copy of the champion model on the demo laptop.

## Reaching RDS from each laptop

Each laptop runs `uv run restart-drill` once on the demo-site network and once on the backup hotspot, after adding that network's port 3306 rule (see "Demo day").

| Laptop | Demo-site network | Backup hotspot |
| --- | --- | --- |
| Richard Go's laptop | Reached RDS and MLflow, tested before 25 Nov 2026 | Reached RDS and MLflow, tested before 25 Nov 2026 |
| Richard Go's second computer, the demo machine | Not tested | Not tested |
| Tung Shi An's laptop | Not recorded | Not recorded |

## Sandbox budget

At every sprint review, open the sandbox's start page, note the budget used and remaining, and add a row here. If the remaining budget would not last until the release, stop the services between work sessions.

| Sprint review | Used | Remaining |
| --- | --- | --- |
| Sprint 2, 24 Nov 2026 | $15 of $50 | About $35 |
