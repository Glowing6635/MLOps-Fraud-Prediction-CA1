# MLOps-Fraud-Prediction-CA1

### Documentation
[Project documentation](docs/MAIN.md)
[Contributor guide](docs/contributors/README.md)

### Setup
`uv` 0.12 is the package manager for this environment. On Windows, install it in PowerShell with `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/0.12.24/install.ps1 | iex"`, then open a new terminal.

1. `uv sync` creates `.venv` with Python 3.12 and every pinned dependency.
2. Copy `.env.example` to `.env` and fill in the database and MLflow settings. Your MLflow login is your own; the admin creates it as in `mlflow-server/README.md`, "Logins". `.env` is git-ignored: never commit it. Variables set in the environment override it.
3. Put the three base CSV files in `data/`. They are git-ignored.
4. `uv run pytest` runs the tests.
5. `uv run log-test-run` logs a test run to the MLflow server in `.env`, to check your laptop can reach it. `mlflow-server/README.md` says how to deploy the server.

### Batch command

From the repository root, run `uv sync` and configure `.env` or your environment
using `.env.example`. The preflight requires the database host, name, username
and password, plus the MLflow tracking URI, experiment name, username
(`MLFLOW_TRACKING_USERNAME`) and password (`MLFLOW_TRACKING_PASSWORD`).

Set `CUSTOMERS_URL`, `ORDERS_URL` and `LABELS_URL` to the corresponding CSV URLs.
Pass them in that order and quote each URL to preserve its query string.

```sh
uv run run-batch "$CUSTOMERS_URL" "$ORDERS_URL" "$LABELS_URL"
```

The command first prints the target database host, database name and MLflow experiment, and the run's `etl_run_id`, which tags every MLflow run it logs. It then runs five stages and prints each one's time against its budget (D27-01 in `docs/decision-log.md`), with `OVER BUDGET` after any that overran:

1. Fetch and validate: downloads and checks the three files, then prints the batch ID and the rows in each file, which is the batch size for timing.
2. Clean and load: cleans the batch and loads it as `uv run load-batch` does (`docs/runbook.md`, "Loading a batch").
3. Read training data: reads the labelled training and validation windows from the database.
4. Train and log: trains the baseline model on them and logs the run to MLflow. Per-batch retraining of every candidate (#25) replaces this once it is merged.
5. Report: prints each table's rows before and after, the number of MLflow runs logged with this `etl_run_id`, and a link that lists them on the MLflow server.

It ends with the total time against the 180-second limit. A failed stage prints `failed after` with its time, and the command exits with status 1. A failure before the load ends leaves the database as it was. Running the command again for a loaded batch adds no rows and logs new runs.

To time a run as the three-minute check (#27) asks, rebuild the rehearsal database (#24), point `.env` at it, and time the whole command, start-up included, in PowerShell:

```powershell
Measure-Command { uv run run-batch $env:CUSTOMERS_URL $env:ORDERS_URL $env:LABELS_URL | Out-Default }
```

### Checks
CI (`.github/workflows/ci.yml`) runs on every pull request: `ruff check`, `ruff format --check`, `pytest`, a licence and vulnerability check of the dependencies, a gitleaks scan of the full history, and a check that the pull request's source and target branches follow `docs/contributors/README.md`. Run the first three locally with `uv run`.

Repository rules block direct pushes, force-pushes and deletion on `main`, `dev` and user branches. A pull request into `main` or `dev` needs approval from someone other than its last pusher and resolved review threads, and uses the merge method the contributor guide names.

### Secret scan
`gitleaks git --log-opts="--all"` scans the full history of every branch for secrets.

### Directories
`/data` Contains data files for training
`/output` Contains model outputs

[Fixed temporal reads](docs/time-split.md) — protected CSV reads and SQL contract.

[Deterministic cleaning](docs/cleaning.md) — reusable rules and row-preserving table API.
