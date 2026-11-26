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

The command prints the target database host, database name, and MLflow experiment. Until the fetch and validate stage (#19) is integrated, it then exits with status 1 before downloading data or connecting to either service.

### Final test-window scoring

`uv run final-test --data-dir data` scores the final champion and the rule once on the test window, with the settings committed in `config/final_test.toml`. It refuses to run a second time. `docs/final-test.md` gives the steps before and after it.

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
