# MLOps-Fraud-Prediction-CA1

### Documentation
[Project documentation](docs/MAIN.md)
[Contributor guide](docs/contributors/README.md)

### Setup
`uv` 0.12 is the package manager for this environment. On Windows, install it in PowerShell with `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/0.12.24/install.ps1 | iex"`, then open a new terminal.

1. `uv sync` creates `.venv` with Python 3.12 and every pinned dependency.
2. Copy `.env.example` to `.env` and fill in the database and MLflow settings. `.env` is git-ignored: never commit it. Variables set in the environment override it.
3. Put the three base CSV files in `data/`. They are git-ignored.
4. `uv run pytest` runs the tests.

### Secret scan
`gitleaks git --log-opts="--all"` scans the full history of every branch for secrets.

### Directories
`/data` Contains data files for training
`/output` Contains model outputs

