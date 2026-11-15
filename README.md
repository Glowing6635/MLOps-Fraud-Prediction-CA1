# MLOps-Fraud-Prediction-CA1

Fraud prediction coursework project for MLOps CA1. The repository currently
provides a Python package scaffold and a notebook development environment;
the CLI prints a placeholder greeting.

## Quick start

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and Git,
then run these commands from your cloned project directory:

```bash
uv sync --locked
git config --local core.hooksPath .githooks

# Verify installation; this prints the installed package path.
uv run python -c "import mlops_fraud_prediction_ca1; print(mlops_fraud_prediction_ca1.__file__)"
```

The project requires Python 3.12. `uv sync` installs the project and its
development dependencies, including NotebookHelpers from GitHub. Use `uv run`
to run commands in the project environment without activating it manually.

See [Setup](docs/setup.md) to open a notebook, select its kernel, or use the CLI.
The pre-commit hook syncs saved notebook edits and stages their tracked Python
representations for the commit, including pairs you did not stage manually.
See [Git hooks](docs/setup.md#enable-git-hooks) for setup and conflict handling.

## Directories

| Path                              | Purpose                                                               |
| --------------------------------- | --------------------------------------------------------------------- |
| `src/mlops_fraud_prediction_ca1/` | Installable Python package and reusable project code                  |
| `notebooks/`                      | Notebook work; tracked Python representations use `# %%` cell markers |
| `data/`                           | Local input data                                                      |
| `output/`                         | Generated results                                                     |
| `docs/`                           | Project documentation                                                 |
| `.githooks/`                      | Shared Git hooks, enabled once per clone                              |
| `scripts/`                        | Repository maintenance checks                                         |

## Documentation

- [Setup](docs/setup.md) — Install the environment and start working.
- [Notebook syncing](docs/notebook-syncing.md) — Keep notebook edits ready for Git.
- [Contributing](docs/contributors/README.md) — Follow the branch and pull request rules.
- [Documentation index](docs/MAIN.md) — Find instructions by task.
