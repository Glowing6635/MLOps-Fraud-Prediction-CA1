# Setup

Windows support is required because a teammate develops on Windows.
The workflow has been tested on Linux; Windows setup, manual syncing, and
Git hooks still need verification on a Windows machine. The shared
[optional PyCharm watcher needs a Windows-specific executable path](notebook-syncing.md#editor-automation).

## Install the project

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and Git.
From the cloned project directory, run:

```bash
uv sync --locked
uv run python -c "import inspect; import mlops_fraud_prediction_ca1; print(inspect.getfile(mlops_fraud_prediction_ca1))"
```

The second command verifies installation by printing the package path.
The project requires Python 3.12; uv normally downloads a suitable version
if needed. Git and access to GitHub are required to install NotebookHelpers.
`--locked` rejects an outdated lockfile instead of updating it. Resolve any
installation errors before proceeding.

`uv sync` creates `.venv` and installs the project in editable mode with its
development dependencies. Source edits are available to new Python processes
without reinstalling. See [uv's syncing documentation](https://docs.astral.sh/uv/concepts/projects/sync/).

## Enable Git hooks

After installing the environment, enable this repository's hooks once per clone:

```bash
git config --local core.hooksPath .githooks
```

If you already use custom hooks, integrate `.githooks/pre-commit` with them
before changing `core.hooksPath`, which selects one hooks directory.

Before committing, save your notebooks. The hook syncs existing notebook
pairs and automatically stages their Python representations for the commit,
even when you staged only unrelated files. It never stages `.ipynb` outputs
or runs notebook cells. See [Before committing](notebook-syncing.md#before-committing)
for the workflow, first-time notebooks, and conflict handling.

The hook requires `uv` on the Git process's PATH, including commits from an
editor. It does not install dependencies or save unsaved editor buffers.
On Windows, use Git for Windows and make `uv` available to Git; the optional
PyCharm watcher is unrelated to hook execution. Windows execution still
needs verification on a Windows machine.

To preview required syncs without changing files or staging anything:

```bash
uv run --no-sync python scripts/sync_notebooks.py --check
```

For notebook-only changes, run `git hook run pre-commit` before committing so
Git starts with the synced Python files staged. See [Before committing](notebook-syncing.md#before-committing).

See the [Git hooks documentation](https://git-scm.com/docs/githooks).

## Use notebooks

On a fresh clone, first [create the local notebook](notebook-syncing.md#create-the-local-notebook)
from its tracked Python representation. Open it in your editor and select
the project's Python interpreter as its kernel:

- Linux/macOS: `.venv/bin/python`
- Windows: `.venv\Scripts\python.exe`

The environment includes `ipykernel`, IPython, Jupytext, and NotebookHelpers.
Use your editor's notebook support; a JupyterLab browser server is not
included in the environment.

Verify the kernel and package in a notebook cell:

```python
import inspect
import sys
import mlops_fraud_prediction_ca1

print(sys.executable)
print(inspect.getfile(mlops_fraud_prediction_ca1))
```

Load NotebookHelpers when needed:

```text
%load_ext load_clean
```

Follow [Notebook syncing](notebook-syncing.md) when saving changes for Git
or switching between notebook and Python representations.

## Import project code

Keep reusable code in `src/mlops_fraud_prediction_ca1/` and import it from
notebooks and scripts using the package name, without `src.`.
For example, import the existing CSV loader:

```python
from mlops_fraud_prediction_ca1.data import load_base_data

print(load_base_data.__module__)
```

Installing the package makes it importable independently of the working
directory; avoid notebook `sys.path` edits and IDE source-root workarounds.
See the [Python packaging guide](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/).

## Run commands

Use `uv run` from the project directory to run commands in its environment:

```bash
uv run python -c "import sys; print(sys.executable)"
uv run mlops-fraud-prediction-ca1
```

The CLI entry point is configured in `pyproject.toml`. It currently prints a
placeholder greeting; it does not train a model or run a pipeline.

Alternatively, activate the environment in your current shell after syncing:

```bash
# Linux/macOS (bash or zsh)
source .venv/bin/activate
```

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

After activation, `python` uses the environment in that shell. `uv run`
selects the environment for one command without activating the parent shell.
See [uv's project guide](https://docs.astral.sh/uv/guides/projects/).

## Update NotebookHelpers

NotebookHelpers is installed as `load-clean` from its GitHub `main` branch.
The lockfile pins an exact commit, so ordinary `uv sync` does not fetch new
upstream changes. To update the lockfile and install the latest revision, run:

```bash
uv sync --upgrade-package load-clean
```

Review the `uv.lock` diff and test the helpers. Once the update is shared,
teammates install the same revision with `uv sync --locked`.
A version bump or release tag does not automatically update the locked commit.
See [uv's dependency upgrade behavior](https://docs.astral.sh/uv/concepts/projects/sync/#upgrading-locked-package-versions).

## Troubleshooting

- **Imports fail:** confirm installation succeeded, the package name is
  `mlops_fraud_prediction_ca1`, and the notebook kernel uses this project's
  `.venv`. Activating a terminal does not change an already running kernel.
- **Source edits do not appear:** restart the notebook kernel or reload the
  imported module to pick up changes.
- **The notebook hook fails:** run `uv sync --locked` if dependencies are
  missing. For conflicts or unstaged Python edits, follow
  [Before committing](notebook-syncing.md#before-committing).
- **The hook does not run:** check `git config --get core.hooksPath` returns
  `.githooks` and `.githooks/pre-commit` is executable. On Linux/macOS, restore
  its executable permission with `chmod +x .githooks/pre-commit` if needed.
- **`python -m mlops_fraud_prediction_ca1` fails:** the package has no
  `__main__.py`. Use `uv run mlops-fraud-prediction-ca1` for the CLI.

[Documentation index](MAIN.md)
