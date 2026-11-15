# Notebook syncing

Jupytext keeps a notebook and its Python representation in sync:

| File | Contents | Tracked in Git |
| --- | --- | --- |
| `notebooks/main.ipynb` | Cells, execution counts, and outputs | No |
| `notebooks/main.py` | Code and Markdown cells, separated by `# %%` markers | Yes |

The Python representation makes cell changes readable in Git diffs. Notebook
outputs stay local because `.ipynb` files are ignored by Git.

## Before committing

Save your notebook files. The pre-commit hook checks all indexed Python files
under `notebooks/` that have a local `.ipynb` copy:

- Saved notebook-only edits are converted to Python and **automatically
  staged for the same commit**, even if that Python file was not selected.
- Fully staged Python-only edits update the local notebook, preserving its
  outputs. Outputs may be stale; syncing does not execute cells.
- Matching pairs are left alone. Metadata, output changes, trailing cell
  newlines, and empty trailing cells do not trigger syncing.

When **only ignored notebooks changed**, run the hook before committing:

```bash
git hook run pre-commit
git diff --cached
git commit -m "feat(notebooks): update analysis"
```

Git runs the hook again during the commit; matching pairs need no further
changes. With other changes already staged, committing directly also works.
If you commit directly with nothing staged, the hook stages notebook edits,
but Git may still say "nothing to commit" on that first attempt. Retry the
same commit command; the synced Python files are now staged.

The hook checks every pair before writing. It blocks the commit if Python
files have unstaged edits (including partial staging), both representations
have different edits since `HEAD`, a differing notebook is older than its
Python file, or a notebook cannot be read. Save and reconcile both copies,
review the Python diff, stage the whole Python file, and retry. Ordinary
preflight failures leave the pairs and index unchanged. If a conversion or
staging operation fails after syncing begins, inspect `git diff` and
`git diff --cached` before retrying; some files may have been updated.

For a **new notebook**, first sync it manually and stage its Python counterpart:

```bash
uv run jupytext --sync notebooks/new-notebook.ipynb
git add -- notebooks/new-notebook.py
```

Newly staged pairs must match before their first commit. Notebooks with no
indexed Python counterpart are outside the hook's scope; it does not discover
or stage scratch notebooks. Indexed scripts without a local notebook are skipped.

Hook messages use `[notebook-sync]` with `START`, `SYNC`, `STAGED`, and `OK`
for progress; `NOTE` explains the notebook-only retry case, `ERROR` explains
a problem, and `STOP` means the commit is blocked.
Enable hooks once per clone as described in [Setup](setup.md#enable-git-hooks).

## Create the local notebook

First [install the environment](setup.md#install-the-project). On a fresh
clone, create the missing notebook from the tracked script. Run from the
project root:

```bash
uv run jupytext --sync notebooks/main.py
```

Then [open the notebook and select its kernel](setup.md#use-notebooks).

## Sync changes

The hook handles syncing at commit time. To switch representations or
review changes earlier, sync manually during editing:

Save your edits, then run from the project root:

```bash
uv run jupytext --sync notebooks/main.ipynb
```

Passing `notebooks/main.py` syncs the same pair. Jupytext takes cell contents
from the most recently modified representation and preserves outputs from
the `.ipynb` file. It creates a missing counterpart but does not execute
cells or refresh outputs.

Sync before switching representations, and reload the destination file if
it is already open. **Avoid editing both copies independently:** syncing
selects the newer version rather than merging conflicting edits.

To sync another notebook, pass its path, for example:

```bash
uv run jupytext --sync notebooks/testing.ipynb
```

This creates or updates `notebooks/testing.py`; syncing `main` alone does
not update `testing`.

## NotebookHelpers commands

### Reusable code and Python documentation

Keep reusable loaders and other shared operations in
`src/mlops_fraud_prediction_ca1/`. Keep directory selection, logging
configuration, analysis calls, displayed results, plots, and interpretation
in the notebook. Exploratory code can remain in cells until reuse or
readability justifies extracting it.

For all new or modified Python code, annotate function parameters and return
types, including `-> None`, and document modules, classes, and functions with
docstrings visible through `help()` and `python -m pydoc`. Function docstrings
should explain purpose, arguments, returns, relevant exceptions, and side
effects where applicable. Annotate important configuration values and data
objects; avoid comments that merely repeat the code. Apply this convention
to notebook-defined helpers and tests as well as package code.

### Data-analysis loader

`load_base_data(data_dir: str | Path)` in `mlops_fraud_prediction_ca1.data`
returns `(customers, orders, labels)` as pandas DataFrames. The caller selects
the directory; the three filenames are fixed to `customers_base.csv`,
`orders_base.csv`, and `labels_base.csv`. The loader checks all required files
before reading, logs filenames and dimensions at INFO level, and preserves
the source files. It uses pandas' default type and missing-value inference,
without cleaning, joining, or parsing dates explicitly. It propagates read
and parse errors to the caller. Configure logging in the notebook, not in
the package.

The analysis notebook defaults to the repository's `data/` directory when
started from either the project root or `notebooks/`. Set `data_dir` explicitly
to use another directory or start from another working directory. For example:

```python
data_dir: Path = Path("~/datasets/fraud").expanduser()
```

After loading the extension, the notebook uses the linked import form:

```python
%%load_clean
from mlops_fraud_prediction_ca1.data import load_base_data
```

On execution, NotebookHelpers synchronizes the selected definition, executes
it in its package module context, and requests that the editor replace the
cell with editable source and a baseline. This is different from
`%%load_clean default`, which performs an ordinary import. If the editor does
not replace the cell, copy the generated cell it displays. Retain its generated
header: later executions synchronize edits and detect conflicting changes.
Save the notebook and sync its Jupytext pair after accepting the replacement.

Generated headers contain an absolute source path. After moving the checkout
or sharing it with a teammate, preserve any cell-only edits and regenerate
the link using the short import form above. Do not reuse another machine's
generated path.

Inspect the loader's documentation from the project environment:

```bash
uv run --no-sync python -m pydoc mlops_fraud_prediction_ca1.data
```

### Testing approach

Use `unittest` for the current suite; no additional test framework is required.
Use small, multirow CSV fixtures and independently constructed expected frames
with `pandas.testing.assert_frame_equal` to check the loader's complete results,
including inferred dtypes. Check required-file failures and source preservation;
avoid reproducing pandas' own parser test suite or asserting exact log wording.

NotebookHelpers tests cover selected-definition synchronization in both
directions, execution in module context, and preservation of unrelated source.
They emulate accepting the generated cell; editor replacement itself still
needs manual verification. Actual-data execution and review of analysis findings
remain separate from these automated checks.

Run all tests from the project root:

```bash
uv run --no-sync python -B -m unittest discover -s tests -v
```

### IPython magics

Load the extension in a notebook cell:

```python
%load_ext load_clean
```

For example, this cell uses the normal Python import behaviour:

```python
%%load_clean default
from mlops_fraud_prediction_ca1 import main
```

In the locked NotebookHelpers revision, import-based line helpers require an
explicit mode. For example:

```python
%load_clean_preview default from mlops_fraud_prediction_ca1 import main
%load_clean_check default from mlops_fraud_prediction_ca1 import main
```

Omitting `default` in those line-helper examples raises a usage error.
Magics stay in notebook cells when synced; the Git hook preserves them
without executing them. Run the notebook in an IPython kernel, rather than
executing its Python representation as an ordinary script.

### Pairing and syncing CLI

NotebookHelpers 0.0.3 offers pairing and syncing commands. Each takes the
path of one existing notebook file, not a directory:

```bash
# For a saved notebook script with no counterpart yet:
uv run load-clean pair --source notebooks/main.py

# For a pair whose metadata records ipynb,py:percent:
uv run load-clean sync --source notebooks/main.py
```

`pair` refuses to overwrite an existing counterpart. `sync` requires recorded
pairing metadata in both files when both exist. If the main files lack this
metadata, use the direct Jupytext commands above. Do not delete either copy
to force pairing; back up and reconcile existing copies before changing
pairing metadata.

See [Update NotebookHelpers](setup.md#update-notebookhelpers) to fetch
upstream changes into the lockfile.

## Configuration

The root-level `jupytext.toml` sets the pairing format to `ipynb,py:percent`.
It does not run a background sync process or select which notebooks to sync;
the command's file path or the editor watcher's scope determines that.

## Editor automation

The JetBrains watcher in `.idea/watcherTasks.xml` invokes the local Jupytext
CLI. Its `Jupytext` scope in `.idea/scopes/Jupytext.xml` includes only
`notebooks/main.ipynb` and `notebooks/main.py`. To sync on save, enable the
watcher in an editor with File Watchers support. Add both paths to the scope
when including another notebook. Other editors can use the commands above.

**Windows support required:** the shared watcher currently points to
`$ProjectFileDir$/.venv/bin/jupytext`, which is a Linux/macOS path. It will
not launch the project's Jupytext executable in a standard Windows virtual
environment. This affects only PyCharm's optional save automation. Manual
syncing does not use this watcher. Windows users who want the watcher can
select `.venv\Scripts\jupytext.exe`; otherwise, use the manual
`uv run jupytext --sync ...` commands. Verification of manual syncing and
Git hooks on Windows is still needed.

[Documentation index](MAIN.md)
