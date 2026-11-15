# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Data Analysis

# %% [markdown]
# ## Table of contents
#
# - [Purpose and workflow](#purpose-and-workflow)
# - [1. Load and inspect the datasets](#1-load-and-inspect-the-datasets)
#   - [Configure the data directory and logging](#configure-the-data-directory-and-logging)
#   - [Enable NotebookHelpers](#enable-notebookhelpers)
#   - [Import the reusable loader](#import-the-reusable-loader)
#   - [Load the tables and check their dimensions](#load-the-tables-and-check-their-dimensions)
#   - [Inspect dataset columns](#inspect-dataset-columns)
# - [2. Assess data quality](#2-assess-data-quality)
#   - [Missing values](#missing-values)
#   - [Exact duplicates](#exact-duplicates)

# %% [markdown]
# ## Purpose and workflow
#
# This notebook conducts **exploratory data analysis (EDA)** on the three base
# datasets used for fraud prediction to identify issues and patterns that could
# affect model performance and inform data preparation. The goals are to:
#
# - [ ] **Assess data quality** by measuring missing values and duplicates,
#   checking data types, identifying invalid categories and format variants,
#   and examining outliers, skew, and class imbalance to inform preparation.
# - [ ] **Establish how the datasets link** by checking ID uniqueness alone and
#   with country codes, measuring match rates in both directions, checking
#   country-code agreement, and determining how orders connect to customers.
# - [ ] **Identify logical inconsistencies** such as refunds above order values,
#   cancelled or refunded counts above total orders, orders before a customer's
#   first order, negative amounts, and orders without labels, and determine how
#   affected records should be handled.
# - [ ] **Understand fraud patterns over time and by country** by establishing
#   date ranges and formats, comparing weekly and country-level fraud rates,
#   and dating any observed changes to inform later modelling decisions.
# - [ ] **Document evidence and handling decisions** for each check so later
#   cleaning and modelling choices can be traced to the findings, while
#   preserving the original source data.
#
#
# Run the code cells from top to bottom using the project's `.venv` kernel.
# Each check introduces its output before the code and records findings after
# it. Review those findings when the source data changes. Use the table of
# contents or your editor's outline to navigate; editors that support heading
# folding can collapse sections while you work.

# %% [markdown]
# ## 1. Load and inspect the datasets
#
# Select the directory containing the three base CSVs. Loading uses pandas'
# default type and missing-value inference; source files remain unchanged.

# %% [markdown]
# ### Configure the data directory and logging
#
# Use the repository's `data/` directory when running from the project root or
# `notebooks/`. Set `data_dir` to use another location. INFO logging displays
# the files loaded and their dimensions.
#
# %%
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Change data_dir to use another local directory; no loader edits are needed.
project_root: Path = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
data_dir: Path = project_root / "data"

# %% [markdown]
# ### Enable NotebookHelpers
#
# Load the `load_clean` IPython extension to register the `%%load_clean` cell
# magic used below. Run this cell before importing the loader through the magic.
#
# %%
# %load_ext load_clean

# %% [markdown]
# ### Import the reusable loader
#
# The linked import below uses NotebookHelpers 0.0.7 to synchronize
# `load_base_data` with its package definition, including the module's imports.
# On execution, NotebookHelpers refreshes the editable cell and runs the loader
# in its package context. Keep the identifying import (marked `# noqa: F401`)
# and generated `# Referenced Source: sha256:…` comment so the source can be
# located and conflicting edits detected. Data loading logic stays in the
# package; the notebook selects the directory and displays the results.
# %%
# %%load_clean
from mlops_fraud_prediction_ca1.data import load_base_data  # noqa: F401
# Referenced Source: sha256:d24bbcee437909f6979697f8c69229422d7128971cd1323ea73751eefd9c016b

import logging

from pathlib import Path

import pandas as pd

def load_base_data(data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read the three base CSVs from an explicitly selected directory.

    Args:
        data_dir: Directory containing customers_base.csv, orders_base.csv,
            and labels_base.csv. Relative paths use the current working
            directory. A leading `~` refers to the user's home directory,
            so `~/datasets/fraud` can be used instead of a full absolute path.

    Returns:
        DataFrames in the order (customers, orders, labels). Pandas infers
        types and missing values using its read_csv defaults. No cleaning,
        date parsing, joining, or source-file modification is performed.

    Raises:
        FileNotFoundError: One or more required files are absent. All paths
            are checked before any CSV is read.
        OSError: An input cannot be read.
        pandas.errors.EmptyDataError: An input has no columns to parse.
        pandas.errors.ParserError: An input cannot be parsed as CSV.

    Logs each filename and its loaded dimensions at INFO level. Logging
    handlers and verbosity are configured by the caller.
    """
    # Path() does not expand "~" automatically. Expand it so callers can use
    # home-relative directories such as "~/datasets/fraud".
    directory: Path = Path(data_dir).expanduser()
    paths: tuple[Path, Path, Path] = (
        directory / "customers_base.csv",
        directory / "orders_base.csv",
        directory / "labels_base.csv",
    )
    missing_files: list[str] = [str(path) for path in paths if not path.is_file()]
    if len(missing_files) > 0:
        raise FileNotFoundError("Required CSV files not found: " + ", ".join(missing_files))

    frames: list[pd.DataFrame] = []
    for path in paths:
        logger.info("Loading %s", path)
        frame: pd.DataFrame = pd.read_csv(path)
        logger.info("Loaded %s: %d rows, %d columns", path.name, *frame.shape)
        frames.append(frame)

    return frames[0], frames[1], frames[2]



# %% [markdown]
# ### Load the tables and check their dimensions
#
# Read the customer, order, and label CSVs into separate DataFrames. The summary
# shows each table's row and column counts as an initial loading check;
# data-quality checks follow separately.
#
# %%
customers: pd.DataFrame
orders: pd.DataFrame
labels: pd.DataFrame
customers, orders, labels = load_base_data(data_dir)

pd.DataFrame(
    {
        "dataset": ["customers", "orders", "labels"],
        "rows": [len(customers), len(orders), len(labels)],
        "columns": [len(customers.columns), len(orders.columns), len(labels.columns)],
    }
)

# %% [markdown]
# ### Inspect dataset columns
#
# The table below lists the column names in `customers`, `orders`, and `labels`
# side by side. Inspect these names before checking missing values to identify
# shared fields, identifiers, and the prediction target (`is_fraud`).
#
# Each list retains its dataset's column order; names on the same row do not
# imply a relationship. Blank cells pad the shorter lists and do not represent
# missing values in the source data.

# %%
from IPython.display import display

datasets: dict[str, pd.DataFrame] = {
    "customers": customers,
    "orders": orders,
    "labels": labels,
}

column_names: pd.DataFrame = pd.DataFrame(
    {name: pd.Series(frame.columns, dtype="string") for name, frame in datasets.items()}
).fillna("")

with pd.option_context("display.max_rows", None, "display.max_columns", None):
    display(column_names)


# %% [markdown]
# ## 2. Assess data quality

# %% [markdown]
# ### Missing values
#
# The table below shows the missing-value count and percentage for every column
# in each dataset. Percentages use the number of rows in that dataset, so we can
# compare missingness across datasets of different sizes and identify columns
# that need further inspection.
#
# `isna()` counts missing values recognised by pandas when the CSVs were loaded.
# Placeholder strings that pandas does not recognise as missing need a separate
# check. Columns with no missing values are included; percentages are displayed
# to two decimal places. For an empty dataset, the percentage is undefined.

# %%
missing_summaries: dict[str, pd.DataFrame] = {}

for name, frame in datasets.items():
    na_counts: pd.Series = frame.isna().sum()
    missing_summaries[name] = pd.DataFrame(
        {
            "na_count": na_counts,
            "na_percent": na_counts / len(frame) * 100 if len(frame) else float("nan"),
        }
    )

missing_values: pd.DataFrame = (
    pd.concat(missing_summaries, names=["dataset", "column"])
    .reset_index()
)

with pd.option_context(
    "display.max_rows", None,
    "display.max_columns", None,
    "display.float_format", "{:.2f}".format,
):
    display(missing_values)


# %% [markdown]
# #### Findings
#
# Checks against the current source CSVs found pandas-recognised missing values
# in three columns:
#
# - `customers.first_order_datetime`: 712 missing values (0.08% of customer rows).
# - `orders.payment_method`: 21,501 missing values (2.26% of order rows).
# - `orders.num_items_ordered`: 1 missing value (less than 0.01%; displayed as 0.00%).
#
# All other columns, including every column in `labels`, had zero missing values
# under this definition. Placeholder strings still need investigation. No values
# have been filled or rows removed; handling decisions remain open.
#
# These findings describe the current CSVs and should be reviewed if the inputs
# or missing-value rules change.

# %% [markdown]
# ### Exact duplicates
#
# The summary below shows how many rows are repeated across all columns in each
# dataset.
#
# - `extra_duplicate_rows` counts copies after the first occurrence
#   (`duplicated()`); its percentage uses the dataset's total row count.
# - `rows_in_duplicate_groups` counts every occurrence of a repeated row
#   (`duplicated(keep=False)`), including the first. For three identical rows,
#   these two counts are 2 and 3 respectively.
#
# Up to five rows from duplicate groups are displayed per affected dataset.
# Pandas treats matching missing values as equal for this check. A repeated ID
# with different values is not an exact duplicate; key uniqueness and conflicting
# records need separate checks. These cells inspect the data without removing
# rows. Percentages are undefined for empty datasets.

# %%
from IPython.display import display

duplicate_records: list[dict[str, str | int | float]] = []
duplicate_examples: dict[str, pd.DataFrame] = {}

for name, frame in {"customers": customers, "orders": orders, "labels": labels}.items():
    extra_count: int = int(frame.duplicated().sum())
    duplicate_group_mask: pd.Series = frame.duplicated(keep=False)
    duplicate_records.append(
        {
            "dataset": name,
            "total_rows": len(frame),
            "extra_duplicate_rows": extra_count,
            "extra_duplicate_percent": extra_count / len(frame) * 100 if len(frame) else float("nan"),
            "rows_in_duplicate_groups": int(duplicate_group_mask.sum()),
        }
    )
    duplicate_examples[name] = frame.loc[duplicate_group_mask].head(5)

duplicate_summary: pd.DataFrame = pd.DataFrame(duplicate_records)

with pd.option_context(
    "display.max_columns", None,
    "display.float_format", "{:.2f}".format,
):
    display(duplicate_summary)
    for name, examples in duplicate_examples.items():
        if not examples.empty:
            print(f"{name}: up to five rows from exact duplicate groups")
            display(examples)


# %% [markdown]
# #### Findings
#
# No exact duplicate rows were found in `customers` (901,312 rows), `orders`
# (949,889 rows), or `labels` (950,318 rows). No rows need removal on this basis.
# Repeated keys and conflicting records remain to be checked.
#
# These findings describe the current CSVs and should be reviewed if the inputs
# or comparison rules change.
