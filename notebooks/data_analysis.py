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
# ## Contents
#
# - [1. Load and inspect the datasets](#1-load-and-inspect-the-datasets)
# - [2. Assess data quality](#2-assess-data-quality)
#   - [Missing values](#missing-values)
#   - [Exact duplicates](#exact-duplicates)
# - [Glossary](#glossary)
# %% [markdown]
# ## 1. Load and inspect the datasets
#
# Select the directory containing the three base CSVs. Loading uses pandas'
# default type and missing-value inference; source files remain unchanged.
#
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
# The linked import below uses NotebookHelpers to synchronize `load_base_data`
# with its package definition. Data loading logic stays in the package; the
# notebook selects the directory and displays the results.
#
# %%
# %%load_clean
from mlops_fraud_prediction_ca1.data import load_base_data

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
#
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

# %% [markdown]
# ## Glossary
#
# | Term | Meaning in this notebook |
# | --- | --- |
# | Column / field | A named attribute in a dataset. |
# | Row / record | One entry containing values for the dataset's columns. |
# | Feature | A variable used as an input to a prediction model. |
# | Target | The outcome to predict; here, `is_fraud`. |
# | Key / composite key | One column, or a combination of columns, intended to identify a record. Uniqueness must be checked. |
# | Missing value | A value pandas recognises as absent; placeholder strings may require additional checks. |
# | Exact duplicate | A row matching another row across all columns. |
