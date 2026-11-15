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
# %% [markdown]
# ## Load the source data
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
