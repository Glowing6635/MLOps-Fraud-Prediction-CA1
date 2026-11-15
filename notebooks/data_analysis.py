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
# - [1. Purpose and workflow](#1-purpose-and-workflow)
# - [2. Load and inspect the datasets](#2-load-and-inspect-the-datasets)
#   - [2.1 Configure the data directory and logging][configure-data]
#   - [2.2 Enable NotebookHelpers](#22-enable-notebookhelpers)
#   - [2.3 Import the reusable loader](#23-import-the-reusable-loader)
#   - [2.4 Load the tables and check their dimensions][load-tables]
#   - [2.5 Inspect dataset columns](#25-inspect-dataset-columns)
# - [3. Assess data quality](#3-assess-data-quality)
#   - [3.1 Missing values](#31-missing-values)
#     - [3.1.1 Findings](#311-findings)
#   - [3.2 Exact duplicates](#32-exact-duplicates)
#     - [3.2.1 Findings](#321-findings)
#   - [3.3 Inspect category values](#33-inspect-category-values)
#   - [3.4 Preview category standardization](#34-preview-category-standardization)
#   - [3.5 Numerical outliers and skew](#35-numerical-outliers-and-skew)
#     - [3.5.1 Distribution and screening summaries](#351-distribution-and-screening-summaries)
#     - [3.5.2 Histograms and box plots](#352-histograms-and-box-plots)
#     - [3.5.3 Extreme examples](#353-extreme-examples)
#     - [3.5.4 Findings](#354-findings)
#
# [configure-data]: #21-configure-the-data-directory-and-logging
#
# [load-tables]: #24-load-the-tables-and-check-their-dimensions

# %% [markdown]
# ## 1. Purpose and workflow
#
# This notebook conducts **exploratory data analysis (EDA)** on the three base
# datasets used for fraud prediction to identify issues and patterns that
# could
# affect model performance and inform data preparation. The goals are to:
#
# - [ ] **Assess data quality** by measuring missing values and duplicates,
#   checking data types, identifying invalid categories and format variants,
#   and examining outliers, skew, and class imbalance to inform preparation.
# - [ ] **Establish how the datasets link** by checking ID uniqueness alone
#   and
#   with country codes, measuring match rates in both directions, checking
#   country-code agreement, and determining how orders connect to customers.
# - [ ] **Identify logical inconsistencies** such as refunds above order
#   values,
#   canceled or refunded counts above total orders, orders before a
# customer's
#   first order, negative amounts, and orders without labels, and determine
# how
#   affected records should be handled.
# - [ ] **Understand fraud patterns over time and by country** by establishing
#   date ranges and formats, comparing weekly and country-level fraud rates,
#   and dating any observed changes to inform later modeling decisions.
# - [ ] **Document evidence and handling decisions** for each check so later
#   cleaning and modeling choices can be traced to the findings, while
#   preserving the original source data.
#
#
# Run the code cells from top to bottom using the project's `.venv` kernel.
# Each check introduces its output before the code and records findings after
# it. Review those findings when the source data changes. Use the table of
# contents or your editor's outline to navigate; editors that support heading
# folding can collapse sections while you work.

# %% [markdown]
# ## 2. Load and inspect the datasets
#
# Select the directory containing the three base CSVs. Loading uses pandas'
# default type and missing-value inference; source files remain unchanged.

# %% [markdown]
# ### 2.1 Configure the data directory and logging
#
# Use the repository's `data/` directory when running from the project root or
# `notebooks/`. Set `base_data_dir` to use another location. INFO logging displays
# the files loaded and their dimensions on stdout. Warnings and errors stay
# on stderr, where notebook interfaces can highlight them.
# %%
import logging
import sys
from pathlib import Path

# The linked cell repeats this import so it can run independently.
# noinspection PyUnusedImports
import pandas as pd
from IPython.display import display
from pandas.io.formats.style import Styler
from pandas.io.formats.style_render import CSSDict


def apply_table_theme(styler: Styler) -> Styler:
    """Apply the notebook's shared slate palette to a pandas table.

    Args:
        styler: Table presentation with any existing numeric formatting.

    Returns:
        The same Styler with navy headers, alternating light rows and readable
        captions. Existing cell highlights and underlying data are preserved.
    """
    table_styles: list[CSSDict] = [
        {"selector": "", "props": [("border-collapse", "collapse")]},
        {"selector": "th, td", "props": [
            ("padding", "8px 12px"),
            ("border-bottom", "1px solid #cbd5e1"),
        ]},
        {"selector": "thead th", "props": [
            ("background-color", "#243b53"), ("color", "#ffffff"),
            ("text-align", "left"),
        ]},
        {"selector": "tbody th", "props": [
            ("background-color", "#e2e8f0"), ("color", "#243b53"),
            ("text-align", "left"), ("vertical-align", "top"),
        ]},
        {"selector": "td", "props": [
            ("background-color", "#ffffff"), ("color", "#243b53"),
        ]},
        {"selector": "tbody tr:nth-child(even) td", "props": [
            ("background-color", "#f1f5f9"),
        ]},
        {"selector": "caption", "props": [
            ("background-color", "#e2e8f0"), ("color", "#243b53"),
            ("padding", "10px 12px"), ("text-align", "left"),
            ("font-weight", "600"), ("caption-side", "top"),
        ]},
    ]
    return styler.set_table_styles(table_styles, overwrite=False)


def format_percent(value: float) -> str:
    """Format a percentage without hiding small nonzero values.

    Args:
        value: Percentage on a 0–100 scale, or NaN if undefined.

    Returns:
        A percentage string; an em dash represents an undefined value.
        Formatting does not change the underlying numeric value.
    """
    if pd.isna(value):
        return "—"
    if 0 < value < 0.01:
        return "<0.01%"
    return f"{value:.2f}%"


info_handler = logging.StreamHandler(sys.stdout)
info_handler.addFilter(lambda record: record.levelno < logging.WARNING)
warning_handler = logging.StreamHandler(sys.stderr)
warning_handler.setLevel(logging.WARNING)
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
    handlers=[info_handler, warning_handler],
    force=True,  # Apply the configuration again when this cell is rerun.
)

# Change base_data_dir to use another local directory; no loader edits are needed.
project_root: Path = (
    Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
)
base_data_dir: Path = project_root / "data"

# %% [markdown]
# ### 2.2 Enable NotebookHelpers
#
# Load the `load_clean` IPython extension to register the `%%load_clean` cell
# magic used below. Run this cell before importing the loader through the
# magic.
# %%
# %load_ext load_clean

# %% [markdown]
# ### 2.3 Import the reusable loader
#
# The linked import below uses NotebookHelpers 0.0.7 to synchronize
# `load_base_data` with its package definition, including the module's
# imports.
# On execution, NotebookHelpers refreshes the editable cell and runs the
# loader
# in its package context. Keep the identifying import (marked `# noqa: F401`)
# and generated `# Referenced Source: sha256:…` comment so the source can be
# located and conflicting edits detected. Data loading logic stays in the
# package; the notebook selects the directory and displays the results.
# %%
# %%load_clean
from mlops_fraud_prediction_ca1.data import load_base_data  # noqa: F401
# Referenced Source: sha256:a39409ecb291a66c6beac45c5f6df45cf1fd84af839727204378fa4c9b2da297

import logging

from pathlib import Path

import pandas as pd

def load_base_data(
    data_dir: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
    logger: logging.Logger = logging.getLogger(__name__)

    # Path() does not expand "~" automatically. Expand it so callers can use
    # home-relative directories such as "~/datasets/fraud".
    directory: Path = Path(data_dir).expanduser()
    paths: tuple[Path, Path, Path] = (
        directory / "customers_base.csv",
        directory / "orders_base.csv",
        directory / "labels_base.csv",
    )
    missing_files: list[str] = [
        str(path) for path in paths if not path.is_file()
    ]
    if len(missing_files) > 0:
        raise FileNotFoundError(
            "Required CSV files not found: " + ", ".join(missing_files)
        )

    logger.info("Loading base datasets from %s", directory)
    frames: list[pd.DataFrame] = []
    for path in paths:
        # Pandas' default returns a DataFrame; PyCharm misreads its Unpack overloads.
        # noinspection PyTypeChecker,PyArgumentList
        frame: pd.DataFrame = pd.read_csv(path)
        logger.info(
            "Loaded %-18s | %9s rows | %2d columns",
            path.name, f"{len(frame):,}", len(frame.columns),
        )
        frames.append(frame)

    return frames[0], frames[1], frames[2]



# %% [markdown]
# ### 2.4 Load the tables and check their dimensions
#
# Read the customer, order, and label CSVs into separate DataFrames. The
# summary
# shows each table's row and column counts as an initial loading check;
# data-quality checks follow separately.
#
# The mixed-type warning for `mobile_verified` does not prevent loading.
# Its values still need inspection and a handling decision before modeling.
# %%
customers: pd.DataFrame
orders: pd.DataFrame
labels: pd.DataFrame
customers, orders, labels = load_base_data(base_data_dir)

dataset_dimensions: pd.DataFrame = pd.DataFrame(
    {
        "dataset": ["customers", "orders", "labels"],
        "rows": [len(customers), len(orders), len(labels)],
        "columns": [
            len(customers.columns), len(orders.columns), len(labels.columns)
        ],
    }
)

(
    dataset_dimensions.set_index("dataset")
    .rename_axis("Dataset")
    .rename(columns={"rows": "Rows", "columns": "Columns"})
    .style
    .pipe(apply_table_theme)
    .set_caption("Source dataset dimensions")
    .format("{:,}")
)

# %% [markdown]
# ### 2.5 Inspect dataset columns
#
# The table below lists each dataset's columns, inferred pandas datatypes,
# and notes on their interpretation. A datatype describes storage, not whether
# values fit the column's meaning. In particular, `object` can hold strings or
# mixed Python types; it does not establish a mixture on its own.

# %%
datasets: dict[str, pd.DataFrame] = {
    "customers": customers,
    "orders": orders,
    "labels": labels,
}

dtype_notes: dict[str, str] = {
    "O": "May contain strings or mixed Python types; inspect values.",
    "T": "Text storage; inspect expected categories or date formats.",
    "U": "Text storage; inspect expected categories or date formats.",
    "S": "Text storage; inspect expected categories or date formats.",
    "b": "Boolean storage; confirm the intended two-state meaning.",
    "i": "Integer storage; confirm whether values are counts, codes or IDs.",
    "u": "Integer storage; confirm whether values are counts, codes or IDs.",
    "f": "Numeric storage; confirm meaning and expected precision.",
    "M": "Datetime storage; confirm parsing and expected date range.",
}

column_information: pd.DataFrame = pd.DataFrame(
    [
        {
            "Dataset": name,
            "Column": column,
            "Datatype": str(dtype),
            "Note": (
                "Text storage; inspect expected categories or date formats."
                if pd.api.types.is_string_dtype(dtype)
                else dtype_notes.get(
                    dtype.kind, "Inspect values against the column's meaning."
                )
            )
            + (
                " Mixed types reported during loading; inspect representations before conversion."
                if name == "customers" and column == "mobile_verified"
                else ""
            ),
        }
        for name, frame in datasets.items()
        for column, dtype in frame.dtypes.items()
    ]
)

(
    column_information.style
    .hide(axis="index")
    .pipe(apply_table_theme)
    .set_caption("Column datatypes and inspection notes by dataset")
    .set_properties(**{"text-align": "left"})
    .set_table_styles(
        [CSSDict(selector="th", props=[("text-align", "left")])],
        overwrite=False,
    )
    .format(escape="html")
)

# %% [markdown]
# ## 3. Assess data quality

# %% [markdown]
# ### 3.1 Missing values
#
# The table below shows the missing-value count and percentage for every
# column
# in each dataset. Percentages use the number of rows in that dataset, so we
# can
# compare missingness across datasets of different sizes and identify columns
# that need further inspection.
#
# `isna()` counts missing values recognized by pandas when the CSVs were
# loaded.
# Placeholder strings that pandas does not recognize as missing need a
# separate
# check. Columns with no missing values are included; percentages use two
# decimal places, with small nonzero values shown as
# `<0.01%`. Muted rose cells mark nonzero missingness. For an empty dataset,
# the percentage is undefined.

# %%

missing_summaries: dict[str, pd.DataFrame] = {}

for name, frame in datasets.items():
    na_counts: pd.Series = frame.isna().sum()
    missing_summaries[name] = pd.DataFrame(
        {
            "na_count": na_counts,
            "na_percent": (
                na_counts / len(frame) * 100 if len(frame) else float("nan")
            ),
        }
    )

missing_values: pd.DataFrame = (
    pd.concat(missing_summaries, names=["dataset", "column"])
    .reset_index()
)


(
    missing_values.set_index(["dataset", "column"])
    .rename_axis(["Dataset", "Column"])
    .rename(columns={"na_count": "Missing count", "na_percent": "Missing %"})
    .style
    .map(
        lambda value: (
            "background-color: #fce8ec; color: #8a2942" if value > 0 else ""
        ),
        subset=["Missing count", "Missing %"],
    )
    .pipe(apply_table_theme)
    .set_caption(
        "Missing values by column; "
        "percentages use all rows in each dataset"
    )
    .format({"Missing count": "{:,}", "Missing %": format_percent}, na_rep="—")
)

# %% [markdown]
# #### 3.1.1 Findings
#
# Checks against the current source CSVs found pandas-recognized missing
# values
# in three columns:
#
# - `customers.first_order_datetime`: 712 missing values (0.08% of customer
#   rows).
# - `orders.payment_method`: 21,501 missing values (2.26% of order rows).
# - `orders.num_items_ordered`: 1 missing value (displayed as <0.01%).
#
# All other columns, including every column in `labels`, had zero missing
# values
# under this definition. Placeholder strings still need investigation. No
# values
# have been filled or rows removed; handling decisions remain open.
#
# These findings describe the current CSVs and should be reviewed if the
# inputs
# or missing-value rules change.

# %% [markdown]
# ### 3.2 Exact duplicates
#
# The summary below shows how many rows are repeated across all columns in
# each
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
# with different values is not an exact duplicate; key uniqueness and
# conflicting
# records need separate checks. These cells inspect the data without removing
# rows. Percentages are undefined for empty datasets.

# %%
duplicate_records: list[dict[str, str | int | float]] = []
duplicate_examples: dict[str, pd.DataFrame] = {}

for name, frame in datasets.items():
    extra_count: int = int(frame.duplicated().sum())
    duplicate_group_mask: pd.Series = frame.duplicated(keep=False)
    duplicate_records.append(
        {
            "dataset": name,
            "total_rows": len(frame),
            "extra_duplicate_rows": extra_count,
            "extra_duplicate_percent": (
                extra_count / len(frame) * 100
                if len(frame) else float("nan")
            ),
            "rows_in_duplicate_groups": int(duplicate_group_mask.sum()),
        }
    )
    duplicate_examples[name] = frame.loc[duplicate_group_mask].head(5)

duplicate_summary: pd.DataFrame = pd.DataFrame(duplicate_records)


display(
    duplicate_summary.set_index("dataset")
    .rename_axis("Dataset")
    .rename(columns={
        "total_rows": "Total rows",
        "extra_duplicate_rows": "Extra duplicate rows",
        "extra_duplicate_percent": "Extra duplicate %",
        "rows_in_duplicate_groups": "Rows in duplicate groups",
    })
    .style
    .pipe(apply_table_theme)
    .set_caption(
        "Exact row duplicates; "
        "percentages count extra copies against all dataset rows"
    )
    .format({
        "Total rows": "{:,}",
        "Extra duplicate rows": "{:,}",
        "Extra duplicate %": format_percent,
        "Rows in duplicate groups": "{:,}",
    }, na_rep="—")
)
for name, examples in duplicate_examples.items():
    if not examples.empty:
        display(
            examples.style
            .pipe(apply_table_theme)
            .set_caption(
                f"{name}: up to five rows from exact duplicate groups"
            )
            .format(escape="html", na_rep="—")
        )

# %% [markdown]
# #### 3.2.1 Findings
#
# No exact duplicate rows were found in `customers` (901,312 rows), `orders`
# (949,889 rows), or `labels` (950,318 rows). No rows need removal on this
# basis.
# Repeated keys and conflicting records remain to be checked.
#
# These findings describe the current CSVs and should be reviewed if the
# inputs
# or comparison rules change.

# %% [markdown]
# ### 3.3 Inspect category values
#
# Show the top `category_top_k` distinct values and their counts for text
# categories and discrete flags. Exclude identifiers, dates, amounts and
# numerical counts. Read only these columns as raw text so CSV inference
# does not merge representations or hide empty fields. Quoted values make
# whitespace and empty strings visible; no values are cleaned or converted.
#
# Each caption reports how many distinct values were omitted. Increase
# `category_top_k` to inspect them: frequent values alone cannot establish
# that rare invalid categories are absent. These tables support inspection;
# allowed categories and handling decisions still need confirmation.

# %%
category_top_k: int = 25
category_columns: dict[str, list[str]] = {
    "customers": ["country_code", "mobile_verified"],
    "orders": ["country_code", "collect_type", "payment_method"],
    "labels": ["country_code", "is_fraud"],
}
category_frequencies: dict[tuple[str, str], pd.Series] = {}

if category_top_k < 1:
    raise ValueError("category_top_k must be at least 1")

for name, columns in category_columns.items():
    # PyCharm cannot resolve the pandas Unpack overloads for this call.
    # noinspection PyTypeChecker,PyArgumentList
    raw_categories: pd.DataFrame = pd.read_csv(
        base_data_dir / f"{name}_base.csv",
        usecols=columns,
        dtype="string",
        keep_default_na=False,
    )
    for column in columns:
        frequencies: pd.Series = raw_categories[column].value_counts(dropna=False)
        category_frequencies[name, column] = frequencies
        top_values: pd.DataFrame = (
            frequencies.head(category_top_k)
            .rename_axis("Value")
            .reset_index(name="Count")
        )
        top_values["Value"] = top_values["Value"].map(repr)
        display(
            top_values.style
            .hide(axis="index")
            .pipe(apply_table_theme)
            .set_caption(
                f"{name}.{column}: {len(frequencies):,} distinct values; "
                f"{max(len(frequencies) - category_top_k, 0):,} omitted"
            )
            .format({"Value": "{}", "Count": "{:,}"}, escape="html")
        )

# %% [markdown]
# ### 3.4 Preview category standardization
#
# **Evidence-based formatting:** uppercase country codes and lowercase collection
# and payment categories to bring observed case variants together. Lowercase
# verification representations for comparison; this does not map `yes` to `true`.
#
# **Safeguards:** trim boundary whitespace as a precaution, preserve missing and
# blank values, and keep unexpected categories visible. Whitespace variants have
# not been established by the preceding inspection. No allowed sets or semantic
# aliases are supplied, so nonblank proposals remain `unchecked`. In particular,
# `on Delivery` is not mapped to `payment on delivery`.
#
# The package function returns original values, proposals, counts and statuses.
# All distinct values are shown directly, with whitespace preserved in the
# value cells. These are previews only: source files and the loaded datasets
# stay unchanged. The
# [category standardization notes](../docs/category-standardisation.md) document
# the evidence, safeguards and decisions still requiring confirmation.

# %%
from mlops_fraud_prediction_ca1.categories import standardise_categories

category_previews: dict[tuple[str, str], pd.DataFrame] = {}

for name, columns in category_columns.items():
    # PyCharm cannot resolve the pandas Unpack overloads for this call.
    # noinspection PyTypeChecker,PyArgumentList
    raw_category_values: pd.DataFrame = pd.read_csv(
        base_data_dir / f"{name}_base.csv",
        usecols=columns,
        dtype="string",
        keep_default_na=False,
    )
    for column in columns:
        preview: pd.DataFrame = standardise_categories(
            raw_category_values[column],
            case="upper" if column == "country_code" else "lower",
        )
        category_previews[name, column] = preview
        display(
            preview.rename(columns={
                "original_value": "Original value",
                "proposed_value": "Standardized value",
                "count": "Count",
                "status": "Status",
            }).style
            .hide(axis="index")
            .pipe(apply_table_theme)
            .set_caption(f"{name.title()} — {column.replace('_', ' ').capitalize()}")
            .set_properties(
                subset=["Original value", "Standardized value"],
                **{"white-space": "pre-wrap"},
            )
            .format(
                {"Count": "{:,}"},
                escape="html",
                na_rep="Missing",
            )
        )

# %% [markdown]
# ### 3.5 Numerical outliers and skew
#
# Inspect the eight amount/count features in customers and orders. IDs, country
# codes, verification flags, and the fraud target are excluded.
#
# Potential outliers fall below Q1 − 1.5 × IQR or above Q3 + 1.5 × IQR, where
# IQR = Q3 − Q1. These flags identify unusual values, not proven errors. When
# IQR is zero, every value different from the shared quartile is flagged.
#
# Statistics use finite, nonmissing values; missing and infinite values are
# reported separately. Positive sample skewness indicates right-tailed
# asymmetry, and negative skewness indicates left-tailed asymmetry. Skewness is
# undefined for constant features or fewer than three finite observations.
# No source values are changed.

# %% [markdown]
# #### 3.5.1 Distribution and screening summaries
#
# Compare typical values, tails, and skewness, then inspect the IQR fences and
# flagged proportions. Zero and flagged percentages use each feature's finite
# observation count.

# %%
from mlops_fraud_prediction_ca1.numeric import profile_numeric_features

iqr_multiplier: float = 1.5
numeric_features: dict[str, list[str]] = {
    "customers": [
        "num_orders_last_50days", "num_cancelled_orders_last_50days",
        "num_refund_orders_last_50days", "total_payment_last_50days",
        "num_associated_customers",
    ],
    "orders": ["order_value", "num_items_ordered", "refund_value"],
}
numeric_labels: dict[str, str] = {
    "num_orders_last_50days": "Orders in last 50 days",
    "num_cancelled_orders_last_50days": "Canceled orders in last 50 days",
    "num_refund_orders_last_50days": "Refunded orders in last 50 days",
    "total_payment_last_50days": "Total payments in last 50 days",
    "num_associated_customers": "Associated customers",
    "order_value": "Order value",
    "num_items_ordered": "Items ordered",
    "refund_value": "Refund value",
}
numeric_summary: pd.DataFrame = profile_numeric_features(
    datasets, numeric_features, iqr_multiplier=iqr_multiplier
)
distribution_columns: dict[str, str] = {
    "finite_count": "Finite rows", "missing_count": "Missing",
    "infinite_count": "Infinite", "zero_percent": "Zero %",
    "minimum": "Minimum", "mean": "Mean", "median": "Median",
    "p99": "P99", "maximum": "Maximum", "skewness": "Skewness",
}
screening_columns: dict[str, str] = {
    "q1": "Q1", "q3": "Q3", "lower_fence": "Lower fence",
    "upper_fence": "Upper fence", "flagged_count": "Flagged rows",
    "flagged_percent": "Flagged %", "zero_iqr": "Zero IQR",
}

for dataset_name in numeric_features:
    display(
        numeric_summary.loc[dataset_name, list(distribution_columns)]
        .rename(index=numeric_labels, columns=distribution_columns)
        .rename_axis("Feature")
        .style.pipe(apply_table_theme)
        .set_caption(f"{dataset_name.title()} — numerical distributions")
        .format({
            "Finite rows": "{:,}", "Missing": "{:,}", "Infinite": "{:,}",
            "Zero %": format_percent,
            "Minimum": "{:,.2f}", "Mean": "{:,.2f}", "Median": "{:,.2f}",
            "P99": "{:,.2f}", "Maximum": "{:,.2f}", "Skewness": "{:.2f}",
        }, na_rep="—")
    )
    display(
        numeric_summary.loc[dataset_name, list(screening_columns)]
        .rename(index=numeric_labels, columns=screening_columns)
        .rename_axis("Feature")
        .style.pipe(apply_table_theme)
        .set_caption(f"{dataset_name.title()} — potential outliers ({iqr_multiplier:g} × IQR)")
        .format({
            "Q1": "{:,.2f}", "Q3": "{:,.2f}",
            "Lower fence": "{:,.2f}", "Upper fence": "{:,.2f}",
            "Flagged rows": "{:,}", "Flagged %": format_percent,
            "Zero IQR": lambda value: "Yes" if value else "No",
        }, na_rep="—")
    )

# %% [markdown]
# #### 3.5.2 Histograms and box plots
#
# Each histogram shows values through the feature's 99th percentile so the
# main distribution remains readable. Its title reports how many values lie
# above that view; the count axis is logarithmic. The paired boxplot includes
# the full finite range and uses the same 1.5 × IQR screening rule.
#
# The tables' skewness values use the full finite data, including the tails.
# The restricted histogram view does not remove observations from the data.

# %%
# %matplotlib inline
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns


def plot_numeric_distributions(
    table: pd.DataFrame,
    selected_features: list[str],
    feature_labels: dict[str, str],
    table_title: str,
    fence_multiplier: float,
) -> Figure:
    """Plot distribution bodies beside full-range box plots for selected features.

    Args:
        table: Source table; no values are modified.
        selected_features: Numerical columns selected by the profiling section.
        feature_labels: Human-readable labels for the selected columns.
        table_title: Dataset name used in the figure title.
        fence_multiplier: IQR multiplier matching the summary-table rule.

    Returns:
        Matplotlib figure with one histogram/boxplot pair per feature. Missing
        and infinite values are excluded from plots. Histograms stop at P99
        and report the number above that view; box plots retain the full finite
        range. Count axes are logarithmic, and count features use integer bins.
        The caller displays and closes the figure.
    """
    figure, axes = plt.subplots(
        len(selected_features), 2, figsize=(14, 3.1 * len(selected_features)),
        squeeze=False,
    )
    figure.suptitle(f"{table_title} — numerical feature distributions", fontweight="bold")
    for row_number, plotted_feature in enumerate(selected_features):
        plotted_values: pd.Series = table[plotted_feature]
        finite_plotted_values: pd.Series = plotted_values.loc[
            plotted_values.notna()
            & ~plotted_values.isin([float("-inf"), float("inf")])
        ]
        histogram_axis, boxplot_axis = axes[row_number]
        feature_title: str = feature_labels[plotted_feature]
        if len(finite_plotted_values) == 0:
            for empty_axis in (histogram_axis, boxplot_axis):
                empty_axis.set_title(feature_title)
                empty_axis.text(0.5, 0.5, "No finite observations", ha="center")
            continue
        cutoff: float = float(finite_plotted_values.quantile(0.99))
        body: pd.Series = finite_plotted_values.loc[finite_plotted_values.le(cutoff)]
        hidden_count: int = len(finite_plotted_values) - len(body)
        sns.histplot(
            x=body, bins=40, discrete=plotted_feature.startswith("num_"),
            color="#486581", ax=histogram_axis,
        )
        histogram_axis.set_yscale("log")
        histogram_axis.set_title(
            f"{feature_title}\nThrough P99; {hidden_count:,} values above view"
        )
        histogram_axis.set_xlabel(feature_title)
        histogram_axis.set_ylabel("Rows (log scale)")
        sns.boxplot(
            x=finite_plotted_values, whis=fence_multiplier, fliersize=2,
            color="#9fb3c8", ax=boxplot_axis,
        )
        boxplot_axis.set_title(f"{feature_title}\nFull finite range")
        boxplot_axis.set_xlabel(feature_title)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    return figure


# noinspection SpellCheckingInspection
sns.set_theme(style="whitegrid")
for dataset_name, feature_names in numeric_features.items():
    distribution_figure: Figure = plot_numeric_distributions(
        datasets[dataset_name], feature_names, numeric_labels,
        dataset_name.title(), iqr_multiplier,
    )
    display(distribution_figure)
    plt.close(distribution_figure)

# %% [markdown]
# #### 3.5.3 Extreme examples
#
# Show up to three smallest and three largest flagged observations per feature.
# The source row index is zero-based and refers to the loaded table. An
# observation appearing at both ends of a short list is shown once.

# %%
outlier_examples: dict[tuple[str, str], pd.DataFrame] = {}

for dataset_name, feature_names in numeric_features.items():
    for feature_name in feature_names:
        feature_values: pd.Series = datasets[dataset_name][feature_name]
        feature_stats: pd.Series = numeric_summary.loc[(dataset_name, feature_name)]
        flagged_values: pd.Series = feature_values.loc[
            feature_values.notna()
            & ~feature_values.isin([float("-inf"), float("inf")])
            & (
                feature_values.lt(feature_stats["lower_fence"])
                | feature_values.gt(feature_stats["upper_fence"])
            )
        ].sort_values()
        if len(flagged_values) == 0:
            continue
        extreme_values: pd.Series = pd.concat([
            flagged_values.head(3), flagged_values.tail(3)
        ])
        extreme_values = extreme_values.loc[~extreme_values.index.duplicated()]
        examples_table: pd.DataFrame = (
            extreme_values.rename_axis("Source row index").reset_index(name="Value")
        )
        outlier_examples[dataset_name, feature_name] = examples_table
        display(
            examples_table.style.hide(axis="index")
            .pipe(apply_table_theme)
            .set_caption(f"{dataset_name.title()} — {numeric_labels[feature_name]}: extremes")
            .format({"Source row index": "{:d}", "Value": "{:,.2f}"})
        )

# %% [markdown]
# #### 3.5.4 Findings
#
# - All eight features are right-skewed: sample skewness ranges from **5.42 to
#   55.00**. Canceled-order counts (55.00) and items ordered (41.04) have the
#   strongest asymmetry.
# - Canceled and refunded order counts are **83.14% and 85.76% zero**. Both have
#   zero IQR, so the rule flags every nonzero count: 16.86% and 14.24% of rows,
#   respectively. Those proportions reflect sparse distributions, not established
#   data errors.
# - The other six features have **4.35%–7.48%** of finite observations outside
#   their IQR fences. These are per-feature rates; they must not be added together
#   because the same record can be flagged in several columns.
# - The largest item count is **1,600**, compared with a median of **2** and P99
#   of **22**. Customer payment totals reach **42,569.35**, compared with a median
#   of **164.58** and P99 of **1,876.68**. The examples locate these extremes for
#   further investigation without classifying them as errors.
# - These features contain no negative or infinite values. The one missing item
#   count is excluded from that feature's statistics and percentage denominator.
#
# No capping, removal, or transformation has been applied.
