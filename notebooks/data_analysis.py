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
#   - [2.1 Configure the data directory and logging](#21-configure-the-data-directory-and-logging)
#   - [2.2 Enable NotebookHelpers](#22-enable-notebookhelpers)
#   - [2.3 Import the reusable loader](#23-import-the-reusable-loader)
#   - [2.4 Load the tables and check their dimensions](#24-load-the-tables-and-check-their-dimensions)
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
#   - [3.6 Semantic types and numeric validity](#36-semantic-types-and-numeric-validity)
#     - [3.6.1 Findings and handling](#361-findings-and-handling)
#   - [3.7 Class balance](#37-class-balance)
#     - [3.7.1 Findings and handling](#371-findings-and-handling)
# - [4. Relationships between datasets](#4-relationships-between-datasets)
#   - [4.1 Key uniqueness](#41-key-uniqueness)
#   - [4.2 Match rates in both directions](#42-match-rates-in-both-directions)
#   - [4.3 Country agreement and safe customer linkage](#43-country-agreement-and-safe-customer-linkage)
#   - [4.4 Findings and handling](#44-findings-and-handling)
# - [5. Business logic checks](#5-business-logic-checks)
#   - [5.1 Findings and handling](#51-findings-and-handling)
# - [6. Dates and fraud patterns](#6-dates-and-fraud-patterns)
#   - [6.1 Date ranges and formats](#61-date-ranges-and-formats)
#     - [6.1.1 Findings and handling](#611-findings-and-handling)
#   - [6.2 Fraud rates by country](#62-fraud-rates-by-country)
#     - [6.2.1 Findings](#621-findings)
#   - [6.3 Weekly fraud rates and dated movements](#63-weekly-fraud-rates-and-dated-movements)
#     - [6.3.1 Findings and handling](#631-findings-and-handling)

# %% [markdown]
# ## 1. Purpose and workflow
#
# Profile the three base datasets to identify data issues and patterns that could affect fraud prediction. The analysis covers data quality, relationships between files, business logic, and fraud rates over time and by country. Each section records evidence and a handling decision while preserving source values.
#
# Run all cells from top to bottom with the project's `.venv` kernel. Country normalization and date parsing use working copies; source CSVs and loaded tables are not cleaned in place. Review the written findings if the inputs change. Use the table of contents or your editor's outline to navigate.

# %% [markdown]
# ## 2. Load and inspect the datasets
#
# Select the directory containing the three base CSVs. Loading uses pandas' default type and missing-value inference; source files remain unchanged.

# %% [markdown]
# ### 2.1 Configure the data directory and logging
#
# Use the repository's `data/` directory when running from the project root or `notebooks/`. Set `base_data_dir` to use another location. INFO logging displays the files loaded and their dimensions on stdout. Warnings and errors stay on stderr, where notebook interfaces can highlight them.
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
# Load the `load_clean` IPython extension to register the `%%load_clean` cell magic used below. Run this cell before importing the loader through the magic.
# %%
# %load_ext load_clean

# %% [markdown]
# ### 2.3 Import the reusable loader
#
# NotebookHelpers `inline` mode imports and runs the package definition of `load_base_data`, including its dependencies. This executable import keeps the loading logic in the package and supports a fresh top-to-bottom run. Skeleton mode is intended for drafting an implementation and would leave the loader unavailable here.

# %%
# %%load_clean inline
from mlops_fraud_prediction_ca1.data import load_base_data

# %% [markdown]
# ### 2.4 Load the tables and check their dimensions
#
# Read the customer, order, and label CSVs into separate DataFrames. The summary shows each table's row and column counts as an initial loading check; data-quality checks follow separately.
#
# The mixed-type warning for `mobile_verified` does not prevent loading. Section 3.4 inspects the original representations and records how to handle the unresolved values.
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
# The table below lists each dataset's columns, inferred pandas datatypes, and notes on their interpretation. A datatype describes storage, not whether values fit the column's meaning. In particular, `object` can hold strings or mixed Python types; it does not establish a mixture on its own.

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
# The table below shows the missing-value count and percentage for every column in each dataset. Percentages use the number of rows in that dataset, so we can compare missingness across datasets of different sizes and identify columns that need further inspection.
#
# `isna()` counts missing values recognized by pandas when the CSVs were loaded. Placeholder strings that pandas does not recognize as missing need a separate check. Columns with no missing values are included; percentages use two decimal places, with small nonzero values shown as `<0.01%`. Muted rose cells mark nonzero missingness. For an empty dataset, the percentage is undefined.

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
# Pandas-recognized missingness affects three columns:
#
# - `customers.first_order_datetime`: **712 rows (0.08%)**. Retain missing timestamps and exclude them from date-order comparisons; do not invent a first-order date.
# - `orders.payment_method`: **21,501 rows (2.26%)**. Preserve missingness as an explicit unknown category in later preparation.
# - `orders.num_items_ordered`: **1 row (<0.01%)**. Retain a nullable count and a missing flag. Fit any later imputation on training data only.
#
# All other columns have zero missing values under pandas' loading rules. Raw category inspection below preserves blank fields and original spellings separately. No values have been filled or rows removed.

# %% [markdown]
# ### 3.2 Exact duplicates
#
# The summary below shows how many rows are repeated across all columns in each dataset.
#
# - `extra_duplicate_rows` counts copies after the first occurrence (`duplicated()`); its percentage uses the dataset's total row count.
# - `rows_in_duplicate_groups` counts every occurrence of a repeated row (`duplicated(keep=False)`), including the first. For three identical rows, these two counts are 2 and 3 respectively.
#
# Up to five rows from duplicate groups are displayed per affected dataset. Pandas treats matching missing values as equal for this check. A repeated ID with different values is not an exact duplicate; section 4 checks key uniqueness and conflicting records separately. These cells inspect the data without removing rows. Percentages are undefined for empty datasets.

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
# No exact duplicate rows were found in customers (901,312 rows), orders (949,889 rows), or labels (950,318 rows), so no rows need removal on this basis. Section 4 checks repeated keys and conflicting records separately. If copies appear in future batches, collapse identical full rows only and audit the count; never resolve conflicting keys by selecting an arbitrary first row.

# %% [markdown]
# ### 3.3 Inspect category values
#
# Show the top `category_top_k` distinct values and their counts for text categories and discrete flags. Exclude identifiers, dates, amounts and numerical counts. Read only these columns as raw text so CSV inference does not merge representations or hide empty fields. Quoted values make whitespace and empty strings visible; no values are cleaned or converted.
#
# Each caption reports how many distinct values were omitted. Increase `category_top_k` to inspect them: frequent values alone cannot establish that rare invalid categories are absent. These tables support inspection; the next section validates the defined binary domains and records handling decisions.

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
# Uppercase country codes and lowercase collection/payment representations on working copies to compare observed case variants. Original values remain visible. Trimming boundary whitespace is a safeguard, not evidence that whitespace defects were found.
#
# Validate mobile verification against the specified True/False domain and the fraud label against 0/1. The 165 raw `yes` values remain unexpected and unresolved; do not silently treat them as True. Country, collection and payment category domains are open, so their nonblank proposals remain `unchecked`. Preserve `on Delivery` (160 rows) without an unconfirmed semantic alias.
#
# All distinct raw values, counts, proposals and statuses are shown. This is a formatting and validation preview; no Boolean conversion or source cleaning occurs. See the [category standardization notes](../docs/category-standardisation.md) for the formatting safeguards.

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
            allowed_values=({"true", "false"} if column == "mobile_verified"
                            else {"0", "1"} if column == "is_fraud" else None),
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
# Inspect the eight amount/count features in customers and orders. IDs, country codes, verification flags, and the fraud target are excluded.
#
# Potential outliers fall below Q1 − 1.5 × IQR or above Q3 + 1.5 × IQR, where IQR = Q3 − Q1. These flags identify unusual values, not proven errors. When IQR is zero, every value different from the shared quartile is flagged.
#
# Statistics use finite, nonmissing values; missing and infinite values are reported separately. Positive sample skewness indicates right-tailed asymmetry, and negative skewness indicates left-tailed asymmetry. Skewness is undefined for constant features or fewer than three finite observations. No source values are changed.

# %% [markdown]
# #### 3.5.1 Distribution and screening summaries
#
# Compare typical values, tails, and skewness, then inspect the IQR fences and flagged proportions. Zero and flagged percentages use each feature's finite observation count.

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
# Each histogram shows values through the feature's 99th percentile so the main distribution remains readable. Its title reports how many values lie above that view; the count axis is logarithmic. The paired boxplot includes the full finite range and uses the same 1.5 × IQR screening rule.
#
# The tables' skewness values use the full finite data, including the tails. The restricted histogram view does not remove observations from the data.

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
# Show up to three smallest and three largest flagged observations per feature. The source row index is zero-based and refers to the loaded table. An observation appearing at both ends of a short list is shown once.

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
# - All eight features are right-skewed: sample skewness ranges from **5.42 to 55.00**. Canceled-order counts (55.00) and items ordered (41.04) have the strongest asymmetry.
# - Canceled and refunded order counts are **83.14% and 85.76% zero**. Both have zero IQR, so the rule flags every nonzero count: 16.86% and 14.24% of rows, respectively. Those proportions reflect sparse distributions, not established data errors.
# - The other six features have **4.35%–7.48%** of finite observations outside their IQR fences. These are per-feature rates; they must not be added together because the same record can be flagged in several columns.
# - The largest item count is **1,600**, compared with a median of **2** and P99 of **22**. Customer payment totals reach **42,569.35**, compared with a median of **164.58** and P99 of **1,876.68**. The examples locate these extremes for further investigation without classifying them as errors.
# - These features contain no negative or infinite values. The one missing item count is excluded from that feature's statistics and percentage denominator.
#
# Retain raw values and use IQR flags for investigation. Do not cap or delete extremes without evidence of an error. Evaluate any learned transformation on training data only. No capping, removal, or transformation has been applied.

# %% [markdown]
# ### 3.6 Semantic types and numeric validity
#
# Compare inferred storage types with the meaning of each column. IDs should remain text; dates require explicit parsing, amounts require numeric precision, and counts should be finite integers. The numeric audit separates source missingness, conversion failures, infinities, negative values and nonintegral counts. Counts use source rows; a missing item count is not a failed conversion.

# %%
from mlops_fraud_prediction_ca1.quality import quality_reports

quality = quality_reports(datasets)


def show_report(frame, caption):
    """Show every report row with the shared theme and explicit rate formatting."""
    formats = {}
    for column in frame.columns:
        if column.endswith("rate"):
            formats[column] = "{:.4%}"
        elif column.endswith("_pp"):
            formats[column] = "{:+.4f}"
        elif pd.api.types.is_integer_dtype(frame[column]):
            formats[column] = "{:,}"
        elif pd.api.types.is_datetime64_any_dtype(frame[column]):
            formats[column] = lambda value: value.isoformat(sep=" ")
    display(frame.style.hide(axis="index").pipe(apply_table_theme)
            .set_caption(caption).format(formats, escape="html", na_rep="—"))


show_report(quality["types"], "Inferred storage and semantic types")
show_report(quality["numeric_validity"], "Numeric validity — source-row counts")

# %% [markdown]
# #### 3.6.1 Findings and handling
#
# All eight selected numeric features have zero conversion failures, infinities and negative values; count features have no nonintegral observations. The item count has one source-missing value. Keep IDs as text and money as decimal-valued numbers. Convert finite integral counts to nullable integers during later preparation, preserving missing/invalid flags; quarantine failed conversions rather than substituting zero.
#
# Original country case variants include customer `pk` (225) and `my` (165), and label `ph` (180) and `my` (121). Normalize country formatting only on working copies for the relationship checks. Collection and payment case variants can be compared in lowercase, but their semantic domains remain unconfirmed. The full raw inventories above expose blank values and original spellings; no missing-value alias or semantic mapping is applied.

# %% [markdown]
# ### 3.7 Class balance
#
# Measure fraud prevalence in the raw label table, independently of whether labels match an order. The denominator includes only valid 0/1 labels; missing and invalid labels are counted separately. Section 6 uses safely matched orders, so its denominator differs.

# %%
show_report(quality["class_balance"], "Class balance — raw label rows")

# %% [markdown]
# #### 3.7.1 Findings and handling
#
# There are **104,504 fraud** and **845,814 nonfraud** labels: **10.9967% fraud** among 950,318 valid labels, with no missing or invalid labels. Preserve this prevalence in evaluation. Compare fraud precision, recall and PR-AUC against a baseline; apply any weighting or resampling within training folds only, leaving validation and test populations representative.

# %% [markdown]
# ## 4. Relationships between datasets
#
# Orders contain no customer ID. Establish key uniqueness, coverage and country agreement before linking **orders → labels → customers**. Report raw country values alongside uppercase/trimmed proposals; the original tables remain unchanged.
#
# ### 4.1 Key uniqueness
#
# Check IDs alone and with `country_code`. `duplicate_rows` includes every row in a repeated-key group; `extra_key_rows` counts rows beyond one per key. `conflicting_keys` identifies keys with differing full records. Uniqueness refers to complete keys; null-key rows are disclosed separately.

# %%
from mlops_fraud_prediction_ca1.relationships import (
    key_profile, bidirectional_matches, country_agreement, link_orders_customers,
)

key_records = []
for normalized in (False, True):
    for name, frame in datasets.items():
        for identifier in ("order_id", "customer_id"):
            if identifier not in frame:
                continue
            for keys in ([identifier], ["country_code", identifier]):
                key_records.append({
                    "mode": "normalized_country" if normalized else "raw",
                    "dataset": name, "keys": "+".join(keys),
                    **key_profile(frame, keys, normalize_country=normalized),
                })
key_evidence = pd.DataFrame(key_records)
for mode, group in key_evidence.groupby("mode", sort=False):
    show_report(group.drop(columns="mode"), f"Key uniqueness — {mode}")

# %% [markdown]
# ### 4.2 Match rates in both directions
#
# For each file pair, `left_to_right` follows the names in the relationship column; `right_to_left` reverses them. Row match rates use complete-key source rows, retaining duplicate row weight. Distinct-key rates use distinct complete keys. Null keys are reported separately and never match. Existence of a match alone does not establish a safe join.

# %%
coverage_parts = []
relationship_pairs = (("orders", "labels", "order_id"),
                      ("labels", "customers", "customer_id"))
for normalized in (False, True):
    for left, right, identifier in relationship_pairs:
        for keys in ([identifier], ["country_code", identifier]):
            coverage = bidirectional_matches(
                datasets[left], datasets[right], keys,
                normalize_country=normalized,
            ).reset_index()
            coverage.insert(0, "relationship", f"{left} → {right}")
            coverage.insert(0, "keys", "+".join(keys))
            coverage.insert(0, "mode", "normalized_country" if normalized else "raw")
            coverage_parts.append(coverage)
coverage_evidence = pd.concat(coverage_parts, ignore_index=True)
for mode, group in coverage_evidence.groupby("mode", sort=False):
    show_report(group.drop(columns="mode"), f"Bidirectional coverage — {mode}")

# %% [markdown]
# ### 4.3 Country agreement and safe customer linkage
#
# Compare country sets for shared IDs without a Cartesian join. `unequal_country_sets` counts differing sets; the status counts distinguish agreement, disagreement, multiple-country ambiguity and missing countries. Equal sets with multiple countries still cannot establish a safe ID-only join.
#
# Safe linkage uses country plus order ID for labels, then country plus customer ID for customers. Collapse exact copies only, exclude null or conflicting keys, and validate one-to-one and many-to-one cardinalities. The audit exposes all source exclusions and retained orders.

# %%
agreement_records = []
for normalized in (False, True):
    for left, right, identifier in relationship_pairs:
        agreement = country_agreement(
            datasets[left], datasets[right], identifier,
            normalize_country=normalized,
        )
        statuses = agreement.status.value_counts()
        agreement_records.append({
            "mode": "normalized_country" if normalized else "raw",
            "relationship": f"{left} → {right}", "shared_ids": len(agreement),
            "unequal_country_sets": int((~agreement.sets_equal).sum()),
            **{status: int(statuses.get(status, 0))
               for status in ("agree", "disagree", "ambiguous", "missing_country")},
        })
show_report(pd.DataFrame(agreement_records), "Country agreement on shared IDs")

linked_orders, link_audit = link_orders_customers(
    orders, labels, customers, normalize_country=True,
)
show_report(pd.DataFrame([{"dataset": name, **link_audit[name]}
                         for name in datasets]), "Safe linkage — source-key audit")
show_report(pd.DataFrame([{
    key: value for key, value in link_audit.items() if key not in datasets
}]), "Safe linkage — retained order coverage")

# %% [markdown]
# ### 4.4 Findings and handling
#
# Customer IDs are globally unique. Three order IDs each occur in two countries; country plus order ID is unique in orders and labels. Customer IDs in labels repeat because one customer can have many orders. Use the validated composite-key bridge rather than an ID-only join.
#
# After country normalization:
#
# - Orders → labels: **949,874 / 949,889** matched; **15** unmatched.
# - Labels → orders: **949,874 / 950,318** matched; **444** unmatched.
# - Labels → customers: **950,304 / 950,318** matched; **14** unmatched.
# - Customers → labels: **622,258 / 901,312** matched; **279,054** unmatched.
#
# Country-set differences on shared IDs fall from **303 to 2** for orders/labels and **569 to 0** for labels/customers after normalization. Multi-country IDs remain ambiguous even when their sets agree. The safe bridge retains **949,889 orders**, of which **949,874 have labels** and **949,872 also have customers**.
#
# Keep unlinked source rows visible in coverage reports. Exclude unlabelled orders from supervised targets and fraud-rate denominators; never interpret an absent label as nonfraud. Mark unavailable customer features explicitly. Quarantine conflicting keys if future data introduces them.

# %% [markdown]
# ## 5. Business logic checks
#
# Check refunds against order values, cancelled/refunded counts against total orders, negative numeric values, missing labels/customer links, and orders before the customer's first order. Each result gives flagged, evaluable and excluded counts with its counting unit. Numeric comparisons exclude missing, nonnumeric and infinite values. The date comparison uses safely linked unique orders and calendar days, so an order on the same day as a later first-order timestamp is not a false violation.

# %%
from mlops_fraud_prediction_ca1.logic import business_logic_report

logic_evidence = business_logic_report(
    customers, orders, labels, normalize_country=True,
).reset_index()
logic_evidence["flagged_rate"] = logic_evidence.flagged.div(
    logic_evidence.evaluable.where(logic_evidence.evaluable.ne(0))
)
show_report(logic_evidence, "Business logic — flagged counts and evaluable denominators")

# %% [markdown]
# ### 5.1 Findings and handling
#
# Refunds exceed order values in **199,687 / 949,889 orders (21.0221%)**. Retain and flag these records for source confirmation; exceedances are not established errors, and not all can be explained by cent rounding. Do not silently clamp refunds or remove possible fraud signals.
#
# No cancelled/refunded counts exceed total orders, and no selected numeric values are negative. The missing item count is excluded from its numeric comparison. **105 / 949,101 evaluable orders** precede the customer's first-order calendar day; **788 orders** cannot be evaluated because of missing linkage or timestamps. Retain and flag the date violations rather than correcting dates without evidence.
#
# **15 orders lack a label key**, and **2 labelled orders lack a customer**. Preserve these coverage exceptions and follow the exclusions and unavailable-feature handling in section 4.

# %% [markdown]
# ## 6. Dates and fraud patterns
#
# ### 6.1 Date ranges and formats
#
# Parse only the three observed explicit order-date formats and the specified customer timestamp shape. The summaries distinguish missing, unrecognized and invalid values, and count each recognized format. Ranges use valid parsed dates; the raw date columns remain unchanged.

# %%
from mlops_fraud_prediction_ca1.temporal import (
    date_profile, fraud_rates, weekly_fraud_rates, largest_weekly_change,
)

date_profiles = {
    "orders.order_date": date_profile(orders.order_date),
    "customers.first_order_datetime": date_profile(
        customers.first_order_datetime, kind="customer",
    ),
}
show_report(pd.DataFrame([
    {"column": name, **{key: value for key, value in report.items() if key != "formats"}}
    for name, report in date_profiles.items()
]), "Date status counts and valid ranges")
show_report(pd.DataFrame([
    {"column": name, "format": fmt, "rows": count}
    for name, report in date_profiles.items() for fmt, count in report["formats"].items()
]), "Recognized date formats — source-row counts")

# %% [markdown]
# #### 6.1.1 Findings and handling
#
# All **949,889 order dates** parse: `%Y/%m/%d` (949,180), `%Y-%m-%d` (200) and `%Y/%b/%d` (509). They span **2022-11-12 to 2023-05-11**. Customer timestamps use `%Y-%m-%d %H:%M:%S`: **900,600 valid**, **712 missing**, and no invalid or unrecognized values, spanning **2012-03-25 12:53:43 to 2023-05-10 20:59:37**.
#
# Retain raw dates, parse these explicit formats on working copies, and leave missing timestamps missing. Isolate any invalid or unrecognized dates in later batches rather than guessing a format.

# %% [markdown]
# ### 6.2 Fraud rates by country
#
# Use safely linked unique orders. Fraud rates divide valid fraud labels by valid 0/1 labels; missing and invalid labels are excluded from the denominator and reported separately. Customer linkage is not required for a labelled order to contribute to its country's rate. These are descriptive rates and do not establish the cause of country differences.

# %%
country_rates = fraud_rates(linked_orders, ["country_code"])
show_report(country_rates, "Country fraud rates — safely linked orders")

# %% [markdown]
# #### 6.2.1 Findings
#
# The matched-order fraud rates are **BD 29.2706%**, **MY 3.8950%**, **PH 8.4721%**, **PK 24.5201%** and **TH 5.9677%**. Together the groups contain **104,422 fraud labels / 949,874 valid labels**; 15 unlabelled orders are excluded. This population differs from the 950,318 raw labels in section 3.7. Preserve country counts alongside rates when comparing groups.

# %% [markdown]
# ### 6.3 Weekly fraud rates and dated movements
#
# Weeks run Monday–Sunday. Show the aggregate series and every country/week with fraud counts, valid-label denominators, excluded labels and partial-week flags. Date exclusions are disclosed separately. The first and last observed weeks are partial; other weeks are not guaranteed to be fully sampled.
#
# The movement table selects the largest absolute change between consecutive calendar weeks for each country and the aggregate. Missing weeks and undefined rates are not compared. Changes are shown in percentage points. These are descriptive candidates, not statistically established change points or causal evidence.

# %%
weekly_country_rates = weekly_fraud_rates(linked_orders)
print(f"Orders excluded from weekly rates for missing/invalid dates: "
      f"{weekly_country_rates.attrs['excluded_date_rows']:,}")
count_columns = ["rows", "valid_labels", "fraud_labels", "missing_labels", "invalid_labels"]
weekly_overall_rates = weekly_country_rates.groupby("week_start", as_index=False)[count_columns].sum()
weekly_overall_rates["country_code"] = "ALL"
weekly_overall_rates["week_end"] = weekly_overall_rates.week_start + pd.Timedelta(days=6)
weekly_overall_rates["partial_week"] = weekly_overall_rates.week_start.isin(
    weekly_country_rates.loc[weekly_country_rates.partial_week, "week_start"]
)
weekly_overall_rates["fraud_rate"] = weekly_overall_rates.fraud_labels.div(
    weekly_overall_rates.valid_labels.where(weekly_overall_rates.valid_labels.ne(0))
)
show_report(weekly_overall_rates, "Weekly fraud rates — all countries")
for country, group in weekly_country_rates.groupby("country_code", sort=True):
    show_report(group, f"Weekly fraud rates — {country}")

weekly_changes = largest_weekly_change(pd.concat(
    [weekly_country_rates, weekly_overall_rates], ignore_index=True,
))
weekly_changes["rate_change_pp"] = weekly_changes.pop("rate_change") * 100
show_report(weekly_changes, "Largest absolute adjacent-week movements — descriptive candidates")

# %% [markdown]
# #### 6.3.1 Findings and handling
#
# No order dates are excluded from weekly rates. The aggregate's largest absolute adjacent-week movement is the rise from the week starting **2023-03-06** to **2023-03-13**: **11.7950% (4,540 / 38,491)** to **14.2392% (5,295 / 37,186)**, an increase of **2.4443 percentage points**. The movement table dates the largest candidate for each country and identifies whether a partial boundary week is involved.
#
# Preserve the weekly counts, denominators and boundary flags. Investigate dated movements alongside country mix and data coverage before attributing them to changing fraud behavior; this descriptive screen does not prove a change point. Keep any later model validation chronological where appropriate and fit preparation steps on training data only.
