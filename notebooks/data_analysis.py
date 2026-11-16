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
# This notebook profiles SP-buy’s customer, order and fraud-label files to identify risks to fraud modelling and decide how to handle them. It examines data quality, safe joins, business rules and fraud patterns over time.
#
# ## Findings at a glance
#
# - [Missingness and categories](#3-assess-data-quality): three columns contain missing values; `yes` verification and `on Delivery` payments need source confirmation.
# - [Safe linkage](#4-relationships-between-datasets): use country plus order ID, then link customers through labels. Fifteen orders have no label.
# - [Business rules](#5-business-logic-checks): refunds exceed order values in 21.0221% of orders. Retain and flag these records for investigation.
# - [Fraud patterns](#6-dates-and-fraud-patterns): 10.9967% of raw labels are fraud. Matched-order rates vary by country and week; the March rise is descriptive evidence.

# %% [markdown]
# ## Table of contents
#
# - [1. Workflow](#1-workflow)
# - [2. Load and inspect the datasets](#2-load-and-inspect-the-datasets)
#   - [Configuration](#21-configure-the-data-directory-and-logging) · [Loader](#22-import-the-reusable-loader) · [Dimensions](#23-load-the-tables-and-check-their-dimensions)
# - [3. Assess data quality](#3-assess-data-quality)
#   - [Missing values](#31-missing-values) · [Duplicates](#32-exact-duplicates)
#   - [Categories](#33-inspect-category-values) · [Standardization preview](#34-preview-category-standardization)
#   - [Numerical distributions](#35-numerical-outliers-and-skew) · [Types and validity](#36-semantic-types-and-numeric-validity) · [Class balance](#37-class-balance)
# - [4. Relationships between datasets](#4-relationships-between-datasets)
# - [5. Business logic checks](#5-business-logic-checks)
# - [6. Dates and fraud patterns](#6-dates-and-fraud-patterns)
#   - [Date parsing](#61-date-ranges-and-formats) · [Country rates](#62-fraud-rates-by-country) · [Weekly movements](#63-weekly-fraud-rates-and-dated-movements)
#   - [Daily volume and country changes](#64-daily-volume-and-country-changes)
# - [7. Preparation decisions and limitations](#7-preparation-decisions-and-limitations)

# %% [markdown]
# ## 1. Workflow
#
# Install with `uv sync --locked`, then convert this percent-format source with `uv run jupytext --to notebook notebooks/data_analysis.py`. Open the generated local notebook and select the project `.venv` kernel.
#
# Run all cells from top to bottom with the project's `.venv` kernel. Country normalization, date parsing and validated joins use working copies; source CSVs and loaded tables are not cleaned in place. The handling decisions guide later data preparation rather than constitute a complete cleaning pipeline. Written numeric findings describe the recorded base-file snapshot. The fingerprint check stops execution on changed inputs so those findings cannot silently go stale. To analyse another snapshot, review and update the written findings and fingerprints together; the shared package functions accept other batches independently. Use the table of contents or your editor's outline to navigate.
#
# This is a full-history exploratory analysis. It chooses no train/validation/test split and exports no prepared dataset. Saved outputs preserve the evidence, not DataFrames for a later pipeline. Splitting, deterministic cleaning and training belong in reusable modules called by a later configurable workflow.
#
# Section 6.4 reports daily volume through the final observed date and scans each eligible weekly boundary. Historical full-extract findings are identified explicitly; a later protected-reader rerun displays its current scope and flags policy exclusions. Earlier full-history exploration means a later holdout cannot be described as never seen.

# %% [markdown]
# ## 2. Load and inspect the datasets
#
# Select the directory containing the three base CSVs. Loading preserves IDs, categories, flags and dates as strings. Numeric columns and missing values follow pandas defaults.

# %% [markdown]
# ### 2.1 Configure the data directory and logging
#
# Use the repository's `data/` directory when running from the project root or `notebooks/`. Set `base_data_dir` to use another location. INFO logging displays the files loaded and their dimensions on stdout. Warnings and errors stay on stderr, where notebook interfaces can highlight them.
# %%
import logging
import sys
from pathlib import Path

import pandas as pd
from IPython.display import display
from pandas.io.formats.style import Styler


def apply_table_theme(styler: Styler) -> Styler:
    """Apply the notebook's shared slate palette to a pandas table.

    Args:
        styler: Table presentation with any existing numeric formatting.

    Returns:
        The same Styler with navy headers, alternating light rows and readable
        captions. Existing cell highlights and underlying data are preserved.
    """
    table_styles = [
        {"selector": "", "props": [("border-collapse", "collapse")]},
        {
            "selector": "th, td",
            "props": [
                ("padding", "8px 12px"),
                ("border-bottom", "1px solid #cbd5e1"),
            ],
        },
        {
            "selector": "thead th",
            "props": [
                ("background-color", "#243b53"),
                ("color", "#ffffff"),
                ("text-align", "left"),
            ],
        },
        {
            "selector": "tbody th",
            "props": [
                ("background-color", "#e2e8f0"),
                ("color", "#243b53"),
                ("text-align", "left"),
                ("vertical-align", "top"),
            ],
        },
        {
            "selector": "td",
            "props": [
                ("background-color", "#ffffff"),
                ("color", "#243b53"),
            ],
        },
        {
            "selector": "tbody tr:nth-child(even) td",
            "props": [
                ("background-color", "#f1f5f9"),
            ],
        },
        {
            "selector": "caption",
            "props": [
                ("background-color", "#e2e8f0"),
                ("color", "#243b53"),
                ("padding", "10px 12px"),
                ("text-align", "left"),
                ("font-weight", "600"),
                ("caption-side", "top"),
            ],
        },
    ]
    numeric_columns = styler.data.select_dtypes(include="number").columns.tolist()
    styler = styler.set_table_styles(table_styles, overwrite=False)
    if numeric_columns:
        styler = styler.set_properties(
            subset=numeric_columns,
            **{"text-align": "right"},
        )
    return styler


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


def style_report(frame: pd.DataFrame, caption: str) -> Styler:
    """Format audit evidence without changing values or dropping report rows.

    Args:
        frame: Report with snake-case column names, counts and fractional rates.
        caption: Reader-facing title explaining population or denominator.

    Returns:
        A themed Styler with readable headings, aligned numbers, four-decimal
        rates and explicit undefined values. Nullable count columns stay integral
        in the display; other floating-point measurements use two decimals.
    """
    labels: dict[str, str] = {
        column: column.replace("_", " ").capitalize() for column in frame.columns
    }
    labels.update(
        {
            key: value
            for key, value in {
                "inferred_dtype": "Storage",
                "meaning": "Meaning / required interpretation",
                "rate_change_pp": "Change (percentage points)",
                "nonintegral_count": "Nonintegral count values",
            }.items()
            if key in frame.columns
        }
    )
    formats: dict = {}
    for column in frame.columns:
        values: pd.Series = frame[column]
        heading: str = labels[column]
        if column.endswith("rate"):
            formats[heading] = "{:.4%}"
        elif column.endswith("_pp"):
            formats[heading] = "{:+.4f}"
        elif pd.api.types.is_bool_dtype(values):
            formats[heading] = lambda value: "Yes" if value else "No"
        elif pd.api.types.is_integer_dtype(values):
            formats[heading] = "{:,}"
        elif column == "nonintegral_count":
            formats[heading] = "{:,.0f}"
        elif pd.api.types.is_float_dtype(values):
            formats[heading] = "{:,.2f}"
        elif pd.api.types.is_datetime64_any_dtype(values):
            formats[heading] = lambda value: value.isoformat(sep=" ")
    return (
        frame.rename(columns=labels)
        .style.hide(axis="index")
        .pipe(apply_table_theme)
        .set_caption(caption)
        .format(formats, escape="html", na_rep="—")
    )


def show_report(frame: pd.DataFrame, caption: str) -> None:
    """Display a complete audit report using the shared formatting helper.

    Args:
        frame: Evidence rows, preserved without rounding underlying values.
        caption: Title identifying the report and its population.

    Returns:
        None. Displays HTML in the notebook; writes no files.
    """
    display(style_report(frame, caption))


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
project_root: Path = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
base_data_dir: Path = project_root / "data"

# %% [markdown]
# ### 2.2 Import the reusable loader
#
# Import the package implementation directly so notebooks and later workflows share the same loading contract.

# %%
from mlops_fraud_prediction_ca1.data import load_base_data

# %% [markdown]
# ### 2.3 Load the tables and check their dimensions
#
# Read the customer, order, and label CSVs into separate DataFrames. The summary shows each table's row and column counts as an initial loading check; data-quality checks follow separately.
#
# Explicit string dtypes preserve leading zeros in IDs and keep `True`, `False` and `yes` as text. Section 3.4 records the verification handling decision.
# %%
customers: pd.DataFrame
orders: pd.DataFrame
labels: pd.DataFrame
# Written findings are tied to this source snapshot, not arbitrary future batches.
# Stop before displaying their numbers when any source file has changed.
from hashlib import file_digest

expected_source_sha256 = {
    "customers": "501c06997d496e127d3e1b2c28d34b47dfe0dcf124f90abf83de60867f557c7e",
    "orders": "934d9143861feac2c00aaa9bf782418e52959d13cfa7fd47b67ebf79f10e67f8",
    "labels": "cde4cc62977ba48f4402a8bac756518fceb5e899cdbeb420a773936d376d5ce4",
}
for dataset, expected_digest in expected_source_sha256.items():
    with (base_data_dir / f"{dataset}_base.csv").open("rb") as source_file:
        actual_digest = file_digest(source_file, "sha256").hexdigest()
    if actual_digest != expected_digest:
        raise ValueError(
            f"{dataset} differs from the analysed base snapshot. Review and update "
            "the written findings and source fingerprints before rerunning this notebook."
        )

customers, orders, labels = load_base_data(base_data_dir)

dataset_dimensions: pd.DataFrame = pd.DataFrame(
    {
        "dataset": ["customers", "orders", "labels"],
        "rows": [len(customers), len(orders), len(labels)],
        "columns": [len(customers.columns), len(orders.columns), len(labels.columns)],
    }
)

(
    dataset_dimensions.set_index("dataset")
    .rename_axis("Dataset")
    .rename(columns={"rows": "Rows", "columns": "Columns"})
    .style.pipe(apply_table_theme)
    .set_caption("Source dataset dimensions")
    .format("{:,}")
)

# %% [markdown]
# ### 2.4 Inspect dataset columns
#
# The semantic-type audit pairs each column's inferred storage with its intended meaning. Storage alone does not establish validity: `object` can hold text or mixed Python types, date strings need explicit parsing, and numeric IDs must not be treated as measurements. The loader prevents mixed Boolean/string storage for `mobile_verified`; sections 3.3–3.4 inspect the original tokens; section 3.6 checks numeric validity.

# %%
from mlops_fraud_prediction_ca1.quality import quality_reports

datasets: dict[str, pd.DataFrame] = {
    "customers": customers,
    "orders": orders,
    "labels": labels,
}
quality: dict[str, pd.DataFrame] = quality_reports(datasets)
show_report(
    quality["types"], "Semantic-type audit — inferred storage and intended meaning"
)

# %% [markdown]
# ## 3. Assess data quality
#
# Identify missing, inconsistent and unusual values that could affect model inputs or evaluation. The following checks distinguish observed issues from possible fraud signals and record how to handle them during data preparation.

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

missing_values: pd.DataFrame = pd.concat(
    missing_summaries, names=["dataset", "column"]
).reset_index()


(
    missing_values.set_index(["dataset", "column"])
    .rename_axis(["Dataset", "Column"])
    .rename(columns={"na_count": "Missing count", "na_percent": "Missing %"})
    .style.map(
        lambda value: (
            "background-color: #fce8ec !important; color: #8a2942 !important"
            if value > 0
            else ""
        ),
        subset=["Missing count", "Missing %"],
    )
    .pipe(apply_table_theme)
    .set_caption("Missing values by column; percentages use all rows in each dataset")
    .format({"Missing count": "{:,}", "Missing %": format_percent}, na_rep="—")
)

# %% [markdown]
# #### 3.1.1 Missingness findings
#
# Pandas-recognized missingness affects three columns:
#
# - `customers.first_order_datetime`: 712 rows (0.08%). Retain missing timestamps and exclude them from date-order comparisons; do not invent a first-order date.
# - `orders.payment_method`: 21,501 rows (2.26%). Preserve missingness as an explicit unknown category in later preparation.
# - `orders.num_items_ordered`: 1 row (<0.01%). Retain a nullable count and a missing flag. Fit any later imputation on training data only.
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
                extra_count / len(frame) * 100 if len(frame) else float("nan")
            ),
            "rows_in_duplicate_groups": int(duplicate_group_mask.sum()),
        }
    )
    duplicate_examples[name] = frame.loc[duplicate_group_mask].head(5)

duplicate_summary: pd.DataFrame = pd.DataFrame(duplicate_records)


display(
    duplicate_summary.set_index("dataset")
    .rename_axis("Dataset")
    .rename(
        columns={
            "total_rows": "Total rows",
            "extra_duplicate_rows": "Extra duplicate rows",
            "extra_duplicate_percent": "Extra duplicate %",
            "rows_in_duplicate_groups": "Rows in duplicate groups",
        }
    )
    .style.pipe(apply_table_theme)
    .set_caption(
        "Exact row duplicates; percentages count extra copies against all dataset rows"
    )
    .format(
        {
            "Total rows": "{:,}",
            "Extra duplicate rows": "{:,}",
            "Extra duplicate %": format_percent,
            "Rows in duplicate groups": "{:,}",
        },
        na_rep="—",
    )
)
for name, examples in duplicate_examples.items():
    if not examples.empty:
        display(
            examples.style.pipe(apply_table_theme)
            .set_caption(f"{name}: up to five rows from exact duplicate groups")
            .format(escape="html", na_rep="—")
        )

# %% [markdown]
# #### 3.2.1 Exact-duplicate findings
#
# No exact duplicate rows were found in customers (901,312 rows), orders (949,889 rows), or labels (950,318 rows), so no rows need removal on this basis. Section 4 checks repeated keys and conflicting records separately. If copies appear in future batches, collapse identical full rows only and audit the count; never resolve conflicting keys by selecting an arbitrary first row.

# %% [markdown]
# ### 3.3 Inspect category values
#
# The decision summary highlights observed country variants and unresolved verification/payment values. It is a focused exception list, not a closed list of valid categories. Collection and payment case variants are formatting proposals; their semantic domains remain unconfirmed.
#
# The supporting inventories retain **every raw value and its exact count**, including blank fields and original spellings. Category columns are reread as text; quotes make empty values and whitespace visible. Dates, identifiers, amounts and numerical counts are inspected separately.

# %%
from html import escape

from IPython.display import HTML

category_columns: dict[str, list[str]] = {
    "customers": ["country_code", "mobile_verified"],
    "orders": ["country_code", "collect_type", "payment_method"],
    "labels": ["country_code", "is_fraud"],
}
category_frequencies: dict[tuple[str, str], pd.Series] = {}
category_tables: dict[tuple[str, str], str] = {}

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
        top_values: pd.DataFrame = frequencies.rename_axis("Value").reset_index(
            name="Count"
        )
        top_values["Value"] = top_values["Value"].map(repr)
        category_table: str = (
            top_values.style.hide(axis="index")
            .pipe(apply_table_theme)
            .set_table_styles(
                [
                    {
                        "selector": "",
                        "props": [
                            ("width", "100%"),
                            ("table-layout", "fixed"),
                            ("margin", "0"),
                        ],
                    },
                    {
                        "selector": "th, td, caption",
                        "props": [
                            ("white-space", "normal"),
                            ("overflow-wrap", "anywhere"),
                        ],
                    },
                ],
                overwrite=False,
            )
            .set_caption(
                '<span style="display: block;">'
                f"{escape(name.title())}: {escape(column)}</span>"
                '<span style="display: block; font-size: 0.85em; '
                'font-weight: 400;">'
                f"Distinct values: {len(frequencies):,} · All values shown</span>"
            )
            .format({"Value": "{}", "Count": "{:,}"}, escape="html")
            .to_html()
        )
        category_tables[name, column] = category_table

category_layout: list[list[tuple[str, str]]] = [
    [("customers", "country_code"), ("customers", "mobile_verified")],
    [("orders", "country_code"), ("orders", "collect_type")],
    [("orders", "payment_method")],
    [("labels", "country_code"), ("labels", "is_fraud")],
]
category_stacks: list[str] = [
    '<div style="display: flex; flex-direction: column; flex: 1 1 260px; '
    'min-width: 0; max-width: 100%; gap: 20px;">'
    + "".join(
        '<div style="width: 100%; min-width: 0;">' + category_tables[key] + "</div>"
        for key in stack
    )
    + "</div>"
    for stack in category_layout
]
category_decision_records: list[dict[str, str | int]] = []
for name, column, value, decision in [
    ("customers", "country_code", "pk", "Uppercase to PK on working copies"),
    ("customers", "country_code", "my", "Uppercase to MY on working copies"),
    ("labels", "country_code", "ph", "Uppercase to PH on working copies"),
    ("labels", "country_code", "my", "Uppercase to MY on working copies"),
    (
        "customers",
        "mobile_verified",
        "yes",
        "Retain unresolved; confirm before mapping",
    ),
    ("orders", "payment_method", "", "Keep missing; explicit unknown policy later"),
    (
        "orders",
        "payment_method",
        "on Delivery",
        "Retain separately; equivalence unconfirmed",
    ),
]:
    category_decision_records.append(
        {
            "dataset / column": f"{name}.{column}",
            "raw_value": repr(value),
            "rows": int(category_frequencies[name, column].get(value, 0)),
            "handling_proposal": decision,
        }
    )
show_report(
    pd.DataFrame(category_decision_records),
    "Category exceptions and handling proposals",
)
display(
    HTML(
        "<details><summary>Supporting evidence: complete raw category inventories</summary>"
        '<div style="display: flex; flex-wrap: wrap; align-items: flex-start; '
        'gap: 20px; width: 100%; max-width: 100%; margin: 20px 0;">'
        + "".join(category_stacks)
        + "</div></details>"
    )
)

# %% [markdown]
# ### 3.4 Preview category standardization
#
# This compares each original category value with its proposed formatting, frequency and validation status. The preview trims surrounding whitespace, uppercases country codes and lowercases other categories. The complete original/proposed-value tables are retained as supporting evidence.
#
# This also checks `mobile_verified` against true/false and `is_fraud` against 0/1. Other nonblank categories show `unchecked` because no allowed values are defined. Whitespace trimming is a precaution, not a confirmed data issue. See the [standardization notes](../docs/category-standardisation.md) for details.

# %%
from mlops_fraud_prediction_ca1.categories import standardise_categories

category_previews: dict[tuple[str, str], pd.DataFrame] = {}
category_preview_tables: dict[tuple[str, str], str] = {}

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
            allowed_values=(
                {"true", "false"}
                if column == "mobile_verified"
                else {"0", "1"}
                if column == "is_fraud"
                else None
            ),
        )
        category_previews[name, column] = preview
        category_preview_tables[name, column] = (
            preview.rename(
                columns={
                    "original_value": "Original value",
                    "proposed_value": "Standardized value",
                    "count": "Count",
                    "status": "Status",
                }
            )
            .style.hide(axis="index")
            .pipe(apply_table_theme)
            .set_table_styles(
                [
                    {
                        "selector": "",
                        "props": [
                            ("width", "100%"),
                            ("table-layout", "fixed"),
                            ("margin", "0"),
                        ],
                    },
                    {
                        "selector": "th, td, caption",
                        "props": [
                            ("white-space", "normal"),
                            ("overflow-wrap", "anywhere"),
                        ],
                    },
                ],
                overwrite=False,
            )
            .set_caption(f"{escape(name.title())}: {escape(column)}")
            .set_properties(
                subset=["Original value", "Standardized value"],
                **{"white-space": "pre-wrap"},
            )
            .format(
                {"Count": "{:,}"},
                escape="html",
                na_rep="Missing",
            )
            .to_html()
        )

category_preview_stacks: list[str] = [
    '<div style="display: flex; flex-direction: column; flex: 1 1 420px; '
    'min-width: 0; max-width: 100%; gap: 20px;">'
    + "".join(
        '<div style="width: 100%; min-width: 0;">'
        + category_preview_tables[key]
        + "</div>"
        for key in stack
    )
    + "</div>"
    for stack in category_layout
]
display(
    HTML(
        "<details><summary>Supporting evidence: complete category-formatting proposals</summary>"
        '<div style="display: flex; flex-wrap: wrap; align-items: flex-start; '
        'gap: 20px; width: 100%; max-width: 100%; margin-bottom: 20px;">'
        + "".join(category_preview_stacks)
        + "</div></details>"
    )
)

# %% [markdown]
# #### 3.4.1 Category findings and handling
#
# The preview brings country, collection and payment case variants into a consistent format. All fraud labels pass the 0/1 check, but 165 `mobile_verified` values contain `yes` and remain unexpected. These require confirmation before mapping to true.
#
# **Verification handling decision:** Load the column as strings rather than a mix of Python booleans and strings. In later preparation, trim and case-normalize a working copy, map only `true`/`false` to nullable booleans, and leave `yes` as unknown with an investigation flag until its meaning is confirmed. Never filter the raw column with `== True`: text `True` would be silently excluded. Keep the original tokens for audit.
#
# Payment methods include 21,501 blanks and 160 `on Delivery` values. The blanks need an explicit missing-value policy; `on Delivery` remains separate from `payment on delivery` until their equivalence is confirmed. Formatting alone does not establish category validity.

# %% [markdown]
# ### 3.5 Numerical outliers and skew
#
# Inspect the eight amount/count features in customers and orders. IDs, country codes, verification flags, and the fraud target are excluded.
#
# The interquartile range is $\mathrm{IQR}=Q_3-Q_1$. Flag values satisfying:
#
# $$
# x < Q_1 - 1.5\,\mathrm{IQR}
# \quad \text{or} \quad
# x > Q_3 + 1.5\,\mathrm{IQR}.
# $$
#
# These flags identify unusual values, not proven errors. When IQR is zero, every value different from the shared quartile is flagged.
#
# Statistics use finite, nonmissing values; missing and infinite values are reported separately. Positive sample skewness indicates right-tailed asymmetry, and negative skewness indicates left-tailed asymmetry. Skewness is undefined for constant features or fewer than three finite observations.

# %% [markdown]
# #### 3.5.1 Distribution and screening summaries
#
# The summaries describe typical values, tails and skewness, alongside the IQR fences and flagged proportions. They help distinguish sparse or heavily skewed features from observations needing investigation. Zero and flagged percentages use each feature's finite observation count.

# %%
from mlops_fraud_prediction_ca1.numeric import profile_numeric_features

iqr_multiplier: float = 1.5
numeric_features: dict[str, list[str]] = {
    "customers": [
        "num_orders_last_50days",
        "num_cancelled_orders_last_50days",
        "num_refund_orders_last_50days",
        "total_payment_last_50days",
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
    "finite_count": "Finite rows",
    "missing_count": "Missing",
    "infinite_count": "Infinite",
    "zero_percent": "Zero %",
    "minimum": "Minimum",
    "mean": "Mean",
    "median": "Median",
    "p99": "P99",
    "maximum": "Maximum",
    "skewness": "Skewness",
}
screening_columns: dict[str, str] = {
    "q1": "Q1",
    "q3": "Q3",
    "lower_fence": "Lower fence",
    "upper_fence": "Upper fence",
    "flagged_count": "Flagged rows",
    "flagged_percent": "Flagged %",
    "zero_iqr": "Zero IQR",
}

for dataset_name in numeric_features:
    display(
        numeric_summary.loc[dataset_name, list(distribution_columns)]
        .rename(index=numeric_labels, columns=distribution_columns)
        .rename_axis("Feature")
        .style.pipe(apply_table_theme)
        .set_caption(f"{dataset_name.title()} — numerical distributions")
        .format(
            {
                "Finite rows": "{:,}",
                "Missing": "{:,}",
                "Infinite": "{:,}",
                "Zero %": format_percent,
                "Minimum": "{:,.2f}",
                "Mean": "{:,.2f}",
                "Median": "{:,.2f}",
                "P99": "{:,.2f}",
                "Maximum": "{:,.2f}",
                "Skewness": "{:.2f}",
            },
            na_rep="—",
        )
    )
    display(
        numeric_summary.loc[dataset_name, list(screening_columns)]
        .rename(index=numeric_labels, columns=screening_columns)
        .rename_axis("Feature")
        .style.pipe(apply_table_theme)
        .set_caption(
            f"{dataset_name.title()} — potential outliers ({iqr_multiplier:g} × IQR)"
        )
        .format(
            {
                "Q1": "{:,.2f}",
                "Q3": "{:,.2f}",
                "Lower fence": "{:,.2f}",
                "Upper fence": "{:,.2f}",
                "Flagged rows": "{:,}",
                "Flagged %": format_percent,
                "Zero IQR": lambda value: "Yes" if value else "No",
            },
            na_rep="—",
        )
    )

# %% [markdown]
# #### 3.5.2 Histograms and box plots
#
# Each histogram shows values through the feature's 99th percentile so the main distribution remains readable. Its title reports how many values lie above that view; the count axis is logarithmic. The paired boxplot includes the full finite range and uses the same 1.5 × IQR screening rule.
#
# The tables' skewness values use the full finite data, including the tails. The restricted histogram view does not remove observations from the data.
#
# Monetary axes use SGD, as specified for `total_payment_last_50days`, `order_value` and `refund_value` in Appendix A1 of the supplied specification; count axes use integer ticks. The cancelled/refunded-count boxes collapse at zero because Q1 = Q3 = 0. Their nonzero points reflect sparse distributions, not established errors.

# %%
# %matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator


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
        len(selected_features),
        2,
        figsize=(14, 3.1 * len(selected_features)),
        squeeze=False,
    )
    figure.suptitle(
        f"{table_title} — numerical feature distributions", fontweight="bold"
    )
    for row_number, plotted_feature in enumerate(selected_features):
        plotted_values: pd.Series = table[plotted_feature]
        finite_plotted_values: pd.Series = plotted_values.loc[
            plotted_values.notna() & ~plotted_values.isin([float("-inf"), float("inf")])
        ]
        histogram_axis, boxplot_axis = axes[row_number]
        feature_title: str = feature_labels[plotted_feature]
        is_count: bool = plotted_feature.startswith("num_")
        axis_label: str = feature_title if is_count else f"{feature_title} (SGD)"
        if len(finite_plotted_values) == 0:
            for empty_axis in (histogram_axis, boxplot_axis):
                empty_axis.set_title(feature_title)
                empty_axis.text(0.5, 0.5, "No finite observations", ha="center")
            continue
        cutoff: float = float(finite_plotted_values.quantile(0.99))
        body: pd.Series = finite_plotted_values.loc[finite_plotted_values.le(cutoff)]
        hidden_count: int = len(finite_plotted_values) - len(body)
        sns.histplot(
            x=body,
            bins=40,
            discrete=plotted_feature.startswith("num_"),
            color="#486581",
            ax=histogram_axis,
        )
        histogram_axis.set_yscale("log")
        histogram_axis.set_title(
            f"{feature_title}\nThrough P99; {hidden_count:,} values above view"
        )
        histogram_axis.set_xlabel(axis_label)
        histogram_axis.set_ylabel("Rows (log scale)")
        sns.boxplot(
            x=finite_plotted_values,
            whis=fence_multiplier,
            fliersize=2,
            color="#9fb3c8",
            ax=boxplot_axis,
        )
        boxplot_axis.set_title(f"{feature_title}\nFull finite range")
        boxplot_axis.set_xlabel(axis_label)
        if is_count:
            for count_axis in (histogram_axis, boxplot_axis):
                count_axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        if finite_plotted_values.quantile(0.75) == finite_plotted_values.quantile(0.25):
            boxplot_axis.text(
                0.98,
                0.85,
                "Q1 = Q3: box collapses at zero",
                ha="right",
                transform=boxplot_axis.transAxes,
                fontsize=9,
                color="#243b53",
            )
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    return figure


# noinspection SpellCheckingInspection
sns.set_theme(style="whitegrid")
for dataset_name, feature_names in numeric_features.items():
    distribution_figure: Figure = plot_numeric_distributions(
        datasets[dataset_name],
        feature_names,
        numeric_labels,
        dataset_name.title(),
        iqr_multiplier,
    )
    display(distribution_figure)
    plt.close(distribution_figure)

# %% [markdown]
# #### 3.5.3 Extreme examples
#
# The examples locate up to three smallest and three largest flagged observations per feature for investigation. The source row index is zero-based and refers to the loaded table. An observation appearing at both ends of a short list is shown once.

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
        extreme_values: pd.Series = pd.concat(
            [flagged_values.head(3), flagged_values.tail(3)]
        )
        extreme_values = extreme_values.loc[~extreme_values.index.duplicated()]
        examples_table: pd.DataFrame = extreme_values.rename_axis(
            "Source row index"
        ).reset_index(name="Value")
        outlier_examples[dataset_name, feature_name] = examples_table
        display(
            examples_table.style.hide(axis="index")
            .pipe(apply_table_theme)
            .set_caption(
                f"{dataset_name.title()} — {numeric_labels[feature_name]}: extremes"
            )
            .format({"Source row index": "{:d}", "Value": "{:,.2f}"})
        )

# %% [markdown]
# #### 3.5.4 Distribution findings and handling
#
# - All eight features are right-skewed: sample skewness ranges from 5.42 to 55.00. Canceled-order counts (55.00) and items ordered (41.04) have the strongest asymmetry.
# - Canceled and refunded order counts are 83.14% and 85.76% zero. Both have zero IQR, so the rule flags every nonzero count: 16.86% and 14.24% of rows, respectively. Those proportions reflect sparse distributions, not established data errors.
# - The other six features have 4.35%–7.48% of finite observations outside their IQR fences. These are per-feature rates; they must not be added together because the same record can be flagged in several columns.
# - The largest item count is 1,600, compared with a median of 2 and P99 of 22. Customer payment totals reach 42,569.35, compared with a median of 164.58 and P99 of 1,876.68. The examples locate these extremes for further investigation without classifying them as errors.
# - These features contain no negative or infinite values. The one missing item count is excluded from that feature's statistics and percentage denominator.
#
# Retain raw values and use IQR flags for investigation. Do not cap or delete extremes without evidence of an error. Evaluate any learned transformation on training data only. No capping, removal, or transformation has been applied.

# %% [markdown]
# ### 3.6 Semantic types and numeric validity
#
# The semantic-type audit in section 2.4 provides the column inventory. IDs should remain text; dates require explicit parsing, amounts require numeric precision, and counts should be finite integers. The numeric audit separates source missingness, conversion failures, infinities, negative values and nonintegral counts. Counts use source rows; a missing item count is not a failed conversion.

# %%
show_report(
    quality["numeric_validity"],
    "Numeric validity — source-row counts; — means not applicable",
)

# %% [markdown]
# #### 3.6.1 Numeric validity and category handling
#
# **Numeric findings:** All eight selected numeric features have zero conversion failures, infinities and negative values; count features have no nonintegral observations. The item count has one source-missing value.
#
# **Preparation:** Keep IDs as text and money as decimal-valued numbers. Convert finite integral counts to nullable integers during later preparation, preserving missing/invalid flags; quarantine failed conversions rather than substituting zero.
#
# **Country formatting:** Original country case variants include customer `pk` (225) and `my` (165), and label `ph` (180) and `my` (121). Normalize country formatting only on working copies for the relationship checks.
#
# **Other categories:** Collection and payment case variants can be compared in lowercase, but their semantic domains remain unconfirmed. The full raw inventories above expose blank values and original spellings; no missing-value alias or semantic mapping is applied.

# %% [markdown]
# ### 3.7 Class balance
#
# Measure fraud prevalence in the raw label table, independently of whether labels match an order. The denominator includes only valid 0/1 labels; missing and invalid labels are counted separately. Section 6 uses safely matched orders, so its denominator differs.

# %%
show_report(quality["class_balance"], "Class balance — raw label rows")

# %% [markdown]
# #### 3.7.1 Class balance and evaluation implications
#
# There are 104,504 fraud and 845,814 nonfraud labels: **10.9967% fraud** among 950,318 valid labels, with no missing or invalid labels.
#
# Preserve this prevalence in evaluation. Compare fraud precision, recall and PR-AUC against a baseline; apply any weighting or resampling within training folds only, leaving validation and test populations representative.

# %% [markdown]
# ## 4. Relationships between datasets
#
# Orders contain no customer ID. Establish key uniqueness, coverage and country agreement before linking **orders → labels → customers**. Report raw country values alongside uppercase/trimmed proposals.
#
# ### 4.1 Key uniqueness
#
# Check IDs alone and with `country_code`. `duplicate_rows` includes every row in a repeated-key group; `extra_key_rows` counts rows beyond one per key. `conflicting_keys` identifies keys with differing full records. Uniqueness refers to complete keys; null-key rows are disclosed separately.

# %%
from mlops_fraud_prediction_ca1.relationships import (
    bidirectional_matches,
    country_agreement,
    key_profile,
    link_orders_customers,
)

key_records = []
for normalized in (False, True):
    for name, frame in datasets.items():
        for identifier in ("order_id", "customer_id"):
            if identifier not in frame:
                continue
            for keys in ([identifier], ["country_code", identifier]):
                key_records.append(
                    {
                        "mode": "normalized_country" if normalized else "raw",
                        "dataset": name,
                        "keys": "+".join(keys),
                        **key_profile(frame, keys, normalize_country=normalized),
                    }
                )
key_evidence = pd.DataFrame(key_records)
for mode, group in key_evidence.groupby("mode", sort=False):
    show_report(group.drop(columns="mode"), f"Key uniqueness — {mode}")

# %% [markdown]
# ### 4.2 Match rates in both directions
#
# For each file pair, `left_to_right` follows the names in the relationship column; `right_to_left` reverses them. Row match rates use complete-key source rows, retaining duplicate row weight. Distinct-key rates use distinct complete keys. Null keys are reported separately and never match. Existence of a match alone does not establish a safe join.

# %%
coverage_parts = []
relationship_pairs = (
    ("orders", "labels", "order_id"),
    ("labels", "customers", "customer_id"),
)
for normalized in (False, True):
    for left, right, identifier in relationship_pairs:
        for keys in ([identifier], ["country_code", identifier]):
            coverage = bidirectional_matches(
                datasets[left],
                datasets[right],
                keys,
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
            datasets[left],
            datasets[right],
            identifier,
            normalize_country=normalized,
        )
        statuses = agreement.status.value_counts()
        agreement_records.append(
            {
                "mode": "normalized_country" if normalized else "raw",
                "relationship": f"{left} → {right}",
                "shared_ids": len(agreement),
                "unequal_country_sets": int((~agreement.sets_equal).sum()),
                **{
                    status: int(statuses.get(status, 0))
                    for status in ("agree", "disagree", "ambiguous", "missing_country")
                },
            }
        )
show_report(pd.DataFrame(agreement_records), "Country agreement on shared IDs")

linked_orders, link_audit = link_orders_customers(
    orders,
    labels,
    customers,
    normalize_country=True,
)
show_report(
    pd.DataFrame([{"dataset": name, **link_audit[name]} for name in datasets]),
    "Safe linkage — source-key audit",
)
show_report(
    pd.DataFrame(
        [{key: value for key, value in link_audit.items() if key not in datasets}]
    ),
    "Safe linkage — retained order coverage",
)

# %% [markdown]
# ### 4.4 Linkage findings and handling
#
# Customer IDs are globally unique. Three order IDs each occur in two countries; country plus order ID is unique in orders and labels. Customer IDs in labels repeat because one customer can have many orders. Use the validated composite-key bridge rather than an ID-only join.
#
# After country normalization:
#
# - Orders → labels: 949,874 / 949,889 matched; 15 unmatched.
# - Labels → orders: 949,874 / 950,318 matched; 444 unmatched.
# - Labels → customers: 950,304 / 950,318 matched; 14 unmatched.
# - Customers → labels: 622,258 / 901,312 matched; 279,054 unmatched.
#
# Country-set differences on shared IDs fall from 303 to 2 for orders/labels and 569 to 0 for labels/customers after normalization. Multi-country IDs remain ambiguous even when their sets agree. The safe bridge retains 949,889 orders, of which 949,874 have labels and 949,872 also have customers.
#
# Keep unlinked source rows visible in coverage reports. Exclude unlabelled orders from supervised targets and fraud-rate denominators; never interpret an absent label as nonfraud. Mark unavailable customer features explicitly. Quarantine conflicting keys if future data introduces them.

# %% [markdown]
# ## 5. Business logic checks
#
# Check refunds against order values, cancelled/refunded counts against total orders, negative numeric values, missing labels/customer links, and orders before the customer's first order. Each result gives flagged, evaluable and excluded counts with its counting unit. Numeric comparisons exclude missing, nonnumeric and infinite values. The date comparison uses safely linked unique orders and calendar days, so an order on the same day as a later first-order timestamp is not a false violation.

# %%
from mlops_fraud_prediction_ca1.logic import business_logic_report

logic_evidence = business_logic_report(
    customers,
    orders,
    labels,
    normalize_country=True,
).reset_index()
logic_evidence["flagged_rate"] = logic_evidence.flagged.div(
    logic_evidence.evaluable.where(logic_evidence.evaluable.ne(0))
)
show_report(
    logic_evidence, "Business logic — flagged counts and evaluable denominators"
)

# %% [markdown]
# ### 5.1 Business-rule findings and handling
#
# Refunds exceed order values in 199,687 / 949,889 orders (21.0221%). Retain and flag these records for source confirmation; exceedances are not established errors, and not all can be explained by cent rounding. Do not silently clamp refunds or remove possible fraud signals.
#
# No cancelled/refunded counts exceed total orders, and no selected numeric values are negative. The missing item count is excluded from its numeric comparison. 105 / 949,101 evaluable orders precede the customer's first-order calendar day; 788 orders cannot be evaluated because of missing linkage or timestamps. Retain and flag the date violations rather than correcting dates without evidence.
#
# 15 orders lack a label key, and 2 labelled orders lack a customer. Preserve these coverage exceptions and follow the exclusions and unavailable-feature handling in section 4.

# %% [markdown]
# ## 6. Dates and fraud patterns
#
# ### 6.1 Date ranges and formats
#
# Parse only the three observed explicit order-date formats and the specified customer timestamp shape. The summaries distinguish missing, unrecognized and invalid values, and count each recognized format. Ranges use valid parsed dates.

# %%
from mlops_fraud_prediction_ca1.temporal import (
    date_profile,
    fraud_rates,
    largest_weekly_change,
    weekly_fraud_rates,
)

date_profiles = {
    "orders.order_date": date_profile(orders.order_date),
    "customers.first_order_datetime": date_profile(
        customers.first_order_datetime,
        kind="customer",
    ),
}
show_report(
    pd.DataFrame(
        [
            {
                "column": name,
                **{key: value for key, value in report.items() if key != "formats"},
            }
            for name, report in date_profiles.items()
        ]
    ),
    "Date status counts and valid ranges",
)
show_report(
    pd.DataFrame(
        [
            {"column": name, "format": fmt, "rows": count}
            for name, report in date_profiles.items()
            for fmt, count in report["formats"].items()
        ]
    ),
    "Recognized date formats — source-row counts",
)

# %% [markdown]
# #### 6.1.1 Date-parsing findings and handling
#
# All 949,889 order dates parse: `%Y/%m/%d` (949,180), `%Y-%m-%d` (200) and `%Y/%b/%d` (509). They span 2022-11-12 to 2023-05-11. Customer timestamps use `%Y-%m-%d %H:%M:%S`: 900,600 valid, 712 missing, and no invalid or unrecognized values, spanning 2012-03-25 12:53:43 to 2023-05-10 20:59:37.
#
# **Numeric date order:** Both numeric formats are interpreted as year/month/day. In the base snapshot, the middle field never exceeds 12, while the third exceeds 12 in 569,616 slash rows and 41 dash rows; this supports the same month/day order across each format. Another 379,564 slash rows and 159 dash rows are ambiguous individually, so their interpretation depends on that dataset-level convention. Confirm it again for later sources. English month abbreviations are parsed case-insensitively; all 509 current values are title case.
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
# #### 6.2.1 Country fraud-rate findings
#
# The matched-order fraud rates are BD 29.2706%, MY 3.8950%, PH 8.4721%, PK 24.5201% and TH 5.9677%. Together the groups contain 104,422 fraud labels / 949,874 valid labels; 15 unlabelled orders are excluded. This population differs from the 950,318 raw labels in section 3.7. Preserve country counts alongside rates when comparing groups.

# %% [markdown]
# ### 6.3 Weekly fraud rates and dated movements
#
# Weeks run Monday–Sunday. Show the aggregate series and every country/week with fraud counts, valid-label denominators, excluded labels and partial-week flags. Date exclusions are disclosed separately. The first and last observed weeks are partial; other weeks are not guaranteed to be fully sampled.
#
# The movement table selects the largest absolute change between consecutive calendar weeks for each country and the aggregate. Missing weeks and undefined rates are not compared. Changes are shown in percentage points. These are descriptive candidates, not statistically established change points or causal evidence.
#
# The chart compares the aggregate and five countries on common axes. Open markers and shaded bands identify partial boundary weeks. Expand the supporting evidence for every weekly count, denominator and exclusion; the movement summary remains visible below the chart.

# %%
weekly_country_rates = weekly_fraud_rates(linked_orders)
logging.getLogger(__name__).info(
    f"Orders excluded from weekly rates for missing/invalid dates: "
    f"{weekly_country_rates.attrs['excluded_date_rows']:,}"
)
count_columns = [
    "rows",
    "valid_labels",
    "fraud_labels",
    "missing_labels",
    "invalid_labels",
]
weekly_overall_rates = weekly_country_rates.groupby("week_start", as_index=False)[
    count_columns
].sum()
weekly_overall_rates["country_code"] = "ALL"
weekly_overall_rates["week_end"] = weekly_overall_rates.week_start + pd.Timedelta(
    days=6
)
weekly_overall_rates["partial_week"] = weekly_overall_rates.week_start.isin(
    weekly_country_rates.loc[weekly_country_rates.partial_week, "week_start"]
)
weekly_overall_rates["fraud_rate"] = weekly_overall_rates.fraud_labels.div(
    weekly_overall_rates.valid_labels.where(weekly_overall_rates.valid_labels.ne(0))
)
weekly_changes = largest_weekly_change(
    pd.concat(
        [weekly_country_rates, weekly_overall_rates],
        ignore_index=True,
    )
)
weekly_changes["rate_change_pp"] = weekly_changes.pop("rate_change") * 100


import math

import matplotlib.dates as mdates
from matplotlib.ticker import PercentFormatter


def plot_weekly_rates(
    country_rates: pd.DataFrame,
    overall_rates: pd.DataFrame,
    changes: pd.DataFrame,
    excluded_date_rows: int,
) -> Figure:
    """Plot aggregate and country rates on common axes with coverage caveats.

    Args:
        country_rates: One row per country/week, including counts and boundaries.
        overall_rates: Aggregate counts and fractional rates for the same weeks.
        changes: Largest adjacent-week movements with percentage-point changes.
        excluded_date_rows: Orders excluded because dates are missing or invalid.

    Returns:
        Figure with gap-preserving weekly series, partial-week markers and the
        aggregate movement annotation. Rates use only valid matched-order
        labels. Does not modify inputs or write files; caller closes the figure.
    """
    rates: pd.DataFrame = pd.concat([overall_rates, country_rates], ignore_index=True)
    weeks: pd.DatetimeIndex = pd.date_range(
        rates.week_start.min(),
        rates.week_start.max(),
        freq="W-MON",
    )
    countries: list[str] = ["ALL", *sorted(country_rates.country_code.unique())]
    panel_rows: int = math.ceil(len(countries) / 2)
    figure, axes = plt.subplots(
        panel_rows,
        2,
        figsize=(14, 3.3 * panel_rows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    upper_rate: float = max(0.6, math.ceil(rates.fraud_rate.max() * 10) / 10)
    for axis, country in zip(axes.flat, countries, strict=False):
        group: pd.DataFrame = (
            rates.loc[rates.country_code.eq(country)]
            .set_index("week_start")
            .reindex(weeks)
        )
        axis.plot(
            group.index,
            group.fraud_rate,
            color="#176b87",
            linewidth=2,
            marker="o",
            markersize=3,
        )
        partial: pd.DataFrame = group.loc[group.partial_week.eq(True)]
        axis.scatter(
            partial.index,
            partial.fraud_rate,
            facecolors="white",
            edgecolors="#176b87",
            zorder=4,
            s=55,
        )
        for week in partial.index:
            axis.axvspan(
                week - pd.Timedelta(days=3),
                week + pd.Timedelta(days=3),
                color="#e2e8f0",
                zorder=0,
            )
        axis.set_title(
            "All countries" if country == "ALL" else country,
            loc="left",
            fontweight="bold",
        )
        axis.set_ylim(0, upper_rate)
        axis.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        axis.grid(axis="y", alpha=0.22)
        axis.grid(axis="x", visible=False)
        axis.spines[["top", "right"]].set_visible(False)
        axis.xaxis.set_major_locator(mdates.MonthLocator())
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        axis.tick_params(axis="x", rotation=25)
    for unused_axis in axes.flat[len(countries) :]:
        unused_axis.set_visible(False)
    aggregate_changes: pd.DataFrame = changes.loc[changes.country_code.eq("ALL")]
    if not aggregate_changes.empty:
        movement: pd.Series = aggregate_changes.iloc[0]
        annotation: str = (
            f"{movement.previous_week:%d %b} → {movement.week_start:%d %b %Y}: "
            f"{movement.rate_change_pp:+.4f} pp\n"
            f"{movement.previous_fraud_labels:,.0f} / {movement.previous_valid_labels:,.0f}"
            f" → {movement.fraud_labels:,.0f} / {movement.valid_labels:,.0f}"
        )
        axes[0, 0].annotate(
            annotation,
            xy=(movement.week_start, movement.fraud_rate),
            xytext=(weeks[0] + (weeks[-1] - weeks[0]) * 0.15, upper_rate * 0.49),
            fontsize=9,
            arrowprops={"arrowstyle": "->", "color": "#243b53"},
        )
    excluded_labels: int = int(
        (overall_rates.missing_labels + overall_rates.invalid_labels).sum()
    )
    figure.suptitle(
        "Weekly fraud rates — safely linked orders",
        fontsize=17,
        fontweight="bold",
        y=0.99,
    )
    figure.supylabel("Fraud labels / valid 0–1 labels")
    figure.supxlabel("Week starting Monday", y=0.055)
    figure.text(
        0.07,
        0.006,
        f"Valid labels only; {excluded_labels:,} missing/invalid labels excluded; "
        f"{excluded_date_rows:,} date exclusions. Open markers / shaded bands: partial boundary weeks.\n"
        f"Common 0–{upper_rate:.0%} axes; missing weeks leave gaps. "
        "Descriptive movements do not establish a change point or cause.",
        fontsize=10,
    )
    figure.tight_layout(rect=(0.02, 0.085, 1, 0.96))
    return figure


weekly_figure: Figure = plot_weekly_rates(
    weekly_country_rates,
    weekly_overall_rates,
    weekly_changes,
    weekly_country_rates.attrs["excluded_date_rows"],
)
display(weekly_figure)
plt.close(weekly_figure)
show_report(
    weekly_changes, "Largest absolute adjacent-week movements — descriptive candidates"
)

weekly_reference_tables: list[str] = [
    style_report(weekly_overall_rates, "Weekly fraud rates — all countries").to_html(),
    *[
        style_report(group, f"Weekly fraud rates — {country}").to_html()
        for country, group in weekly_country_rates.groupby("country_code", sort=True)
    ],
]
display(
    HTML(
        "<details><summary>Supporting evidence: all weekly counts, denominators, exclusions and boundary flags</summary>"
        '<div style="overflow-x: auto;">'
        + "".join(weekly_reference_tables)
        + "</div></details>"
    )
)


# %% [markdown]
# #### 6.3.1 Weekly movement and interpretation
#
# The largest adjacent-week jump identifies a possible onset; it is not a test of a lasting regime change. The base snapshot also shows a broader elevation: weekly aggregate rates are roughly 9–11% through February, 11.8% in the transition week starting 6 March, and roughly 12–14% from mid-March to May (including weeks just below 12%). This persistence matters alongside the single-week screen.
#
# The summary dates the largest adjacent-week aggregate movement and shows PK and BD in those weeks. Section 6.4 compares all countries across eligible boundaries and decomposes the broader March elevation; PK provides its largest positive contribution.

# %%
from IPython.display import Markdown

aggregate_movement = weekly_changes.loc[weekly_changes.country_code.eq("ALL")]
if not aggregate_movement.empty:
    movement = aggregate_movement.iloc[0]
    display(
        Markdown(
            f"The aggregate's largest adjacent-week movement is "
            f"**{movement.previous_week:%Y-%m-%d} → {movement.week_start:%Y-%m-%d}**: "
            f"{movement.previous_rate:.4%} ({movement.previous_fraud_labels:,.0f} / "
            f"{movement.previous_valid_labels:,.0f}) to {movement.fraud_rate:.4%} "
            f"({movement.fraud_labels:,.0f} / {movement.valid_labels:,.0f}), "
            f"a change of **{movement.rate_change_pp:+.4f} percentage points**."
        )
    )
    for country in ("PK", "BD"):
        country_weeks = weekly_country_rates.loc[
            weekly_country_rates.country_code.eq(country)
            & weekly_country_rates.week_start.isin(
                [movement.previous_week, movement.week_start]
            )
        ].sort_values("week_start")
        if len(country_weeks) == 2:
            previous_country, current_country = (
                country_weeks.iloc[0],
                country_weeks.iloc[1],
            )
            display(
                Markdown(
                    f"{country} in those same weeks: {previous_country.fraud_rate:.2%} "
                    f"({previous_country.fraud_labels:,} / {previous_country.valid_labels:,}) to "
                    f"{current_country.fraud_rate:.2%} ({current_country.fraud_labels:,} / "
                    f"{current_country.valid_labels:,}). Section 6.4 compares all countries "
                    "and separates rate changes from country mix; these comparisons do not establish causes."
                )
            )

display(
    Markdown(
        f"Orders excluded for missing/invalid dates: "
        f"{weekly_country_rates.attrs['excluded_date_rows']:,}. "
        "The movement table dates each country's largest candidate and flags partial weeks."
    )
)

# %% [markdown]
#
# Preserve the weekly counts, denominators and boundary flags. Investigate dated movements alongside country mix and data coverage before attributing them to changing fraud behavior; this descriptive screen does not prove a change point. Keep any later model validation chronological where appropriate and fit preparation steps on training data only.

# %% [markdown]
# ### 6.4 Daily volume and country changes
#
# Daily counts cover every calendar date between the first and last observed orders, including the final observed day. Boundary flags indicate potentially partial coverage; low volume alone does not establish an incomplete day. Missing/invalid dates are counted separately.
#
# Compare each country's pooled valid-label rate before and after every eligible complete-week boundary, with the boundary week included in the later period. Require at least two complete weeks on each side; exclude partial boundary weeks and undefined rates. Display denominators and every comparison, rather than selecting a March-only search interval. These overlapping descriptive comparisons do not establish significance, a causal mechanism or an exact change-point date. Short endpoint periods can produce large contrasts.
#
# The daily table reports the current reader's scope. If a downstream protected reader withholds the test interval, its zero-count days are flagged as policy exclusions and the country scan uses the contiguous pre-test prefix. No withheld rows are opened for this analysis. The full-history findings below refer to the recorded historical snapshot.

# %%
from IPython.display import HTML

from mlops_fraud_prediction_ca1.temporal import (
    daily_order_counts,
    parse_dates,
    weekly_country_changes,
    weekly_fraud_rates,
)

review_audit = orders.attrs.get("split_audit", {})
review_locked = bool(review_audit.get("test_locked", False))
review_daily = daily_order_counts(linked_orders)
review_daily["withheld_by_policy"] = False
if review_locked:
    cutoffs = review_audit["cutoffs"]
    review_daily["withheld_by_policy"] = review_daily.date.ge(
        pd.Timestamp(cutoffs["test_start"])
    ) & review_daily.date.lt(pd.Timestamp(cutoffs["test_end_exclusive"]))
assert int(review_daily.orders.sum()) == len(linked_orders)
show_report(
    pd.DataFrame(
        [
            {
                "orders": len(linked_orders),
                "calendar_days": len(review_daily),
                "first_observed": review_daily.date.min(),
                "last_observed": review_daily.date.max(),
                "zero_count_days": int(review_daily.orders.eq(0).sum()),
                "policy_withheld_days": int(review_daily.withheld_by_policy.sum()),
                "date_exclusions": review_daily.attrs["excluded_date_rows"],
                "test_locked": review_locked,
            }
        ]
    ),
    "Daily-volume reconciliation — current reader scope",
)
display(
    HTML(
        "<details><summary>Every calendar date and policy/boundary flags</summary>"
        + style_report(review_daily, "Observed daily orders").to_html()
        + "</details>"
    )
)
# A scan needs a contiguous series: locked reruns use the pre-test prefix,
# while the standalone #3 branch scans all complete historical weeks.
review_scan_orders = linked_orders
if review_locked:
    dates = parse_dates(linked_orders.order_date)
    review_scan_orders = linked_orders.loc[
        dates.date.lt(pd.Timestamp(review_audit["cutoffs"]["test_start"]))
    ].copy()
review_weekly = weekly_fraud_rates(review_scan_orders)
review_changes = weekly_country_changes(review_weekly)
review_largest = (
    review_changes.loc[
        review_changes.groupby("country_code").difference_pp.apply(
            lambda x: x.abs().idxmax()
        )
    ]
    if not review_changes.empty
    else review_changes
)
show_report(review_largest, "Largest pooled before/after contrast per country")
display(
    HTML(
        "<details><summary>Every eligible country/week boundary, with denominators</summary>"
        + style_report(review_changes, "Pooled complete-week comparisons").to_html()
        + "</details>"
    )
)
# Preserve the previously previewed baseline: complete weeks 14 Nov–26 Feb
# versus 13 Mar–16 Apr. Show each country's additive contribution.
review_counts = ["valid_labels", "fraud_labels"]
review_baseline = (
    review_weekly.loc[review_weekly.week_start.between("2022-11-14", "2023-02-20")]
    .groupby("country_code")[review_counts]
    .sum()
)
review_comparison = (
    review_weekly.loc[review_weekly.week_start.between("2023-03-13", "2023-04-10")]
    .groupby("country_code")[review_counts]
    .sum()
)
review_baseline_rate = review_baseline.fraud_labels / review_baseline.valid_labels
review_comparison_rate = review_comparison.fraud_labels / review_comparison.valid_labels
review_baseline_share = (
    review_baseline.valid_labels / review_baseline.valid_labels.sum()
)
review_comparison_share = (
    review_comparison.valid_labels / review_comparison.valid_labels.sum()
)
review_contributions = pd.DataFrame(
    {
        "before_valid": review_baseline.valid_labels,
        "before_fraud": review_baseline.fraud_labels,
        "after_valid": review_comparison.valid_labels,
        "after_fraud": review_comparison.fraud_labels,
        "before_rate": review_baseline_rate,
        "after_rate": review_comparison_rate,
        "within_country_pp": review_baseline_share
        * (review_comparison_rate - review_baseline_rate)
        * 100,
        "mix_pp": (review_comparison_share - review_baseline_share)
        * review_comparison_rate
        * 100,
    }
).reset_index()
show_report(review_contributions, "March rise — within-country and mix contributions")


# %% [markdown]
# In the full historical snapshot, daily counts reconcile 949,889 orders across 181 dates, 12 November 2022–11 May 2023, with no excluded dates or zero-observation days. The median is 5,240 orders/day; 11 May contains 2,341 orders. This final-day volume warrants caution but does not prove incomplete source coverage.
#
# Across all eligible full-history weekly boundaries, BD's largest pooled rise is at 28 November (14.11% before, 30.43% after) and PK's at 6 March (20.55% before, 31.14% after). PH's largest pooled decline is at 27 February (9.54% to 6.84%); it also has lower rates across the April comparisons, including 8.94% before 3 April versus 6.53% afterward. These are descriptive contrasts, not proof that each movement began on its selected boundary. A locked rerun can yield different contrasts because its scope is shorter.
#
# For the March decomposition, the baseline is the complete weeks 14 November–26 February and the comparison is 13 March–16 April. PK is the main contributor: its rate rises from 20.3127% to 38.4699%, adding +3.3777 percentage points at baseline country shares. BD adds +0.5799 points; PH offsets the rise by -0.8936 points. The aggregate rises from 9.9813% to 13.3058%, decomposed into +3.2284 points within countries and +0.0962 from mix. The former March-only threshold check also classified an early-February BD week as high, so it could not establish BD's onset. The all-week comparison replaces that restricted screen.


# %% [markdown]
# ## 7. Preparation decisions and limitations
#
# **Preparation decisions:** Use validated composite keys and keep coverage exceptions visible. Preserve unknown categories and missing values; parse only the documented date formats. Retain unusual numeric values with investigation flags until source evidence supports correction. Learned imputation, scaling, feature selection and any resampling must fit on training data only.
#
# **Later workflow:** Later preparation should call shared package functions directly, with explicit configuration and audited outputs. This notebook demonstrates reporting and working-copy transformations; it does not persist a cleaned dataset or train a model. Feature availability still needs a separate leakage audit, including the customer-history fields.
#
# **Limitations:** The full historical label period has been explored here. A later chronological split must disclose that earlier exploration and define fixed cutoffs separately. Weekly movements, country differences and IQR screens are descriptive evidence; they establish neither causes nor data errors. The final boundary weeks are partial, and full sampling is not guaranteed elsewhere.
