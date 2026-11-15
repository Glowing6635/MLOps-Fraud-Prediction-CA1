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
# value cells. These are previews only: source files and the loaded datasets stay unchanged. The
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
