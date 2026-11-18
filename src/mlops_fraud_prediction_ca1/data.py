"""Guarded base CSV reads; source bytes and values remain unchanged."""

import logging
from copy import deepcopy
from pathlib import Path

import pandas as pd

from mlops_fraud_prediction_ca1.quality import CATEGORY_COLUMNS
from mlops_fraud_prediction_ca1.splitting import split_orders


def _read_base_data(data_dir: str | Path, *, raw_strings: bool = False) -> tuple:
    """Private ingestion: callers must scope rows before exposing any frame."""
    directory = Path(data_dir).expanduser()
    paths = tuple(
        directory / f"{name}_base.csv" for name in ("customers", "orders", "labels")
    )
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required CSV files not found: " + ", ".join(missing))
    # IDs are opaque keys: inference can merge e.g. customers "001" and "1",
    # exposing a test-only customer's payload through an allowed label.
    options = (
        {"dtype": "string", "keep_default_na": False}
        if raw_strings
        else {
            "dtype": {
                name: "string"
                for name in (
                    "order_id",
                    "customer_id",
                    "country_code",
                    "mobile_verified",
                    "is_fraud",
                    "order_date",
                    "first_order_datetime",
                    "collect_type",
                    "payment_method",
                )
            }
        }
    )
    frames = tuple(pd.read_csv(path, **options) for path in paths)
    for path, frame in zip(paths, frames, strict=True):
        logging.getLogger(__name__).info(
            "Ingested %s | %s rows | %s columns",
            path.name,
            len(frame),
            len(frame.columns),
        )
    return frames


def _key_index(frame: pd.DataFrame, id_column: str) -> pd.MultiIndex:
    country = (
        frame["country_code"]
        .astype("string")
        .str.strip()
        .str.upper()
        .replace("", pd.NA)
    )
    identity = frame[id_column].astype("string")
    identity = identity.mask(identity.str.strip().eq(""), pd.NA)
    return pd.MultiIndex.from_arrays([country, identity])


def _scope_related(
    customers: pd.DataFrame,
    labels: pd.DataFrame,
    allowed: pd.DataFrame,
    denied: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # A business key appearing in both allowed and withheld orders is unsafe:
    # its undated label cannot be assigned to either side. Preserve order rows,
    # but quarantine that key's related payload rather than guessing.
    allowed_keys = _key_index(allowed, "order_id")
    denied_keys = _key_index(denied, "order_id")
    complete_orders = allowed_keys.to_frame(index=False).notna().all(axis=1).to_numpy()
    eligible_keys = allowed_keys[complete_orders & ~allowed_keys.isin(denied_keys)]
    label_keys = _key_index(labels, "order_id")
    label_mask = (
        label_keys.isin(eligible_keys)
        & label_keys.to_frame(index=False).notna().all(axis=1).to_numpy()
    )
    scoped_labels = labels.iloc[label_mask].copy()
    customer_keys = _key_index(scoped_labels, "customer_id")
    complete_customers = (
        customer_keys.to_frame(index=False).notna().all(axis=1).to_numpy()
    )
    scoped_customers = customers.iloc[
        _key_index(customers, "customer_id").isin(customer_keys[complete_customers])
    ].copy()
    return scoped_customers, allowed.copy(), scoped_labels


def load_base_data(
    data_dir: str | Path,
    *,
    purpose: str = "analysis",
    config_path: str | Path | None = None,
    raw_strings: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read base CSVs and return authorized customers, orders and labels.

    Args:
        data_dir: Directory containing customers_base.csv, orders_base.csv and
            labels_base.csv. A leading ~ is expanded to the user's home directory.
        purpose: "analysis" by default, excluding the locked test interval.
            "train" and "validation" use fixed windows; "final_evaluation"
            requires a separate final_evaluation_enabled=true configuration.
        config_path: Temporal-policy TOML path. None resolves config/time_split.toml
            relative to this source checkout, independent of the working directory.
            Installed consumers must supply an explicit configuration path.
        raw_strings: False uses normal CSV missing-value/numeric inference while
            preserving opaque IDs and category/date/label text. True reads every
            column as literal strings, preserving blanks under the same guard.

    Returns:
        Tuple of customers, orders and labels DataFrames, in that order.
        Orders are scoped first; labels use normalized country/order keys and
        customers use country/customer keys referenced by retained labels.
        Incomplete or cross-boundary ambiguous keys cannot expose related rows.
        Shared customers may remain visible: windows are not customer-disjoint.
        Each frame has an independent, aggregate-only split_audit in attrs,
        including nested cutoffs and counts. No withheld payload enters audits.
        Feature availability remains the caller's responsibility.

    Raises:
        FileNotFoundError: Any required CSV or the policy file is absent. All
            three CSV paths are checked before reading any of them.
        pandas.errors.EmptyDataError: An input CSV has no columns.
        pandas.errors.ParserError: An input CSV cannot be parsed.
        KeyError: Required country, identity or order_date columns are absent.
        ValueError: Configuration, purpose or access settings are invalid,
            final evaluation is disabled, or the reserved positional marker is
            already present. Malformed TOML also raises a ValueError subclass.
    """
    customers, orders, labels = _read_base_data(data_dir, raw_strings=raw_strings)
    # Use a separate positional marker to detect cross-boundary duplicate keys.
    marked = orders.copy()
    marker = "__pbi6_position__"
    if marker in marked:
        raise ValueError(f"Reserved input column: {marker}")
    marked[marker] = range(len(marked))
    selected, audit = split_orders(marked, purpose=purpose, config_path=config_path)
    denied = marked.loc[~marked[marker].isin(selected[marker])]
    selected = selected.drop(columns=marker)
    frames = _scope_related(customers, labels, selected, denied)
    for name, frame in zip(("customers", "orders", "labels"), frames, strict=True):
        frame.attrs = {"split_audit": deepcopy(audit)}
        logging.getLogger(__name__).info(
            "Scoped %s_base.csv | %s rows | %s columns",
            name,
            len(frame),
            len(frame.columns),
        )
    return frames


def load_category_data(
    data_dir: str | Path, *, config_path: str | Path | None = None
) -> dict[str, pd.DataFrame]:
    """Literal category values, scoped through the same locked analysis read."""
    frames = load_base_data(data_dir, config_path=config_path, raw_strings=True)
    return {
        name: frame[list(CATEGORY_COLUMNS[name])].copy()
        for name, frame in zip(("customers", "orders", "labels"), frames, strict=True)
    }
