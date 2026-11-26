"""Skew and outlier handling for the eligible scoring inputs (PBI #16, criterion 2).

The treatment is a fixed, row-wise signed log transform. It learns nothing, so
training and held-out rows are treated identically and no threshold can leak
from later data. Raw cleaned values are never changed, capped or dropped.
"""

from types import MappingProxyType

import numpy as np
import pandas as pd

from .leakage import ORDER_FEATURE_COLUMNS

SUFFIX = "_log1p"

# Numeric subset of leakage.ORDER_FEATURE_COLUMNS (decision D4-01). Evidence is
# notebook section 3.5.4: right-skewed, no negative/infinite values, extremes
# located but not established as errors.
TRANSFORMED_FEATURES: tuple[str, ...] = ("order_value", "num_items_ordered")

TREATMENTS = MappingProxyType(
    {
        "order_value": (
            "signed log1p, no cap: right-skewed amount whose large orders are "
            "plausible fraud-relevant signal; extremes are not established "
            "errors, so values are compressed, not clipped or removed"
        ),
        "num_items_ordered": (
            "signed log1p, no cap: strongest tail (max 1,600 vs median 2, P99 22) "
            "but not shown to be an error; the log scale limits its leverage "
            "while the raw count stays available. One missing value stays missing"
        ),
        "collect_type": "not numeric: category handling, outside this treatment",
        "payment_method": "not numeric: category handling, outside this treatment",
        "refund_value": (
            "not transformed: evaluation-only under D4-01, never a model input"
        ),
        "customer numeric fields": (
            "not transformed: withheld from champion inputs under D4-02; revisit "
            "only if #14 admits them, where their skew (5.4-55.0) would need a "
            "fresh stated reason"
        ),
    }
)
if not set(TRANSFORMED_FEATURES) <= set(ORDER_FEATURE_COLUMNS):
    raise ValueError("Skew transforms must use eligible order features")


def transform_skewed_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``frame`` with ``<feature>_log1p`` columns appended.

    Each value x becomes ``sign(x) * log1p(|x|)`` (identical to ``log1p`` for the
    observed non-negative data; defined for finite negatives). Missing values
    stay missing; infinities become missing. Existing columns, index, row order
    and ``attrs`` are untouched, so raw cleaned values remain available.

    Raises:
        KeyError: A required feature is absent.
        TypeError: A feature is non-numeric or Boolean.
        ValueError: A generated column name already exists.
    """
    result = frame.copy(deep=True)
    for feature in TRANSFORMED_FEATURES:
        if feature not in frame.columns:
            raise KeyError(f"{feature} is required")
        source = frame[feature]
        if not pd.api.types.is_numeric_dtype(
            source.dtype
        ) or pd.api.types.is_bool_dtype(source.dtype):
            raise TypeError(f"{feature} must be numeric")
        name = feature + SUFFIX
        if name in frame.columns:
            raise ValueError(f"{name} already exists")
        values = source.astype("Float64").to_numpy(dtype="float64", na_value=np.nan)
        values[np.isinf(values)] = np.nan
        out = np.sign(values) * np.log1p(np.abs(values))
        result[name] = (
            pd.Series(out, index=frame.index).astype("Float64").mask(np.isnan(values))
        )
    return result
