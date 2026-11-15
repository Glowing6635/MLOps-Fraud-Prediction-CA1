"""Summarize numerical features without changing or filtering source tables."""

from collections.abc import Mapping, Sequence
import math

import pandas as pd


def profile_numeric_features(
    datasets: Mapping[str, pd.DataFrame],
    feature_columns: Mapping[str, Sequence[str]],
    *,
    iqr_multiplier: float = 1.5,
) -> pd.DataFrame:
    """Measure distributions and count observations outside IQR fences.

    Args:
        datasets: Named source tables. No table is modified.
        feature_columns: Explicit numerical amounts/counts to inspect per table;
            callers must exclude identifiers, category codes, and target labels.
        iqr_multiplier: Positive finite multiplier applied below Q1 and above Q3.

    Returns:
        One row per (dataset, feature), indexed by those names. Quantiles,
        sample standard deviation, and pandas sample skewness use finite,
        nonmissing observations. Missing values and infinities are counted
        separately. Zero and outlier percentages use the finite count.
        Fences are inclusive: only strictly outside values are flagged.
        A zero IQR is reported explicitly; its flags do not establish errors.
        Skewness is undefined for fewer than three finite values or a constant
        feature. Undefined statistics and percentages are NaN. Source values
        are never coerced, corrected, capped, or removed.

    Raises:
        ValueError: The multiplier is nonpositive/nonfinite, or no features
            were selected.
        KeyError: A requested table or column is absent.
        TypeError: A selected column is nonnumeric, Boolean, or complex.
    """
    if not math.isfinite(iqr_multiplier) or iqr_multiplier <= 0:
        raise ValueError("iqr_multiplier must be positive and finite")

    records: list[dict[str, object]] = []
    for dataset, columns in feature_columns.items():
        for column in columns:
            source: pd.Series = datasets[dataset][column]
            if (
                not pd.api.types.is_numeric_dtype(source.dtype)
                or pd.api.types.is_bool_dtype(source.dtype)
                or pd.api.types.is_complex_dtype(source.dtype)
            ):
                raise TypeError(f"{dataset}.{column} must contain real numerical values")

            infinite: pd.Series = source.isin([float("-inf"), float("inf")])
            values: pd.Series = source.loc[source.notna() & ~infinite]
            count: int = len(values)
            q1: float = float(values.quantile(0.25)) if count else float("nan")
            q3: float = float(values.quantile(0.75)) if count else float("nan")
            iqr: float = q3 - q1
            lower_fence: float = q1 - iqr_multiplier * iqr
            upper_fence: float = q3 + iqr_multiplier * iqr
            below: int = int(values.lt(lower_fence).sum())
            above: int = int(values.gt(upper_fence).sum())
            records.append({
                "dataset": dataset,
                "feature": column,
                "rows": len(source),
                "finite_count": count,
                "missing_count": int(source.isna().sum()),
                "infinite_count": int(infinite.sum()),
                "zero_percent": float(values.eq(0).sum()) / count * 100 if count else float("nan"),
                "negative_count": int(values.lt(0).sum()),
                "minimum": float(values.min()) if count else float("nan"),
                "q1": q1,
                "median": float(values.median()) if count else float("nan"),
                "mean": float(values.mean()) if count else float("nan"),
                "q3": q3,
                "p99": float(values.quantile(0.99)) if count else float("nan"),
                "maximum": float(values.max()) if count else float("nan"),
                "std": float(values.std()) if count > 1 else float("nan"),
                "skewness": (
                    float(values.skew())
                    if count >= 3 and values.nunique() > 1 else float("nan")
                ),
                "iqr": iqr,
                "lower_fence": lower_fence,
                "upper_fence": upper_fence,
                "below_fence": below,
                "above_fence": above,
                "flagged_count": below + above,
                "flagged_percent": (below + above) / count * 100 if count else float("nan"),
                "zero_iqr": bool(count > 0 and iqr == 0),
            })
    if len(records) == 0:
        raise ValueError("Select at least one numerical feature")
    return pd.DataFrame.from_records(records).set_index(["dataset", "feature"])
