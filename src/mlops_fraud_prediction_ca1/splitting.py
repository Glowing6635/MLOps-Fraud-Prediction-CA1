"""Fixed calendar membership and purpose-specific exposure shared by CSV/SQL.

This is a workflow guard, not a security boundary against arbitrary Python.
SQL adapters must supply a nullable, validated calendar DATE named order_date.
"""

import logging
import tomllib
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from mlops_fraud_prediction_ca1.temporal import parse_date_rows


@dataclass(frozen=True)
class SplitPolicy:
    train_start: date
    validation_start: date
    test_start: date
    test_end_exclusive: date
    test_locked: bool = True
    final_evaluation_enabled: bool = False

    def __post_init__(self):
        bounds = (
            self.train_start,
            self.validation_start,
            self.test_start,
            self.test_end_exclusive,
        )
        if any(type(value) is not date for value in bounds):
            raise ValueError("Cut-offs must be calendar dates")
        if list(bounds) != sorted(set(bounds)):
            raise ValueError("Cut-offs must be distinct and strictly increasing")
        if any(
            type(value) is not bool
            for value in (self.test_locked, self.final_evaluation_enabled)
        ):
            raise ValueError("Access settings must be Booleans")


def load_policy(config_path: str | Path | None = None) -> SplitPolicy:
    """Load fixed dates and access settings; invalid configuration fails closed.

    The repository default is resolved relative to this module, not cwd. An
    installed consumer must explicitly supply its versioned configuration path.
    """
    path = (
        Path(config_path)
        if config_path is not None
        else Path(__file__).resolve().parents[2] / "config/time_split.toml"
    )
    with path.open("rb") as source:
        settings = tomllib.load(source)
    names = ("train_start", "validation_start", "test_start", "test_end_exclusive")
    if set(settings) != {*names, "test_locked", "final_evaluation_enabled"}:
        raise ValueError(
            "Expected four cut-offs, test_locked and final_evaluation_enabled"
        )
    dates = {}
    for name in names:
        value = settings[name]
        if not isinstance(value, str):
            # Keep the public ValueError contract for malformed configuration.
            raise ValueError(f"{name} must be an ISO calendar-date string")  # noqa: TRY004
        parsed = date.fromisoformat(value)
        if parsed.isoformat() != value:
            raise ValueError(f"{name} must use YYYY-MM-DD")
        dates[name] = parsed
    return SplitPolicy(
        **dates,
        test_locked=settings["test_locked"],
        final_evaluation_enabled=settings["final_evaluation_enabled"],
    )


def calendar_dates(values: pd.Series) -> pd.Series:
    """Parse order days by position, accepting explicit text and naive native dates.

    Returns a RangeIndex with NaT for missing, invalid, out-of-range or
    timezone-aware inputs. Native times are reduced to their calendar day.
    """
    # Reset before parsing: duplicate caller indices must never align rows.
    source = values.reset_index(drop=True)
    parsed = parse_date_rows(source).date
    native = source.map(
        lambda value: (
            isinstance(value, (date, datetime, pd.Timestamp)) and not pd.isna(value)
        )
    )
    for position in source.index[native]:
        value = source.iloc[position]
        # No implicit timezone conversion: SQL DATE and naive datetimes only.
        if isinstance(value, datetime) and value.tzinfo is not None:
            continue
        try:
            day = pd.Timestamp(value.year, value.month, value.day).as_unit("ns")
        except pd.errors.OutOfBoundsDatetime:
            continue
        parsed.iloc[position] = day
    return parsed


def _intervals(policy: SplitPolicy, purpose: str) -> list[tuple[date, date | None]]:
    if purpose == "train":
        return [(policy.train_start, policy.validation_start)]
    if purpose == "validation":
        return [(policy.validation_start, policy.test_start)]
    if purpose == "final_evaluation":
        if not policy.final_evaluation_enabled:
            raise ValueError("Final evaluation requires final_evaluation_enabled=true")
        logging.getLogger(__name__).warning(
            "Final-evaluation access | test_start=%s | test_end_exclusive=%s | test_locked=%s",
            policy.test_start,
            policy.test_end_exclusive,
            policy.test_locked,
        )
        return [(policy.test_start, policy.test_end_exclusive)]
    if purpose == "analysis":
        return (
            [(policy.train_start, policy.test_start), (policy.test_end_exclusive, None)]
            if policy.test_locked
            else [(policy.train_start, None)]
        )
    raise ValueError(f"Unknown read purpose: {purpose}")


def split_orders(
    orders: pd.DataFrame,
    *,
    purpose: str = "analysis",
    config_path: str | Path | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Return exposed rows, aggregate counts and resolved policy; no withheld rows.

    Training/validation membership stays fixed even after unlock. Analysis can
    include later batches outside the frozen test interval. Final evaluation
    requires its own configuration opt-in and logs access. Bad dates and
    pre-start rows are always withheld.
    """
    policy = load_policy(config_path)
    intervals = _intervals(policy, purpose)
    dates = calendar_dates(orders["order_date"])
    allowed = pd.Series(False, index=dates.index)
    for start, end in intervals:
        allowed |= dates.ge(pd.Timestamp(start)) & (
            dates.lt(pd.Timestamp(end)) if end else dates.notna()
        )
    membership = pd.Series("after_end", index=dates.index)
    membership.loc[dates.lt(pd.Timestamp(policy.train_start))] = "before_start"
    for name, start, end in (
        ("train", policy.train_start, policy.validation_start),
        ("validation", policy.validation_start, policy.test_start),
        ("test", policy.test_start, policy.test_end_exclusive),
    ):
        membership.loc[dates.ge(pd.Timestamp(start)) & dates.lt(pd.Timestamp(end))] = (
            name
        )
    membership.loc[dates.isna()] = "invalid_date"
    counts = {
        name: int(membership.eq(name).sum())
        for name in (
            "train",
            "validation",
            "test",
            "before_start",
            "after_end",
            "invalid_date",
        )
    }
    return orders.iloc[allowed.to_numpy()].copy(), {
        "purpose": purpose,
        "test_locked": policy.test_locked,
        "final_evaluation_enabled": policy.final_evaluation_enabled,
        "cutoffs": {
            "train_start": policy.train_start.isoformat(),
            "validation_start": policy.validation_start.isoformat(),
            "test_start": policy.test_start.isoformat(),
            "test_end_exclusive": policy.test_end_exclusive.isoformat(),
        },
        "source_rows": len(orders),
        "returned_rows": int(allowed.sum()),
        "excluded_rows": int((~allowed).sum()),
        "membership_counts": counts,
    }


def sql_date_predicate(
    *,
    purpose: str = "analysis",
    config_path: str | Path | None = None,
) -> tuple[str, tuple[str, ...]]:
    """Return a parameterized predicate and ordered parameters for validated DATE.

    Fixed order_date and DB-API %s placeholders support MySQL adapters.
    Raw-text dates must first be validated/converted to nullable DATE. This
    predicate alone is not a raw-date parser or live RDS integration.
    """
    token = "%s"
    clauses, parameters = [], []
    for start, end in _intervals(load_policy(config_path), purpose):
        clause = f"order_date >= {token}"
        parameters.append(start.isoformat())
        if end:
            clause += f" AND order_date < {token}"
            parameters.append(end.isoformat())
        clauses.append(f"({clause})")
    return "order_date IS NOT NULL AND (" + " OR ".join(clauses) + ")", tuple(
        parameters
    )
