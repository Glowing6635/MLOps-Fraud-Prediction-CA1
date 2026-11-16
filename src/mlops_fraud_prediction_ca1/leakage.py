"""Conservative availability inventory and nonmutating timing evidence for PBI #4.

Reports do not certify production availability or clean data. The separate
column selector implements the provisional PBI #4 policy in docs/decision-log.md;
a customer timestamp check cannot date other fields.
"""

from numbers import Integral

import pandas as pd

from .relationships import link_orders_customers
from .temporal import parse_dates

CUSTOMER_WINDOW_COLUMNS = (
    "num_orders_last_50days",
    "num_cancelled_orders_last_50days",
    "num_refund_orders_last_50days",
    "total_payment_last_50days",
)

CUSTOMER_FEATURE_COLUMNS: tuple[str, ...] = (
    "mobile_verified",
    *CUSTOMER_WINDOW_COLUMNS,
    "num_associated_customers",
    "first_order_datetime",
)
ORDER_FEATURE_COLUMNS: tuple[str, ...] = (
    "collect_type",
    "payment_method",
    "order_value",
    "num_items_ordered",
)
POLICY_DECISION_LOG: str = "docs/decision-log.md"
ASSUMED_LABEL_DELAY_DAYS: int = 30


def select_feature_inputs(
    frame: pd.DataFrame,
    *,
    source: str,
    exploratory: bool = False,
) -> pd.DataFrame:
    """Copy allowed source columns under decisions D4-01 through D4-04.

    Args:
        frame: One source's columns, before combining feature groups.
        source: ``orders``, ``customers`` or ``labels``; unknown sources fail.
        exploratory: Explicitly opt into the seven unconfirmed customer fields
            for #14's grouped comparison. Never use that branch for a champion.

    Returns:
        An independent copy in source-column order. Unknown fields, identifiers,
        refund_value and is_fraud are always excluded. order_date is temporal
        context for history/splitting, not a direct feature. Order attributes
        rely on D4-01's unconfirmed placement/original-value assumptions.

    Raises:
        ValueError: For an unknown source or ambiguous duplicate column names.
        TypeError: If exploratory is not an explicit boolean.

    This selects columns only: callers retain keys/dates/target/evaluation data
    separately and enforce linkage, date flags, missingness and train-only
    transformations. In exploratory customer rows, withhold inconsistent or
    unevaluable values under D4-03. No input mutation or training occurs.
    """
    if source not in {"orders", "customers", "labels"}:
        raise ValueError("source must be orders, customers or labels")
    if not isinstance(exploratory, bool):
        raise TypeError("exploratory must be a boolean")
    if not frame.columns.is_unique:
        raise ValueError("duplicate column names make feature selection ambiguous")
    allowed = (
        ORDER_FEATURE_COLUMNS
        if source == "orders"
        else (CUSTOMER_FEATURE_COLUMNS if source == "customers" and exploratory else ())
    )
    return frame.loc[:, [column for column in frame.columns if column in allowed]].copy(
        deep=True
    )


def availability_inventory(
    customers: pd.DataFrame,
    orders: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    reviewed_on: str,
) -> pd.DataFrame:
    """List every actual source column, with conservative proposed dispositions.

    ``reviewed_on`` is the actual review date, not a historical confirmation.
    Definitions come from spec.txt Appendix A1; definitions alone do not prove
    availability. No Operations response is supplied to this function, so it
    never marks a field confirmed. Unknown/new fields fail closed. Repeated
    identifiers in different sources remain separate entries. The result is
    an audit proposal, not an executable feature allowlist.
    """
    date = pd.Timestamp(reviewed_on)
    if pd.isna(date):
        raise ValueError("reviewed_on must be an actual review date")
    records = []
    for source, frame in (
        ("customers", customers),
        ("orders", orders),
        ("labels", labels),
    ):
        for column in frame.columns:
            role, timing, use = (
                "candidate feature",
                "Unconfirmed",
                "Withhold pending evidence",
            )
            evidence = "spec.txt A1 defines the field; no timing confirmation supplied"
            scope = "Order placement; source version at that time needs confirmation"
            if source == "customers":
                scope = "Customer snapshot anchor/order unknown, including later orders"
            if column in {"country_code", "customer_id", "order_id"}:
                role, use = "key", "Link/group only; not a proposed model input"
                timing = "Identity at order placement assumed; source availability unconfirmed"
            elif source == "customers":
                use = "Unconfirmed customer group: #14 exploratory comparison only"
                if column in CUSTOMER_WINDOW_COLUMNS:
                    timing = "50-day window endpoint and event availability unconfirmed"
                elif column == "first_order_datetime":
                    timing = "First order event; stored value availability unconfirmed"
                    evidence = "Date-order check supports this field only; see customer_timing_audit"
                elif column in {"mobile_verified", "num_associated_customers"}:
                    timing = "Verification/association state at each order unconfirmed"
                else:
                    role, use = (
                        "unknown",
                        "Withhold pending field definition and timing evidence",
                    )
            elif source == "orders" and column == "refund_value":
                role, timing, use = (
                    "evaluation only",
                    "Refund event; pre-shipping availability not established",
                    "Never a feature; evaluation only",
                )
            elif source == "labels" and column == "is_fraud":
                role, timing, use = (
                    "target",
                    f"After fraud confirmation; {ASSUMED_LABEL_DELAY_DAYS}-day delay assumed, unconfirmed",
                    "Target only; history uses the recorded provisional delay",
                )
            elif source == "orders" and column == "order_date":
                role, timing, use = (
                    "temporal context",
                    "Order placement date; stored version unconfirmed",
                    "Splits/history only; not a direct feature",
                )
            elif source == "orders" and column in ORDER_FEATURE_COLUMNS:
                timing = "Order placement assumed; original value/version unconfirmed"
                use = (
                    "Candidate only under a dated, explicit assumption or confirmation"
                )
            else:
                role, use = (
                    "unknown",
                    "Withhold pending field definition and timing evidence",
                )
            records.append(
                {
                    "source": source,
                    "column": column,
                    "role": role,
                    "when_known": timing,
                    "order_scope": scope,
                    "status": "unconfirmed",
                    "evidence": evidence,
                    "proposed_use": use,
                    "reviewed_on": date.date().isoformat(),
                }
            )
    return pd.DataFrame(
        records,
        columns=[
            "source",
            "column",
            "role",
            "when_known",
            "order_scope",
            "status",
            "evidence",
            "proposed_use",
            "reviewed_on",
        ],
    )


def customer_timing_audit(
    customers: pd.DataFrame,
    orders: pd.DataFrame,
    labels: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """Return one row per safe order and counts of limited temporal evidence.

    Reuse normalized composite-key linkage, excluding conflicting/null keys
    and collapsing exact copies. Earliest *observed* order dates describe only
    this extract: they neither identify the customer's true first order nor
    anchor its snapshot. Same-day orders cannot be sequenced from date-only
    order_date. Missing/invalid dates or customer links remain unevaluable.
    Flagged customer values should be withheld pending resolution, not silently
    corrected or treated as grounds to reject the whole order.
    """
    linked, link_audit = link_orders_customers(
        orders,
        labels,
        customers,
        normalize_country=True,
    )
    result = linked[["country_code", "order_id", "customer_id"]].copy()
    result["customer_linked"] = linked.customer_link.eq("both")
    result["order_day"] = parse_dates(linked.order_date).date.dt.normalize()
    result["first_order_day"] = parse_dates(
        linked.first_order_datetime,
        kind="customer",
    ).date.dt.normalize()
    valid = (
        result.customer_linked
        & result.order_day.notna()
        & result.first_order_day.notna()
    )
    result["date_evidence"] = "unevaluable"
    result.loc[valid & result.order_day.lt(result.first_order_day), "date_evidence"] = (
        "before_first_order"
    )
    result.loc[valid & result.order_day.eq(result.first_order_day), "date_evidence"] = (
        "same_day_intraday_unknown"
    )
    result.loc[valid & result.order_day.gt(result.first_order_day), "date_evidence"] = (
        "after_first_order_day"
    )
    # Limit group evidence to safely customer-linked orders; do not group null IDs.
    matched = result.loc[result.customer_linked]
    groups = matched.groupby(["country_code", "customer_id"])
    result["observed_order_count"] = groups.order_id.transform("size").reindex(
        result.index
    )
    result["earliest_observed_order_day"] = groups.order_day.transform("min").reindex(
        result.index
    )
    result["strictly_later_observed_day"] = (
        result.customer_linked & result.order_day.gt(result.earliest_observed_order_day)
    )
    counts = result.date_evidence.value_counts()
    sizes = groups.size()
    summary = {
        "safe_orders": len(result),
        "customer_linked_orders": int(result.customer_linked.sum()),
        "date_evaluable_orders": int(valid.sum()),
        **{
            name: int(counts.get(name, 0))
            for name in (
                "before_first_order",
                "same_day_intraday_unknown",
                "after_first_order_day",
                "unevaluable",
            )
        },
        "observed_linked_customers": len(sizes),
        "customers_with_multiple_observed_orders": int(sizes.gt(1).sum()),
        "orders_for_multiple_order_customers": int(sizes.loc[sizes.gt(1)].sum()),
        "orders_strictly_after_earliest_observed_day": int(
            result.strictly_later_observed_day.sum()
        ),
        "snapshot_anchor": "unknown; earliest observed order is not snapshot evidence",
        "proposed_exception_handling": "Withhold inconsistent/unevaluable customer values; retain orders",
        "link_audit": link_audit,
    }
    return result, summary


def historical_order_mask(
    prior_order_dates: pd.Series,
    current_order_date: str,
    *,
    label_delay_days: int,
) -> pd.Series:
    """Mark orders strictly older than the caller's required calendar-day delay.

    Pass zero explicitly for history without labels, or the recorded label
    maturity assumption for labelled history. Equality and bad dates are
    excluded. This checks timing only, not actual label availability, validity,
    customer matching or window membership. Input indices are preserved.
    """
    if (
        isinstance(label_delay_days, bool)
        or not isinstance(label_delay_days, Integral)
        or label_delay_days < 0
    ):
        raise ValueError("label_delay_days must be a nonnegative whole number")
    current = parse_dates(pd.Series([current_order_date])).date.iloc[0]
    if pd.isna(current):
        raise ValueError("current_order_date must be a valid supported order date")
    prior = parse_dates(prior_order_dates).date
    age = (current - prior).dt.days
    return age.gt(label_delay_days).rename(prior_order_dates.name)
