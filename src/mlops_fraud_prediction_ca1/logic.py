"""Count business-rule anomalies without interpreting them as proven errors."""

from collections.abc import Mapping, Sequence

import pandas as pd

from .relationships import bidirectional_matches, link_orders_customers
from .temporal import parse_dates

DEFAULT_NUMERIC_COLUMNS = {
    "customers": (
        "num_orders_last_50days",
        "num_cancelled_orders_last_50days",
        "num_refund_orders_last_50days",
        "total_payment_last_50days",
        "num_associated_customers",
    ),
    "orders": ("order_value", "num_items_ordered", "refund_value"),
}


def business_logic_report(
    customers: pd.DataFrame,
    orders: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    normalize_country: bool = False,
    numeric_columns: Mapping[str, Sequence[str]] | None = None,
) -> pd.DataFrame:
    """Return check/flagged/evaluable/excluded counts and explicit counting units.

    Amount/count checks retain source-row weights (including copies). Missing,
    nonnumeric and infinite values are excluded from numeric comparisons.
    Date checks use safe unique order→label→customer mappings and calendar
    dates, avoiding false positives from first-order times on the same day.
    Raw absence of label keys differs from an unresolved/conflicting mapping.
    A flagged observation needs investigation; this function never corrects it.
    ``attrs['link_audit']`` discloses join exclusions and original multiplicity.
    """
    records = []

    def add(check, flagged, evaluable, population, unit):
        records.append(
            {
                "check": check,
                "flagged": int(flagged),
                "evaluable": int(evaluable),
                "excluded": int(population - evaluable),
                "population": int(population),
                "unit": unit,
            }
        )

    def finite(values):
        converted = pd.to_numeric(values, errors="coerce")
        return converted.where(~converted.isin([float("inf"), -float("inf")]))

    def greater(frame, lhs, rhs, check, dataset):
        a, b = finite(frame[lhs]), finite(frame[rhs])
        valid = a.notna() & b.notna()
        add(
            check,
            (a.gt(b) & valid).sum(),
            valid.sum(),
            len(frame),
            f"{dataset} source rows",
        )

    greater(orders, "refund_value", "order_value", "refund_above_order_value", "orders")
    greater(
        customers,
        "num_cancelled_orders_last_50days",
        "num_orders_last_50days",
        "cancelled_above_total_orders",
        "customers",
    )
    greater(
        customers,
        "num_refund_orders_last_50days",
        "num_orders_last_50days",
        "refunded_above_total_orders",
        "customers",
    )
    frames = {"customers": customers, "orders": orders}
    for name, columns in (
        DEFAULT_NUMERIC_COLUMNS if numeric_columns is None else numeric_columns
    ).items():
        for column in columns:
            values = finite(frames[name][column])
            add(
                f"negative:{name}.{column}",
                values.lt(0).sum(),
                values.notna().sum(),
                len(values),
                f"{name} source rows",
            )

    coverage = bidirectional_matches(
        orders,
        labels,
        ["country_code", "order_id"],
        normalize_country=normalize_country,
    ).loc["left_to_right"]
    add(
        "orders_without_label_key",
        coverage.unmatched_rows,
        coverage.complete_key_rows,
        len(orders),
        "orders source rows",
    )
    joined, audit = link_orders_customers(
        orders, labels, customers, normalize_country=normalize_country
    )
    add(
        "orders_without_safe_label",
        joined.label_link.ne("both").sum(),
        len(joined),
        len(joined),
        "safe unique orders",
    )
    label_present = joined.label_link.eq("both")
    add(
        "labelled_orders_without_customer",
        (label_present & joined.customer_link.ne("both")).sum(),
        label_present.sum(),
        len(joined),
        "safe unique orders",
    )
    order_dates = parse_dates(joined.order_date)
    first_dates = parse_dates(joined.first_order_datetime, kind="customer")
    valid = (
        order_dates.status.eq("valid")
        & first_dates.status.eq("valid")
        & joined.customer_link.eq("both")
    )
    before = order_dates.date.dt.normalize().lt(first_dates.date.dt.normalize()) & valid
    add(
        "order_before_customer_first_order",
        before.sum(),
        valid.sum(),
        len(joined),
        "safe unique orders",
    )
    result = pd.DataFrame(records).set_index("check")
    result.attrs["link_audit"] = audit
    return result
