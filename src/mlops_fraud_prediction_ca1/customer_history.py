"""Customer-history counts for the v2 candidate (PBI #22, decision D22-01).

Each count for a scored order reads only earlier labelled orders of the same
customer (`country_code` + `customer_id`). The fraud counts apply the label
delay of D4-04 (`leakage.ASSUMED_LABEL_DELAY_DAYS`): a label is used only when its
order is more than that many days older, the rule `leakage.historical_order_mask`
checks. The customer ID is never a model input, only the derived counts are.
"""

import pandas as pd

from mlops_fraud_prediction_ca1.leakage import ASSUMED_LABEL_DELAY_DAYS
from mlops_fraud_prediction_ca1.temporal import parse_dates

HISTORY_COLUMNS = ["n_prior_orders", "n_prior_labelled_30d", "n_prior_fraud_30d"]
_CUSTOMER = ["country_code", "customer_id"]
_NEEDED = ["order_date", *_CUSTOMER]


def customer_history_features(
    orders: pd.DataFrame,
    history: pd.DataFrame,
    *,
    label_delay_days: int = ASSUMED_LABEL_DELAY_DAYS,
) -> pd.DataFrame:
    """Return the three history counts for `orders`, indexed like `orders`.

    `orders` needs `order_date`, `country_code` and `customer_id`; `history` also
    needs the 0/1 `is_fraud` label and holds the labelled orders that may be read
    (for validation, the training and earlier validation orders). Columns:

    - `n_prior_orders`: history orders of the customer dated strictly before the order;
    - `n_prior_labelled_30d`: those dated more than `label_delay_days` before it;
    - `n_prior_fraud_30d`: how many of the latter are fraud.

    Orders without a customer ID, and customers absent from `history`, get zeros.
    A missing or unrecognised order date raises `ValueError`, as does a negative or
    non-integer delay. Neither input is modified.
    """
    if (
        isinstance(label_delay_days, bool)
        or not isinstance(label_delay_days, int)
        or label_delay_days < 0
    ):
        raise ValueError("label_delay_days must be a nonnegative whole number")
    for frame, extra in ((orders, []), (history, ["is_fraud"])):
        missing = [c for c in [*_NEEDED, *extra] if c not in frame.columns]
        if missing:
            raise ValueError(f"Missing columns: {', '.join(missing)}.")

    left = _dated(orders, "orders")
    right = _dated(history, "history")
    right["is_fraud"] = history["is_fraud"].astype(int).to_numpy()

    daily = (
        right.dropna(subset=["customer_id"])
        .groupby([*_CUSTOMER, "date"], as_index=False)
        .agg(n=("is_fraud", "size"), fraud=("is_fraud", "sum"))
        .sort_values("date")
    )
    by = daily.groupby(_CUSTOMER, sort=False)
    daily["cum_n"] = by["n"].cumsum()
    daily["cum_fraud"] = by["fraud"].cumsum()

    result = pd.DataFrame(0, index=orders.index, columns=HISTORY_COLUMNS)
    scored = left.dropna(subset=["customer_id"]).copy()
    if scored.empty or daily.empty:
        return result
    scored["row"] = range(len(scored))
    scored["cutoff"] = scored["date"] - pd.Timedelta(days=label_delay_days)

    def lookup(on: str, columns: list[str]) -> pd.DataFrame:
        # Latest cumulative row strictly before `on`; no such row means no history.
        merged = pd.merge_asof(
            scored.sort_values(on),
            daily[[*_CUSTOMER, "date", *columns]].rename(columns={"date": on}),
            on=on,
            by=_CUSTOMER,
            allow_exact_matches=False,
        )
        return merged.set_index(scored.index[merged["row"].to_numpy()])[columns]

    # merge_asof keeps the left rows, so align both lookups on the order index.
    recent = lookup("date", ["cum_n"]).fillna(0)
    matured = lookup("cutoff", ["cum_n", "cum_fraud"]).fillna(0)
    result.loc[recent.index, "n_prior_orders"] = recent["cum_n"].astype(int)
    result.loc[matured.index, "n_prior_labelled_30d"] = matured["cum_n"].astype(int)
    result.loc[matured.index, "n_prior_fraud_30d"] = matured["cum_fraud"].astype(int)
    return result


def _dated(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    parsed = parse_dates(frame["order_date"].astype("string")).date
    if parsed.isna().any():
        raise ValueError(f"{name} has a missing or unrecognised order_date.")
    out = frame[_CUSTOMER].copy()
    out["date"] = parsed
    return out
