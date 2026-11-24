"""Compare a model with today's rule at the same number of flags per group (PBI #18).

Selection groups are the #5 groups (`top_k`): one `order_date` when K is a whole
number, one `order_date` and `country_code` when K is a mapping. Per group:

1. The rule's flags are ranked by `order_value` (highest first, ties by
   `country_code`, `order_id`) and cut at K.
2. The model is cut to the rule's count in that group, so both flag the same number
   of orders. The model at the full K is reported alongside.

`select_threshold` fits X on the supplied (validation) window with the predeclared
objective in docs/baseline-rule.md: most fraud caught after the cut, then fewer
flags, then the larger X. The result is a validation-tuned proxy unless a confirmed
X is passed to `compare` directly.
"""

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from mlops_fraud_prediction_ca1.rules import baseline_rule_flags
from mlops_fraud_prediction_ca1.splitting import calendar_dates
from mlops_fraud_prediction_ca1.top_k import ORDER_COLUMNS, evaluate_top_k

THRESHOLD_GRID = (50, 100, 200, 500, 1000, 2000)
_ORDER_COLUMNS = (*ORDER_COLUMNS, "order_value")
_ORDER_KEY = ["country_code", "order_id"]


def _comparison_orders(orders: pd.DataFrame, k) -> pd.DataFrame:
    missing = [c for c in _ORDER_COLUMNS if c not in orders]
    if missing:
        raise ValueError(f"The orders have no {', '.join(missing)} column.")
    capacities = list(k.values()) if isinstance(k, Mapping) else [k]
    if any(isinstance(c, bool) or not isinstance(c, int) or c < 1 for c in capacities):
        raise ValueError(f"K must be a whole number of orders, at least 1: {k}.")

    _validate_labels(orders)
    if orders[_ORDER_KEY].isna().any(axis=None):
        raise ValueError("Every order needs a country_code and order_id.")
    if orders.duplicated(_ORDER_KEY).any():
        raise ValueError(
            "Duplicate order key: country_code and order_id must be unique."
        )
    dates = calendar_dates(orders["order_date"])
    if dates.isna().any():
        raise ValueError("Every order needs a valid order_date.")
    if isinstance(k, Mapping):
        unknown = set(orders["country_code"]) - k.keys()
        if unknown:
            countries = ", ".join(sorted(map(str, unknown)))
            raise ValueError(f"K is not set for country {countries}.")
    return orders.assign(order_date=dates.to_numpy())


def _validate_labels(orders: pd.DataFrame) -> None:
    if not orders["is_fraud"].isin([0, 1]).all():
        raise ValueError("Each order's is_fraud label must be 0 or 1.")


def _group_k(orders: pd.DataFrame, k) -> pd.Series:
    """Each row's K, and so the group's, by position."""
    if isinstance(k, int):
        return pd.Series(k, index=range(len(orders)))
    return pd.Series([k[c] for c in orders["country_code"]], index=range(len(orders)))


def _group_keys(orders: pd.DataFrame, k) -> list:
    columns = ["order_date"] if isinstance(k, int) else ["order_date", "country_code"]
    return [orders[c].tolist() for c in columns]


def _ranked_positions(orders: pd.DataFrame, primary: np.ndarray, k) -> pd.DataFrame:
    """Rows in rank order (primary descending, then country_code, order_id) with group
    and within-group rank. Positions refer to the rows of `orders`."""
    frame = pd.DataFrame(
        {
            "pos": np.arange(len(orders)),
            "primary": np.asarray(primary, dtype=float),
            "country_code": orders["country_code"].to_numpy(),
            "order_id": orders["order_id"].to_numpy(),
        }
    )
    for name, values in zip(
        ["g1", "g2"][: len(_group_keys(orders, k))],
        _group_keys(orders, k),
        strict=False,
    ):
        frame[name] = values
    keys = [c for c in ("g1", "g2") if c in frame]
    frame = frame.sort_values(
        [*keys, "primary", "country_code", "order_id"],
        ascending=[*[True] * len(keys), False, True, True],
        kind="mergesort",
    )
    frame["rank"] = frame.groupby(keys, sort=False).cumcount()
    frame["group"] = frame.groupby(keys, sort=False).ngroup()
    return frame


def cut_rule_flags(orders: pd.DataFrame, flags: Sequence[bool], k) -> np.ndarray:
    """Boolean array: the rule's flags after keeping each group's top K by order_value."""
    orders = _comparison_orders(orders, k)
    return _cut_rule_flags(orders, flags, k)


def _cut_rule_flags(orders, flags, k):
    flags = np.asarray(flags, dtype=bool)
    if len(flags) != len(orders):
        raise ValueError("The flags and the orders must have the same length.")
    flagged = orders.reset_index(drop=True)[flags]
    out = np.zeros(len(orders), dtype=bool)
    if flagged.empty:
        return out
    positions = np.flatnonzero(flags)
    ranked = _ranked_positions(flagged, flagged["order_value"].to_numpy(), k)
    capacity = _group_k(flagged, k).to_numpy()[ranked["pos"].to_numpy()]
    keep = ranked["rank"].to_numpy() < capacity
    out[positions[ranked["pos"].to_numpy()[keep]]] = True
    return out


def match_model_to_rule(
    orders: pd.DataFrame, scores: Sequence[float], rule_selected: Sequence[bool], k
) -> np.ndarray:
    """Boolean array: per group, the model's top-n orders where n is the rule's count
    (already at most K) in that group, ranked by score then country_code, order_id."""
    orders = _comparison_orders(orders, k)
    scores = _scores(orders, scores)
    return _match_model_to_rule(_ranked_positions(orders, scores, k), rule_selected)


def _match_model_to_rule(ranked, rule_selected):
    if len(rule_selected) != len(ranked):
        raise ValueError("The rule selection and the orders must have the same length.")
    counts = (
        pd.Series(np.asarray(rule_selected, dtype=bool))
        .groupby(ranked.sort_values("pos")["group"].to_numpy())
        .sum()
    )
    limit = counts.reindex(ranked["group"].to_numpy()).to_numpy()
    keep = ranked["rank"].to_numpy() < limit
    out = np.zeros(len(ranked), dtype=bool)
    out[ranked["pos"].to_numpy()[keep]] = True
    return out


def top_k_selection(orders: pd.DataFrame, scores: Sequence[float], k) -> np.ndarray:
    """Boolean array: each group's top K orders by score (the model at K)."""
    orders = _comparison_orders(orders, k)
    scores = _scores(orders, scores)
    return _top_k_mask(orders, _ranked_positions(orders, scores, k), k)


def _scores(orders, scores):
    scores = np.asarray(scores, dtype=float)
    if scores.shape != (len(orders),):
        raise ValueError("Provide one score per order.")
    if not np.isfinite(scores).all():
        raise ValueError("Every score must be finite.")
    return scores


def _top_k_mask(orders, ranked, k):
    capacity = _group_k(orders, k).to_numpy()[ranked["pos"].to_numpy()]
    out = np.zeros(len(orders), dtype=bool)
    out[ranked["pos"].to_numpy()[ranked["rank"].to_numpy() < capacity]] = True
    return out


def selection_metrics(
    orders: pd.DataFrame, selected: Sequence[bool], loyalty_status: Sequence[str]
) -> dict:
    """Criterion 5 counts for one selection of flagged orders."""
    _validate_labels(orders)
    selected = np.asarray(selected, dtype=bool)
    if selected.shape != (len(orders),) or len(loyalty_status) != len(orders):
        raise ValueError("Provide one selection flag and loyalty status per order.")
    fraud = orders["is_fraud"].to_numpy() == 1
    loyal = np.asarray(loyalty_status) == "loyal"
    value = pd.to_numeric(orders["order_value"], errors="coerce").to_numpy()
    caught = selected & fraud
    false_flags = selected & ~fraud
    return {
        "flags": int(selected.sum()),
        "fraud_caught": int(caught.sum()),
        "fraud_exposure": float(np.nansum(value[caught])),
        "false_flags": int(false_flags.sum()),
        "false_flags_on_loyal": int((false_flags & loyal).sum()),
    }


def select_threshold(
    orders: pd.DataFrame, account_age: pd.Series, k, grid=THRESHOLD_GRID
) -> dict:
    """Fit X on `orders` with the predeclared objective; return the chosen X and the
    candidate table (X, flags before and after the cut, fraud caught after the cut)."""
    orders = _comparison_orders(orders, k)
    rows = []
    for x in grid:
        flags = baseline_rule_flags(orders["order_value"], account_age, x)[
            "rule_flag"
        ].to_numpy(dtype=bool)
        flagged = orders.loc[flags]
        at_k = evaluate_top_k(flagged, flagged["order_value"].astype(float), k).at_k
        rows.append(
            {
                "x": x,
                "flags_before_cut": int(flags.sum()),
                "flags": at_k.selected,
                "fraud_caught": at_k.fraud_caught,
            }
        )
    table = pd.DataFrame(rows)
    best = table.sort_values(
        ["fraud_caught", "flags", "x"], ascending=[False, True, False]
    ).iloc[0]
    return {"x": best["x"].item(), "candidates": table}


def compare(
    orders: pd.DataFrame,
    scores: Sequence[float],
    account_age: pd.Series,
    loyalty_status: Sequence[str],
    k,
    threshold,
    threshold_source: str,
    *,
    candidates: pd.DataFrame | None = None,
) -> dict:
    """Rule versus model at the rule's count, plus the model at K.

    `threshold_source` is stated in the result (for example "validation-tuned proxy"
    or the name of the confirmation). `beats_rule` is whether the matched model
    caught strictly more fraud than the rule.
    """
    orders = _comparison_orders(orders, k)
    scores = _scores(orders, scores)
    flags = baseline_rule_flags(orders["order_value"], account_age, threshold)
    rule = _cut_rule_flags(orders, flags["rule_flag"].to_numpy(dtype=bool), k)
    ranked = _ranked_positions(orders, scores, k)
    matched = _match_model_to_rule(ranked, rule)
    at_k = _top_k_mask(orders, ranked, k)
    evaluation = evaluate_top_k(orders, scores, k).at_k
    model_at_k = selection_metrics(orders, at_k, loyalty_status)
    model_at_k.update(
        flags=evaluation.selected,
        fraud_caught=evaluation.fraud_caught,
        false_flags=evaluation.selected - evaluation.fraud_caught,
    )
    result = {
        "k": k,
        "threshold": threshold,
        "threshold_source": threshold_source,
        "rule_status_counts": flags["rule_status"].value_counts().to_dict(),
        "rule_flags_before_cut": int(flags["rule_flag"].sum()),
        "rule": selection_metrics(orders, rule, loyalty_status),
        "model_matched": selection_metrics(orders, matched, loyalty_status),
        "model_at_k": model_at_k,
        "candidates": candidates,
        "loyalty_counts": pd.Series(loyalty_status).value_counts().to_dict(),
    }
    result["beats_rule"] = (
        result["model_matched"]["fraud_caught"] > result["rule"]["fraud_caught"]
    )
    return result


def format_comparison(result: dict) -> str:
    """Plain-text report for criteria 1 and 4 to 6."""
    k = result["k"]
    capacity = f"K = {k}"
    if k == 250:
        capacity += " orders/day across all countries (D5-01, Product Owner assumption)"
    lines = [
        (
            f"Rule: order_value > {result['threshold']} SGD and account age 0 days"
            f" ({result['threshold_source']}); {capacity}."
        ),
        (
            f"Rule flags before the K cut: {result['rule_flags_before_cut']};"
            f" status counts: {result['rule_status_counts']}."
        ),
        "first_order_datetime is unconfirmed. It is used only for rule account age.",
        (
            "Loyalty history starts on 2022-11-12 (12 Nov 2022). Earlier orders are"
            " left-censored, so early loyal customers can be undercounted."
        ),
        f"Loyalty counts: {result['loyalty_counts']}.",
        f"{'':26}{'flags':>7}{'caught':>8}{'exposure':>12}{'false':>7}{'false+loyal':>13}",
    ]
    candidates = result.get("candidates")
    if candidates is None:
        lines.insert(1, "No X candidates were fitted for this comparison.")
    else:
        lines[1:1] = [
            "Candidate X values tested on validation (SGD):",
            candidates.to_string(index=False),
        ]
    for label, key in (
        ("Rule", "rule"),
        ("Model, matched count", "model_matched"),
        ("Model at K", "model_at_k"),
    ):
        m = result[key]
        lines.append(
            f"{label:26}{m['flags']:>7}{m['fraud_caught']:>8}"
            f"{m['fraud_exposure']:>12.2f}{m['false_flags']:>7}"
            f"{m['false_flags_on_loyal']:>13}"
        )
    verdict = "did" if result["beats_rule"] else "did not"
    lines.append(
        f"The model {verdict} beat the rule on fraud caught at the same number of flags."
    )
    return "\n".join(lines)
