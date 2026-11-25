"""SGD impact comparison of the model and the baseline rule (PBI #34).

Input contract: one row per scored order with ``order_date``, ``country_code``,
``order_id``, ``customer_id``, ``order_value`` (numeric SGD), ``is_fraud`` (0/1),
``loyal`` (bool), ``score`` (higher = riskier) and precomputed ``rule_flag``
(bool, #18). #23 owns the frozen inputs, including the rule threshold and loyalty
definition. Reading rules are those of D34-01 in docs/decision-log.md.

Selection follows #18's matched-count method using the shared #5 evaluator.
The reporting stage can also accept its positional masks independently.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping

import numpy as np
import pandas as pd

from mlops_fraud_prediction_ca1.splitting import _calendar_dates
from mlops_fraud_prediction_ca1.top_k import select_top_k

MEASURES = ("exposure_sgd", "loyal_wrongly_flagged")
CUSTOMER_KEY = ["country_code", "customer_id"]


def select_flags(frame: pd.DataFrame, k: int | Mapping[str, int]) -> pd.DataFrame:
    """Attach the shared policy masks in row order, preserving the caller's index.

    K may be a global daily integer or a mapping of country to daily capacity.
    The #18 rule flags must already be computed; this does not tune X.
    """
    flags = frame["rule_flag"]
    if not pd.api.types.is_bool_dtype(flags.dtype) or flags.isna().any():
        raise ValueError("Each rule_flag must be a non-missing boolean.")

    dates = _calendar_dates(frame["order_date"]).to_numpy()
    values = pd.to_numeric(frame["order_value"], errors="raise")
    scores = pd.to_numeric(frame["score"], errors="raise")
    orders = frame.assign(order_date=dates, order_value=values.to_numpy())
    rule = select_top_k(
        orders,
        values.to_numpy(dtype=float, na_value=np.nan),
        k,
        eligible=flags.to_numpy(dtype=bool),
    )
    keys = orders["order_date"].tolist()
    if isinstance(k, Mapping):
        keys = list(zip(keys, orders["country_code"], strict=True))
    limits = Counter(dict.fromkeys(keys, 0))
    limits.update(key for key, selected in zip(keys, rule, strict=True) if selected)

    scores = scores.to_numpy(dtype=float, na_value=np.nan)
    matched = select_top_k(orders, scores, k, group_limits=limits)
    at_k = select_top_k(orders, scores, k)
    return frame.assign(
        order_value=orders["order_value"].to_numpy(),
        rule_sel=rule,
        model_sel=matched,
        model_k_sel=at_k,
    )


def _linked(frame: pd.DataFrame) -> pd.Series:
    return frame[CUSTOMER_KEY].notna().all(axis=1)


def _per_customer(sel: pd.DataFrame, col: str) -> pd.DataFrame:
    """Return exposure and loyal false flags for linked customers only."""
    s = sel.loc[sel[col] & _linked(sel)].set_index(CUSTOMER_KEY)
    fraud = s["is_fraud"] == 1
    exposure = s["order_value"].where(fraud, 0.0).groupby(level=CUSTOMER_KEY).sum()
    wrong = (
        (s["loyal"].astype(bool) & ~fraud).groupby(level=CUSTOMER_KEY).any().astype(int)
    )
    return pd.DataFrame({"exposure_sgd": exposure, "loyal_wrongly_flagged": wrong})


def _unlinked_fraud(sel: pd.DataFrame) -> pd.DataFrame:
    return sel.loc[~_linked(sel) & (sel["is_fraud"] == 1)]


def customer_table(
    sel: pd.DataFrame, customers: pd.MultiIndex
) -> dict[str, pd.DataFrame]:
    """Align each policy to the same population of composite customer keys."""
    out = {}
    for name, col in (
        ("rule", "rule_sel"),
        ("model", "model_sel"),
        ("model_k", "model_k_sel"),
    ):
        out[name] = _per_customer(sel, col).reindex(customers, fill_value=0)
    return out


def point_estimates(sel: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Unresampled values: exposure SGD and distinct loyal customers flagged."""
    result = {}
    unlinked = _unlinked_fraud(sel)
    for name, col in (
        ("rule", "rule_sel"),
        ("model", "model_sel"),
        ("model_k", "model_k_sel"),
    ):
        t = _per_customer(sel, col)
        unlinked_hits = unlinked.loc[unlinked[col]]
        unlinked_exposure = float(unlinked_hits["order_value"].sum())
        result[name] = {
            "exposure_sgd": float(t["exposure_sgd"].sum()) + unlinked_exposure,
            "unlinked_exposure_sgd": unlinked_exposure,
            "unlinked_fraud_caught": len(unlinked_hits),
            "loyal_wrongly_flagged": int(t["loyal_wrongly_flagged"].sum()),
            "flags": int(sel[col].sum()),
            "fraud_caught": int((sel[col] & (sel["is_fraud"] == 1)).sum()),
        }
    result["window"] = {
        "fraud_orders": int((sel["is_fraud"] == 1).sum()),
        "unlinked_fraud_orders": len(unlinked),
    }
    return result


def bootstrap_differences(
    sel: pd.DataFrame, seed: int, n_boot: int = 1000
) -> dict[str, dict[str, float | None]]:
    """Model-minus-rule differences with 95% percentile intervals.

    Customers are resampled with replacement; a customer drawn m times keeps
    multiplicity m on both measures (D34-01). The same draw is applied to the
    rule and the model. Unlinked fraud exposure is held fixed in each draw:
    its customer clustering is unknown. No interval is reported without linked
    customers. Returns also the logged seed and draw count.
    """
    keys = (
        sel.loc[_linked(sel), CUSTOMER_KEY].drop_duplicates().sort_values(CUSTOMER_KEY)
    )
    customers = pd.MultiIndex.from_frame(keys)
    tabs = customer_table(sel, customers)
    rule = tabs["rule"][list(MEASURES)].to_numpy(dtype=float)
    model = tabs["model"][list(MEASURES)].to_numpy(dtype=float)
    diff = model - rule  # per customer, shape (n, 2)
    unlinked = _unlinked_fraud(sel)
    fixed = np.array(
        [
            unlinked.loc[unlinked["model_sel"], "order_value"].sum()
            - unlinked.loc[unlinked["rule_sel"], "order_value"].sum(),
            0.0,
        ]
    )
    rng = np.random.default_rng(seed)
    n = len(customers)
    draws = np.empty((n_boot, 2))
    if n:
        for b in range(n_boot):
            idx = rng.integers(0, n, size=n)
            draws[b] = diff[idx].sum(axis=0) + fixed
    out: dict[str, dict[str, float | None]] = {}
    for j, m in enumerate(MEASURES):
        result = {
            "difference": float(diff[:, j].sum() + fixed[j]),
            "ci_low": None,
            "ci_high": None,
            "includes_zero": None,
        }
        if n:
            lo, hi = np.percentile(draws[:, j], [2.5, 97.5])
            result.update(
                ci_low=float(lo), ci_high=float(hi), includes_zero=bool(lo <= 0 <= hi)
            )
        out[m] = result
    out["meta"] = {"seed": seed, "n_boot": n_boot, "n_customers": n}
    return out


def wording(measure: str, row: dict[str, float]) -> str:
    """D34-01 wording: an interval including 0 is 'no difference was shown'."""
    if row["includes_zero"]:
        return f"{measure}: no difference was shown on that measure"
    direction = "higher" if row["difference"] > 0 else "lower"
    return f"{measure}: model {direction} than rule (interval excludes 0)"


def capacity_curve(frame: pd.DataFrame, capacities: list[int]) -> pd.DataFrame:
    """Fraud caught (count and SGD exposure) at each capacity K, same scores.

    For each daily K, use the #18 rule cut and model at K (not the matched
    model), with the same input scores at every capacity.
    """
    rows = []
    for k in capacities:
        sel = select_flags(frame, k)
        for name, col in (("rule", "rule_sel"), ("model", "model_k_sel")):
            hit = sel[sel[col] & (sel["is_fraud"] == 1)]
            rows.append(
                {
                    "capacity_k": k,
                    "policy": name,
                    "fraud_caught": len(hit),
                    "exposure_sgd": float(hit["order_value"].sum()),
                }
            )
    return pd.DataFrame(rows)


def plot_capacity(curve: pd.DataFrame, path: str) -> None:
    """Save the count and exposure capacity comparison as a plot."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
    for ax, col, label in (
        (axes[0], "fraud_caught", "Fraud orders caught"),
        (axes[1], "exposure_sgd", "Fraud exposure identified (SGD)"),
    ):
        for name, g in curve.groupby("policy"):
            ax.plot(g["capacity_k"], g[col], marker="o", label=name)
        ax.set_xlabel("Review capacity K (flags per selection group)")
        ax.set_ylabel(label)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def render_report(
    est: dict,
    diffs: dict,
    curve_path: str,
    prior_comparisons: int | None,
    *,
    mocked_inputs: bool = False,
) -> str:
    """Markdown report body; run-dependent items stay marked as placeholders.

    Fill ``prior_comparisons`` from the MLflow run records (never from memory);
    None renders a PENDING marker. Label and delay assumptions are cited from
    #30 / D4-04 / D4-05 and are not restated as findings. Use
    ``mocked_inputs=True`` for a synthetic report with preselected flags.
    """
    prior = (
        "PENDING (count from MLflow run records)"
        if prior_comparisons is None
        else prior_comparisons
    )
    lines = [
        "# SGD impact report (#34)",
        "",
    ]
    if mocked_inputs:
        lines += [
            (
                "**Synthetic dry run only.** The inputs are synthetic; "
                "this is not final-window or holdout evidence."
            ),
            "",
        ]
    lines += [
        (
            "| Policy | Flags | Fraud exposure identified (SGD) | Unlinked fraud caught "
            "(orders) | Unlinked fraud exposure (SGD, included in total) | Distinct "
            "loyal customers wrongly flagged |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, label in (
        ("rule", "Rule"),
        ("model", "Model, rule's count"),
        ("model_k", "Model at K"),
    ):
        e = est[key]
        lines.append(
            f"| {label} | {e['flags']} | {e['exposure_sgd']:.2f} | "
            f"{e['unlinked_fraud_caught']} | {e['unlinked_exposure_sgd']:.2f} | "
            f"{e['loyal_wrongly_flagged']} |"
        )
    window = est["window"]
    lines += [
        "",
        (
            f"Fraud orders in the window: {window['fraud_orders']} "
            f"({window['unlinked_fraud_orders']} without a linked customer)."
        ),
        "",
    ]
    meta = diffs["meta"]
    for m, label in zip(
        MEASURES,
        ("fraud exposure identified (SGD)", "distinct loyal customers wrongly flagged"),
        strict=True,
    ):
        d = diffs[m]
        prefix = f"- Model minus rule, {label}: {d['difference']:.2f}"
        if d["ci_low"] is None:
            lines.append(f"{prefix} (no interval: no linked customers).")
        else:
            lines.append(
                f"{prefix} (95% interval {d['ci_low']:.2f} to {d['ci_high']:.2f}). "
                f"{wording(label, d)}."
            )
    lines += [
        "",
        (
            f"Interval: {meta['n_boot']} customer-level bootstrap resamples over "
            f"{meta['n_customers']} linked customers, seed {meta['seed']}."
            if meta["n_customers"]
            else f"No customer-level resampling was possible; seed {meta['seed']}."
        ),
        (
            "Unlinked exposure is held fixed across draws because customer "
            "clustering is unknown. The interval does not quantify its "
            "uncertainty; unlinked orders do not count as distinct customers."
        ),
        (
            "#30 assumptions (D4-04 and D4-05): use a provisional "
            "30-calendar-day delay for eligible historical label-derived features, "
            "not a measured review delay. Supplied "
            "labels use 1 for fraud and 0 for not fraud; review coverage and "
            "finalization remain unverified."
        ),
        (
            "Exposure is order value of flagged fraud orders, not savings or "
            "prevented loss. The result excludes review costs."
        ),
        f"Test-window comparisons run before the final one: {prior}.",
        f"![Fraud caught across review capacities]({curve_path})",
    ]
    return "\n".join(lines) + "\n"
