"""SGD impact comparison of the model and the baseline rule (PBI #34).

Input contract: one row per scored order in the final evaluation window with
columns ``group`` (#5 selection group), ``customer_id``, ``order_id``,
``country_code``, ``order_value`` (SGD), ``is_fraud`` (0/1), ``loyal`` (bool,
loyal definition frozen under #23), ``score`` (champion score, higher = riskier)
and ``rule_flag`` (bool, baseline rule from #18). Reading rules are those of
decision D34-01 in docs/decision-log.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED = (
    "group", "customer_id", "order_id", "country_code",
    "order_value", "is_fraud", "loyal", "score", "rule_flag",
)  # fmt: skip
MEASURES = ("exposure_sgd", "loyal_wrongly_flagged")


def _check(frame: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED if c not in frame.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")


def select_flags(frame: pd.DataFrame, k: int) -> pd.DataFrame:
    """Return boolean columns ``rule_sel``, ``model_sel`` and ``model_k_sel``.

    The rule keeps its flags ranked by order value (ties by country, order id),
    cut at K per group. The model flags the same number as the rule per group
    (``model_sel``); ``model_k_sel`` is the model at K.
    """
    _check(frame)
    f = frame.reset_index(drop=True)
    rule_sel = np.zeros(len(f), dtype=bool)
    model_sel = np.zeros(len(f), dtype=bool)
    model_k = np.zeros(len(f), dtype=bool)
    for idx in f.groupby("group", sort=True).groups.values():
        g = f.loc[idx]
        flagged = g[g["rule_flag"].astype(bool)].sort_values(
            ["order_value", "country_code", "order_id"],
            ascending=[False, True, True],
            kind="mergesort",
        )
        rule_rows = flagged.index[:k]
        rule_sel[rule_rows] = True
        ranked = g.sort_values(
            ["score", "country_code", "order_id"],
            ascending=[False, True, True],
            kind="mergesort",
        ).index
        model_sel[ranked[: len(rule_rows)]] = True
        model_k[ranked[:k]] = True
    return f.assign(rule_sel=rule_sel, model_sel=model_sel, model_k_sel=model_k)


def _per_customer(sel: pd.DataFrame, col: str) -> pd.DataFrame:
    """Per-customer exposure and loyal-wrongly-flagged indicator for ``col``."""
    s = sel[sel[col]]
    fraud = s["is_fraud"] == 1
    exposure = s["order_value"].where(fraud, 0.0).groupby(s["customer_id"]).sum()
    wrong = (
        (s["loyal"].astype(bool) & ~fraud).groupby(s["customer_id"]).any().astype(int)
    )
    return pd.DataFrame({"exposure_sgd": exposure, "loyal_wrongly_flagged": wrong})


def customer_table(sel: pd.DataFrame, customers: pd.Index) -> dict[str, pd.DataFrame]:
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
    for name, col in (
        ("rule", "rule_sel"),
        ("model", "model_sel"),
        ("model_k", "model_k_sel"),
    ):
        t = _per_customer(sel, col)
        result[name] = {
            "exposure_sgd": float(t["exposure_sgd"].sum()),
            "loyal_wrongly_flagged": int(t["loyal_wrongly_flagged"].sum()),
            "flags": int(sel[col].sum()),
            "fraud_caught": int((sel[col] & (sel["is_fraud"] == 1)).sum()),
        }
    result["window"] = {"fraud_orders": int((sel["is_fraud"] == 1).sum())}
    return result


def bootstrap_differences(
    sel: pd.DataFrame, seed: int, n_boot: int = 1000
) -> dict[str, dict[str, float]]:
    """Model-minus-rule differences with 95% percentile intervals.

    Customers are resampled with replacement; a customer drawn m times keeps
    multiplicity m on both measures (D34-01). The same draw is applied to the
    rule and the model. Returns also the logged seed and draw count.
    """
    customers = pd.Index(sorted(sel["customer_id"].unique()))
    tabs = customer_table(sel, customers)
    rule = tabs["rule"][list(MEASURES)].to_numpy(dtype=float)
    model = tabs["model"][list(MEASURES)].to_numpy(dtype=float)
    diff = model - rule  # per customer, shape (n, 2)
    rng = np.random.default_rng(seed)
    n = len(customers)
    draws = np.empty((n_boot, 2))
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        draws[b] = diff[idx].sum(axis=0)
    out: dict[str, dict[str, float]] = {}
    for j, m in enumerate(MEASURES):
        lo, hi = np.percentile(draws[:, j], [2.5, 97.5])
        out[m] = {
            "difference": float(diff[:, j].sum()),
            "ci_low": float(lo),
            "ci_high": float(hi),
            "includes_zero": bool(lo <= 0 <= hi),
        }
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

    Rule flags are cut at K per group as in :func:`select_flags`; the model
    flags its top K per group.
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
    est: dict, diffs: dict, curve_path: str, prior_comparisons: int | None
) -> str:
    """Markdown report body; run-dependent items stay marked as placeholders.

    Fill ``prior_comparisons`` from the MLflow run records (never from memory);
    None renders a PENDING marker. Label and delay assumptions are cited from
    #30 / D4-04 / D4-05 and are not restated as findings.
    """
    prior = (
        "PENDING (count from MLflow run records)"
        if prior_comparisons is None
        else prior_comparisons
    )
    lines = [
        "# SGD impact report (#34)",
        "",
        "| Policy | Flags | Fraud exposure identified (SGD) | Distinct loyal customers wrongly flagged |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key, label in (
        ("rule", "Rule"),
        ("model", "Model, rule's count"),
        ("model_k", "Model at K"),
    ):
        e = est[key]
        lines.append(
            f"| {label} | {e['flags']} | {e['exposure_sgd']:.2f} | {e['loyal_wrongly_flagged']} |"
        )
    lines += ["", f"Fraud orders in the window: {est['window']['fraud_orders']}.", ""]
    meta = diffs["meta"]
    for m in MEASURES:
        d = diffs[m]
        lines.append(
            f"- Model minus rule, {m}: {d['difference']:.2f} "
            f"(95% interval {d['ci_low']:.2f} to {d['ci_high']:.2f}). {wording(m, d)}."
        )
    lines += [
        "",
        (
            f"Interval: {meta['n_boot']} customer-level bootstrap resamples over "
            f"{meta['n_customers']} customers, seed {meta['seed']}."
        ),
        (
            "Exposure is order value of flagged fraud orders, not savings or "
            "prevented loss. The result excludes review costs."
        ),
        f"Test-window comparisons run before the final one: {prior}.",
        f"![Fraud caught across review capacities]({curve_path})",
    ]
    return "\n".join(lines) + "\n"
