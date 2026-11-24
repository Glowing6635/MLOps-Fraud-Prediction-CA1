import numpy as np
import pandas as pd

from mlops_fraud_prediction_ca1 import impact


def _frame() -> pd.DataFrame:
    # Synthetic, illustrative only. One group, K = 2.
    rows = [
        # customer, order, value, fraud, loyal, score, rule_flag
        ("A", "o1", 100.0, 1, True, 0.1, True),
        ("A", "o2", 10.0, 0, True, 0.2, True),
        ("B", "o3", 50.0, 1, False, 0.9, False),
        ("C", "o4", 5.0, 0, True, 0.8, False),
        ("D", "o5", 1.0, 0, False, 0.0, False),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "customer_id",
            "order_id",
            "order_value",
            "is_fraud",
            "loyal",
            "score",
            "rule_flag",
        ],
    ).assign(group="d1", country_code="SG")


def test_rule_cut_by_value_and_model_matched_count():
    sel = impact.select_flags(_frame(), k=1)
    assert sel["rule_sel"].tolist() == [True, False, False, False, False]
    assert sel["model_sel"].tolist() == [False, False, True, False, False]


def test_rule_flags_fewer_than_k_and_model_at_k():
    sel = impact.select_flags(_frame(), k=3)
    assert sel["rule_sel"].sum() == 2
    assert sel["model_sel"].sum() == 2
    assert sel["model_k_sel"].sum() == 3


def test_point_estimates_distinct_customers_and_exposure():
    est = impact.point_estimates(impact.select_flags(_frame(), k=2))
    assert est["rule"]["exposure_sgd"] == 100.0
    assert est["rule"]["loyal_wrongly_flagged"] == 1  # A once, not per order
    assert est["model"]["exposure_sgd"] == 50.0
    assert est["model"]["loyal_wrongly_flagged"] == 1  # C
    assert est["window"]["fraud_orders"] == 2


def test_bootstrap_is_reproducible_and_logs_seed():
    sel = impact.select_flags(_frame(), k=2)
    a = impact.bootstrap_differences(sel, seed=7, n_boot=200)
    b = impact.bootstrap_differences(sel, seed=7, n_boot=200)
    assert a == b
    assert a["meta"] == {"seed": 7, "n_boot": 200, "n_customers": 4}
    assert a["exposure_sgd"]["difference"] == -50.0


def test_bootstrap_multiplicity_matches_decision_example():
    # D34-01 worked example: draw A, A, B, B gives exposure -100 and loyal -2.
    frame = pd.DataFrame(
        {
            "group": "g", "country_code": "SG",
            "customer_id": ["A", "A", "B", "C", "D"],
            "order_id": ["1", "2", "3", "4", "5"],
            "order_value": [50.0, 50.0, 50.0, 1.0, 1.0],
            "is_fraud": [1, 1, 1, 0, 0],
            "loyal": [True, True, False, False, False],
            "score": [0.0, 0.0, 1.0, 0.0, 0.0],
            "rule_flag": [True, True, False, False, False],
        }
    )  # fmt: skip
    sel = impact.select_flags(frame, k=2)
    custs = pd.Index(["A", "B", "C", "D"])
    tabs = impact.customer_table(sel, custs)
    diff = tabs["model"].to_numpy(float) - tabs["rule"].to_numpy(float)
    total = diff[[0, 0, 1, 1]].sum(axis=0)
    # per-customer diffs: A (rule 100, model 50), B (rule 0, model 50)
    assert total[0] == 2 * (50 - 100) + 2 * (50 - 0)
    assert total[1] == 2 * (1 - 1) + 2 * (0 - 0)


def test_zero_interval_wording():
    row = {"difference": 0.0, "ci_low": -1.0, "ci_high": 2.0, "includes_zero": True}
    assert "no difference was shown" in impact.wording("exposure", row)


def test_capacity_curve_and_plot(tmp_path):
    curve = impact.capacity_curve(_frame(), [1, 2, 3])
    assert set(curve["policy"]) == {"rule", "model"}
    assert len(curve) == 6
    out = tmp_path / "c.png"
    impact.plot_capacity(curve, str(out))
    assert out.stat().st_size > 0
    assert np.isfinite(curve["exposure_sgd"]).all()


def test_render_report_marks_pending_and_seed():
    sel = impact.select_flags(_frame(), k=2)
    est = impact.point_estimates(sel)
    diffs = impact.bootstrap_differences(sel, seed=3, n_boot=50)
    text = impact.render_report(est, diffs, "capacity.png", None)
    assert "PENDING" in text and "seed 3" in text and "excludes review costs" in text
