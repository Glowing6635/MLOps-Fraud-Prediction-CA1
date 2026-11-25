"""Produce #34 aggregate artifacts from a scored export or built-in synthetic data."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from mlops_fraud_prediction_ca1 import impact

BOOTSTRAP_SEED = 34
BOOTSTRAP_DRAWS = 1000
COLUMNS = (
    "order_date",
    "country_code",
    "order_id",
    "customer_id",
    "order_value",
    "is_fraud",
    "loyal",
    "score",
    "rule_flag",
)


def _fixture() -> pd.DataFrame:
    """Synthetic only: duplicate IDs across countries; unlinked and zero-rule groups."""
    rows = [
        ("2024-01-01", "SG", "001", "A", 100, 1, False, 0.1, True),
        ("2024-01-01", "SG", "002", "A", 10, 0, True, 0.2, True),
        ("2024-01-01", "MY", "001", "A", 60, 0, True, 0.8, True),
        ("2024-01-01", "SG", "003", "", 50, 1, False, 0.9, False),
        ("2024-01-02", "SG", "004", "B", 40, 1, False, 0.7, False),
        ("2024-01-02", "MY", "002", "C", 5, 0, True, 0.6, False),
    ]
    return pd.DataFrame(rows, columns=COLUMNS)


def _parse_export(raw: bytes) -> pd.DataFrame:
    # Start from strings to avoid losing leading zeros or mapping ID "NA" to null.
    rows = pd.read_csv(io.BytesIO(raw), dtype=str, keep_default_na=False)
    if len(rows.columns) != len(COLUMNS) or set(rows.columns) != set(COLUMNS):
        raise ValueError("Export must have exactly the columns documented for #34.")
    if not rows.index.equals(pd.RangeIndex(len(rows))):
        raise ValueError("Export contains rows with extra fields.")
    if rows.empty:
        raise ValueError("Export has no scored orders.")
    for col in ("order_date", "country_code", "order_id"):
        if rows[col].str.strip().eq("").any():
            raise ValueError(f"Each {col} is required.")
    rows["customer_id"] = rows["customer_id"].replace(r"^\s*$", pd.NA, regex=True)
    for col in ("loyal", "rule_flag"):
        values = (
            rows[col]
            .str.lower()
            .map({"true": True, "false": False, "1": True, "0": False})
        )
        if values.isna().any():
            raise ValueError(f"{col} must be true/false or 1/0 for every order.")
        rows[col] = values.astype(bool)
    if not rows["is_fraud"].isin(("0", "1")).all():
        raise ValueError("is_fraud must be 0 or 1; unknown labels cannot be scored.")
    rows["is_fraud"] = rows["is_fraud"].astype(int)
    for col in ("score", "order_value"):
        values = pd.to_numeric(rows[col], errors="coerce")
        if not np.isfinite(values.to_numpy(dtype=float, na_value=np.nan)).all():
            raise ValueError(f"{col} must be finite and numeric for every order.")
        rows[col] = values
    return rows


def main(argv: list[str] | None = None) -> None:
    """Create the report, chart and reproduction metadata in a new directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--synthetic", action="store_true")
    mode.add_argument("--scored-csv", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--capacities", type=int, nargs="+", required=True)
    parser.add_argument("--evaluation-run-id")
    parser.add_argument("--freeze-commit")
    parser.add_argument("--prior-comparisons", type=int)
    args = parser.parse_args(argv)

    capacities = sorted(set(args.capacities))
    if args.k < 1 or len(capacities) < 2 or capacities[0] < 1:
        parser.error("K must be positive and the chart needs at least two capacities.")
    prior = args.prior_comparisons
    run = args.evaluation_run_id
    freeze = args.freeze_commit
    if args.synthetic:
        if any(x is not None for x in (prior, run, freeze)):
            parser.error("Synthetic runs cannot claim final evaluation provenance.")
    elif (
        prior is None
        or prior < 0
        or not re.fullmatch(r"[0-9a-f]{32}", run or "")
        or not re.fullmatch(r"[0-9a-f]{40}", freeze or "")
    ):
        parser.error(
            "A scored export requires a 32-character MLflow run ID, a full "
            "40-character freeze commit and a nonnegative prior-comparison count."
        )
    if args.output_dir.exists():
        parser.error("Output directory already exists; choose a new directory.")

    if args.synthetic:
        raw = _fixture().to_csv(index=False, lineterminator="\n").encode("utf-8")
    else:
        raw = args.scored_csv.read_bytes()
    frame = _parse_export(raw)
    selected = impact.select_flags(frame, args.k)
    estimates = impact.point_estimates(selected)
    differences = impact.bootstrap_differences(
        selected, seed=BOOTSTRAP_SEED, n_boot=BOOTSTRAP_DRAWS
    )
    curve = impact.capacity_curve(frame, capacities)
    report = impact.render_report(
        estimates, differences, "capacity.png", prior, mocked_inputs=args.synthetic
    )
    receipt = {
        "mode": "synthetic" if args.synthetic else "scored export",
        "input_sha256": hashlib.sha256(raw).hexdigest(),
        "evaluation_run_id": run,
        "freeze_commit": freeze,
        "prior_test_window_comparisons": prior,
        "daily_k": args.k,
        "capacities": capacities,
        "seed": BOOTSTRAP_SEED,
        "requested_bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_draws_used": BOOTSTRAP_DRAWS
        if differences["meta"]["n_customers"]
        else 0,
    }
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "report.md").write_text(report, encoding="utf-8")
    (args.output_dir / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    curve.to_csv(args.output_dir / "capacity.csv", index=False)
    impact.plot_capacity(curve, str(args.output_dir / "capacity.png"))


if __name__ == "__main__":
    main()
