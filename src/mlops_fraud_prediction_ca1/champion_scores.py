"""Re-score the frozen validation window with the champion's recipe, offline.

    uv run champion-scores --data-dir data --out output/champion-validation-scores.csv

This is a deterministic RE-SCORE, not the original run's artifact. It refits the
`boosted_trees_model` recipe of the registered champion (`fraud-candidate` v1, run
`9c3a59771cf8499d92972f5c7873e280`) with seed 0 on the frozen training window and
scores the frozen validation window. The CSV is accepted by `rule-report --scores`.
"""

import argparse
import hashlib
from pathlib import Path

import pandas as pd

from mlops_fraud_prediction_ca1.models import boosted_trees_model
from mlops_fraud_prediction_ca1.training import (
    K,
    _git,
    dataset_hashes,
    fit_and_score,
    load_window,
)

CHAMPION_RUN_ID = "9c3a59771cf8499d92972f5c7873e280"
MODEL_NAME = "fraud-candidate"
MODEL_VERSION = "1"
# The champion record in docs/champion.md; the re-score must reproduce it.
EXPECTED_FRAUD_CAUGHT = 2893
EXPECTED_AVERAGE_PRECISION = 0.3716


def rescore(data_dir, *, k: int = K, seed: int = 0) -> pd.DataFrame:
    """One row per labelled validation order, with provenance on every row."""
    result = fit_and_score(boosted_trees_model(), data_dir, k=k, seed=seed)
    keys, features = load_window(data_dir, "validation", False)
    hashes = dataset_hashes(data_dir)
    rows = keys[["order_date", "country_code", "order_id"]].astype(str)
    rows["score"] = result.model.predict_proba(features)[:, 1]
    rows["model_name"] = MODEL_NAME
    rows["model_version"] = MODEL_VERSION
    rows["source_run_id"] = CHAMPION_RUN_ID
    rows["artifact_kind"] = "re-score"
    rows["k"] = k
    rows["seed"] = seed
    for name, value in hashes.items():
        rows[name] = value
    rows["code_commit"] = _git("rev-parse", "HEAD")
    rows.attrs["report"] = result.report
    return rows.sort_values(["country_code", "order_id"], kind="stable").reset_index(
        drop=True
    )


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out", required=True)
    parser.add_argument("--k", type=int, default=K)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    rows = rescore(args.data_dir, k=args.k, seed=args.seed)
    report = rows.attrs["report"]
    if (
        args.k == K
        and args.seed == 0
        and (
            report.at_k.fraud_caught != EXPECTED_FRAUD_CAUGHT
            or report.average_precision is None
            or round(report.average_precision, 4) != EXPECTED_AVERAGE_PRECISION
        )
    ):
        raise SystemExit(
            "The re-score does not reproduce the champion aggregate "
            f"({EXPECTED_FRAUD_CAUGHT}, {EXPECTED_AVERAGE_PRECISION}); "
            "the data or code differs from the champion run. No CSV was written."
        )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(out, index=False, float_format="%.17g", lineterminator="\n")
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    print(f"rows: {len(rows)}  sha256: {digest}  path: {out}")
    ap = report.average_precision
    ap_text = "N/A" if ap is None else f"{ap:.4f}"
    print(f"fraud caught at K={args.k}: {report.at_k.fraud_caught}  AP: {ap_text}")
