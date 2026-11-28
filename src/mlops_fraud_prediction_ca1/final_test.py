"""One-shot scoring of the final champion and today's rule on the frozen test window (PBI #23).

    uv run final-test --freeze-commit SHA --k 250 --x 13.36 \\
        --loyal-min-prior-orders 2 --champion-mode as-trained \\
        --champion-uri models:/fraud-candidate@champion --dry-run

Add no `--dry-run` only on the day the Product Owner authorises the final run. Before
any test row is read the command:

1. requires `--freeze-commit`: a commit that is an ancestor of HEAD and whose
   `config/final_test_freeze.toml` holds exactly the K, X and loyal definition passed
   (and the loyal definition this code applies);
2. requires `--champion-mode` (no default) and appends it to the ledger BEFORE scoring;
3. refuses when the ledger already holds any entry for the frozen window, whether
   scored, failed or interrupted. A window is scored once, ever.

It then scores the champion and the rule on the same orders and logs the primary
measure (fraud caught at K) with precision and recall at K to the ledger and, with
`--mlflow`, to MLflow. `--dry-run` runs the three checks and stops: it reads no data,
claims nothing and scores nothing. docs/final-test.md has the procedure.
"""

import argparse
import fcntl
import json
import math
import os
import subprocess
import sys
import tomllib
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone

from mlops_fraud_prediction_ca1.loyalty import LOYAL_MIN_PRIOR_ORDERS
from mlops_fraud_prediction_ca1.rule_comparison import compare
from mlops_fraud_prediction_ca1.splitting import load_policy
from mlops_fraud_prediction_ca1.top_k import evaluate_top_k

FREEZE_FILE = "config/final_test_freeze.toml"
DEFAULT_LEDGER = "output/final_test_ledger.jsonl"
AS_TRAINED = "as-trained"
REFIT = "refit-train-validation"
MODES = (AS_TRAINED, REFIT)


class FinalTestRefused(Exception):
    """A gate failed. Nothing was scored and, for a refusal, nothing was recorded."""


@dataclass(frozen=True)
class Window:
    """Labelled orders of one window: keys with `is_fraud`, and the model features."""

    keys: pd.DataFrame
    features: pd.DataFrame


@dataclass(frozen=True)
class FinalInputs:
    """Everything scoring needs, read only after the gates pass.

    `rule_orders` holds the test orders with `order_value`; `account_age` and
    `loyalty` align with its rows. It must cover the same orders as `test`.
    """

    train: Window
    validation: Window
    test: Window
    rule_orders: pd.DataFrame
    account_age: pd.Series
    loyalty: np.ndarray


def _git(repo: Path, *args: str) -> str:
    done = subprocess.run(
        ["git", *args], capture_output=True, text=True, cwd=repo, check=False
    )
    if done.returncode:
        raise FinalTestRefused(f"git {' '.join(args)} failed: {done.stderr.strip()}")
    return done.stdout.strip()


def verify_freeze(
    freeze_commit: str | None,
    *,
    k: int,
    x: float,
    loyal_min_prior_orders: int,
    repo: str | Path = ".",
) -> str:
    """Return the full freezing commit SHA, or raise `FinalTestRefused`.

    The commit must be named, exist, be an ancestor of HEAD, and hold a freeze file
    with exactly this K, X and loyal definition. The loyal definition must also be
    the one this code applies (`loyalty.LOYAL_MIN_PRIOR_ORDERS`).
    """
    repo = Path(repo)
    if not freeze_commit or not freeze_commit.strip():
        raise FinalTestRefused(
            "Name the freezing commit for K, X and the loyal definition."
        )
    sha = _git(repo, "rev-parse", "--verify", f"{freeze_commit.strip()}^{{commit}}")
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", sha, "HEAD"], cwd=repo, check=False
    ).returncode:
        raise FinalTestRefused(f"Freezing commit {sha} is not an ancestor of HEAD.")
    try:
        frozen = tomllib.loads(_git(repo, "show", f"{sha}:{FREEZE_FILE}"))
    except tomllib.TOMLDecodeError as error:
        raise FinalTestRefused(f"{FREEZE_FILE} at {sha} is not valid TOML.") from error
    if set(frozen) != {"k", "x", "loyal_min_prior_orders"}:
        raise FinalTestRefused(
            f"{FREEZE_FILE} at {sha} must hold exactly k, x and loyal_min_prior_orders."
        )
    asked = {"k": k, "x": x, "loyal_min_prior_orders": loyal_min_prior_orders}
    for name, value in asked.items():
        recorded = frozen[name]
        if isinstance(recorded, bool) or recorded != value:
            raise FinalTestRefused(
                f"{name} is {value!r} but the freezing commit froze {recorded!r}."
            )
    if loyal_min_prior_orders != LOYAL_MIN_PRIOR_ORDERS:
        raise FinalTestRefused(
            f"The frozen loyal definition ({loyal_min_prior_orders} prior orders) is not"
            f" the one this code applies ({LOYAL_MIN_PRIOR_ORDERS})."
        )
    return sha


def window_id(config_path: str | Path | None = None) -> str:
    """The frozen test window as `start/end_exclusive`, from the split configuration."""
    policy = load_policy(config_path)
    return f"{policy.test_start.isoformat()}/{policy.test_end_exclusive.isoformat()}"


def _entries(ledger: Path) -> list[dict]:
    if not ledger.exists():
        return []
    return [
        json.loads(line) for line in ledger.read_text().splitlines() if line.strip()
    ]


def check_unscored(ledger: str | Path, window: str) -> None:
    """Raise `FinalTestRefused` when the ledger has any entry for `window`."""
    used = [e for e in _entries(Path(ledger)) if e.get("window") == window]
    if used:
        events = ", ".join(e["event"] for e in used)
        raise FinalTestRefused(
            f"Window {window} is already used (ledger events: {events}). A frozen"
            " window is scored once. Releasing it needs an owner decision recorded in"
            " docs/decision-log.md."
        )


def _append(ledger: Path, entry: dict, *, claim_window: str | None = None) -> None:
    """Append one JSON line under an exclusive lock; a claim re-checks the window."""
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        if claim_window is not None:
            check_unscored(ledger, claim_window)
        handle.write(json.dumps(entry, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _params_hash(estimator) -> str:
    text = json.dumps(estimator.get_params(deep=True), sort_keys=True, default=repr)
    return sha256(text.encode()).hexdigest()


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _fit_rows(*windows: Window) -> tuple[pd.DataFrame, pd.Series]:
    features = pd.concat([w.features for w in windows], ignore_index=True)
    labels = pd.concat([w.keys["is_fraud"] for w in windows], ignore_index=True)
    return features, labels.astype(int)


def prepare_champion(champion, mode: str, inputs: FinalInputs):
    """The estimator to score: as trained, or a fresh copy refit on train + validation.

    A refit clones the champion, so it keeps the stored hyperparameters (including its
    seed) but none of the fitted state, and sees only rows before the test window.
    """
    if mode == AS_TRAINED:
        return champion
    features, labels = _fit_rows(inputs.train, inputs.validation)
    return clone(champion).fit(features, labels)


def score_final(estimator, inputs: FinalInputs, *, k: int, x: float, source: str):
    """Score the champion and the rule on the same test orders; return the results."""
    orders = inputs.rule_orders.reset_index(drop=True)
    test_keys = inputs.test.keys[["country_code", "order_id"]].astype(str)
    scores = pd.Series(
        estimator.predict_proba(inputs.test.features)[:, 1],
        index=pd.MultiIndex.from_frame(test_keys),
    )
    order_keys = pd.MultiIndex.from_frame(
        orders[["country_code", "order_id"]].astype(str)
    )
    if scores.index.duplicated().any() or set(scores.index) != set(order_keys):
        raise ValueError(
            "The test features and the rule orders cover different orders."
        )
    aligned = scores.loc[order_keys].to_numpy()
    age = inputs.account_age.reset_index(drop=True)
    result = compare(orders, aligned, age, inputs.loyalty, k, x, source)
    champion_at_k = evaluate_top_k(orders, aligned, k)
    total_fraud = champion_at_k.at_k.total_fraud
    rule = result["rule"]
    return {
        "orders": champion_at_k.at_k.total_orders,
        "fraud": total_fraud,
        "k": k,
        "x": x,
        "primary_measure": "fraud caught at K",
        "champion": {
            "fraud_caught_at_k": champion_at_k.at_k.fraud_caught,
            "fraud_catchable_at_k": champion_at_k.at_k.fraud_catchable,
            "flags": champion_at_k.at_k.selected,
            "precision_at_k": champion_at_k.at_k.precision,
            "recall_at_k": champion_at_k.at_k.recall,
            "average_precision": champion_at_k.average_precision,
        },
        "rule": {
            "fraud_caught_at_k": rule["fraud_caught"],
            "flags": rule["flags"],
            "precision_at_k": _ratio(rule["fraud_caught"], rule["flags"]),
            "recall_at_k": _ratio(rule["fraud_caught"], total_fraud),
        },
        "model_matched_to_rule_fraud_caught": result["model_matched"]["fraud_caught"],
        "champion_beats_rule_at_matched_flags": bool(result["beats_rule"]),
    }


def run_final_test(
    *,
    freeze_commit: str | None,
    k: int,
    x: float,
    loyal_min_prior_orders: int,
    mode: str | None,
    champion,
    champion_id: str,
    load_inputs: Callable[[], FinalInputs],
    ledger: str | Path = DEFAULT_LEDGER,
    config_path: str | Path | None = None,
    repo: str | Path = ".",
    dry_run: bool = False,
    log_mlflow: Callable[[dict], None] | None = None,
) -> dict:
    """Gate, record, then score once. `load_inputs` runs only after every gate passes.

    Order: freeze check, mode check, window unused check, (stop here for a dry run),
    claim the window and record the mode, read the data, fit if refit, score, log.
    Any exception after the claim leaves the claim in place, so the window stays used.
    """
    repo, ledger = Path(repo), Path(ledger)
    sha = verify_freeze(
        freeze_commit,
        k=k,
        x=x,
        loyal_min_prior_orders=loyal_min_prior_orders,
        repo=repo,
    )
    if mode not in MODES:
        raise FinalTestRefused(
            f"Record the champion mode first: one of {', '.join(MODES)}."
        )
    if not isinstance(k, int) or isinstance(k, bool) or k < 1 or not math.isfinite(x):
        raise FinalTestRefused("K must be a whole number of at least 1 and X finite.")
    window = window_id(config_path)
    check_unscored(ledger, window)
    code_commit = _git(repo, "rev-parse", "HEAD")
    dirty = bool(_git(repo, "status", "--porcelain", "--untracked-files=no"))
    plan = {
        "window": window,
        "mode": mode,
        "k": k,
        "x": x,
        "loyal_min_prior_orders": loyal_min_prior_orders,
        "freeze_commit": sha,
        "code_commit": code_commit,
        "champion_id": champion_id,
    }
    if dry_run:
        return {"dry_run": True, **plan, "code_dirty": dirty}
    if dirty:
        raise FinalTestRefused("Tracked files have uncommitted changes; commit first.")

    claim = {
        **plan,
        "event": "mode_recorded",
        "at": _now(),
        "champion_params_sha256": _params_hash(champion),
    }
    _append(ledger, claim, claim_window=window)
    try:
        inputs = load_inputs()
        estimator = prepare_champion(champion, mode, inputs)
        result = score_final(
            estimator, inputs, k=k, x=x, source=f"frozen X at {sha[:12]}"
        )
    except BaseException as error:
        _append(
            ledger,
            {"event": "failed", "window": window, "at": _now(), "error": repr(error)},
        )
        raise
    entry = {
        "event": "scored",
        "window": window,
        "at": _now(),
        **plan,
        "result": result,
    }
    _append(ledger, entry)
    if log_mlflow is not None:
        log_mlflow(entry)
    return entry


def format_result(entry: dict) -> str:
    """The logged result as plain text."""
    if entry.get("dry_run"):
        return (
            f"DRY RUN: gates passed for window {entry['window']} (mode {entry['mode']}, "
            f"K={entry['k']}, X={entry['x']}, freeze {entry['freeze_commit'][:12]}). "
            "No data read, nothing recorded, nothing scored."
        )
    r = entry["result"]

    def show(value):
        return "N/A" if value is None else f"{value:.4f}"

    lines = [
        f"Window {entry['window']}  mode {entry['mode']}  K={r['k']}  X={r['x']}",
        f"Orders {r['orders']}  fraud {r['fraud']}  primary measure: {r['primary_measure']}",
    ]
    for name in ("champion", "rule"):
        m = r[name]
        lines.append(
            f"{name:9} caught {m['fraud_caught_at_k']:>5}  flags {m['flags']:>5}"
            f"  precision@K {show(m['precision_at_k'])}  recall@K {show(m['recall_at_k'])}"
        )
    return "\n".join(lines)


def log_to_mlflow(entry: dict) -> None:
    """Log the scored entry as one MLflow run (needs MLFLOW_TRACKING_URI)."""
    import mlflow
    from dotenv import load_dotenv

    load_dotenv()
    uri, name = os.getenv("MLFLOW_TRACKING_URI"), os.getenv("MLFLOW_EXPERIMENT_NAME")
    if not uri or not name:
        raise FinalTestRefused("Set MLFLOW_TRACKING_URI and MLFLOW_EXPERIMENT_NAME.")
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(name)
    r = entry["result"]
    with mlflow.start_run(run_name="final-one-shot-test"):
        mlflow.set_tags({"final_test": "true", "window": entry["window"]})
        mlflow.log_params(
            {
                key: entry[key]
                for key in (
                    "mode",
                    "k",
                    "x",
                    "loyal_min_prior_orders",
                    "freeze_commit",
                    "code_commit",
                    "champion_id",
                )
            }
        )
        for who in ("champion", "rule"):
            for key, value in r[who].items():
                if value is not None:
                    mlflow.log_metric(f"{who}_{key}", float(value))


def _real_inputs(data_dir, config_path) -> FinalInputs:
    """Read the final window through the protected reader (raises while it is disabled)."""
    from mlops_fraud_prediction_ca1.rule_report import validation_inputs
    from mlops_fraud_prediction_ca1.training import _load_window

    windows = {
        purpose: Window(*_load_window(data_dir, purpose, False, config_path))
        for purpose in ("train", "validation", "final_evaluation")
    }
    rule = validation_inputs(
        data_dir,
        purpose="final_evaluation",
        config_path=config_path,
        history_purposes=("train", "validation"),
    )
    return FinalInputs(
        train=windows["train"],
        validation=windows["validation"],
        test=windows["final_evaluation"],
        rule_orders=rule["orders"],
        account_age=rule["age"],
        loyalty=rule["loyalty"],
    )


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--freeze-commit", required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--x", type=float, required=True)
    parser.add_argument("--loyal-min-prior-orders", type=int, required=True)
    parser.add_argument("--champion-mode", choices=MODES, required=True)
    parser.add_argument("--champion-uri", required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--config", default=None, help="time_split.toml path")
    parser.add_argument("--ledger", default=DEFAULT_LEDGER)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--mlflow", action="store_true")
    args = parser.parse_args(argv)

    try:
        champion = None
        if not args.dry_run:
            import mlflow.sklearn

            champion = mlflow.sklearn.load_model(args.champion_uri)
        entry = run_final_test(
            freeze_commit=args.freeze_commit,
            k=args.k,
            x=args.x,
            loyal_min_prior_orders=args.loyal_min_prior_orders,
            mode=args.champion_mode,
            champion=champion,
            champion_id=args.champion_uri,
            load_inputs=lambda: _real_inputs(args.data_dir, args.config),
            ledger=args.ledger,
            config_path=args.config,
            dry_run=args.dry_run,
            log_mlflow=log_to_mlflow if args.mlflow else None,
        )
    except FinalTestRefused as error:
        sys.exit(f"final-test refused: {error}")
    print(format_result(entry))


if __name__ == "__main__":
    main()
