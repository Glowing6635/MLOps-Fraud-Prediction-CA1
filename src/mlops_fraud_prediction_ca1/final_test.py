"""Score the final champion and today's rule once on the locked test window (issue #23).

    uv run final-test --data-dir data

The settings come from `config/final_test.toml`: the registered model and version,
whether it is scored as trained or refit first, and the frozen K, loyal count and rule
threshold X. Scoring needs a clean git tree, so the settings and the code are exactly a
commit, and `final_evaluation_enabled = true` in `config/time_split.toml`.

The test window is scored once per MLflow store: a run tagged `final_test = true` in
any experiment, deleted ones included, stops a new attempt. Every check that reads no
test row runs first, and a refit is fitted before the test window is read. The run
tagged `final_test = true` is created just before the first test row is read: a
failure before that leaves no run and can be retried, and from then on the attempt is
used up, even if it fails. A second attempt stops before reading anything.
"""

import argparse
import sys
import tomllib
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
import skops.io
from mlflow.entities import ViewType
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from sklearn.base import clone

from mlops_fraud_prediction_ca1.loyalty import LOYAL_MIN_PRIOR_ORDERS
from mlops_fraud_prediction_ca1.rule_comparison import compare, format_comparison
from mlops_fraud_prediction_ca1.rule_report import (
    clean_window,
    labelled_orders,
    window_inputs,
)
from mlops_fraud_prediction_ca1.splitting import load_policy
from mlops_fraud_prediction_ca1.top_k import K, evaluate_top_k, format_report
from mlops_fraud_prediction_ca1.training import (
    _dataset_hashes,
    _git,
    _model_inputs,
    _use_experiment,
    check_feature_columns,
)

FREEZE_PATH = Path(__file__).resolve().parents[2] / "config/final_test.toml"
_SETTINGS = {
    "model_name": str,
    "model_version": str,
    "mode": str,
    "k": int,
    "loyal_min_prior_orders": int,
    "threshold_x": float,
}
_MODES = ("as_trained", "refit")


def score_final_test(
    data_dir: str | Path,
    *,
    freeze_path: str | Path = FREEZE_PATH,
    config_path: str | Path | None = None,
) -> tuple[str, str]:
    """Score the frozen model and rule on the test window; return the run ID and the
    report logged as `final_test.txt`.

    `config_path` is the time-split configuration, as in `data.load_base_data`. Raises,
    with no run created, when a check fails, the model version can't be loaded, or the
    test window was already scored in this MLflow store.
    """
    settings = _frozen_settings(Path(freeze_path))
    policy = load_policy(config_path)
    if not policy.final_evaluation_enabled:
        raise ValueError(
            "Set final_evaluation_enabled = true in the time-split configuration"
            " and commit it before the final test."
        )
    _use_experiment()
    client = MlflowClient()
    experiments = [
        experiment.experiment_id
        for experiment in client.search_experiments(view_type=ViewType.ALL)
    ]
    earlier = client.search_runs(
        experiments,
        "tags.final_test = 'true'",
        run_view_type=ViewType.ALL,
        max_results=1,
    )
    if earlier:
        run_id = earlier[0].info.run_id
        raise RuntimeError(
            f"The test window was already scored, in run {run_id}. A second scoring"
            " needs a new decision-log entry saying why, and then the old run's tag"
            f' changed by hand: MlflowClient().set_tag("{run_id}", "final_test",'
            ' "superseded").'
        )
    version = client.get_model_version(
        settings["model_name"], settings["model_version"]
    )
    model = mlflow.sklearn.load_model(
        f"models:/{settings['model_name']}/{settings['model_version']}"
    )
    columns = list(model.feature_names_in_)
    history = [
        clean_window(data_dir, purpose, config_path=config_path)
        for purpose in ("train", "validation")
    ]
    # Stops on a missing or forbidden model column before any test row is read.
    _features(labelled_orders(history[1]), columns)
    refit_orders = None
    if settings["mode"] == "refit":
        # clone keeps every hyperparameter, random_state (the training seed) included.
        model = clone(model)
        rows = pd.concat(map(labelled_orders, history), ignore_index=True)
        model.fit(_features(rows, columns), rows["is_fraud"])
        refit_orders = len(rows)

    with mlflow.start_run(
        run_name="final test",
        tags={
            "final_test": "true",
            "mode": settings["mode"],
            "model_name": settings["model_name"],
            "model_version": settings["model_version"],
            "source_run_id": version.run_id,
            "git_commit": _git("rev-parse", "HEAD"),
            "freeze_commit": _git("log", "-1", "--format=%H", "--", str(freeze_path)),
        },
    ) as run:
        test = clean_window(data_dir, "final_evaluation", config_path=config_path)
        inputs = window_inputs([*history, test], test)
        orders = inputs["orders"]
        scores = model.predict_proba(_features(orders, columns))[:, 1]
        report = evaluate_top_k(orders, scores, settings["k"])
        result = compare(
            orders,
            scores,
            inputs["age"],
            inputs["loyalty"],
            settings["k"],
            settings["threshold_x"],
            "frozen in config/final_test.toml",
        )
        mlflow.log_params(
            {
                "k": settings["k"],
                "threshold_x": settings["threshold_x"],
                "loyal_min_prior_orders": settings["loyal_min_prior_orders"],
                "test_start": policy.test_start.isoformat(),
                "test_end_exclusive": policy.test_end_exclusive.isoformat(),
                **_dataset_hashes(data_dir),
            }
        )
        mlflow.log_metrics(
            {
                name: value
                for name, value in _metrics(
                    report, result, len(orders), refit_orders
                ).items()
                if value is not None
            }
        )
        text = (
            f"Model {settings['model_name']} version {settings['model_version']},"
            f" scored {settings['mode']}.\n\n{format_report(report)}\n\n"
            f"{format_comparison(result)}\n"
        )
        mlflow.log_text(text, "final_test.txt")
        if refit_orders is not None:
            # The model was fitted in this process, so every type in it is trusted.
            mlflow.sklearn.log_model(
                model,
                name="refit_model",
                skops_trusted_types=skops.io.get_untrusted_types(
                    data=skops.io.dumps(model)
                ),
            )
    return run.info.run_id, text


def _frozen_settings(path: Path) -> dict:
    """The committed settings, checked against the code and the git tree."""
    settings = tomllib.loads(path.read_text(encoding="utf-8"))
    if set(settings) != set(_SETTINGS) or not all(
        isinstance(settings[name], kind) for name, kind in _SETTINGS.items()
    ):
        raise ValueError(
            f"{path} needs exactly these settings: {', '.join(_SETTINGS)}."
        )
    if settings["mode"] not in _MODES:
        raise ValueError(f"mode is {settings['mode']!r}; use one of {_MODES}.")
    if (settings["k"], settings["loyal_min_prior_orders"]) != (
        K,
        LOYAL_MIN_PRIOR_ORDERS,
    ):
        raise ValueError(
            f"{path} sets k = {settings['k']} and loyal_min_prior_orders ="
            f" {settings['loyal_min_prior_orders']}, but the code uses {K} and"
            f" {LOYAL_MIN_PRIOR_ORDERS}. Commit matching values first."
        )
    if _git("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError(
            "The git tree has uncommitted changes. Commit the frozen settings and"
            " code, so the final test is exactly a commit."
        )
    if not _git("log", "-1", "--format=%H", "--", str(path)):
        raise RuntimeError(f"{path} is not committed.")
    return settings


def _features(orders: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """The columns the model was trained on, as the training harness gives them."""
    missing = [name for name in columns if name not in orders.columns]
    if missing:
        raise ValueError(f"The test rows lack the model's columns: {missing}.")
    features = orders[columns]
    check_feature_columns(features)
    return _model_inputs(features)


def _metrics(report, result, test_orders, refit_orders) -> dict:
    total_fraud = report.at_k.total_fraud
    rule = result["rule"]
    metrics = {
        "test_orders": test_orders,
        "refit_train_orders": refit_orders,
        "average_precision": report.average_precision,
        # The rule's flags, cut to each day's K orders of highest order value.
        "rule_selected_at_k": rule["flags"],
        "rule_fraud_caught_at_k": rule["fraud_caught"],
        "rule_precision_at_k": rule["fraud_caught"] / rule["flags"]
        if rule["flags"]
        else None,
        "rule_recall_at_k": rule["fraud_caught"] / total_fraud if total_fraud else None,
        "model_fraud_caught_at_rule_flags": result["model_matched"]["fraud_caught"],
        "beats_rule": int(result["beats_rule"]),
    }
    for suffix, at in (("at_k", report.at_k), ("at_2k", report.at_2k)):
        for name in ("fraud_caught", "fraud_catchable", "selected", "total_fraud"):
            metrics[f"{name}_{suffix}"] = getattr(at, name)
        metrics[f"precision_{suffix}"] = at.precision
        metrics[f"recall_{suffix}"] = at.recall
    return metrics


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--data-dir", default="data", help="Folder with the three base CSV files."
    )
    args = parser.parse_args(argv)
    # MLflow prints an emoji when a run ends, which Windows cannot encode when piped.
    sys.stdout.reconfigure(errors="replace")
    try:
        run_id, text = score_final_test(args.data_dir)
    except (ValueError, RuntimeError, MlflowException) as error:
        sys.exit(str(error))
    print(f"{text}Logged run {run_id}.")  # noqa: T201


if __name__ == "__main__":
    main()
