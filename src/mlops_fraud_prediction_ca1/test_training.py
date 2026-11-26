"""Tests for the training harness on small made-up base files in a temporary folder."""

import re
import sys

import mlflow
import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin

from mlops_fraud_prediction_ca1 import training
from mlops_fraud_prediction_ca1.models import (
    baseline_model,
    boosted_trees_model,
    extra_trees_model,
    random_forest_model,
)
from mlops_fraud_prediction_ca1.top_k import K
from mlops_fraud_prediction_ca1.training import (
    _load_window,
    check_feature_columns,
    check_label_history,
    compare_customer_columns,
    fit_and_score,
    main,
    rerun,
    train,
)

# A value only test-window orders have, so it shows if one reaches a model.
TEST_WINDOW_VALUE = 999_999.0
WINDOWS = {
    "train": ("2022-11-20", 300),
    "validation": ("2023-03-28", 120),
    "test": ("2023-04-18", 60),
}


@pytest.fixture
def data_dir(tmp_path):
    rng = np.random.default_rng(0)
    customers = pd.DataFrame(
        {
            "country_code": "MY",
            "customer_id": [f"c{i}" for i in range(80)],
            "mobile_verified": "true",
            "num_orders_last_50days": 1,
            "num_cancelled_orders_last_50days": 0,
            "num_refund_orders_last_50days": 0,
            "total_payment_last_50days": 10.0,
            "num_associated_customers": 1,
            "first_order_datetime": "2022-01-01 10:00:00",
        }
    )
    # Customer c0's first order comes after all its orders, so its values are withheld.
    customers.loc[0, "first_order_datetime"] = "2024-01-01 10:00:00"
    orders, labels = [], []
    for window, (start, count) in WINDOWS.items():
        for i in range(count):
            order_id = f"{window}{i}"
            fraud = int(rng.random() < 0.2)
            orders.append(
                {
                    "country_code": "MY",
                    "order_id": order_id,
                    "collect_type": rng.choice(["delivery", "pickup"]),
                    "payment_method": "credit card",
                    "order_value": TEST_WINDOW_VALUE
                    if window == "test"
                    else round(rng.normal(50 + 50 * fraud, 10), 2),
                    "num_items_ordered": int(rng.integers(1, 5)),
                    "refund_value": 0,
                    "order_date": str(
                        (pd.Timestamp(start) + pd.Timedelta(days=i % 14)).date()
                    ),
                }
            )
            labels.append(
                {
                    "country_code": "MY",
                    "order_id": order_id,
                    "customer_id": f"c{rng.integers(80)}",
                    "is_fraud": fraud,
                }
            )
    for name, frame in (
        ("customers", customers),
        ("orders", pd.DataFrame(orders)),
        ("labels", pd.DataFrame(labels)),
    ):
        frame.to_csv(tmp_path / f"{name}_base.csv", index=False)
    return tmp_path


class _Recorder(ClassifierMixin, BaseEstimator):
    """Scores by order value and records every feature matrix it is given."""

    def fit(self, X, y):
        self.seen_ = [X]
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        self.seen_.append(X)
        score = X["order_value"].fillna(0).to_numpy() / 1e6
        return np.column_stack([1 - score, score])


@pytest.mark.parametrize("exploratory", [False, True])
def test_no_test_window_row_reaches_training_or_validation(data_dir, exploratory):
    for purpose in ("train", "validation"):
        keys, _ = _load_window(data_dir, purpose, exploratory)
        assert not keys.order_id.str.startswith("test").any()
        assert keys.order_id.str.startswith(purpose).all()
    model = _Recorder()
    result = fit_and_score(model, data_dir, k=5, seed=0, exploratory=exploratory)
    fitted, scored = model.seen_
    assert (len(fitted), len(scored)) == (300, 120)
    assert (result.train_orders, result.validation_orders) == (300, 120)
    for features in (fitted, scored):
        assert not features.order_value.eq(TEST_WINDOW_VALUE).any()
        check_feature_columns(features)


def test_exploratory_runs_add_customer_columns_only_on_request(data_dir):
    _, order_only = _load_window(data_dir, "train", exploratory=False)
    _, with_customers = _load_window(data_dir, "train", exploratory=True)
    assert list(order_only) == [
        "collect_type",
        "payment_method",
        "order_value",
        "num_items_ordered",
    ]
    keys, _ = _load_window(data_dir, "train", exploratory=True)
    withheld = keys.customer_id.eq("c0").to_numpy()
    assert withheld.any()
    assert with_customers.loc[withheld, "mobile_verified"].isna().all()
    assert with_customers.loc[~withheld, "mobile_verified"].eq(1.0).all()
    assert with_customers.loc[~withheld, "num_refund_orders_last_50days"].eq(0).all()
    # The pipelines drop date columns, so the first order date must arrive as a number.
    assert "first_order_datetime" not in with_customers
    assert with_customers.loc[withheld, "days_since_first_order"].isna().all()
    first_train_day = pd.Timestamp(WINDOWS["train"][0]) - pd.Timestamp("2022-01-01")
    assert (
        with_customers.loc[~withheld, "days_since_first_order"] >= first_train_day.days
    ).all()


@pytest.mark.parametrize(
    "column",
    [
        "is_fraud",
        "refund_value",
        "raw_refund_value",
        "refund_value_share",
        "order_id",
        "customer_id",
        "batch_id",
    ],
)
def test_forbidden_columns_fail(column):
    with pytest.raises(ValueError, match=column):
        check_feature_columns(pd.DataFrame(columns=["order_value", column]))


def test_allowed_columns_pass():
    check_feature_columns(
        pd.DataFrame(columns=["order_value", "num_refund_orders_last_50days"])
    )


def test_history_feature_using_an_immature_label_fails():
    # A made-up history feature: the customer's earlier fraud labels. A label is
    # usable only when its order_date plus 30 days is before the scored order's date.
    order_dates = pd.Series(["2023-03-01", "2023-03-01"])
    check_label_history(order_dates, pd.Series(["2023-01-15", "2023-01-29"]))
    for label_date in ("2023-01-30", "2023-02-20", "2023-03-01", "2023-03-05", None):
        with pytest.raises(ValueError):
            check_label_history(order_dates, pd.Series(["2023-01-15", label_date]))
    with pytest.raises(ValueError):
        check_label_history(
            pd.to_datetime(order_dates), pd.to_datetime(pd.Series(["2023-01-30"] * 2))
        )


def test_a_run_logs_every_field_and_reruns_within_tolerance(
    data_dir, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)  # MLflow stores artifacts under the working folder.
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "harness-test")
    # No git identity, as on a CI runner.
    (tmp_path / "gitconfig").touch()
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(tmp_path / "gitconfig"))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    run_id, result = train(baseline_model(), data_dir, k=5, seed=7)
    run = mlflow.get_run(run_id)
    params, metrics, tags = run.data.params, run.data.metrics, run.data.tags
    assert params["k"] == "5" and params["seed"] == "7"
    assert params["exploratory"] == "False"
    assert params["logisticregression__max_iter"] == "1000"
    assert params["cleaning_policy_version"]
    assert params["train_start"] == "2022-11-12"
    assert params["test_end_exclusive"] == "2023-05-11"
    assert all(
        len(params[f"dataset_sha256_{name}"]) == 64
        for name in ("customers", "orders", "labels")
    )
    assert len(tags["git_commit"]) == 40
    assert tags["author"] == "unknown" and tags["git_dirty"] in ("True", "False")
    assert metrics["fraud_caught_at_k"] == result.report.at_k.fraud_caught
    for name in (
        "fraud_catchable_at_k",
        "fraud_caught_at_2k",
        "average_precision",
        "validation_customer_overlap",
        "fit_seconds",
        "model_size_bytes",
    ):
        assert name in metrics
    assert 0 < metrics["validation_customer_overlap"] <= 1
    assert metrics["model_size_bytes"] > 0
    assert mlflow.sklearn.load_model(f"runs:/{run_id}/model") is not None

    again = rerun(run_id, data_dir)
    assert again["reproduced"]
    assert again["fraud_caught_at_k"] == metrics["fraud_caught_at_k"]

    (data_dir / "labels_base.csv").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="labels"):
        rerun(run_id, data_dir)


def test_customer_columns_are_compared_under_one_parent_run(
    data_dir, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "harness-test")
    parent_id, (without_id, without), (with_id, with_columns) = (
        compare_customer_columns(
            boosted_trees_model(),
            data_dir,
            k=5,
            seed=3,
            tags={"k_status": "provisional"},
        )
    )
    children = mlflow.search_runs(
        filter_string=f"tags.mlflow.parentRunId = '{parent_id}'",
        output_format="list",
    )
    assert {run.info.run_id for run in children} == {without_id, with_id}
    parent, without_run, with_run = map(
        mlflow.get_run, (parent_id, without_id, with_id)
    )
    assert "HistGradientBoostingClassifier" in parent.data.tags["mlflow.runName"]
    assert without_run.data.params["exploratory"] == "False"
    assert with_run.data.params["exploratory"] == "True"
    assert with_run.data.tags["champion_eligible"] == "False"
    for run in (parent, without_run, with_run):
        assert run.data.tags["k_status"] == "provisional"
    assert without_run.data.metrics["fraud_caught_at_k"] == (
        without.report.at_k.fraud_caught
    )
    assert parent.data.metrics["fraud_caught_at_k_difference"] == (
        with_columns.report.at_k.fraud_caught - without.report.at_k.fraud_caught
    )
    assert rerun(with_id, data_dir)["reproduced"]


def test_the_train_command_uses_the_decided_k_and_its_status_by_default(
    data_dir, tmp_path, monkeypatch, capsys
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "harness-test")
    monkeypatch.setattr(
        sys, "argv", ["train", "boosted-trees", "--data-dir", str(data_dir)]
    )
    main()
    run_id = re.findall(r"Logged run (\w+)\.", capsys.readouterr().out)[-1]
    run = mlflow.get_run(run_id)
    assert run.data.params["k"] == str(K)
    assert run.data.tags["k_status"] == f"D5-01: product owner assumption, K = {K}"


def test_candidate_cli_tags_cannot_replace_dirty_git_provenance(
    data_dir, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "provenance-test")
    actual_commit = "a" * 40
    git_results = {
        "rev-parse": actual_commit,
        "status": " M training.py",
        "config": "Fixture Author",
    }
    monkeypatch.setattr(training, "_git", lambda command, *args: git_results[command])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train",
            "boosted-trees",
            "--data-dir",
            str(data_dir),
            "--tag",
            "git_dirty=False",
            "--tag",
            "git_commit=overridden",
            "--tag",
            "author=overridden",
            "--tag",
            "pr=164",
        ],
    )
    with pytest.warns(UserWarning, match="working tree has uncommitted changes"):
        main()

    runs = mlflow.search_runs(
        experiment_names=["provenance-test"], output_format="list"
    )
    assert len(runs) == 3
    for run in runs:
        tags = run.data.tags
        assert tags["git_dirty"] == "True"
        assert tags["git_commit"] == actual_commit
        assert tags["author"] == "Fixture Author"
        assert tags["pr"] == "164"
    assert sum(run.data.tags.get("champion_eligible") == "True" for run in runs) == 1


def test_the_train_command_needs_a_k_status_for_another_k(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["train", "boosted-trees", "--k", "100"])
    with pytest.raises(SystemExit):
        main()
    assert "--k-status" in capsys.readouterr().err


def test_balanced_weighting_leaves_the_validation_fraud_rate_natural(data_dir):
    # Criterion 2 of #17: weights change the fit only; the scored window is untouched.
    results = {
        weighting: fit_and_score(
            baseline_model(class_weight=weighting), data_dir, k=K, seed=0
        )
        for weighting in (None, "balanced")
    }
    validation = pd.read_csv(data_dir / "labels_base.csv")
    validation = validation[validation.order_id.str.startswith("validation")]
    natural = validation.is_fraud.mean()
    assert 0 < natural < 0.5  # imbalanced, so a rebalanced window would differ
    for result in results.values():
        assert result.validation_orders == len(validation)
        assert result.report.at_k.total_fraud / result.validation_orders == natural
    assert (
        results[None].report.at_k.total_fraud
        == results["balanced"].report.at_k.total_fraud
    )


@pytest.mark.parametrize("weighting", [None, "balanced"])
def test_feature_group_cli_logs_weighting_for_both_runs(
    weighting, data_dir, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "feature-weights-test")
    argv = ["train", "compare-feature-groups", "--data-dir", str(data_dir)]
    if weighting is not None:
        argv.extend(["--weighting", weighting])
    monkeypatch.setattr(sys, "argv", argv)
    main()

    runs = mlflow.search_runs(
        experiment_names=["feature-weights-test"], output_format="list"
    )
    assert len(runs) == 2
    assert {run.data.params["features__group"] for run in runs} == {
        "order_raw",
        "order_log",
    }
    for run in runs:
        assert run.data.params["model__logisticregression__class_weight"] == str(
            weighting
        )


def test_rerun_without_mlflow_settings_says_what_to_set(
    data_dir, tmp_path, monkeypatch
):
    # Without the check, MLflow would quietly open a new local store in the working folder.
    monkeypatch.chdir(tmp_path)
    # An empty value also stops load_dotenv from filling it in from a developer's .env.
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "")
    with pytest.raises(RuntimeError, match="MLFLOW_TRACKING_URI"):
        rerun("0" * 32, data_dir)


@pytest.mark.parametrize("build", [random_forest_model, extra_trees_model])
def test_forest_candidates_log_both_runs_and_rerun(
    build, data_dir, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "forest-test")
    _, (without_id, _), (with_id, _) = compare_customer_columns(
        build(), data_dir, k=5, seed=3, tags={"k_status": "provisional"}
    )
    assert mlflow.get_run(without_id).data.tags["champion_eligible"] == "True"
    assert mlflow.get_run(with_id).data.tags["champion_eligible"] == "False"
    assert rerun(without_id, data_dir)["reproduced"]
