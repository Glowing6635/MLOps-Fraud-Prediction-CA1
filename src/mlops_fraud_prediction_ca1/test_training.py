"""Tests for the training harness on small made-up base files in a temporary folder."""

import json
import re
import subprocess
import sys
from decimal import Decimal
from types import SimpleNamespace

import mlflow
import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.class_weight import compute_class_weight

from mlops_fraud_prediction_ca1 import training
from mlops_fraud_prediction_ca1.models import (
    baseline_model,
    boosted_trees_model,
    extra_trees_model,
    random_forest_model,
)
from mlops_fraud_prediction_ca1.top_k import K
from mlops_fraud_prediction_ca1.training import (
    check_feature_columns,
    check_label_history,
    compare_customer_columns,
    compare_groups,
    fit_and_score,
    load_window,
    main,
    rerun,
    score_labelled_rows,
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

    def fit(self, X, y, sample_weight=None):
        self.seen_ = [X]
        self.sample_weight_ = sample_weight
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        self.seen_.append(X)
        score = X["order_value"].fillna(0).to_numpy() / 1e6
        return np.column_stack([1 - score, score])


@pytest.mark.parametrize("exploratory", [False, True])
def test_no_test_window_row_reaches_training_or_validation(data_dir, exploratory):
    for purpose in ("train", "validation"):
        keys, _ = load_window(data_dir, purpose, exploratory)
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


def _database_rows(data_dir):
    """The fixture's labelled training and validation orders, as the database gives them."""
    rows = pd.read_csv(data_dir / "orders_base.csv").merge(
        pd.read_csv(data_dir / "labels_base.csv"), on=["country_code", "order_id"]
    )
    rows["order_value"] = rows.order_value.map(lambda value: Decimal(str(value)))
    rows["order_date"] = pd.to_datetime(rows.order_date).dt.date
    rows.index += 1000  # A filtered frame keeps its row numbers.
    return tuple(
        rows[rows.order_id.str.startswith(window)] for window in ("train", "validation")
    )


def test_labelled_rows_from_the_database_score_like_the_base_files(data_dir):
    windows = _database_rows(data_dir)
    from_files = fit_and_score(_Recorder(), data_dir, k=5, seed=0)
    model = _Recorder()
    from_rows = fit_and_score(model, k=5, seed=0, labelled_windows=windows)
    assert from_rows.report == from_files.report
    assert (from_rows.train_orders, from_rows.validation_orders) == (300, 120)
    assert model.seen_[0].order_value.dtype == "float64"
    assert score_labelled_rows(model, windows[1], k=5) == from_files.report
    for wrong in (
        {},
        {"data_dir": data_dir, "labelled_windows": windows},
        {"labelled_windows": windows, "exploratory": True},
    ):
        with pytest.raises(ValueError):
            fit_and_score(_Recorder(), k=5, seed=0, **wrong)


def test_exploratory_runs_add_customer_columns_only_on_request(data_dir):
    _, order_only = load_window(data_dir, "train", exploratory=False)
    _, with_customers = load_window(data_dir, "train", exploratory=True)
    assert list(order_only) == [
        "collect_type",
        "payment_method",
        "order_value",
        "num_items_ordered",
    ]
    keys, _ = load_window(data_dir, "train", exploratory=True)
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


def test_a_run_on_labelled_rows_logs_the_callers_metrics_and_model(
    data_dir, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "harness-test")
    windows = _database_rows(data_dir)
    model = boosted_trees_model()
    model.set_params(
        **{name: 3 for name in model.get_params() if name.endswith("random_state")}
    )
    run_id, result = train(
        model,
        k=5,
        seed=None,
        labelled_windows=windows,
        tags={"mlflow.runName": "trees-b1", "batch_id": "b1"},
        extra_metrics=lambda fit: {"caught_less_one": fit.report.at_k.fraud_caught - 1},
        max_model_size_bytes=50 * 1024 * 1024,
    )
    run = mlflow.get_run(run_id)
    params = run.data.params
    assert run.data.tags["mlflow.runName"] == "trees-b1"
    assert run.data.tags["batch_id"] == "b1"
    assert params["data_source"] == "labelled_windows"
    assert not [name for name in params if name.startswith("dataset_sha256")]
    # With no seed, the model keeps the random state it was given.
    assert {value for name, value in params.items() if "random_state" in name} == {"3"}
    assert run.data.metrics["caught_less_one"] == result.report.at_k.fraud_caught - 1
    assert mlflow.sklearn.load_model(f"runs:/{run_id}/model") is not None
    with pytest.raises(ValueError, match="rows passed in"):
        rerun(run_id, data_dir)

    for limit, metrics, message in (
        (1, None, "over the limit"),
        (None, lambda fit: {"train_orders": 1}, "train_orders"),
    ):
        with pytest.raises(ValueError, match=message):
            train(
                baseline_model(),
                k=5,
                seed=0,
                labelled_windows=windows,
                extra_metrics=metrics,
                max_model_size_bytes=limit,
            )
    assert len(mlflow.search_runs(experiment_names=["harness-test"])) == 1


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


def test_balanced_fit_matches_explicit_weights_computed_from_training_labels(data_dir):
    train_rows, validation_rows = _database_rows(data_dir)
    train_rows = train_rows.assign(
        is_fraud=lambda rows: (rows.index % 5 == 0).astype(int)
    )
    validation_rows = validation_rows.assign(
        is_fraud=lambda rows: (rows.index % 5 != 0).astype(int)
    )
    classes = np.array([0, 1])
    train_weights = compute_class_weight(
        "balanced", classes=classes, y=train_rows.is_fraud
    )
    pooled_weights = compute_class_weight(
        "balanced",
        classes=classes,
        y=pd.concat([train_rows.is_fraud, validation_rows.is_fraud]),
    )
    assert train_weights == pytest.approx([0.625, 2.5])

    # Fit the references directly so a regression in the harness cannot change them.
    train_keys, train_x = training._labelled_inputs(train_rows)
    _, validation_x = training._labelled_inputs(validation_rows)
    train_only = baseline_model(
        class_weight=dict(zip(classes, train_weights, strict=True))
    ).fit(train_x, train_keys.is_fraud)
    pooled = baseline_model(
        class_weight=dict(zip(classes, pooled_weights, strict=True))
    ).fit(train_x, train_keys.is_fraud)
    result = fit_and_score(
        baseline_model(class_weight="balanced"),
        k=K,
        seed=0,
        labelled_windows=(train_rows, validation_rows),
    )
    assert result.validation_orders == len(validation_rows)
    expected = train_only.predict_proba(validation_x)
    assert not np.allclose(expected, pooled.predict_proba(validation_x))
    np.testing.assert_allclose(result.model.predict_proba(validation_x), expected)


def test_compare_groups_imputes_from_training_with_validation_present(
    data_dir, monkeypatch
):
    orders_path = data_dir / "orders_base.csv"
    orders = pd.read_csv(orders_path)
    train_mask = orders.order_id.str.startswith("train")
    validation_mask = orders.order_id.str.startswith("validation")
    orders.loc[train_mask, "order_value"] = np.tile([10, 20, 30], train_mask.sum() // 3)
    orders.loc[validation_mask, "order_value"] = 1000
    orders.loc[orders.order_id.isin(["train0", "validation0"]), "order_value"] = np.nan
    orders.to_csv(orders_path, index=False)
    assert orders.loc[train_mask, "order_value"].median() == 20
    assert orders.loc[train_mask | validation_mask, "order_value"].median() == 30

    def fit_without_logging(estimator, data_dir, *, k, seed):
        result = fit_and_score(estimator, data_dir, k=k, seed=seed)
        assert result.train_orders == train_mask.sum()
        assert result.validation_orders == validation_mask.sum()
        preprocessor = result.model.named_steps["model"].named_steps[
            "columntransformer"
        ]
        _, numeric_steps, numeric_columns = preprocessor.transformers_[0]
        medians = dict(
            zip(
                numeric_columns, numeric_steps["simpleimputer"].statistics_, strict=True
            )
        )
        assert medians["order_value"] == 20
        group = result.model.named_steps["features"].group
        if group == "order_log":
            assert medians["order_value_log1p"] == pytest.approx(np.log1p(20))
        return group, result

    monkeypatch.setattr(training, "train", fit_without_logging)
    assert set(compare_groups(data_dir, k=K).group) == {"order_raw", "order_log"}


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


def test_history_features_reach_both_windows_without_leaking(data_dir):
    plain, history = _Recorder(), _Recorder()
    fit_and_score(plain, data_dir, k=5, seed=0)
    fit_and_score(history, data_dir, k=5, seed=0, history=True)
    counts = ["n_prior_orders", "n_prior_labelled_30d", "n_prior_fraud_30d"]
    fitted, scored = history.seen_
    for features in (fitted, scored):
        assert list(features.columns) == [*plain.seen_[0].columns, *counts]
        check_feature_columns(features)
    # Training orders see only earlier training orders (14 day spread, 30 day delay),
    # so their delayed counts are zero; validation orders see the training labels.
    assert fitted.n_prior_labelled_30d.eq(0).all()
    assert scored.n_prior_labelled_30d.gt(0).any()
    assert scored.n_prior_fraud_30d.le(scored.n_prior_labelled_30d).all()
    assert scored.n_prior_labelled_30d.le(scored.n_prior_orders).all()


def test_history_comparison_logs_description_and_reruns(
    data_dir, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "harness-test")
    parent_id, (v1_id, v1), (v2_id, v2) = training.compare_history(
        boosted_trees_model(), data_dir, k=5, seed=0, tags={}
    )
    parent = mlflow.get_run(parent_id)
    assert mlflow.get_run(v1_id).data.params["history_features"] == "False"
    assert mlflow.get_run(v2_id).data.params["history_features"] == "True"
    for child_id in (v1_id, v2_id):
        assert mlflow.get_run(child_id).data.tags["champion_eligible"] == "True"
    note = parent.data.tags["mlflow.note.content"]
    assert "Weakness:" in note and "n_prior_fraud_30d" in note
    assert f"{v1.report.at_k.fraud_caught} -> {v2.report.at_k.fraud_caught}" in note
    assert parent.data.metrics["fraud_caught_at_k_difference"] == (
        v2.report.at_k.fraud_caught - v1.report.at_k.fraud_caught
    )
    assert rerun(v2_id, data_dir)["reproduced"]


def _move_even_orders_to_thailand(data_dir, order_values, label_values):
    """Move orders whose ID ends in an even digit to TH, setting the given columns."""
    orders = pd.read_csv(data_dir / "orders_base.csv")
    labels = pd.read_csv(data_dir / "labels_base.csv")
    moved = orders.order_id.str[-1].isin(list("02468"))
    orders.loc[moved, ["country_code", *order_values]] = ["TH", *order_values.values()]
    labels.loc[
        labels.order_id.isin(orders.order_id[moved]), ["country_code", *label_values]
    ] = ["TH", *label_values.values()]
    orders.to_csv(data_dir / "orders_base.csv", index=False)
    labels.to_csv(data_dir / "labels_base.csv", index=False)
    return orders, labels


def test_country_input_reaches_each_order_only_on_request(data_dir):
    # TH orders are marked by an item count no MY order has.
    _move_even_orders_to_thailand(data_dir, {"num_items_ordered": 99}, {})
    plain, country = _Recorder(), _Recorder()
    fit_and_score(plain, data_dir, k=5, seed=0)
    fit_and_score(country, data_dir, k=5, seed=0, country_input=True)
    for without, features in zip(plain.seen_, country.seen_, strict=True):
        assert list(features.columns) == [*without.columns, "country_code"]
        assert features.country_code.eq("TH").equals(features.num_items_ordered.eq(99))
        assert features.country_code.eq("TH").any()


def test_recency_weighting_decays_training_rows_only_and_rejects_bad_half_life(
    data_dir,
):
    uniform, weighted = _Recorder(), _Recorder()
    fit_and_score(uniform, data_dir, k=5, seed=0)
    fit_and_score(weighted, data_dir, k=5, seed=0, recency_half_life_days=14.0)
    assert uniform.sample_weight_ is None
    weights = weighted.sample_weight_
    assert weights is not None and len(weights) == 300
    assert np.isclose(weights.max(), 1.0)
    # Training dates span 13 days (i % 14 -> 0..13), so oldest weight is 0.5**(13/14).
    assert np.isclose(weights.min(), 0.5 ** (13 / 14))
    assert np.all((weights > 0) & (weights <= 1.0))
    for bad in (0, -7, True, False, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="positive finite"):
            fit_and_score(
                _Recorder(), data_dir, k=5, seed=0, recency_half_life_days=bad
            )
    bad_orders = pd.read_csv(data_dir / "orders_base.csv")
    bad_orders.loc[bad_orders.order_id.eq("train0"), "order_date"] = "not-a-date"
    with pytest.raises(ValueError, match="valid order_date"):
        training._recency_weights(bad_orders["order_date"], 14.0)


def test_country_command_logs_ablation_and_recency_runs_rerun_and_score(
    data_dir, tmp_path, monkeypatch, capsys
):
    # Every TH order is fraud, so only a model that sees the country can tell.
    orders, labels = _move_even_orders_to_thailand(data_dir, {}, {"is_fraud": 1})
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "harness-test")
    monkeypatch.setattr(
        sys,
        "argv",
        ["train", "boosted-trees-country", "--data-dir", str(data_dir), "--k", "5"],
    )
    main()
    out = capsys.readouterr().out
    order_only_id, country_id, recency_id = re.findall(r"Logged run (\w+)\.", out)
    (parent_id,) = re.findall(r"Logged parent run (\w+)\.", out)
    order_only, country, recency = map(
        mlflow.get_run, (order_only_id, country_id, recency_id)
    )
    assert order_only.data.params["country_input"] == "False"
    assert order_only.data.params["recency_half_life_days"] == "None"
    assert country.data.params["country_input"] == "True"
    assert country.data.params["recency_half_life_days"] == "None"
    assert recency.data.params["country_input"] == "True"
    assert float(recency.data.params["recency_half_life_days"]) == 14.0
    for run in (order_only, country, recency):
        assert run.data.tags["champion_eligible"] == "True"
    before, mid, after = (
        int(run.data.metrics["fraud_caught_at_k"])
        for run in (order_only, country, recency)
    )
    assert mid > before and after > before
    parent = mlflow.get_run(parent_id)
    note = parent.data.tags["mlflow.note.content"]
    assert "Weakness:" in note and "D22-03" in note and "Cost: MY gets" in note
    assert f"fraud caught {before} -> {mid} (country) -> {after}" in note
    assert parent.data.metrics["fraud_caught_at_k_difference"] == after - before
    assert (
        parent.data.metrics["country_only_fraud_caught_at_k_difference"] == mid - before
    )
    assert parent.data.metrics["recency_fraud_caught_at_k_difference"] == after - mid
    again = rerun(recency_id, data_dir)
    assert again["fraud_caught_at_k"] == after and again["reproduced"]
    validation = orders.merge(labels, on=["country_code", "order_id"]).loc[
        lambda rows: rows.order_id.str.startswith("validation")
    ]
    model = mlflow.sklearn.load_model(f"runs:/{recency_id}/model")
    assert training.score_labelled_rows(model, validation, k=5).at_k.fraud_caught == (
        after
    )


def test_country_command_without_recency_weighting_logs_two_runs(
    data_dir, tmp_path, monkeypatch, capsys
):
    _move_even_orders_to_thailand(data_dir, {}, {"is_fraud": 1})
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "harness-test")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train",
            "boosted-trees-country",
            "--no-recency-weighting",
            "--data-dir",
            str(data_dir),
            "--k",
            "5",
        ],
    )
    main()
    out = capsys.readouterr().out
    order_only_id, country_id = re.findall(r"Logged run (\w+)\.", out)
    assert mlflow.get_run(order_only_id).data.params["recency_half_life_days"] == "None"
    assert mlflow.get_run(country_id).data.params["recency_half_life_days"] == "None"


def test_country_command_custom_half_life_rejects_conflicts_and_rerun_handles_omitted_params(
    data_dir, tmp_path, monkeypatch, capsys
):
    _move_even_orders_to_thailand(data_dir, {}, {"is_fraud": 1})
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "harness-test")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train",
            "boosted-trees-country",
            "--recency-half-life",
            "7",
            "--no-recency-weighting",
            "--data-dir",
            str(data_dir),
        ],
    )
    with pytest.raises(SystemExit):
        main()
    with pytest.raises(ValueError, match="positive finite"):
        training.compare_country(
            _Recorder(), data_dir, k=5, seed=0, tags={}, recency_half_life_days=-1
        )
    assert (
        mlflow.search_runs(experiment_names=["harness-test"], output_format="list")
        == []
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train",
            "boosted-trees-country",
            "--recency-half-life",
            "7",
            "--data-dir",
            str(data_dir),
            "--k",
            "5",
        ],
    )
    main()
    out = capsys.readouterr().out
    _, _, recency_id = re.findall(r"Logged run (\w+)\.", out)
    run = mlflow.get_run(recency_id)
    assert run.data.tags["mlflow.runName"] == "v2 country input + 7d recency"
    assert float(run.data.params["recency_half_life_days"]) == 7.0
    assert rerun(recency_id, data_dir)["reproduced"]

    # Simulate an earlier logged run missing `recency_half_life_days` and `country_input`.
    order_run_id = re.findall(r"Logged run (\w+)\.", out)[0]
    order_run = mlflow.get_run(order_run_id)
    prior_params = {
        k: v
        for k, v in order_run.data.params.items()
        if k not in ("recency_half_life_days", "country_input")
    }
    monkeypatch.setattr(
        mlflow,
        "get_run",
        lambda rid: SimpleNamespace(
            info=order_run.info,
            data=SimpleNamespace(
                metrics=order_run.data.metrics,
                params=prior_params,
                tags=order_run.data.tags,
            ),
        ),
    )
    assert rerun(order_run_id, data_dir)["reproduced"]


def test_boosted_trees_v2_command_logs_two_runs(
    data_dir, tmp_path, monkeypatch, capsys
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "harness-test")
    monkeypatch.setattr(
        sys,
        "argv",
        ["train", "boosted-trees-v2", "--data-dir", str(data_dir), "--k", "5"],
    )
    main()
    out = capsys.readouterr().out
    assert out.count("Logged run") == 2 and "Logged parent run" in out


def test_packaged_pipeline_replay_uses_an_isolated_registry(data_dir, monkeypatch):
    # Keep optional MLflow agent diagnostics out of the CLI stderr assertion.
    monkeypatch.setenv("MLFLOW_DISABLE_AGENT_HINT", "1")
    configured_tracking = data_dir / "configured-tracking.db"
    configured_registry = data_dir / "configured-registry.db"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{configured_tracking}")
    monkeypatch.setenv("MLFLOW_REGISTRY_URI", f"sqlite:///{configured_registry}")
    registry = data_dir / "private registry"
    command = [
        sys.executable,
        "-m",
        "mlops_fraud_prediction_ca1.verify_packaged_pipeline",
        "--data-dir",
        str(data_dir),
        "--registry-dir",
        str(registry),
    ]

    resolved_registry = registry.resolve()
    run = subprocess.run(
        command, capture_output=True, text=True, check=True, timeout=120
    )
    evidence = json.loads(run.stdout)
    assert evidence["tracking_uri"] == f"sqlite:///{resolved_registry / 'mlflow.db'}"
    assert evidence["registered_model_uri"] == "models:/packaged-pipeline-reload/1"
    assert evidence["registered_status"] == "READY"
    assert evidence["registered_source"].startswith(
        (resolved_registry / "artifacts").as_uri()
    )
    assert evidence["feature_group"] == "order_log"
    assert evidence["train_rows"] == WINDOWS["train"][1]
    assert evidence["validation_rows"] == WINDOWS["validation"][1]
    assert evidence["ordered_validation_inputs_identical"]
    assert evidence["probabilities_identical"]
    assert evidence["predictions_identical"]
    assert evidence["max_probability_abs_delta"] == 0
    assert not configured_tracking.exists() and not configured_registry.exists()

    retry = subprocess.run(
        command, capture_output=True, text=True, check=False, timeout=120
    )
    assert retry.returncode == 1
    assert retry.stdout == ""
    assert retry.stderr.splitlines() == [
        f"verify_packaged_pipeline.py: Registry directory already exists: {registry}"
    ]
