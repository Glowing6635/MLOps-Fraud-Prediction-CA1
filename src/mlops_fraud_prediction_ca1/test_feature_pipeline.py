import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from .feature_pipeline import FEATURE_GROUPS, FeatureGroup, build_pipeline
from .models import baseline_model


def _frame(values, items):
    n = len(values)
    return pd.DataFrame(
        {
            "order_value": np.array(values, dtype="float64"),
            "num_items_ordered": np.array(items, dtype="float64"),
            "collect_type": ["a", "b"] * (n // 2) + ["a"] * (n % 2),
            "payment_method": ["x"] * n,
        }
    )


def test_log_group_adds_columns_and_keeps_raw():
    out = (
        FeatureGroup("order_log")
        .fit(_frame([1, 2], [1, 2]))
        .transform(_frame([0, np.e - 1], [1, 3]))
    )
    assert out["order_value_log1p"].tolist() == pytest.approx([0.0, 1.0])
    assert out["order_value"].tolist() == pytest.approx([0.0, np.e - 1])
    assert out["order_value_log1p"].dtype == "float64"


def test_raw_group_adds_nothing():
    frame = _frame([1, 2], [1, 2])
    assert list(FeatureGroup("order_raw").fit(frame).transform(frame).columns) == list(
        frame.columns
    )


def test_unknown_group_rejected():
    with pytest.raises(ValueError):
        FeatureGroup("nope").fit(_frame([1, 2], [1, 2]))


@pytest.mark.parametrize("group", FEATURE_GROUPS)
def test_imputer_fitted_on_training_rows_only(group):
    train = _frame([10, 20, 30, 40], [1, 2, 3, 4])
    train.loc[0, "order_value"] = np.nan
    y = [0, 1, 0, 1]
    pipe = build_pipeline(baseline_model(), group).fit(train, y)
    imputer = pipe.named_steps["model"][0].transformers_[0][1][0]
    train_median = train["order_value"].median()
    assert train_median in imputer.statistics_


def test_pipeline_applies_steps_to_new_raw_rows_and_clones():
    train = _frame([10, 20, 30, 40], [1, 2, 3, 4])
    pipe = build_pipeline(baseline_model(), "order_log").fit(train, [0, 1, 0, 1])
    proba = pipe.predict_proba(_frame([15, np.nan], [2, np.nan]))
    assert proba.shape == (2, 2) and not np.isnan(proba).any()
    assert clone(pipe).get_params()["features__group"] == "order_log"


@pytest.mark.parametrize("dtype", ["object", "str", "string"])
@pytest.mark.parametrize("model", ["feature_pipeline", "baseline"])
def test_preprocessor_keeps_both_categorical_columns(dtype, model):
    train = _frame([10, 20, 30, 40], [1, 2, 3, 4]).astype(
        {"collect_type": dtype, "payment_method": dtype}
    )
    pipe = (
        build_pipeline(baseline_model())
        if model == "feature_pipeline"
        else baseline_model()
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", pd.errors.Pandas4Warning)
        pipe.fit(train, [0, 1, 0, 1])
    candidate = pipe.named_steps["model"] if model == "feature_pipeline" else pipe
    preprocessor = candidate[0]
    names = {name.split("__", 1)[1] for name in preprocessor.get_feature_names_out()}
    assert {"collect_type_a", "collect_type_b", "payment_method_x"} <= names


def test_logged_model_loads_and_applies_the_same_steps(tmp_path, monkeypatch):
    import mlflow
    import skops.io

    train = _frame([10, 20, 30, 40], [1, 2, 3, 4])
    pipe = build_pipeline(baseline_model(), "order_log").fit(train, [0, 1, 0, 1])
    trusted = skops.io.get_untrusted_types(data=skops.io.dumps(pipe))
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path}/m.db")
    experiment_id = mlflow.create_experiment("feature-pipeline")
    with mlflow.start_run(experiment_id=experiment_id) as run:
        mlflow.sklearn.log_model(pipe, name="model", skops_trusted_types=trusted)
    run_uri = f"runs:/{run.info.run_id}/model"
    version = mlflow.register_model(run_uri, "feature-pipeline")
    new = _frame([15, np.nan], [2, 3])
    for uri in (run_uri, f"models:/{version.name}/{version.version}"):
        loaded = mlflow.sklearn.load_model(uri)
        assert loaded.predict_proba(new) == pytest.approx(pipe.predict_proba(new))
        assert loaded.named_steps["features"].group == "order_log"


def test_build_pipeline_fits_boosted_trees_model():
    from .models import boosted_trees_model

    train = _frame([10, 20, 30, 40], [1, 2, 3, 4])
    pipe = build_pipeline(boosted_trees_model()).fit(train, [0, 1, 0, 1])
    assert pipe.predict_proba(train).shape == (4, 2)
