"""Packaged feature pipeline (PBI #16).

Every feature step is a scikit-learn step inside one ``Pipeline`` together with the
model, so the object the training harness logs to MLflow (``training.train``) applies the
same steps to new rows when a registered version is loaded. Steps that learn anything
(imputer, scaler, encoder) are fitted by ``Pipeline.fit`` on the training rows only.
``FeatureGroup`` learns nothing, so it cannot leak validation or test rows.
"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .skew_outliers import SUFFIX, TRANSFORMED_FEATURES, transform_skewed_features

# Feature groups compared on the primary measure (decision D16-02).
FEATURE_GROUPS: dict[str, str] = {
    "order_raw": "the allowed order columns as cleaned (decision D4-01)",
    "order_log": "order_raw plus the signed log1p of order_value and num_items_ordered",
}


class FeatureGroup(TransformerMixin, BaseEstimator):
    """Select a feature group from a frame of the allowed order columns.

    Stateless: ``fit`` stores nothing, and each row's output depends only on that row.
    The output has plain ``float64`` columns with NaN for missing values, as the
    harness's model inputs do.
    """

    def __init__(self, group: str = "order_log"):
        self.group = group

    def fit(self, X, y=None):
        if self.group not in FEATURE_GROUPS:
            raise ValueError(f"unknown feature group {self.group!r}")
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        if self.group not in FEATURE_GROUPS:
            raise ValueError(f"unknown feature group {self.group!r}")
        if self.group == "order_raw":
            return X.copy()
        result = transform_skewed_features(X)
        for feature in TRANSFORMED_FEATURES:
            name = feature + SUFFIX
            result[name] = result[name].astype("float64")
        return result

    def set_output(self, *, transform=None):  # always a DataFrame
        return self


def build_pipeline(estimator, group: str = "order_log") -> Pipeline:
    """One pipeline: feature group, train-fitted preprocessing, then ``estimator``.

    ``estimator`` is any scikit-learn classifier with ``predict_proba``. The result is
    what ``training.train`` fits, scores and logs as the single MLflow model.
    """
    preprocess = make_column_transformer(
        (
            make_pipeline(SimpleImputer(strategy="median"), StandardScaler()),
            make_column_selector(dtype_include="number"),
        ),
        (
            make_pipeline(
                SimpleImputer(strategy="constant", fill_value="missing"),
                OneHotEncoder(handle_unknown="ignore"),
            ),
            make_column_selector(dtype_include=["object", "string"]),
        ),
    )
    return Pipeline(
        [
            ("features", FeatureGroup(group)),
            ("preprocess", preprocess),
            ("model", estimator),
        ]
    )


def compare_groups(data_dir: str | Path, *, k: int, seed: int = 0) -> pd.DataFrame:
    """Train the baseline classifier on each feature group; log one MLflow run each.

    Each run is scored on the validation window by the harness, so the groups are
    compared on the primary measure (fraud caught at K) out of sample.
    """
    from sklearn.linear_model import LogisticRegression

    from .training import train

    rows = []
    for group in FEATURE_GROUPS:
        run_id, result = train(
            build_pipeline(LogisticRegression(max_iter=1000), group),
            data_dir,
            k=k,
            seed=seed,
        )
        rows.append(
            {
                "group": group,
                "run_id": run_id,
                "fraud_caught_at_k": result.report.at_k.fraud_caught,
                "average_precision": result.report.average_precision,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=compare_groups.__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    table = compare_groups(args.data_dir, k=args.k, seed=args.seed)
    sys.stdout.write(table.to_string(index=False) + "\n")


if __name__ == "__main__":
    # Run through the package name, so a logged model's classes are not pickled as
    # `__main__.FeatureGroup`, which no other process could load.
    from mlops_fraud_prediction_ca1.feature_pipeline import main as _main

    _main()
