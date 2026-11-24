"""Packaged feature pipeline (PBI #16).

Every feature step is a scikit-learn step inside one ``Pipeline`` together with the
model, so the object the training harness logs to MLflow (``training.train``) applies the
same steps to new rows when a registered version is loaded. Steps that learn anything
(imputer, scaler, encoder) are fitted by ``Pipeline.fit`` on the training rows only.
``FeatureGroup`` learns nothing, so it cannot leak validation or test rows.
"""

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from .skew_outliers import SUFFIX, TRANSFORMED_FEATURES, transform_skewed_features

# Feature groups compared on the primary measure (decision D16-02).
FEATURE_GROUPS: dict[str, str] = {
    "order_raw": "the allowed order columns as cleaned (decision D4-01)",
    "order_log": "order_raw plus the signed log1p of order_value and num_items_ordered",
}


class FeatureGroup(TransformerMixin, BaseEstimator):
    """Select a feature group from a frame of the allowed order columns.

    No parameters are learned from the data. Original columns retain their dtypes,
    and the generated columns are ``float64`` with NaN for missing values.
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
    """Add the feature group ahead of a candidate pipeline that accepts a DataFrame.

    The candidate supplies its own preprocessing and classifier with ``predict_proba``.
    Pass the combined pipeline to ``training.train`` to fit, score and log it.
    """
    return Pipeline([("features", FeatureGroup(group)), ("model", estimator)])
