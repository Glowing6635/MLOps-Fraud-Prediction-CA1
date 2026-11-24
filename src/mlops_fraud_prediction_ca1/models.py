"""The candidate models the training harness (`training.py`) fits.

Each function returns an unfitted scikit-learn pipeline that takes the harness's feature
DataFrame: numbers as floats, text as strings and NaN for every missing value. Each
pipeline fills or encodes missing values itself, from the training window only.
"""

from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def baseline_model():
    """Logistic regression on the numeric columns and the one-hot-encoded text columns.

    Missing values are filled from the training window only. Date columns are dropped.
    """
    return make_pipeline(
        make_column_transformer(
            (
                make_pipeline(SimpleImputer(strategy="median"), StandardScaler()),
                make_column_selector(dtype_include="number"),
            ),
            (
                make_pipeline(
                    SimpleImputer(strategy="constant", fill_value="missing"),
                    OneHotEncoder(handle_unknown="ignore"),
                ),
                make_column_selector(dtype_include=object),
            ),
        ),
        LogisticRegression(max_iter=1000),
    )


def boosted_trees_model():
    """Histogram gradient-boosted trees on the numeric and one-hot-encoded text columns.

    The trees handle missing numbers themselves; missing text becomes its own category.
    Date columns are dropped. Early stopping is off, so the whole training window fits
    the model and none of it is held back at random.
    """
    return make_pipeline(
        make_column_transformer(
            ("passthrough", make_column_selector(dtype_include="number")),
            (
                make_pipeline(
                    SimpleImputer(strategy="constant", fill_value="missing"),
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
                make_column_selector(dtype_include=object),
            ),
        ),
        HistGradientBoostingClassifier(early_stopping=False),
    )
