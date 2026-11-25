# Feature pipeline (PBI #16)

Pass an unfitted candidate pipeline to `feature_pipeline.build_pipeline(candidate, group)`.
The returned scikit-learn pipeline applies `FeatureGroup` first and passes the resulting
DataFrame to the candidate's own preprocessing and classifier. Existing columns retain
their dtypes. The training harness logs the whole pipeline as one MLflow model, so
loading a run or registered version applies the same steps to order inputs.

```python
from mlops_fraud_prediction_ca1.feature_pipeline import build_pipeline
from mlops_fraud_prediction_ca1.training import baseline_model

pipeline = build_pipeline(baseline_model(), group="order_log")
```

PR #152 at `6c1326ca` supplies `models.boosted_trees_model()` with the same DataFrame
input contract. It handles missing numeric values in the classifier and encodes text
in its own preprocessor. Use the harness's input convention (floats for numbers,
object columns for text, and NaN for missing values) when scoring outside the harness.

- **Fitted on training rows only:** `Pipeline.fit` runs inside the harness on the training
  window. `FeatureGroup` is stateless (signed `log1p` from `skew_outliers`, reasons per
  feature in `skew_outliers.TREATMENTS`), so no later row can influence it.
- **Imported by the pipeline:** candidates import `build_pipeline`; nothing is re-coded.
- **Comparing groups:** `uv run train compare-feature-groups --k 250` trains the baseline
  logistic regression on each group, one MLflow run each, scored on the validation window. Use the
  same CSV files and MLflow settings as the [training harness](training.md).

Results and the decision are in D16-02 in [the decision log](decision-log.md).

## Load a logged or registered model

The serialized pipeline imports `FeatureGroup` from this project. The model's
generated `requirements.txt` lists third-party dependencies, so the consumer
also needs the project installed. In a checkout of this repository, use the
source run's `git_commit` tag as `RUN_COMMIT` and sync its locked environment:

```bash
git checkout --detach "$RUN_COMMIT"
uv sync --locked
```

Set `MLFLOW_TRACKING_URI` to the server that holds the run and configure access
to its artifacts. For a registered version, use the source run linked from that
version to obtain the tag. From the synced checkout, load with:

```bash
uv run python -c 'import sys; import mlflow.sklearn; model = mlflow.sklearn.load_model(sys.argv[1])' "$MODEL_URI"
```

Set `MODEL_URI` to `runs:/<run-id>/model` or `models:/<registered-name>/<version>`.
For a serving image, install the project from the same run commit into the
serving environment. A run tagged `git_dirty=true` may also depend on local
changes beyond that commit.
