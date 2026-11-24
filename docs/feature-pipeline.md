# Feature pipeline (PBI #16)

`feature_pipeline.build_pipeline(estimator, group)` returns one scikit-learn `Pipeline`:
`features` (`FeatureGroup`) → `preprocess` (median impute + scale numbers, constant impute
+ one-hot text) → `model`. Pass it to `training.train` and the harness logs the whole
pipeline as the single MLflow model `model`, so loading a run or registered version
applies the same steps to raw order rows.

- **Fitted on training rows only:** `Pipeline.fit` runs inside the harness on the training
  window. `FeatureGroup` is stateless (signed `log1p` from `skew_outliers`, reasons per
  feature in `skew_outliers.TREATMENTS`), so no later row can influence it.
- **Imported by the pipeline:** candidates import `build_pipeline`; nothing is re-coded.
- **Comparing groups:** `uv run compare-feature-groups --k 250` trains the baseline
  logistic regression on each group, one MLflow run each, scored on the validation window.

Results and the decision are in D16-02 in [the decision log](decision-log.md).
