# Per-batch retraining (PBI #25)

After #20 commits the batch, run `uv run retrain-batch BATCH_ID package.module:build_inputs`.
Configure the database and MLflow connection with the variables in `.env.example`
and use a database account with read access to the loaded tables. The command
uses the configured MLflow experiment. A pipeline caller can pass
`--etl-run-id ID` or call `run_evaluation_mode(..., etl_run_id=ID)` with its
invocation ID. Supply a fresh ID per invocation. The command generates one when
omitted. Run names have the form `<model>-<batch_id>-<etl_run_id>`.

The trusted, importable input factory returns `EvaluationInputs` with the
expected engineer roster, the candidate manifest, the frozen #6 validation
frame, K, and the fitted #15 champion and its run ID.
Use a fresh estimator factory per manifest entry, such as #14's order-only
pipeline factories, and pass the stored estimator parameter overrides in
`ModelSpec.params`. Factory defaults supply the other stored settings. Include
each logged #22 candidate version as its own entry when available. Both the
champion and candidates must accept the same four order features. The command
checks manifest coverage against the supplied roster before reading data.

The #20/#154 reader fetches order records once with the #6 `train` SQL
predicate. An aggregate query in the same repeatable-read transaction counts
rows withheld by date, including a separate count for rows on or after
`validation_start`. The audit population is orders with a matching label
record. Invalid labels within the train window are counted separately and
omitted from fits. Old and new counts are eligible, valid-label orders from
other batches and the invocation's batch, respectively. The command prints
these counts and records them on every successful candidate run.

Every candidate and the champion use the shared harness's order features and
top-K evaluator on the same validation frame. Each candidate run records its
scores, champion run ID and scores, score deltas, validation digest and
estimator parameters. The command reports a failed candidate, attempts the
remaining candidates, and exits 1 if any candidate failed. Evaluation runs
are tagged `champion_eligible=False` for #161's automatic candidate discovery.

The shared [D27-01 train-and-log target](decision-log.md#d27-01-wall-clock-budget-for-a-practice-batch)
is **85 seconds** across all candidates, including scoring, serialization, and
MLflow artifact uploads. The command reports the stage's elapsed wall time.
The real input factory, live #24 rehearsal and demo-machine timing remain
partial. Release mode depends on #23's final evaluation.

## Shared training harness integration

PR #162's #154 base supplies the protected RDS reader. The shared `training.py`
from #151 is on a separate branch lineage. Before running with real inputs,
apply this change in the shared harness when both branches are available.

1. Add keyword-only `labelled_windows=None` to `fit_and_score()` and
   `train()`, and allow `data_dir=None`. Require exactly one source. Keep
   the CSV path and its hashes for existing callers. For labelled windows,
   accept a `(train_rows, validation_rows)` tuple of already-cleaned, joined
   DataFrames with the harness's `_KEYS` and order features. Use
   `select_feature_inputs(rows, source="orders")` and `_model_inputs()`
   through one shared labelled-row helper. Normalize RDS Decimal numeric
   features to numbers before `_model_inputs()`, and cast labels to integers.
   Reject exploratory mode with labelled windows. Log
   `data_source=rds_labelled_windows` instead of the CSV hashes for this mode.
   Leave configured seed handling intact for CSV calls and skip overriding
   stored estimator random states when the caller explicitly supplies
   `seed=None`.
2. Extract `score_labelled_rows(estimator, rows, *, k)` in `training.py` for
   champion scoring. Use the same row helper and validation scoring function
   as `fit_and_score()`, passing the keys and positive-class probabilities
   to the existing `evaluate_top_k()`.
3. Add keyword-only `run_name=None`, `extra_params=None`,
   `extra_metrics=None` and `max_model_size_bytes=None` to `train()`.
   Pass the name to `mlflow.start_run()`, merge the supplied parameters into
   the run parameters and merge `extra_metrics(result)` into the metrics
   before logging. Keep the `(run_id, FitResult)` return.
4. Reuse the harness's `mlflow.sklearn.save_model()` size measurement with the
   same serialization options as `mlflow.sklearn.log_model()`. Reject a
   missing/empty model or a model over `max_model_size_bytes` before upload.
   Keep the shared sklearn model name `model` so #161's
   `register_top_two()` can consume `runs:/<run_id>/model`. Verify the URI
   resolves from a retraining run in local MLflow before running the rehearsal.

The CLI calls this interface directly once the harness is available. Tests
on the #154 base exercise the CLI with a stand-in for the shared harness.
