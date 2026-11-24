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
frame, the #5 top-K evaluator and K, and the fitted #15 champion and its run ID.
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

Every candidate and the champion are scored on the same supplied validation
frame. The candidate run records its scores, champion run ID and scores, score
deltas, validation digest, fitted parameters and artifact byte count. The
adapter serializes the fitted estimator once and uploads those same bytes as
`model.pkl` only if the size is within 50 MiB (52,428,800 bytes). Only load
pickle artifacts from trusted runs. The command reports a failed candidate,
attempts the remaining candidates, and exits 1 if any candidate failed. The
comparison is informational.

The shared [D27-01 train-and-log target](decision-log.md#d27-01-wall-clock-budget-for-a-practice-batch)
is **85 seconds** across all candidates, including scoring, serialization, and
MLflow artifact uploads. The command reports the stage's elapsed wall time.

Representative fixtures exercise old and new rows, validation and test
exclusions, per-engineer coverage, stored estimator settings, and candidate
artifacts and comparisons in a local MLflow store. The owner roster, final
candidate versions and count contract still need confirmation. Production
champion/validation inputs, the live #24 rehearsal, and demo-machine timing
remain open. Release mode depends on #23's final evaluation.
