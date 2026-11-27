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

## Roster and release mode

`config/retrain_roster.toml` lists the expected engineers and the stored candidate
list; `retrain_roster.load_roster()` returns them and checks every engineer has a
model. Factories resolve only to `*_model` functions in `models.py`. The owners
are provisional until the team confirms the roster; add each logged #22 version as
its own entry.

`uv run retrain-batch BATCH_ID module:factory --mode release` reads every labelled
order dated on or after the training start in `config/time_split.toml` (no upper
date cut-off, later batches included) and is refused while the #6
holdout setting (`test_locked`) is on. It never touches #23 results or the champion
artifact; runs are tagged `retrain_mode=release`, `champion_eligible=False`.
Because release models also fit the frozen validation rows, their harness scores
are in-sample: they carry `scores_out_of_sample=False`. Each run still logs the
comparison against the current champion on the frozen validation rows, but it is
in-sample, so it never supports promotion. The run stops if the requested batch
has no eligible rows. See [D25-02](decision-log.md#d25-02-release-mode-retraining-and-its-in-sample-scores).

## Shared training harness

The command passes the two labelled windows directly to the harness, without
reading base CSV files. The harness records `data_source=labelled_windows`
and records `batch_id`, `etl_run_id`, `champion_run_id`, `validation_sha256`,
`model` and `owner` as run tags. The run name uses `mlflow.runName`, candidate
overrides use `model_params`, and counts and comparisons are metrics. The
harness converts decimal order values to numbers and preserves the estimator's
random state when the seed is omitted. It reports models over 50,000,000 bytes
as candidate failures before artifact upload.

## Acceptance map (offline, 2026-11-26)

| Criterion | State |
| --- | --- |
| Read once, shared | Done: one snapshot per run (`test_cli_trains_and_logs_real_models_after_candidate_failure`). |
| Train only before the validation window; print exclusions | Done in evaluation mode (reader counts, logged). |
| Every model retrained, stored params, <=50 MB, 85 s budget | Done in code (roster, size cap, 85 s timer). Real fit time on RDS rows and the real #22 entries: gap. |
| Evaluation vs champion; release mode, results unchanged | Done offline for both; release scores tagged in-sample. Real #23 final evaluation and holdout switch: gap (depends on #23). |
| Rehearsal with old/new counts | Gap: needs #24 and a live RDS run. |
| Unique `etl_run_id`, run names | Done (`<model>-<batch>-<etl_run_id>`, tags). |
| Failed model reported, others continue, exit non-zero | Done (tested). |
| No auto-promotion, champion comparison logged | Done in both modes: `champion_eligible=False` and champion comparison metrics logged, in-sample in release mode and never promoting. Real #23 champion artifact and holdout switch: gap. |
| Real input factory (frozen validation frame, fitted champion) | Gap: not written; needs #15 and #23 artifacts. |
