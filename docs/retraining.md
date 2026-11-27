# Per-batch retraining (PBI #25)

After #20 commits the batch, run from the repository root:

```sh
uv run --locked retrain-batch BATCH_ID mlops_fraud_prediction_ca1.retrain_inputs:build_inputs
```

Configure the database and MLflow connection with the variables in `.env.example`
and use a database account with read access to the loaded tables. The command
uses the configured MLflow experiment. The checked-in factory also reads these
settings from the process environment or the same local `.env`:

| Setting | Value |
| --- | --- |
| `RETRAIN_VALIDATION_CSV` | Trusted, cleaned, labelled, validation-only CSV fixed for the comparison. Required columns are `order_date`, `country_code`, `order_id`, `customer_id`, `is_fraud`, `collect_type`, `payment_method`, `order_value`, and `num_items_ordered`. |
| `RETRAIN_VALIDATION_SHA256` | SHA-256 of the approved CSV bytes, recorded when the export is frozen. The same file and row order must be supplied for repeats. |
| `RETRAIN_CHAMPION_MODEL_NAME`, `RETRAIN_CHAMPION_MODEL_VERSION` | Trusted #15 registry name and exact numeric version, resolved during approval. |
| `RETRAIN_CHAMPION_RUN_ID` | Exact MLflow source run ID for that version. The factory checks the version belongs to the specified run and loads the fitted artifact. |

Use an approved export matching the champion comparison cohort. The #20 reader
with `purpose="validation"` can supply cleaned rows for an export where the
approved cohort is in the database. Keep only valid labels. Record the source,
cohort and digest before rehearsal. The factory has no path that exports data
or scores the holdout. It verifies the file digest and rejects duplicate order
keys or rows outside the #6 validation window. The harness logs the SHA-256 of
the parsed validation frame serialized as CSV, which can differ from the byte
digest pinned for the original file.

A pipeline caller can pass
`--etl-run-id ID` or call `run_evaluation_mode(..., etl_run_id=ID)` with its
invocation ID. Supply a fresh ID per invocation. The command generates one when
omitted. Run names have the form `<model>-<batch_id>-<etl_run_id>`.

The factory returns `EvaluationInputs` with the result of `load_roster()`,
the frozen #6 validation frame, D5-01's K = 250 and the fitted #15 champion and
run ID. Both the champion and candidates must accept the same four order
features. The command checks manifest coverage against the supplied roster
before reading data.

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
Live #24 rehearsal and demo-machine timing are pending. Release mode depends
on #23's final evaluation.

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

## Practice batch and offline rehearsal

The #25 fixture is independent of the #24 generator. From the repository root,
generate it under a disposable directory:

```sh
uv run --locked python scripts/p25_practice.py --out output/p25-practice
```

Pass `output/p25-practice/base` to #24's `rehearsal-reset --data-dir` on the
authorized rehearsal target. It contains the three #20-format base CSVs,
with 12 rows in each. Load all three `p25-practice/*_p25-practice.csv` files
with #20 `load-batch` in customers, orders, labels order, then invoke
`retrain-batch p25-practice ...` against the same rehearsal database. All
11 incoming order IDs and labels are new to the fixture. Eight orders are
dated 2023-03-26 and three are dated 2025-01-06. The 2025 rows are a negative
control for evaluation mode. The separate synthetic `validation.csv` has six
orders dated 2023-04-01 and is appropriate only for a synthetic comparison.
For an authorized rehearsal against the current team champion, supply its
approved frozen validation export and pinned fitted model instead.

For the fully offline exercise, use a fresh output directory. The setup
loads the same files using the real #20 loader on a disposable SQLite fixture,
fits a synthetic champion on the old rows and pins its local registry version
and validation export in `offline.env`. Run the existing CLI with the
explicitly selected fixture factory:

```sh
uv run --locked python scripts/p25_practice.py --out output/p25-offline --prepare-offline
PYTHONPATH=scripts uv run --locked --env-file output/p25-offline/offline.env \
  retrain-batch p25-practice p25_practice:build_inputs
```

Look for one `SQLite fixture training snapshots=1` and the printed counts.
In the local `p25-offline-retrain` MLflow experiment both roster runs should
have `train_orders=20`, `train_old_rows=12`, `train_new_rows=8`,
`excluded_rows=3`, `excluded_late_rows=3`, and
`excluded_invalid_label_rows=0`. Compare their run names, shared
`validation_sha256`, champion run ID, `model_size_bytes` and `fit_seconds`.
The adapter runs the application's loader and training SELECTs on SQLite
with adapted schema and bound parameters. MySQL strict mode, collation,
isolation semantics and RDS or S3 timings require the actual rehearsal.

Adding `--mode release` must be refused with the #6 holdout locked.
Keep it locked for this exercise. The offline proof uses genuinely new order
keys. A first valid label on an older order can contribute
training data but is counted as old. With only such contributions, the release
guard would reject the batch. The offline evidence records order and label
provenance separately. Production attribution and its effect on the release
guard still need agreement with the #20 owner.

## Acceptance map

| Criterion | Evidence and remaining check |
| --- | --- |
| Read once, shared | One offline snapshot in the practice CLI, with both candidate runs reporting 20 fit rows. Confirm one read against RDS. |
| Train before validation; print exclusions | Both offline candidates include 8 new eligible rows with 3 later rows excluded. |
| Every model retrained, stored params, <=50 MB, 85 s budget | Offline with both provisional roster entries and the shared size cap and timer. Team roster confirmation, #22 versions and demo-size measurements with S3 remain. |
| Evaluation vs champion; release mode, results unchanged | Offline comparison against a pinned synthetic champion. The locked release invocation refused before reading. Real post-#23 comparison, separately logged artifacts and authorized holdout switch remain. |
| Rehearsal with old/new counts (AC5) | Offline fixture evidence. AC5 stays open for the #24 MySQL/RDS rehearsal and authorized post-#23 later-batch proof. Attribution of first labels on prior orders is open. |
| Unique `etl_run_id`, run names | A fresh invocation ID is generated by default, with both runs named `<model>-<batch>-<etl_run_id>` and tagged. |
| Failed model reported, others continue, exit non-zero | Covered by `test_cli_trains_and_logs_real_models_after_candidate_failure`. |
| No auto-promotion, champion comparison logged | Offline candidates tag `champion_eligible=False` and log comparison metrics. Verify against the approved champion in the authorized rehearsal. |
