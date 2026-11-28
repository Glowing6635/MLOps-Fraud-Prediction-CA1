# Per-batch retraining (PBI #25)

After #20 commits the batch, run `uv run retrain-batch BATCH_ID package.module:build_inputs`.
Configure the database and MLflow connection with the variables in `.env.example`
and use a database account with read access to the loaded tables. The command
uses the configured MLflow experiment. A pipeline caller can pass
`--etl-run-id ID` or call `run_evaluation_mode(..., etl_run_id=ID)` with its
invocation ID. The caller is responsible for choosing a fresh ID per invocation.
Retraining uses a supplied ID as-is and generates a UUID when the ID is omitted.
Run names have the form `<model>-<batch_id>-<etl_run_id>`.

The trusted, importable input factory returns `EvaluationInputs` with the
expected engineer roster, the candidate manifest, the frozen #6 validation
frame, K, and the fitted #15 champion and its run ID.
Use a fresh estimator factory per manifest entry, such as #14's order-only
pipeline factories or #22's boosted-trees candidate, and pass any stored
estimator parameter overrides in `ModelSpec.params`, `country_input=True`, or
`recency_half_life_days` (D22-03). `team_evaluation_inputs` provides the team's
two-candidate manifest (`boosted-trees-v2` with country input and 14-day recency
decay, and `baseline-logistic`). Candidates accept the four order features plus
`country_code` when `country_input=True`; the champion may also take the order's
country, which `score_labelled_rows` adds for a model fitted with it. The
command checks manifest coverage against the supplied roster before reading data.

The #20/#154 reader fetches order records once with the #6 `train` SQL
predicate. An aggregate query in the same repeatable-read transaction counts
rows withheld by date, including a separate count for rows on or after
`validation_start`. The audit population is orders with a matching label
record. Invalid labels within the train window are counted separately and
omitted from fits. Old and new counts are eligible, valid-label orders from
other batches and the invocation's batch, respectively. The command prints
these counts and records them on every successful candidate run.

Every candidate and the champion use the shared harness's features (the order
features, plus the country for a model fitted with it) and top-K evaluator on
the same validation frame. Each candidate run records its
scores, champion run ID and scores, score deltas, validation digest and
estimator parameters. The command reports a failed candidate, attempts the
remaining candidates, and exits 1 if any candidate failed. Evaluation runs
are tagged `champion_eligible=False` for #161's automatic candidate discovery.

The shared [D27-01 train-and-log target](decision-log.md#d27-01-wall-clock-budget-for-a-practice-batch)
is **85 seconds** across all candidates, including scoring, serialization, and
MLflow artifact uploads. The command reports the stage's elapsed wall time.
The real input factory, live #24 rehearsal and demo-machine timing remain
partial. Release mode depends on #23's final evaluation.

## Shared training harness

The command passes the two labelled windows directly to the harness, without
reading base CSV files. The harness records `data_source=labelled_windows`
and records the batch details (`batch_id`, `etl_run_id`, `champion_run_id`, `validation_sha256`, `model`, `owner`) as run tags. It converts decimal order values to numbers. An omitted seed preserves the
estimator's random-state settings. Models that exceed the batch model-size limit
are reported as candidate failures before artifact upload.
