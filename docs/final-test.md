# Final test-window scoring (PBI #23)

The final champion and today's rule are scored once on the test window
`[2023-04-17, 2023-05-11)`, which no training or model choice has used. The
result is the estimate the Trust & Safety analyst is shown. Decision D23-01 in
the [decision log](decision-log.md) records the design.

## Run it

1. Set `config/final_test.toml`:
   - `model_name` and `model_version`: the registered final champion. Use the
     version number, because the `champion` alias can move.
   - `mode`: `as_trained` scores the model as it was trained, on rows before
     the validation window. `refit` first fits a fresh copy, with the same
     hyperparameters, including the seed its training run set as
     `random_state`, and any logged `recency_half_life_days` exponential decay
     weights, on the training and validation rows. The test window is read only
     after the refit.
   - `k` and `loyal_min_prior_orders`: must equal the code's K (D5-01) and loyal
     count (D18-01), or the command stops.
   - `threshold_x`: the rule's threshold X in SGD (D18-04).
2. Set `final_evaluation_enabled = true` in `config/time_split.toml`. Leave
   `test_locked = true`, so ordinary reads still leave the test window out.
3. Commit both files. This commit freezes K, X, the loyal count and the mode.
4. One named person, agreed beforehand, so that two people can't start it at
   once, runs this with `.env` pointing at the team's MLflow server, and nowhere
   else (pass `--export-scores <private-path>` outside the repository if handing
   row-level scores to #34):

       uv run final-test --data-dir data

5. Commit `final_evaluation_enabled = false` again, and record the result in
   the decision log.

## What stops it

The command stops before it creates a run or reads a test row when:

- the git tree has uncommitted changes to tracked files, so the run would not
  be exactly a commit;
- the settings file is missing a setting, has an unknown one, has an unknown
  `mode`, or isn't committed;
- `k` or `loyal_min_prior_orders` differs from the code;
- `final_evaluation_enabled` is false;
- the model version can't be loaded, or the base files lack a column the model
  was trained on (its `feature_names_in_`);
- a run tagged `final_test = true` already exists in any experiment of the
  MLflow store, deleted experiments included, in any state, including failed
  and deleted runs.

A failure before the tagged run exists can be fixed and run again. The tagged run
is created just before the first test row is read, so from then on the attempt is
used up, even if it fails, and the command has no option to run again.

A second scoring needs two steps: a new decision-log entry saying why, and then
changing the old run's tag by hand, as
`MlflowClient().set_tag("<run ID>", "final_test", "superseded")`. The refusal
message names the run ID. Purging deleted runs with `mlflow gc` also removes
them from the check, so a purged final-test run no longer blocks a second
scoring.

"Once" holds within one MLflow store: a run against another tracking server
would not see the first. So the real run goes to the team's server only.

## What it logs

One MLflow run named `final test` in the `MLFLOW_EXPERIMENT_NAME` experiment:

| Kind | Recorded |
| --- | --- |
| Tags | `final_test`, `mode`, `model_name`, `model_version`, `source_run_id` (the model's training run), `git_commit`, `freeze_commit` (the last commit of the settings file), and `export_sha256` when `--export-scores` writes a private file |
| Parameters | `k`, `threshold_x`, `loyal_min_prior_orders`, `test_start`, `test_end_exclusive`, `recency_half_life_days` (when a refit source run sets it), and the SHA-256 of each base file |
| Model metrics | The primary measure `fraud_caught_at_k`, with `precision_at_k` and `recall_at_k`, the same at 2K, `fraud_catchable_at_k`, `selected_at_k`, `total_fraud_at_k` and `average_precision` |
| Rule metrics | `rule_selected_at_k`, `rule_fraud_caught_at_k`, `rule_precision_at_k` and `rule_recall_at_k`: the rule's flags, cut to each day's K orders of highest value, as in [the comparison](baseline-rule.md#comparison-with-a-model-criteria-4-to-6) |
| Other metrics | `model_fraud_caught_at_rule_flags` and `beats_rule` (the model at the rule's count), `test_orders`, and `refit_train_orders` for a refit |
| Artifacts | `final_test.txt`, the top-K report and the rule comparison; `refit_model` for a refit |

A precision or recall with no orders selected, or no fraud, is left out.

The test rows come from the base CSV files through `data.load_base_data` with
`purpose="final_evaluation"` ([fixed temporal reads](time-split.md)), the same
files the champion and the rule were fitted on. Account age and loyalty follow
[the rule's definitions](baseline-rule.md), with loyalty counting earlier
orders from the training, validation and test windows.
