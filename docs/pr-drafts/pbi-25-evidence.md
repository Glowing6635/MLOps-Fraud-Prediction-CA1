# Evidence audit: PBI #25 per-batch auto retraining

Status: **partial. Evaluation mode is implemented; release mode, the measured
roster and the #24 rehearsal are not.** Use `Refs #25`, not `Resolves #25`.
Audited against `origin/user/spiralsky` at `fec86b0` (`1c78e83` plus the #17 test and the retrain size-limit alignment), issue body read live.

Code: `src/mlops_fraud_prediction_ca1/retrain.py`; docs: `docs/retraining.md`;
tests: `src/mlops_fraud_prediction_ca1/test_retrain.py`; decision: D25-01.

## Criterion evidence

| # | Criterion | Status | Evidence and gap |
|---|---|---|---|
| 1 | Training data read from RDS once per run, shared by all models | MET | `retrain()` calls `read_rows()` once and passes the same `train_rows` to every `train_candidate` call; `read_training_snapshot` uses one REPEATABLE READ transaction (`rds_reads.py:71`). `test_cli_trains_and_logs_real_models_after_candidate_failure`. |
| 2 | Before #23, train only on rows dated before the #6 validation window; excluded rows counted and printed | MET (evaluation mode) | `sql_date_predicate(purpose="train")` in the reader; `excluded_date_rows` and `excluded_late_rows` come from an aggregate query and are logged in the `counts` line and on each run (`retrain.py`, `log(...)`). `test_training_snapshot_keeps_holdout_rows_in_db` is DB-backed and **skipped here** (needs `TEST_MYSQL_URL`), so the SQL path is not exercised in this audit. |
| 3 | Every model retrained with stored hyperparameters, no search; one per engineer plus each #22 second version; no artifact over 50 MB; total fit time fits the #27 budget | PARTIAL | Met: `validate_manifest` requires owners to equal the roster (`test_manifest_requires_every_expected_engineer`); `ModelSpec.params` applied with no search; `MAX_ARTIFACT_BYTES` (`retrain.py:18`, now `champion.SIZE_NOTE_BYTES` = 50,000,000 bytes, matching D15-03 and `champion.py:35`; it was 50 MiB = 52,428,800) enforced before upload (`test_shared_harness_stops_oversized_model_before_upload`; `test_artifact_limit_matches_the_champion_rule_and_rejects_52_4_mb` rejects a 52.4 MB artifact). **Gaps:** (a) no real input factory or manifest is committed, so the actual roster and #22 versions are caller-supplied and unmeasured (D15-02 says the roster is still to be measured); (b) the 85 s budget (D27-01) is only logged, never checked or failed, and has not been measured on a real roster. Closable offline: a committed manifest and a budget assertion. Needs #22 and the #27 timing for the rest. |
| 4 | Evaluation mode compares each model and the champion on the frozen #6 window; release mode after #23 trains on all eligible data; #23 artifacts unchanged; no training-row score reported as out-of-sample | PARTIAL | Evaluation mode MET: champion and candidates scored on the same validation frame, deltas and `beats_champion` logged, validation checked as frozen and the digest recorded. **Release mode is not implemented** (`docs/retraining.md`: "Release mode depends on #23"). Blocked on #23. |
| 5 | Rehearsal shows eligible new-batch rows reaching every training task, with old/new counts and exclusions; release mode excludes no valid batch; evaluation-window changes in the decision log | PARTIAL | Old/new/excluded counts are computed and logged (`train_old_rows`, `train_new_rows`; asserted in the CLI test with fake rows). No live rehearsal: the issue says Done needs #24. No window changed (D25-01 states this). Release-mode no-silent-cutoff is untestable until release mode exists. |
| 6 | Unique `etl_run_id` per invocation; run names contain it and the batch ID | MET, with a caveat | Default `uuid.uuid4().hex`; names are `<model>-<batch_id>-<etl_run_id>`; also logged as a tag (`test_candidate_size_failures_and_unique_ids`). Caveat: a caller-supplied `--etl-run-id` is not checked for uniqueness; the doc says to supply a fresh one. |
| 7 | One failed model is reported and the rest continue; run exits non-zero | MET | Per-candidate `try/except` records `report.failures`; `exit_code` is 1 if any failed; `main` returns it. CLI test covers a failure followed by successes. |
| 8 | Champion not auto-promoted; each run logs comparison against the champion | MET | Runs are tagged `champion_eligible=False`; nothing registers or promotes; the champion scores, run ID and deltas are logged on every run. |

## Checks (2026-11-26, worktree on `fec86b0` plus the size-limit alignment)

- `uv run pytest`: 310 passed, 14 skipped, 18 warnings, 57 subtests passed.
- `uv run pytest src/mlops_fraud_prediction_ca1/test_retrain.py`: 6 passed, 1 skipped (the DB test above).
- `uv run ruff check`: all checks passed.
- `uv run ruff format --check`: 84 files already formatted.

## Are all criteria MET?

No. Criteria 1, 2, 6, 7 and 8 are MET (2 offline/evaluation mode; 6 with a caveat). Criteria 3, 4 and 5 are PARTIAL, so #25 is not finished under the owner's rule and stays `Refs #25`. The owner ruling on local MLflow runs (D19-01) does not change any #25 criterion.

## What stays open

1. Release mode (criteria 4, 5): waits on #23's final evaluation.
2. Roster, committed input factory and measured fit time against 85 s (criterion 3): needs #22 versions and the #27 timing. A budget check in code is doable now.
3. Live rehearsal with a new batch (criterion 5): needs #24.
4. The skipped DB test should be run against a disposable MySQL before claiming criterion 2 end to end.

Recommend `Refs #25` until items 1 and 3 are done. No independent review has been run on this audit.
