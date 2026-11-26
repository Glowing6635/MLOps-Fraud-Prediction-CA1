# Evidence map: PBIs #15, #16, #17 (audited at fec86b0: origin/user/spiralsky 1c78e83 plus the #17 test)

Audit of the live acceptance criteria (`gh issue view`, 2026-11-26) against
`origin/user/spiralsky`. Paths are relative to the repository root; `S` =
`src/mlops_fraud_prediction_ca1`.

## Checks run (feature branch fec86b0, on top of 1c78e83)

| Check | Result |
| --- | --- |
| `uv run pytest -q` | 310 passed, 14 skipped, 18 warnings, 57 subtests passed (189 s) |
| `uv run ruff check` | All checks passed |
| `uv run ruff format --check` | 84 files already formatted |

The 14 skips come from `S/conftest.py:26` (needs `TEST_MYSQL_URL`, a disposable
MySQL 8.4 server for the RDS tests); they do not touch #15, #16 or #17. The
warning is an MLflow/SQLAlchemy deprecation notice.

## PBI #15 — First review, champion pick (4 criteria: 4 MET offline after G15-a was closed; live registry still open)

| # | Criterion | Status | Evidence |
| --- | --- | --- | --- |
| 1 | Top 2 by #5 measure on validation data, `champion` alias on best, exploratory runs excluded, registry entry names the exact artifact | MET offline; live registry state needs an operator (see G15-b) | Ranking and exploratory/dirty exclusion: `S/champion.py:65-84`; top-two registration: `S/champion.py:376-433`; artifact tags (`artifact`, `resolved_artifact`, `source_run_id`): `docs/champion.md` "Pick at the model review"; tests `S/test_champion.py:106` (exploratory excluded), `:151` (aliases and artifact tags), `:166` (50 MB rule), `:428` (CLI exact source); decisions D15-01, D15-02, D15-03 in `docs/decision-log.md:233-320` |
| 2 | Recorded comparison table gives the reason | MET | `S/champion.py:191` (`comparison_table`), `docs/champion.md` "Team registry record" (three-run table), `docs/model-review.md:8` |
| 3 | Choice made at the first model review with the Trust & Safety analyst present | MET (with a stated limit) | `docs/model-review.md:41-53`: 2026-11-24 15:45 UTC sign-off, the Product Owner acting as the analyst (course persona, D18-04). The limit is that no separate real analyst attended |
| 4 | Each change records the previous version; alias moves to the second model and back by a step documented **in the runbook** | MET (G15-a closed: `docs/runbook.md` "Champion alias move and rollback") | Code and tests: `S/champion.py:437-496` writes `previous_champion_version`; `S/test_champion.py:207` (move and back), `:352`, `:681`, `:732`. The step is in `docs/runbook.md` "Champion alias move and rollback" and in `docs/champion.md` "Move the alias and roll back" |

Gaps:

- **G15-a (closed):** `docs/runbook.md` now has the "Champion alias move and rollback" section.
- **G15-b (needs the registry operator, not offline):** D15-03 says the recorded
  `fraud-candidate` v2 (random forest, 187 MB) exceeds the 50 MB limit, so the
  challenger must be re-registered from the next within-limit run. D15-03 also
  records itself as "Needs the Product Owner's one-line confirmation". D15-01
  lists the historical-artifact and alias/retry/rollback checks as pending on a
  socket-capable host. The offline tests use local SQLite MLflow only.

## PBI #16 — Packaged feature pipeline (4 criteria: 2 MET, 2 GAP/PARTIAL)

| # | Criterion | Status | Evidence |
| --- | --- | --- | --- |
| 1 | Feature groups listed in the decision log; each compared on the primary measure on validation data and logged to MLflow | **GAP (partial)** | Groups: `S/feature_pipeline.py:16-19` and D16-02 (`docs/decision-log.md:407`). Comparison: `uv run train compare-feature-groups` (`S/training.py:497-520`, `:617`), logs one run per group; test `S/test_training.py:365`. Result table in D16-02: `order_raw` 2498 caught at K / AP 0.3259, `order_log` 2450 / 0.3240. The comparison ran against a local SQLite store, not the team MLflow server, and only for the logistic baseline; D16-02 says the server comparison is pending. Gap: the runs are not on the team server |
| 2 | Skew and outlier handling with a stated reason per feature | MET | `S/skew_outliers.py` (`TREATMENTS`, `transform_skewed_features`), D16-01 (`docs/decision-log.md:390`), tests `S/test_skew_outliers.py:32-64` |
| 3 | Steps fitted on training rows only and imported by the pipeline | MET | `S/feature_pipeline.py:22-56` (`FeatureGroup` is stateless; `build_pipeline` composes it); `S/training.py:231` (`estimator.fit(train_x, ...)`) fits on the training window only, and `:233` scores validation; tests `S/test_feature_pipeline.py:48` (imputer fitted on training rows), `:58`, `S/test_training.py:111` (no test-window row reaches training or validation). Documented in `docs/feature-pipeline.md` |
| 4 | Feature steps and the model logged as one MLflow model; loading a registered version applies the same steps | **PARTIAL** (mechanism MET; registered champion not covered, see G16-b) | `S/training.py:354` (`mlflow.sklearn.log_model` of the whole pipeline); test `S/test_feature_pipeline.py:86` logs, registers, then loads by `runs:/` and `models:/` URIs and checks identical `predict_proba` and the same `features` step. Limit: only models built with `build_pipeline` carry the steps; `train baseline` and `train boosted-trees` (the registered champion and baseline) do not use it, so loading their registered versions applies no feature steps |

Gaps:

- **G16-a (not closable offline):** log the group comparison to the team MLflow
  server (needs the credentials and a reachable host). This is the same item
  D16-02 lists as pending.
- **G16-b (open, scope decision, needs the owner):** D16-02 notes that `train baseline`
  and `train boosted-trees` do not yet wrap their models in `build_pipeline`, so
  models they register have no skew handling. Criterion 4 is met for the
  packaging mechanism only; the registered champion (boosted trees, v1) does not
  use it. Wrapping them would change the frozen champion settings, so it is not
  done here; the owner decides whether to accept or schedule a follow-up.

## PBI #17 — Train-only imbalance fix (3 criteria: 2 MET, 1 GAP)

| # | Criterion | Status | Evidence |
| --- | --- | --- | --- |
| 1 | Resampling or class weights applied to training rows only | MET | Weights come from the estimator's `class_weight` (`S/models.py:20,43,84,102`); the harness fits only on the training window (`S/training.py:231`); D17-01 (`docs/decision-log.md:431`); `docs/training.md:67-81`. The estimator derives weights during `fit` from training labels only. No resampling is used |
| 2 | Validation and test keep the natural fraud rate | MET | `S/training.py:233` (via `_score_window`, `:182-189`) scores the unmodified validation window; D17-01 table shows 13,066 frauds in 101,190 validation orders (12.91%) for all four runs; test split stays locked. Test `S/test_training.py::test_balanced_weighting_leaves_the_validation_fraud_rate_natural` fits with `None` and `balanced` and asserts the validation fraud rate equals the labelled window's natural rate in both; CLI test `S/test_training.py:365` covers the logging |
| 3 | Each method compared on the #5 primary measure on validation data and logged to MLflow | **GAP (partial)** | D17-01 table: LR `None` 2,498 / `balanced` 2,479; boosted trees `None` 2,893 / `balanced` 2,856. `--weighting` logs `class_weight` (test `S/test_training.py:365-390`). Runs were local MLflow, not the team server. The "each method" wording also covers resampling, which was dropped as an alternative without a measured comparison; only class weighting was run |

Gaps:

- **G17-a (not closable offline):** local-SQLite runs only; no team-server log.
- **G17-b (closable offline):** either run a resampling arm on training rows
  only, or record in D17-01 that resampling was deliberately not compared and
  have the PBI owner accept the wording "Resampling **or** class weights".
  Criterion 1 is written as "or", so one method may suffice; criterion 3's
  "each method" then means the methods actually used.
- **G17-c (closed):** `test_balanced_weighting_leaves_the_validation_fraud_rate_natural`
  asserts the validation fraud rate is identical with and without `balanced`.

## Summary

| PBI | MET | GAP | Not closable offline |
| --- | --- | --- | --- |
| #15 | 4 | 0 (G15-a closed) | G15-b (live registry, D15-03 confirmation) |
| #16 | 2 | 2 (c1 via G16-a; c4 via G16-b) | G16-a (team server) |
| #17 | 2 | 1 (G17-a/b; G17-c closed) | G17-a (team server) |

None of #15, #16 or #17 should be closed on the strength of this audit alone:
each has at least one criterion with an unmet or partial evidence item.
