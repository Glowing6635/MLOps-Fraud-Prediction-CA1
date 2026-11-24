# Decision log

This log records the feature-timing assumptions and handling for PBI #4, the temporal policy for PBI #6,
the daily review capacity K for PBI #5 (D5-01), where the MLflow server keeps its data (D8-01),
and the champion registry workflow for PBI #15 (D15-01).
No Operations confirmation has been received, so every timing status in D4-01 to D4-05 remains
unconfirmed. K is a product owner assumption, set on 23 November 2026. Feature-timing decisions were

unconfirmed. The product owner confirmed K on 23 November 2026. Feature-timing decisions were
recorded on 15 November 2026, temporal windows on 16 November and the MLflow server's storage
on 18 November. Owner identifies the accountable engineer.

Sources: [live #4](https://github.com/Glowing6635/MLOps-Fraud-Prediction-CA1/issues/4),
[#14](https://github.com/Glowing6635/MLOps-Fraud-Prediction-CA1/issues/14), and
[#30](https://github.com/Glowing6635/MLOps-Fraud-Prediction-CA1/issues/30), checked
15 November 2026; the supplied field definitions; notebook section 7's executed extract
checks. None supplies field timing or
measured label maturity; the issue criteria permit explicit assumptions
and the unconfirmed/#14 fallback.

## D4-01 — Scoring boundary and order inputs

- Settled: 2026-11-15. Owner: SpiralSky. Affected PBIs: #4, #6, #12, #14, #30.
- Decision/source: **engineering assumptions**, based on #4's pre-shipping
  objective and the supplied field definitions: score at order placement, before shipping;
  `collect_type`, `payment_method`, `order_value`, and `num_items_ordered` are
  the original placement-time values, available then. In particular, assume
  `order_value` has not been overwritten by payment/refund adjustments. These
  statements are not Operations confirmation. The inventory stays unconfirmed.
- Reason: a stated, reviewable boundary is needed for an initial candidate;
  stored historical values alone cannot prove it. Confirm original versions
  and availability with Operations before treating these assumptions as facts.
- Consumer setting: `leakage.ORDER_FEATURE_COLUMNS` and
  `select_feature_inputs(frame, source="orders")` select just those four.
  Keep `order_date` separately as temporal context for splits/history and later
  date derivations; it is not a direct input from this selector.
  Keys (`country_code`, `customer_id`, `order_id`), `refund_value`, `is_fraud`,
  and unknown columns are excluded, even if present in the wrong source.
  `refund_value` is evaluation-only, because a later refund is not established
  pre-shipping information. `is_fraud` is target-only for the current order.
- Code: `src/mlops_fraud_prediction_ca1/leakage.py`; audit inventory remains
  evidence, distinct from the executable column-selection policy. Selection
  does not clean rows, encode features or enforce a training pipeline.

## D4-02 — Individual customer timing gaps and snapshot scope

- Settled: 2026-11-15. Owner: SpiralSky. Affected PBIs: #4, #12, #14, #30.
- Decision: every customer field's historical availability remains
  **unconfirmed**. No supplied dated evidence establishes its historical
  availability. The snapshot anchor and which order the row describes are
  **unknown**, for both initial and later observed orders.
- Per-field evidence and required follow-up:
  - `country_code`: The supplied field definitions describe country identity; normalized linkage checks
    support historical matching only. Placement-time customer record/version
    remains unconfirmed. Link/group only, never a feature.
  - `customer_id`: The supplied field definitions describe customer identity; composite-key/label-bridge
    checks support historical identity only. The scoring-time order-to-customer
    mapping is unconfirmed. Link/group only, never a feature.
  - `mobile_verified`: The supplied field definitions describe verification state; no historical verification
    timestamp/version establishes the state at placement. Unconfirmed; #14 group.
  - `num_orders_last_50days`: The supplied field definitions describe an order count; window endpoint,
    event inclusion, and snapshot/order scope are unknown for every order,
    including later orders. Unconfirmed; #14 group.
  - `num_cancelled_orders_last_50days`: The supplied field definitions describe cancellations; cancellation
    event availability, window endpoint, and snapshot/order scope are unknown,
    including for later orders. Unconfirmed; #14 group.
  - `num_refund_orders_last_50days`: The supplied field definitions describe refund-request counts;
    request-event availability, window endpoint, and snapshot/order scope are
    unknown, including for later orders. Unconfirmed; #14 group.
  - `total_payment_last_50days`: The supplied field definitions describe total payments; payment-event
    availability, window endpoint, and snapshot/order scope are unknown,
    including for later orders. Unconfirmed; #14 group.
  - `num_associated_customers`: The supplied field definitions describe associations; no historical edge
    creation/version evidence dates the recorded count. Unconfirmed; #14 group.
  - `first_order_datetime`: The supplied field definitions describe the first-order event. The date-order
    test in section 7.2 is limited evidence for this column alone: 105 orders
    precede that calendar day, 14,274 are same-day with unknown intraday order,
    934,722 are later, and 788 are unevaluable (949,889 safe unique orders,
    949,101 evaluable). Even a consistent date does not establish when the
    stored value became available. Unconfirmed; #14 group and D4-03 flags.
- Reason/evidence: 154,284 of 622,097 safely linked observed customers have
  multiple orders, accounting for 482,059 orders. There are 319,952 orders after
  their customer's earliest observed day and 6,020 earliest-day ties. Earliest
  observed order and batch arrival order do not anchor a snapshot. No field's
  consistency can confirm another field. Obtain field-specific historical
  versions/window endpoints; do not reconstruct the four 50-day fields from an
  extract with unknown history completeness or anchor.
- #14 handover: each engineer compares models with and without **all seven**
  non-key columns above as one group on the shared downstream split. Label
  with-column results exploratory and report their cost/effect; none of these
  columns enters a champion while unconfirmed. Keys stay outside that group.
  Actual experiments and candidate logging belong to #14, not this audit.
- Code: `CUSTOMER_FEATURE_COLUMNS` and `select_feature_inputs` default to no
  customer inputs; `source="customers", exploratory=True` explicitly selects
  the seven fields present. New columns remain withheld. Consumers must apply
  D4-03 row handling before exploratory use; column selection does not do it.

## D4-03 — Exceptions and customer linkage

- Settled: 2026-11-15. Owner: SpiralSky. Affected PBIs: #4, #6, #12, #14, #30.
- Decision/source: retain orders, flag the 105 inconsistent and 788 unevaluable
  customer-date rows, and withhold their customer values pending resolution.
  Preserve the 14,274 same-day uncertainty flags; they cannot establish an
  intraday ordering. All seven fields are already withheld from champion inputs
  on all rows under D4-02. In the exploratory branch, affected customer values
  must be missing/withheld; downstream missing-value handling must fit on the
  training partition. Never silently repair dates or drop an entire order
  because this date test fails. Missingness is not proof of leakage.
- Reason: the test identifies limited consistency evidence, not source truth.
  `customer_timing_audit` exposes per-order `date_evidence` and retains safe
  unique orders. Its normalized order → label → customer bridge is valid for
  historical analysis only. Labels containing customer IDs do not establish
  the mapping needed to score a new, unlabelled order. Operations must supply
  that mapping before customer-based scoring can rely on it.
- Code: `customer_timing_audit`, `relationships.link_orders_customers`, and
  notebook section 7.2. Later preparation owns row enforcement and audit output;
  this unit does not implement a training or scoring harness.

## D4-04 — Provisional label delay and strict historical eligibility

- Settled: 2026-11-15. Owner: SpiralSky. Affected PBIs: #4, #6, #12, #14, #30.
- Decision/source: **30 calendar days**, an explicit provisional engineering
  assumption for label maturity. No Operations delay or subsequent answer was
  supplied during completion. This is a round one-month planning margin,
  chosen to make the missing maturity assumption explicit and testable; it is
  not estimated from these files, a measured percentile, or a guarantee that
  reviews finish within 30 days. Operations evidence must replace it when
  available. No owner/Operations endorsement is implied.
- Reason: #4 permits a recorded assumption; silently using immediate labels or
  leaving a label-based caller to take the no-label branch hides leakage risk.
  Keep label-derived features out of the current four-column input policy.
  Later consumers experimenting under this assumption must disclose it and
  explicitly pass `ASSUMED_LABEL_DELAY_DAYS`; it is not an automatic default.
- Strict contract: no-label history requires `prior order_date < current
  order_date`. Label history requires `prior order_date + 30 days < current
  order_date` under this assumption, plus an actually available valid prior
  label. Equality, same-day/future orders, and missing/invalid prior dates are
  ineligible. A different supported delay must be explicitly recorded/passed.
  Use `historical_order_mask(..., label_delay_days=ASSUMED_LABEL_DELAY_DAYS)`.
  The delay argument is required; pass zero explicitly for unlabelled history.
  Existing explicit-delay calls retain the same strict boundary.
- Code: `ASSUMED_LABEL_DELAY_DAYS` and `historical_order_mask` in `leakage.py`.
  The mask checks temporal eligibility only: the consumer separately validates
  labels, matches customers, establishes as-of availability and any window
  endpoints. Passing 30 days does not prove an absent or unreviewed label exists.

## D4-05 — Supplied target definition versus review completeness

- Settled: 2026-11-15. Owner: SpiralSky. Affected PBIs: #4, #6, #14, #30.
- Source/decision: The supplied field definitions explicitly define `is_fraud=1` as fraud and `is_fraud=0`
  as **not fraud**. Preserve that supplied definition. Do not reinterpret every
  zero as missing or unreviewed.
- Separate uncertainty/reason: fraud-review coverage, subsequent reversals,
  finalization timestamps and the completeness of reviews are unknown. The
  binary definition alone cannot establish those properties or validate D4-04's
  delay. Keep this gap separate from target meaning and seek Operations evidence.
- Code/evidence: the supplied field definitions, `availability_inventory` target role and notebook
  section 7.3. No target relabelling or source-file changes are introduced.

## Reopening and downstream ownership

Reopen the relevant entry on new timing evidence or changed input schema/data;
record the actual evidence date, owner, reason and changed code/configuration.
#6/#12 own later preparation/split integration; #14 owns the grouped model
comparison and champion enforcement. Operations timing and label evidence may
replace assumptions when field-specific evidence becomes available.
Other #30 decisions and its engineer handover remain pending.

## D5-01: The top-K measure uses K = 250 orders a day

- Settled: 2026-11-23, as a product owner assumption. Owner: Glowing6635. Affected PBIs: #5, #13, #14, #15, #30.
- Decision: K, the daily review capacity, is 250 orders a day, one total for all countries, so the evaluator selects the top 250 orders of each order date and also reports 2K = 500. It is a product owner assumption, since Trust & Safety has given no capacity yet. The #13 harness and every #14 candidate use it, and each #14 candidate run records where K comes from in its `k_status` tag.
- Reason: 250 is about 5% of daily orders, roughly two or three analysts' work. 50, the earlier example, is about 1% of the roughly 4,800 validation orders a day and about 8% of their roughly 620 fraud, which is too few for a 13% fraud rate. At 50, a good model would catch close to the most it can each day, so fraud caught would hardly separate candidates and average precision, the tie-breaker, would decide instead. At 250 and 500, every validation day has more fraud than K (at least 546), so the most catchable is always the full K. One total, not a K per country, follows #5's default selection group of one order date; no country has a capacity of its own, and a single K applied to every country wouldn't work, since Thailand has only about 19 fraud a day, so any K above 19 could never be filled with fraud there.
- Set before any model result was seen, from the training and validation windows' daily order and fraud counts only. The team MLflow server's run list for the experiment `fraud-prediction` (checked on 25 November 2026) shows that before 23 November it holds only the 17 November server tests. The first model runs are the #14 boosted-trees runs, all tagged `git_commit` 83d2fea: the parent run, "HistGradientBoostingClassifier: customer columns compared" (`7d57012eafcc4f278070bd4368eba4a4`), started on 23 November 2026 at 15:06 (UTC+8), and its two child runs at 15:09 and 15:15. Commit fa0a687 (22 November, 20:51) already used K = 250 provisionally, but no model run reached the server that day.

- Settled: 2026-11-23, when the product owner confirmed it. Owner: Glowing6635. Affected PBIs: #5, #13, #14, #15, #30.
- Decision: K, the daily review capacity, is 250 orders a day, one total for all countries, so the evaluator selects the top 250 orders of each order date and also reports 2K = 500. It is a product owner assumption, since Trust & Safety has given no capacity yet. The #13 harness and every #14 candidate use it, and each #14 candidate run records where K comes from in its `k_status` tag.
- Reason: 250 is about 5% of daily orders, roughly two or three analysts' work. 50, the earlier example, is about 1% of the roughly 4,800 validation orders a day and about 8% of their roughly 620 fraud, which is too few for a 13% fraud rate. At 50, a good model would catch close to the most it can each day, so fraud caught would hardly separate candidates and average precision, the tie-breaker, would decide instead. At 250 and 500, every validation day has more fraud than K (at least 546), so the most catchable is always the full K. One total, not a K per country, follows #5's default selection group of one order date; no country has a capacity of its own, and a single K applied to every country wouldn't work, since Thailand has only about 19 fraud a day, so any K above 19 could never be filled with fraud there.
- Set before any model result was seen, from the training and validation windows' daily order and fraud counts only.
- If Trust & Safety gives a capacity that differs, the candidates are re-run at the new K and the change is logged here, as #5 says.

## D6-01 — Frozen retrospective windows and protected exposure

- Recorded: 2026-11-16. Accountable engineer: SpiralSky. Affected PBIs:
  #6, #13, #14, #21, #23, #25, #30.
- Decision: freeze half-open windows in `config/time_split.toml`: train
  [2022-11-12, 2023-03-27), validation [2023-03-27, 2023-04-17), test
  [2023-04-17, 2023-05-11). Incoming batches never move these dates.
- Reason: validation and test follow the descriptive March elevation; training
  retains 70,653 orders on/after 13 March. This is a retrospective evaluation
  choice, not a statistically established change point. The 2,341 orders on
  11 May are excluded conservatively; low volume does not prove a partial day.
  Counts, country mix and target denominators appear in the
  [frozen-window evidence](time-split.md#frozen-windows-and-country-mix).
- Enforcement: `split_orders` shares one policy with CSV reads and parameterized
  SQL predicates. Related labels/customers are scoped by allowed composite keys.
  `test_locked=true` withholds the test interval from ordinary analysis. Train
  and validation membership remains fixed after an analysis unlock.
- Final-evaluation access additionally requires
  `final_evaluation_enabled=true`, default false, after model, threshold and
  preprocessing freeze. Both selection and SQL-predicate access log the test
  interval and ordinary lock state. The setting does not unlock ordinary reads.
  Caller-owned run logs must retain access and final results. Release training
  is outside the current API.
- Provenance: loader audits record purpose, both access settings, resolved ISO
  cutoffs and aggregate membership counts. Consumers retain their code and
  configuration revisions with this metadata in their run log.
- Interpretation: preserve the supplied zero = not fraud definition; unmatched labels remain
  unknown. D4-04's provisional 30 days applies to historical label-derived
  features, not proof of historical target availability or a required gap.
  As-of evaluation requires actual label timestamps and a separate decision.
- Limitations: freeze and implementation were committed on 16 November, after
  the 12 November target. External first-training chronology is unverified.
  Earlier full-extract EDA means this is not an untouched holdout. SQLite
  semantic tests do not establish live RDS or downstream consumer enforcement.
  No model or final scoring was run for this implementation.
- Code and checks: `splitting.py`, `data.py`, `test_splitting.py`,
  `test_guarded_data.py`, `test_data.py` and `test_eda_integration.py` under
  `src/mlops_fraud_prediction_ca1/`. Consumer contract:
  [fixed temporal reads](time-split.md).

Date or interpretation changes require a new decision before subsequent
training. Preserve existing final-evaluation artifacts when policy changes.

## D8-01: The MLflow server keeps its data in RDS and S3

- Settled: 2026-11-18. Owner: Glowing6635. Affected PBIs: #7, #8, #9, #11.
- Decision: runs, models and logins live in RDS, and artifacts in S3 through the server. The server runs one pinned image from ECR, reads its secrets from Parameter Store, and reaches S3 through `LabInstanceProfile`, which attaches `LabRole`.
- Reason: the sandbox replaces the server's instance on every redeploy and at the start of a new session, so anything stored on it is lost.
- Source: the #8 research conversation, https://claude.ai/share/02998146-af10-472d-a21f-05e215dc7623.
- Details, the limits found and where we didn't follow the advice: [MLflow server design](mlflow-design.md).

## D15-01 — Approved comparison and exact registry sources

- Recorded 2026-11-25 for [PBI #15](https://github.com/Glowing6635/MLOps-Fraud-Prediction-CA1/issues/15).
  Accountable engineer: SpiralSky.
- Decision: record the Comparison ID with the analyst's sign-off and require
  it for registration. It binds the tracking store, ranking and scores, and
  the resolved top two sources. Recheck it before any registry writes. A
  logged model uses its exact artifact location. A legacy run uses its run
  artifact location. Reuse a ready version only when the run, source reference
  and download URI match the elected artifact.
- Rationale and alternatives: repeating an unbound search can register late
  runs that were absent at the review. Saving a roster of run IDs alone
  preserves membership but can miss changes in scores or model identity.
  The ID checks the comparison without a separate snapshot file. The sign-off
  still documents who approved it. A run ID or a version tag alone cannot
  distinguish two versions of different artifacts from the same run.
- Alias recovery: write challenger first and champion last so an interrupted
  operation can be retried using the same arguments. Attempt to restore the
  challenger on a reported error when the visible state still matches the
  first write. MLflow provides separate alias operations. Operators serialize
  writes and verify both aliases after retry.
- Scope and open verification: the historical Lane A comparison and acting-analyst
  sign-off stand. The installed CLI and local SQLite checks support offline
  verification. The team registry operator needs to verify the exact historical
  artifact and run the approved source, alias, retry and rollback checks on a
  socket-capable host. SpiralSky owns #15 integration, which depends on #165.

## D15-02 — Rank for review and qualify retraining separately

- Settled 2026-11-25. Decision owner: SpiralSky (Product Owner), response to
  `agile-mlops-20261124-15`. Affected PBIs: #15 and #25.
- Decision: #15 ranks every otherwise eligible validation candidate by #5
  fraud caught at K, then average precision, then ascending run ID for exact
  ties. Keep #14 exploratory runs out. Annotate measured sizes above 50 MB
  and report unavailable measurements as unknown. The Comparison ID binds the
  reviewed scores, cohort and elected sources. Size measurements are
  informational and may be updated without a new approval.
- Rationale and alternative: applying the #25 cap to #15 would omit an
  otherwise eligible winner or challenger and change the validation record.
  The two decisions answer different questions: which models caught most
  fraud, and which models the pipeline can afford to retrain within its budget.
- Impact: #25 still needs a measured roster within its 50 MB and #27 fit-time
  limits. Baseline logistic regression and qualifying boosted-tree or light
  candidates can enter that roster once measured. Oversized random-forest
  artifacts stay out unless reduced and remeasured within the limits. The
  saved-copy measurement from the harness or another fit does not measure the
  exact registry artifact. SpiralSky owns the #25 roster and its measured
  evidence, including confirmation of the cap's implementation boundary for
  #25. The #15 display uses 50,000,000 bytes for its MB note.
- Validation evidence: AP is undefined for fraud-free cohorts, as in #5.
  The current #13 harness records inputs, cutoffs, policy version and totals.
  Accept wholly legacy comparisons with matching recorded evidence and an
  explicit scored-row identity limitation. Reject malformed, mixed or
  conflicting row fingerprints. Accept matching fingerprints when present.
  Requiring them for every run now would reject the live producer's output.
  Backfilling them from equal counts would invent provenance. SpiralSky and
  the #13 owner (Glowing6635) still need to agree on the producer's fingerprint
  contract before requiring it for fresh exact-cohort evidence.
- The historical Lane A selection, exact run references, analyst sign-off
  and rollback remain the record of that review. The runbook describes new
  comparisons and the [operator checks](champion.md#pick-at-the-model-review).

## D15-03 — Register the best two ranked models within 50 MB

- Recorded 2026-11-25 for [PBI #15](https://github.com/Glowing6635/MLOps-Fraud-Prediction-CA1/issues/15)
  after teammate review of PR #161. Accountable engineer: SpiralSky.
  **Needs the Product Owner's one-line confirmation.**
- Decision: keep D15-02's ranking of every eligible validation candidate, with
  size notes. The team registry gets the best two ranked candidates whose
  model size is not known to exceed 50,000,000 bytes (exact artifact size when
  the inventory gives it, otherwise the harness's saved-copy size). Larger
  candidates stay in the table with their note and are passed over
  for registration. An unmeasured size shows as unknown and does not block it. The Comparison ID binds the registered run IDs as well as
  the ranking.
- Rationale: the registry holds what the team can actually retrain and serve
  under #25 and #27. A 187 MB forest as challenger (v2) fails that, yet
  dropping it from the ranking would hide the validation result that D15-02
  keeps visible.
- Alternatives: (1) exclude oversized models from the ranking: rejected by
  D15-02. (2) Register the unfiltered top two and note the size: this was
  the state teammate review rejected. (3) Reduce and remeasure the forest:
  possible later, and it would re-enter by rank once measured.
- Effect on the team registry: the recorded `fraud-candidate` v2 (random
  forest, 187 MB) is not within the limit. Under this rule the pair is the
  boosted-trees run and the next-ranked within-limit run. Registering it needs
  a live registry operation, which this change does not perform.

## D25-01: Evaluation-mode retraining uses the protected reader and frozen validation

- Recorded: 2026-11-24. Affected PBI: #25.
- Decision: use the #20 reader and #6 train predicate for the shared RDS snapshot. Count exclusions in the database and score all candidates and the current champion on the same frozen validation input. No window changed. Release mode follows #23.
- Reason: this lets retraining run independently of the #24 rehearsal tooling and keeps the frozen test rows out of the training process.

## D27-01: Wall-clock budget for a practice batch

Recorded 2026-11-20 for [the three-minute pipeline (#27)](https://github.com/Glowing6635/MLOps-Fraud-Prediction-CA1/issues/27)
and [per-batch retraining (#25)](https://github.com/Glowing6635/MLOps-Fraud-Prediction-CA1/issues/25).
Use these initial targets when choosing the candidate list and instrumenting the pipeline.

| Stage | Target (seconds) | Work covered |
| --- | ---: | --- |
| Fetch and validate | 25 | Download and validate all three inputs. |
| Clean and load | 30 | Clean the batch, capture starting counts and commit the database writes. |
| Read training data | 15 | Read the eligible rows from RDS once for all candidates. |
| Train and log | 85 | Fit the full candidate list, evaluate it as required and finish MLflow logging, including artifact uploads. |
| Report | 15 | Obtain final counts and print the batch summary and filtered MLflow URL. |

The five stages have 170 seconds in total. The remaining 10 seconds cover process
startup, configuration and shutdown within the 180-second command limit. Training
gets the largest allocation because each additional candidate requires a fit and
an upload. The 85 seconds are shared across all candidates.

For each timed rehearsal, print elapsed wall-clock seconds alongside each stage
target and measure the whole command from invocation to exit. Record the input
row counts for all three files with each result so the timing has a batch size.
These allocations are planning assumptions. Verification requires three runs on
the demo machine and network, each starting from a rebuilt rehearsal database.

## D26-01 — Apply and verify the deployment settings before rollout

- Settled: 2026-11-24. Owner: SpiralSky. Affected PBI: #26.
- Decision: use a uniquely named EB saved configuration for creation. Check all
  seven #9 settings in the deployed configuration before redeploying and stop
  if any are missing or invalid. Require one artifact setting in the active
  application environment namespace and apply one offline URI check on both
  rollout paths. Require a literal S3 key prefix without raw `?` or `#`.
  The check reports setting names and leaves configuration changes to the
  operator.
- Rationale: the offline EB CLI 3.27.5 resolver uploads a saved configuration
  under its basename and passes `--cfg` verbatim to `CreateEnvironment`. A
  local path therefore selects a different name. A CNAME match alone lets an
  incomplete environment attempt a rollout that the container cannot start.
  [DescribeConfigurationSettings](https://docs.aws.amazon.com/elasticbeanstalk/latest/api/API_DescribeConfigurationSettings.html)
  can include a draft configuration alongside the active one. A URI can end in
  `/artifacts` and still have an invalid bucket name. Validating the bucket
  component against the [S3 general purpose naming rules](https://docs.aws.amazon.com/AmazonS3/latest/userguide/bucketnamingrules.html)
  in one place keeps create and redeploy consistent and allows nested key
  prefixes. MLflow 3.18.0 treats raw `?` and `#` as URI delimiters, so either
  can redirect artifact writes to a different key even when rollout passes.
- Alternatives: applying settings after creation leaves first boot without
  required values. Automatically repairing an existing configuration would
  change a running environment and require selecting a destination and secret
  references on the operator's behalf. Keeping a second URI rule in the AWS
  query can lead to different behavior on create and redeploy. Percent-encoding
  the delimiters would store the escape sequences literally in the S3 key.
  Probing S3 would add a network call and test the deployer's permissions, which can
  differ from the instance role's.
- Impact: creation receives the settings at launch. Redeploy requires read
  access to the environment's configuration and leaves a mismatched
  environment untouched. The URI check verifies the name and path syntax.
  Bucket existence and instance-role access still require the live persistence
  check.
- Open: SpiralSky owns the clean sandbox deployment and stopped-session billing
  evidence and will arrange the unaided teammate deployment and quickstart.
  The live evidence for #26 AC1–4 is tracked in owner request
  `agile-mlops-20261124-10`.

## D16-01 — Skew and outlier handling for scoring inputs

- Recorded: 2026-11-23. Owner: SpiralSky. Affected PBIs: #16 (criterion 2), #119.
- Decision: the two numeric scoring inputs from D4-01, `order_value` and
  `num_items_ordered`, receive a fixed signed `log1p` (`skew_outliers.transform_skewed_features`),
  appended as `<feature>_log1p`. Raw cleaned values are kept; nothing is capped, removed or imputed.
- Reason: notebook section 3.5.4 shows every numeric feature right-skewed (items ordered
  41.04), with a largest item count of 1,600 against a median of 2 and P99 of 22, and no
  negative or infinite values. The extremes are located but not established as errors, and
  large orders may carry fraud signal, so the log scale limits leverage without discarding them.
- Training-only fitting: the transform learns nothing, so no threshold is fitted and a
  held-out row's output never depends on other rows. Missing values stay missing;
  infinities become missing. Missing-value imputation remains a later, train-fitted step.
- Not transformed: `collect_type`/`payment_method` (categorical), `refund_value`
  (evaluation-only, D4-01) and the customer fields (withheld, D4-02). `skew_outliers.TREATMENTS`
  holds the per-feature reasons. This does not claim model performance, #118/#120 or registration.

## D16-02: Feature groups and their validation comparison

- Recorded: 2026-11-24. Owner: SpiralSky. Affected PBIs: #16 (criteria 1, 3, 4).
- Groups (`feature_pipeline.FEATURE_GROUPS`): `order_raw`, the four allowed order columns (D4-01); `order_log`, `order_raw` plus signed log1p of `order_value` and `num_items_ordered` (D16-01). Customer columns stay out (D4-02).
- Comparison: baseline logistic regression, K = 250, seed 0, validation window, run through the #13 harness into a local SQLite MLflow store (not the team server).

| Group | Fraud caught at K | Average precision |
| --- | --- | --- |
| `order_raw` | 2498 | 0.3259 |
| `order_log` | 2450 | 0.3240 |

- Decision: `order_log` stays the packaging default so a candidate using the default
  exercises the signed-log treatment end to end, including after model loading. It
  keeps the original values available and compresses plausible extremes in the added
  columns. For the measured logistic baseline, choose `order_raw` on validation
  evidence. It caught 48 more frauds at K (about 2% of the raw group's catch) and had
  higher average precision. Choose the group for each candidate using its validation
  results. The local comparison did not test boosted-tree performance. A comparison
  against the team MLflow server is pending.
- Which candidates use it: `uv run train compare-feature-groups` puts it in front of the
  baseline. `uv run train baseline` and `uv run train boosted-trees` do not yet wrap
  their models in `build_pipeline`, so models they register have no skew handling until a
  candidate is chosen using the validation comparison above (tracked in #16).

## D17-01 — Training-only class weighting

- Recorded: 2026-11-23. Updated: 2026-11-25. Accountable engineer: SpiralSky.
  Affected PBIs: #17, #13, #5, #14.
- Decision: Keep `class_weight=None` as the default and use the final
  estimator's `class_weight="balanced"` for the opt-in comparison. Both
  candidates in pinned PR #152 (`6c1326c`), logistic regression and histogram
  gradient boosting, support this parameter. Each computes a weight of
  n / (2 × class count) from the training labels. The harness scores the
  original validation orders at their natural fraud rate.
- Reason: Both candidates produced identical probabilities on a synthetic
  fixture when native balancing was compared with explicit balanced sample
  weights. The harness already logs the classifier parameter and restores
  it when it clones the saved model for replay. Omitted weighting keeps
  the existing estimator fit. Explicit unit weights were dropped because
  supplying them can change seeded tree binning.
- Alternatives: Resampling changes the training rows. A fit adapter adds
  weight routing and serialization handling that neither candidate needs.
- Scope: The production weighting decision stays open. This comparison uses
  the frozen split, seed 0, and the product owner's K=250/day assumption
  in D5-01. The final holdout stays locked. See
  [training invocation](training.md#comparing-training-class-weights).

### Frozen validation comparison

The training window had 75,449 fraud labels across 721,354 orders. All four
runs and their replays scored the same 101,190 validation orders, including
13,066 fraud cases (12.91%). The local MLflow runs used the standard fit,
parameter logging, artifact loading and replay. Each replay exactly matched
its logged fraud caught at K and average precision.

| Candidate | Class weight | Fraud caught at K | Fraud caught at 2K | Average precision |
| --- | --- | ---: | ---: | ---: |
| Logistic regression | `None` | 2,498 | 3,985 | 0.325879 |
| Logistic regression | `balanced` | 2,479 | 3,933 | 0.322586 |
| Boosted trees | `None` | 2,893 | 4,370 | 0.371637 |
| Boosted trees | `balanced` | 2,856 | 4,395 | 0.371064 |

## D18-01 — Baseline rule, account age, loyalty and the threshold selection procedure

- Recorded: 2026-11-23. Accountable engineer: SpiralSky. Affected PBIs: #18
  (criteria 1–3), with inputs to #5, #15 and #125. No value was fitted, no
  order value was examined and no model or rule was scored for this entry.
- Settled: (a) the rule is `order_value > X` strictly and account age under one
  day (age 0), unusable inputs never flag and are counted by status; (b) account
  age is calendar days between `order_date` and the date part of
  `first_order_datetime`, an exception to D4-02 because today's rule already
  uses that field; (c) a loyal order has at least 2 distinct orders of the
  same (country, customer) on strictly earlier calendar dates, with `unknown`
  kept separate from `not_loyal` and loyalty used for reporting only; (d) X is
  chosen by the predeclared procedure in
  [the threshold protocol](baseline-rule.md#threshold-x-candidate-grid-and-selection-protocol):
  grid 50, 100, 200, 500, 1000, 2000 SGD, fitted on the frozen D6-01 validation
  window only, objective and tie-breaks as written there.
- Source of X: `spec.txt:135` gives only the symbol X, and no value has been
  received from Trust & Safety, so any X used is a validation-tuned proxy and
  is labelled as one. A confirmed value replaces it as the headline.
- Provenance: the age-only use of `first_order_datetime` (excluded from loyalty)
  is anchored at the definition freeze 466f080, before any result.
- Rationale: fixing the grid, the objective and the loyal count before results
  prevents tuning the baseline or the loyal group after seeing outcomes. The
  age and loyalty definitions follow the data limits in D4-02 to D4-04.
- Alternatives not taken: age in elapsed hours (clock times are unconfirmed);
  `num_orders_last_50days` or `first_order_datetime` as loyalty evidence
  (both withheld, D4-02); a loyal count of 1 (a single earlier order could be a
  failed first attempt); a grid or X chosen after seeing results.
- Impact: the rule is a labelled proxy baseline; loyalty left-censoring from the
  2022-11-12 history start is a stated limit; the loyal count of 2 is a project
  assumption Operations may replace by a new dated entry before results.
- Open, not settled here: **#5 implementation** (owner: Glowing6635, the #5
  assignee: the shared evaluator, and K and the selection grouping fixed before
  any results); **Trust & Safety capacity confirmation** (the real capacity K
  and deployed X are not received; until then K is a Product Owner (PO)
  assumption, to be confirmed or replaced by a dated entry); **X** (deciding
  owner: Trust & Safety supplies the deployed value, otherwise the procedure
  above selects the proxy once the #5 evaluator exists); **#15 champion** (deciding owner: SpiralSky, the #15
  assignee, not decided here and not affected by this entry).
- Code and checks: `rules.py`, `loyalty.py`, `test_rules.py`, `test_loyalty.py`;
  rule and loyalty reference in [baseline-rule.md](baseline-rule.md).


## D18-02 — Matching rule and model flags per group

- Recorded: 2026-11-24. Accountable engineer: SpiralSky. Affected PBIs: #18
  (criteria 4 to 6). No real data was scored for this entry.
- Settled: per #5 selection group the rule's flags are ranked by `order_value`
  (ties by `country_code`, `order_id`) and cut at K; the model is cut to the
  rule's resulting count and also shown at K. "Beats the rule" means strictly more
  fraud caught at the matched count. X is fitted on the D6-01 validation window with the
  D18-01 grid and objective; K is 250 a day (D5-01), so if K changes the comparison
  is rerun. A group with no rule flags gets no matched model flags.
- Code and checks: `rule_comparison.py`, `test_rule_comparison.py`.

## D18-03 — Post-hoc replacement grid for the proxy threshold X

- Recorded: 2026-11-24, **after** the first real comparison and before any rerun.
  Accountable engineer: SpiralSky; approved by the owner (option B), who is also
  asking Trust & Safety for the real X. Affected PBI: #18 criterion 1.
- Why: the D18-01 grid (50 to 2000 SGD) was fixed before any order value was
  examined. The first real run (validation window, K = 250, #152 boosted-trees
  scores, `a410f31`) showed it sits almost entirely above the order values: the
  rule flagged 7 orders at X = 50 and none at X = 100 or more, and the matched
  comparison was a 5-5 tie at 7 flags. That result is kept and reported as the
  original.
- Settled: the second grid is the training window's `order_value` quantiles
  0.90, 0.95, 0.99, 0.995, 0.999 and 0.9999, computed on the training window
  `[2022-11-12, 2023-03-27)` only (all its cleaned orders, no labels, no
  validation rows), rounded to 2 decimals and de-duplicated. Selection is
  unchanged (D18-01 objective and tie-breaks, fitted on the validation window,
  K = 250 from D5-01). The grid rule was written here before the values were
  computed, so the quantile list is not tuned to results.
- Status: the result is a **post-hoc, validation-tuned proxy** comparison. It is
  not the deployed threshold and does not replace the original result. The real
  X from Trust & Safety, when it arrives, supersedes both.
- Limit: the grid is still chosen after seeing that the first grid failed, and
  it is fitted on the same validation window that scores the comparison, so it
  favours the rule and a model win would be conservative.
