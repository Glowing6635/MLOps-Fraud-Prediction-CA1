# Decision log

This log records the feature-timing assumptions and handling for PBI #4, the temporal policy for PBI #6,
the daily review capacity K for PBI #5 (D5-01) and where the MLflow server keeps its data (D8-01).
No Operations confirmation has been received, so every timing status in D4-01 to D4-05 remains
unconfirmed. K is a product owner assumption, set on 23 November 2026. Feature-timing decisions were
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
