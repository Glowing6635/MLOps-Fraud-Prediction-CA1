# Champion fraud model card

**Draft for [PBI #35][35].** Update for the final champion by 29 November
2026. The final artifact and its evaluation are pending.

## Intended use

Prioritise SP-buy orders for daily pre-shipping review by Trust & Safety.
The score ranks orders for analyst attention. Analysts investigate flags,
make fraud determinations and decide any action on an order. The review
capacity limits how many orders can be flagged. [D4-01][timing], [PBI #5][5].

## Data period and primary measure

The frozen retrospective windows use order dates. Each start date is
inclusive and each end date is exclusive. Replaying batches leaves those
boundaries fixed. The final champion's observed training period and whether
it was refit on training plus validation remain pending. [D6-01][split],
[time-split policy][policy], [PBI #23][23].

| Window | Start | End (exclusive) |
| --- | --- | --- |
| Training | 2022-11-12 | 2023-03-27 |
| Validation | 2023-03-27 | 2023-04-17 |
| Final evaluation | 2023-04-17 | 2023-05-11 |

The primary measure is the total number of labelled fraud orders caught
in the top K across the daily selection groups. Report it beside the
maximum catchable at K, the sum across groups of the smaller of K and the
group's labelled fraud count. [PBI #5][5].

**Result against the rule at K: pending.** Report the champion and rule
counts and their difference under the final frozen comparison method.
The rule's flags are capped at K per group by ranking them by order value.
In groups where the rule flags fewer than K, show the model at K and
separately compare the model at the rule's actual flag count. Label the
comparator as a validation-tuned proxy if the actual threshold remains
unconfirmed. [PBI #18][18], [PBI #23][23], [PBI #34][34].

## Limitations

- **Feature timing.** The scoring policy assumes collection type, payment
  method, order value and item count reflect the original values at order
  placement. Operations has yet to verify their historical availability,
  including whether later adjustments overwrote order value. Refund value
  is evaluation-only. Customer snapshots have unknown timing and order
  scope, so all seven non-key customer fields are withheld from champion
  inputs. Historical linkage through labels supplies no verified customer
  mapping for new unlabelled orders. The rule still uses first-order date
  for account age, with that field's historical availability unconfirmed.
  [D4-01][timing], [D4-02][customer], [D4-03][linkage], [PBI #18][18].
- **Label meaning and completeness.** The supplied target defines
  `is_fraud=1` as fraud and `is_fraud=0` as not fraud. Unmatched labels
  remain unknown. Review coverage, later reversals and finalisation dates
  are unknown. The provisional 30-calendar-day delay is an engineering
  assumption for eligibility of historical label-derived features. It
  supplies no evidence of label maturity or historical target availability.
  The measures describe the supplied labels. [D4-04][delay], [D4-05][labels],
  [D6-01][split].
- **Daily capacity.** The Product Owner assumed 250 orders per day across
  countries before model results, as recorded in D5-01 in PR #148.
  Trust & Safety capacity confirmation is pending. The final report must
  cite the frozen K and its source, include the 2K sensitivity, and log and
  report results at any later confirmed capacity that differs. [PBI #5][5],
  [PR #148][148], [PBI #23][23].
- **Retrospective evidence.** Full-extract exploratory analysis preceded
  the test lock and included the final evaluation interval. The final
  report must give the number of formal test-window comparisons before
  the final one, based on run records. [D6-01][split], [PBI #34][34].
- **Replay and manual steps.** Batch exercises replay labelled inputs in
  rehearsal, including synthetic practice batches. They do not establish
  live unlabelled intake or review turnaround. An operator starts the
  rehearsal and supplies the three URLs. The #27 preflight documented in
  PR #99 stops at fetch/validation pending integration. End-to-end
  automation and timing await #27's rehearsal evidence. Champion selection
  and promotion are review decisions, and Trust & Safety owns review and
  action on flags. [PBI #24][24], [PBI #27][27], [PR #99][99],
  [PBI #15][15].
- **Business impact.** #34 measures order value identified as fraud
  exposure and counts distinct loyal customers wrongly flagged, at
  matched flag counts. The comparison excludes review costs and does
  not measure prevented loss or savings. A 95% interval that includes
  zero means no difference was shown on that measure. The report also
  shows the model at K where the rule flags fewer than K. [PBI #34][34].

## Evidence required to finalise

- **#15 and final review** — The registry entry, exact model version and
  artifact, originating MLflow run and recorded champion choice, including
  the final review under #22. [PBI #15][15], [PBI #22][22].
- **#18** — The rule's X and source (or documented validation-tuned proxy
  and candidate thresholds), loyal definition, account-age handling and
  matched-flag comparison method. [PBI #18][18].
- **#23** — The freeze commit for K, X and the loyal definition, the
  trained-versus-refit decision, the artifact's observed fit-date range and
  split provenance, and the final run link with actual fraud caught and
  maximum catchable at K, rule comparison, precision and recall. Verify
  the provenance against the frozen schedule above. [PBI #23][23].
- **#27** — The operator procedure and actual automated stage coverage,
  supported by the rehearsal run records and timings. Use these to update
  the remaining manual steps. [PBI #27][27].
- **#34** — The final matched-count exposure and loyal-customer differences
  with intervals, logged seed and prior test-comparison count. Link the
  final report. [PBI #34][34].
- **D34-01 publication (SpiralSky)** — Before finalising this card, link
  directly to the published decision and verify that it was recorded
  before the comparison. Then use it to support the interpretation of
  the final matched-count results and document who decides whether to
  switch. The decision-specific interpretation, decision ownership and
  chronology are pending this evidence. [PBI #34][34].

[timing]: decision-log.md#d4-01--scoring-boundary-and-order-inputs
[customer]: decision-log.md#d4-02--individual-customer-timing-gaps-and-snapshot-scope
[linkage]: decision-log.md#d4-03--exceptions-and-customer-linkage
[delay]: decision-log.md#d4-04--provisional-label-delay-and-strict-historical-eligibility
[labels]: decision-log.md#d4-05--supplied-target-definition-versus-review-completeness
[split]: decision-log.md#d6-01--frozen-retrospective-windows-and-protected-exposure
[policy]: ../config/time_split.toml
[5]: https://github.com/Glowing6635/MLOps-Fraud-Prediction-CA1/issues/5
[15]: https://github.com/Glowing6635/MLOps-Fraud-Prediction-CA1/issues/15
[18]: https://github.com/Glowing6635/MLOps-Fraud-Prediction-CA1/issues/18
[22]: https://github.com/Glowing6635/MLOps-Fraud-Prediction-CA1/issues/22
[23]: https://github.com/Glowing6635/MLOps-Fraud-Prediction-CA1/issues/23
[24]: https://github.com/Glowing6635/MLOps-Fraud-Prediction-CA1/issues/24
[27]: https://github.com/Glowing6635/MLOps-Fraud-Prediction-CA1/issues/27
[34]: https://github.com/Glowing6635/MLOps-Fraud-Prediction-CA1/issues/34
[35]: https://github.com/Glowing6635/MLOps-Fraud-Prediction-CA1/issues/35
[99]: https://github.com/Glowing6635/MLOps-Fraud-Prediction-CA1/pull/99
[148]: https://github.com/Glowing6635/MLOps-Fraud-Prediction-CA1/pull/148
