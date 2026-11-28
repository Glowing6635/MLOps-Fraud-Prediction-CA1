# Champion fraud model card

**Draft for [PBI #35][35], due 29 November 2026.** The team registry
records `fraud-candidate` version 1 (histogram gradient boosting) as
champion after the rollback check. The registered artifact points to
MLflow run `9c3a59771cf8499d92972f5c7873e280`
(`runs:/9c3a59771cf8499d92972f5c7873e280/model`). That run caught
2,893 labelled fraud orders at K = 250/day, with average precision 0.3716
on validation. The selection is provisional for final evaluation: #22
can change the champion and #23 records the exact scored artifact and
whether it was refit. [First model review and registry record][registry],
[PBI #22][22], [PBI #23][23].

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

**Validation comparison, not final performance.** The #18 comparison
reproduced the boosted-trees candidate on validation. At K = 250/day
it caught 2,893 fraud orders across 5,250 flags. The proxy rule flagged
132 orders in total and caught 73. At the same 132 flags as the rule,
the model caught 102. Compare 102 with 73 for the same review load,
not 2,893 with 73. The rule flags orders above SGD 13.36 when account
age is zero days; this threshold was tuned on validation, not taken
from a deployed rule. [#18 comparison][157], [D18-04][proxy],
[First model review][registry].

**Final result against the rule at K: pending #23 and #34.** Use the
frozen K and X, show the model at K, and compare both at the rule's
actual flag count wherever it flags fewer than K. The validation
numbers above are not substitutes. [PBI #23][23], [PBI #34][34].

## Limitations

- **What the model knows at review time.** The model assumes collection
  type, payment method, order value and item count reflect the original
  order, including that later adjustments did not overwrite order value.
  We have not verified this. The model excludes refund value and seven
  unconfirmed customer fields. The files do not reliably identify the
  customer for a new order before its fraud label arrives. The comparison
  rule still uses the customer's first-order date to calculate account age,
  although its availability at placement is also unconfirmed.
  [D4-01][timing], [D4-02][customer], [D4-03][linkage], [D18-04][proxy].
- **What the labels tell us.** The dataset uses 1 for fraud and 0 for not
  fraud. An order without a matched label is unknown, not cleared. We do
  not know how many orders were reviewed, when reviews finished, or
  whether decisions later changed. The 30-day delay is an assumption about
  when past labels can be used for features, not proof that reviews are
  complete after 30 days. [D4-04][delay], [D4-05][labels], [D6-01][split].
- **How many orders get flagged.** K = 250 reviews per day across all
  countries is the Product Owner's planning assumption, not measured
  analyst capacity. The course has no operational capacity figure to
  verify it against. The final report must cite the frozen K and include
  the 2K sensitivity. [D5-01][capacity], [D18-04][proxy], [PBI #5][5].
- **How to read the final evaluation.** Earlier exploratory analysis
  covered the full historical file, including the period reserved for
  final scoring. The final score is retrospective evidence, not
  evidence from a period the team never saw. Count earlier formal
  comparisons from run records, not from that exploratory work.
  [D6-01][split], [PBI #34][34].
- **What remains manual.** Rehearsals use labelled replay data and
  synthetic practice batches, not live incoming orders. An operator starts
  the run and supplies three batch links. The batch preflight is in
  `dev`; it checks the inputs, not whether all stages finish within three
  minutes. The stage budget is a target, not a measured time. Analysts
  decide what to do with flags; a run does not select or promote a champion.
  [PBI #24][24], [D27-01][budget], [PR #99][99], [PBI #15][15].
- **What the impact figures mean.** Fraud exposure is the order value of
  flagged orders labelled as fraud, not money saved. The comparison
  excludes review costs and counts distinct loyal customers wrongly flagged
  at matched flag counts. If a 95% interval includes zero, the report
  says it did not show a difference on that measure. [D34-01][impact].

## Final evaluation status

- **#22 final review:** Confirm or replace the #15 validation champion;
  record the selected version, artifact and MLflow run. [PBI #22][22].
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
- **D34-01** — Verify that the linked decision remains canonical and was
  recorded before the comparison; apply it to the final matched-count
  results and record who decides whether to switch. [D34-01][impact].

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
[157]: https://github.com/Glowing6635/MLOps-Fraud-Prediction-CA1/pull/157
[registry]: https://github.com/Glowing6635/MLOps-Fraud-Prediction-CA1/blob/c4d75a01deb53cb0cc4b0a5e296212f853101dc7/docs/model-review.md#4-criterion-3-acting-analyst-sign-off
[capacity]: decision-log.md#d5-01-the-top-k-measure-uses-k--250-orders-a-day
[budget]: decision-log.md#d27-01-wall-clock-budget-for-a-practice-batch
[impact]: https://github.com/Glowing6635/MLOps-Fraud-Prediction-CA1/blob/2b0bbf2b9bbf9f3434f88872c741e9367fb4c42d/docs/decision-log.md#d34-01--criterion-for-reading-the-model-versus-rule-result
[proxy]: https://github.com/Glowing6635/MLOps-Fraud-Prediction-CA1/blob/dd749e1b9e294539a82e2d7e9bdc0faf1fd63b47/docs/decision-log.md#d18-04--the-proxy-rule-is-final-no-deployed-threshold-or-capacity-source-exists
