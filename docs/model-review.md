# First model review: materials (PBI #15)

The review picks the top 2 eligible candidates and the champion. The owner or a
teammate acts as the Trust & Safety analyst: Trust & Safety is a persona in the
course brief, so no outside analyst is available, and K = 250 orders a day is a
Product Owner assumption with no source (D5-01, D18-04). Record who acted as the
analyst below.

## 1. Candidates on the table (validation window, K = 250, seed 0)

From the `laneA-live` experiment on the team MLflow server (runs of commit
`77837c6`; these are rehearsal runs, not the team experiment):

| Rank | Run | Fraud caught at K | Average precision | Registered |
|---|---|---|---|---|
| 1 | HistGradientBoostingClassifier: customer columns compared (9c3a59771cf8499d92972f5c7873e280) | 2893 | 0.3716 | yes |
| 2 | RandomForestClassifier: customer columns compared (9a40f8f5e35a48ddb39a97f9f052dcad) | 2811 | 0.3495 | yes |
| 3 | ExtraTreesClassifier: customer columns compared (8af9db9f95e543cfbfebdef82545322c) | 2695 | 0.3434 | no |

Champion: HistGradientBoostingClassifier: customer columns compared (9c3a59771cf8499d92972f5c7873e280): more fraud caught at K.

Exploratory (with-customer-columns) runs are not eligible. Differences with and
without those columns: boosted trees +161, random forest +14, extra trees -81 fraud
caught at K (PR #164).

## 2. Rule versus model (PR #157, post-hoc proxy rule)

The rule (proxy X = 13.36 SGD, account age 0 days) flags 132 validation orders and
catches 73. The boosted-trees model cut to the same 132 flags catches 102, with 30 false
flags (rule: 59) and 6 false flags on loyal orders (rule: 0). The original-grid result
was a 5-5 tie at 7 flags. See PR #157 for the tables and limits.

## 3. Questions for the analyst

1. Is fraud caught at K the right primary measure, and is 250 flags a day a workable load?
2. Is a false flag on a loyal customer (rank 1 model: 6 vs the rule's 0 on the proxy
   comparison) acceptable for the extra fraud caught?
3. Do you accept the top-ranked model as champion, with the second as challenger?

## 4. Record of the review (fill in at the review)

- Date and attendees:
- Who acted as the analyst:
- Champion (run ID, registered version):
- Challenger:
- Reason (from the table above, plus the analyst's view):
- Disagreements or follow-ups:

## 5. Team registry status

At the Product Owner's request, the top two runs above were registered as
`fraud-candidate` versions 1 and 2 on the team server. Version 1 is the
`champion` after the rollback check. The analyst review is pending. See the
[team registry record](champion.md#team-registry-record) for the run IDs and
rollback evidence.
