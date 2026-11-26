# Champion pick and rollback (PBI #15)

The selector ranks finished #13 harness runs tagged `champion_eligible=True`
with `exploratory=False` and `git_dirty=False` at the requested daily K. The
order follows [#5](https://github.com/Glowing6635/MLOps-Fraud-Prediction-CA1/issues/5):
most fraud caught at K, then higher average precision over the full validation
window, then ascending run ID. The table shows fraud catchable at K when the
harness recorded it. Average precision must be present for a cohort with
labelled fraud and is undefined (N/A) when there is no fraud.

Compare candidates with matching input SHA-256 hashes, validation start and end
(the test start), cleaning policy version, and validation order and fraud counts.
If any run records `validation_rows_sha256`, every candidate must record the
same valid fingerprint. Invalid or conflicting evidence stops selection.
If all runs lack the fingerprint, the current harness's input and aggregate
records support a legacy comparison that is labelled as such. Equal counts
alone cannot establish scored-row identity. Use a separate experiment containing
the complete comparable cohort when a producer starts logging fingerprints.

The table reports the harness's measured saved-copy size when available. The
preview reports the exact elected artifact's byte size when its file inventory
provides it. An unavailable measurement is shown as `unknown`. The size note
`>50 MB` means more than 50,000,000 bytes. Size is informational for ranking
and registration approval. The size and fit-time limits govern the #25
retraining roster under [D15-02](decision-log.md#d15-02--rank-for-review-and-qualify-retraining-separately).

## Pick at the model review

Copy `.env.example` to `.env` and set `MLFLOW_TRACKING_URI` to the intended MLflow
server and `MLFLOW_EXPERIMENT_NAME` to the existing experiment containing the
validation runs. Supply the server credentials if required. From the repository
root, print the comparison and elected artifact IDs for the Trust & Safety
analyst to review:

```sh
uv run champion-pick --k 250
```

Keep the comparison and its printed Comparison ID with the analyst's sign-off.
After approval and authorization to write to the intended registry, pass the ID
from that review:

```sh
uv run champion-pick --k 250 --register fraud-candidate --approve-comparison APPROVED_ID
```

The command checks the tracking store, ordered ranking, scores, validation
provenance and two resolved artifact identities against the approved ID before
writing to the registry. If an eligible run arrives, the comparison changes,
or the ID predates this format, print a new comparison and obtain approval for
that result. A size measurement can be added or corrected without changing the
ID. The sign-off records the analyst's approval.

For each of the top two runs, the selector requires one model named `model`
with an `MLmodel` manifest, in either the MLflow 3 logged-model layout or the
legacy run-artifact layout. A run with both layouts or multiple logged models
requires investigation before registration. Each registry version records
`source_run_id`, the logical `artifact` (`runs:/<run>/model`), the
`resolved_artifact` location, `k` and its metrics. Reuse requires the same run
and the verified registry source and download location.

The best two ranked runs whose model is within 50 MB take `champion` and
`challenger` ([D15-03](decision-log.md#d15-03--register-the-best-two-ranked-models-within-50-mb)).
A run known to be larger stays in the table and is passed over. The comparison
shows the validation rank and identifies the elected champion separately. Any displaced
version keeps its registry entry. A new champion records the prior version in
`previous_champion_version`.

## Move the alias and roll back

```python
from mlops_fraud_prediction_ca1 import champion

# Use an MlflowClient configured for the target registry.
champion.move_champion(client, "fraud-candidate", "<version>", "reason")
```

This records `previous_champion_version` and the reason on the new champion and
gives the old champion the `challenger` alias. Promoting the current challenger
swaps the aliases. Promoting a third version displaces the existing challenger,
whose version remains registered. To roll back the champion, read its
`previous_champion_version` tag and pass that version to `move_champion`. The
version you roll back from becomes the challenger. To restore an earlier
challenger as well, record its version before the move and reassign its alias
explicitly after rollback.

Serialize registry operations. The command writes `challenger` first and
`champion` last. A failed write can leave an intermediate alias pair. Retry
the same registration with its approved ID or repeat the same move with its
destination and reason, then read both aliases and the previous version tag.
If a registration retry reports a changed comparison, pause for review of the
new comparison and inspect the current alias pair before another move.

## Team registry record

The Product Owner requested setting the alias and exercising rollback.

The source was the `laneA-live` rehearsal experiment (ID `2`) on the team MLflow
server. Its three eligible #14 runs used commit `77837c6`, seed 0, and the
same validation window at K = 250 orders per day (D5-01). The separate
`fraud-prediction` experiment had one eligible run at this K. The three
exploratory runs with customer columns were excluded.

| Rank | Model | Validation run ID | Fraud caught at K | Average precision | `fraud-candidate` version | Final alias |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | HistGradientBoostingClassifier | `9c3a59771cf8499d92972f5c7873e280` | 2893 | 0.3716 | 1 | `champion` |
| 2 | RandomForestClassifier | `9a40f8f5e35a48ddb39a97f9f052dcad` | 2811 | 0.3495 | 2 | `challenger` |
| 3 | ExtraTreesClassifier | `8af9db9f95e543cfbfebdef82545322c` | 2695 | 0.3434 | — | — |

The first-ranked run caught 82 more fraud cases than the second at the same
daily review capacity. The live rollback check moved `champion` from version 1
to version 2 and restored it to version 1. Both aliases resolved to the expected
ready versions and source runs after each move. The destination version recorded
the previous champion version on each move.

See the [acting-analyst sign-off](model-review.md#4-criterion-3-acting-analyst-sign-off).
Historical version 2 (random forest, 187 MB) is above 50 MB, so it no longer
qualifies as challenger. Re-registering needs a live registry operation.
The exact bytes of historical version 2 remain unverified. Size measurements for
separate PR #164 fits cannot establish that version's size. The historical
sign-off applies to the recorded pair. New team registry changes remain on hold
pending exact artifact and integration evidence. Integration also depends on
[PR #165](https://github.com/Glowing6635/MLOps-Fraud-Prediction-CA1/pull/165).
