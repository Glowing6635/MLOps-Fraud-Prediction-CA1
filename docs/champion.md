# Champion pick and rollback (PBI #15)

The selector ranks finished #13 harness runs tagged `champion_eligible=True`
with `exploratory=False` and `git_dirty=False`. It considers only the requested
daily K and skips runs with a recorded model size above 50 MiB (52,428,800
bytes). At that K, all candidates must have matching input SHA-256 hashes,
validation start and end (the test start), cleaning policy version, and
validation order and fraud counts. Missing, invalid, or conflicting provenance
stops selection. The order follows #5: most fraud caught at K, then average
precision, with run ID breaking exact ties.

## Pick at the model review

Copy `.env.example` to `.env` and set `MLFLOW_TRACKING_URI` to the intended MLflow
server and `MLFLOW_EXPERIMENT_NAME` to the existing experiment containing the
validation runs. Supply the server credentials if required. From the repository
root, print the comparison for the Trust & Safety analyst to review:

```sh
uv run champion-pick --k 250
```

After approval, select and register the top two from the same experiment:

```sh
uv run champion-pick --k 250 --register fraud-candidate
```

Each version is tagged with `source_run_id`, `artifact` (`runs:/<run>/model`), `k`
and its metrics. The best takes `champion` and the second takes `challenger`.
Any displaced version keeps its registry entry. When the champion changes,
the new champion records the prior version in `previous_champion_version`.

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
The recorded random forest challenger exceeds the current size cap.
