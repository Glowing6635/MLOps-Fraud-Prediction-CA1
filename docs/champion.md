# Champion pick and rollback (PBI #15)

`champion.py` ranks eligible runs, registers the top 2 and moves the `champion`
alias. Eligible means a #13 harness run tagged `champion_eligible` = `True`, not
exploratory, evaluated at the same K (the exploratory with-customer-columns runs
of #14 are never eligible). Order: most fraud caught at K, then average precision.

## Pick at the model review

Set `MLFLOW_TRACKING_URI` (the team server) and run, with the Trust & Safety
analyst present:

```python
from mlflow import MlflowClient
from mlops_fraud_prediction_ca1 import champion

client = MlflowClient()
runs = champion.eligible_candidates(client, [EXPERIMENT_ID], k=250)
print(champion.comparison_table(runs))        # paste into the review record
champion.register_top_two(client, "fraud-candidate", runs)
```

Each version is tagged with `source_run_id`, `artifact` (`runs:/<run>/model`), `k`
and its metrics. The best has alias `champion`, the second `challenger`.

## Move the alias and roll back

```python
champion.move_champion(client, "fraud-candidate", "<version>", "reason")
```

This records `previous_champion_version` and the reason on the new champion and
gives the old one the `challenger` alias. To roll back, call it again with the
recorded previous version. The two registered models swap roles each time.

K = 250 (D5-01). Checks: `test_champion.py` against a local MLflow store.
Not yet run against the team server.
