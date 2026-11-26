# SpiralSky v2 run: customer-history counts (PBI #22)

This is the evidence for a reproducible, package-level v2 run. It extends
[model-v2-diagnosis.md](model-v2-diagnosis.md) and decision D22-01 in
[decision-log.md](decision-log.md). The run went to a **local** MLflow store
(SQLite, outside the repository), not the team tracking server, so it is not the
logged version #22 needs and enters no #15 review or #25 roster yet.

## Reproduce

```bash
uv run train boosted-trees-v2 --data-dir data
```

(`train` is `training.main`; with the team server's `MLFLOW_TRACKING_URI` set, the same
command logs there.) It fits boosted trees (`models.boosted_trees_model`, frozen seed 0,
K = 250 from D5-01) twice on the #6 training window and scores the validation window:
v1 on the four order columns, then v2 with `n_prior_orders`, `n_prior_labelled_30d` and
`n_prior_fraud_30d` (`training.compare_history`, `customer_history`). Both runs are
children of one parent run whose description (`mlflow.note.content`) states the weakness,
the change and the validation change. `train rerun RUN_ID` replays a child run, v2
included (`history_features` is a logged parameter).

## Result (commit 688f175, clean tree, seed 0, 101,190 validation orders, 13,066 fraud)

| Run | Fraud caught at K = 250 | Fraud caught at 2K | Average precision |
| --- | ---: | ---: | ---: |
| v1, order columns | 2,893 | 4,370 | 0.3716 |
| v2, + history counts | 3,042 | 4,996 | 0.4239 |
| Change | +149 | +626 | +0.0522 |

v1 reproduces the registered result exactly (2,893 and 0.3716). v2 matches the diagnosis
table's "all three" row (3,042 and 0.4239). The brief's 3,060 and 0.4248 did not come
out of the package: the packaged v2 uses the strict-date, 30-day-delay rule and
reads training plus earlier validation labels for validation orders, so treat 3,042 and
0.4239 as the number of record. Fit time was about 1.6 s for each model and the saved
model about 1.0 MB, so the fit alone is far inside the 85 s #25 budget (a timed #27 run
and scoring-time history are still unmeasured).

## #22 criteria

| Criterion | Status |
| --- | --- |
| At least 2 logged versions per engineer | Open: v1 is logged on the team server; v2 only locally. Needs the team server (credentials). |
| Curve, confusion matrix at K, slices by country, payment and value band | Partly: the confusion matrix at K and 2K prints for each run; slices are in the diagnosis doc but not logged. The PR curve and logged slice tables are not built. |
| Run description states weakness, change, primary measure change | Evidenced in the local parent run (text above). Must be repeated on the team server. |
| #15 review repeated at the second model review; #27 timed run after 25 Nov | Open: needs Glowing6635 and the team server. |
| Second versions in the #25 training list by 25 Nov | Open: #25 roster; needs scoring-time history (#20, #25) and the packaged model (#16). |
