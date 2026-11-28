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
`n_prior_fraud_30d` (`training.compare_history`, `customer_history`). Both children are tagged
`champion_eligible=True`, so the champion selector can rank them (champion.py); the
selector still needs a clean tree and the same K. Both runs are
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
table's "all three" row (3,042 and 0.4239). The packaged v2 uses the strict-date, 30-day-delay rule and
reads training plus earlier validation labels for validation orders; 3,042 and
0.4239 are the number of record. Fit time was about 1.6 s for each model and the saved
model about 1.0 MB, so the fit alone is far inside the 85 s #25 budget (a timed #27 run
and scoring-time history are still unmeasured).

## Logged diagnostics (AC2)

Every `train` run, v1 and v2 alike, now logs the validation diagnostics through one code
path (`diagnostics.build_diagnostics` and `log_diagnostics`), as table artifacts under
`diagnostics/`:

- `precision_recall_curve.json`: precision and recall over the whole validation window,
  thinned to 500 points.
- `confusion_matrix_at_k250.json`: selected / not selected against fraud / not fraud, from
  the same daily top-K selection as `fraud_caught_at_k`.
- `errors_by_country.json`, `errors_by_payment_method.json`,
  `errors_by_order_value_band.json`: orders, fraud, fraud rate, selected, caught,
  `missed_fraud` (fraud not selected), `false_alarms` (selected, not fraud), precision and
  recall at K. A slice's `caught` adds up to the run's `fraud_caught_at_k`.

Order-value bands (currency unit of the data, lower bound included): `[0,3)`, `[3,5)`,
`[5,8)`, `[8,13)`, `[13,30)`, `[30,∞)`, plus `missing`. The run tag `order_value_bands`
records them. Reproduce locally with a SQLite store, no server needed (the experiment name
must be set, as in `.env.example`, or `train` raises):

```bash
MLFLOW_TRACKING_URI=sqlite:///$HOME/mlflow-local.db MLFLOW_EXPERIMENT_NAME=<experiment> \
  uv run train boosted-trees-v2 --data-dir data
```

Before this change neither version logged any of these: v1 on the team server has none, and
the per-country, payment and band tables in the diagnosis doc were a one-off refit. v1
runs logged earlier lack the tables; re-log v1 (the command above does) to get them.

### v2 slices from the local run (K = 250, 101,190 validation orders)

Run with a local SQLite store on `data/` at commit 936bffa (the logging change; the local run ID was not recorded): fraud caught 3,042,
confusion at K: 3,042 caught, 2,208 false alarms, 10,024 fraud missed, 85,916 not fraud
not selected. Caught by country, v1 (diagnosis doc) to v2: PK 2,080 to 1,943; BD 514 to 587;
PH 83 to 131; MY 105 to 296; TH 111 to 85. So the gain is in MY, PH and BD, **not** the
PK/BD pattern the diagnosis predicted, and PK and TH lose fraud. By value band v2 catches
785, 684, 598, 420, 362 and 193 in `[0,3)` to `[30,∞)`; the top band's recall is 0.53.
Against v1's refit counts (561, 522, 528, 502, 534, 246; diagnosis doc), v2 gains in `[0,3)`,
`[3,5)` and `[5,8)` (+224, +162, +70) but **loses** in `[8,13)` (-82), `[13,30)` (-172) and
`[30,∞)` (-53). The v1 counts are from a refit, not a logged v1 run.

## #22 criteria: SpiralSky status (offline)

| Criterion | Status |
| --- | --- |
| AC1: at least 2 logged versions | Partly: v1 and v2 log under one parent run to a local store; not yet on the team server (needs credentials). |
| AC2: curve, confusion matrix at K, slices by country, payment and value band, bands stated | Met in code and tests (`test_diagnostics.py`) for v1 and v2; to be logged on the team server with AC1. |
| AC3: description states weakness, change and primary measure change | Met in the parent run's description (v1 AP 0.3716 to v2 AP 0.4239, fraud caught 2,893 to 3,042); repeat on the team server. |
| AC4: #15 review repeated; #27 timed run after 25 Nov | Open, depends on #25 and Glowing6635; unchecked. |
| AC5: second versions in the #25 training list by 25 Nov | Open, depends on #23/#25; unchecked. |
