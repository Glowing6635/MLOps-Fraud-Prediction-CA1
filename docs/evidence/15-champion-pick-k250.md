# Champion Pick Rerun Evidence (PBI #15, K = 250)

- **Date (UTC):** 2026-11-26T11:22Z
- **Command:** `uv run champion-pick --k 250` (read-only)
- **Target MLflow:** team MLflow server (`laneA-live`, experiment ID `2`)

## Verbatim Output

```text
| Rank | Run | Fraud caught at K | Fraud catchable at K | Average precision | Saved model size (harness) | Top two by rank |
|---|---|---|---|---|---|---|
| 1 | HistGradientBoostingClassifier: customer columns compared (9c3a59771cf8499d92972f5c7873e280) | 2893 | 5250 | 0.3716 | 1,032,734 bytes | yes |
| 2 | RandomForestClassifier: customer columns compared (9a40f8f5e35a48ddb39a97f9f052dcad) | 2811 | 5250 | 0.3495 | 187,236,750 bytes (>50 MB) | yes |
| 3 | ExtraTreesClassifier: customer columns compared (8af9db9f95e543cfbfebdef82545322c) | 2695 | 5250 | 0.3434 | 23,109,033 bytes | no |

Validation provenance: matching harness inputs, window and counts. Exact scored-row identity is unverified (legacy runs).

Rank 1: HistGradientBoostingClassifier: customer columns compared (9c3a59771cf8499d92972f5c7873e280): more fraud caught at K.
Champion: HistGradientBoostingClassifier: customer columns compared (9c3a59771cf8499d92972f5c7873e280) (rank 1).

champion (rank 1) artifact: models:/m-34712ed5add845fe838835bd2107ec47 (artifact size: 1,032,894 bytes)
challenger (rank 3) artifact: models:/m-2f7ebb90292f405c9482ef5e6e0bd90a (artifact size: 23,109,193 bytes)
Comparison ID: 81dbb7f314e29c815c09a468c85f32a8f3079a452f6d9aa5f75614f7c85eb965
```

## `< 50 MB` (`< 50,000,000 bytes`) Check

- **Rank 1 (`HistGradientBoostingClassifier`, `9c3a59771cf8499d92972f5c7873e280`):** `1,032,734` bytes (harness) / `1,032,894` bytes (artifact) — **under 50,000,000 bytes** (elected `champion`).
- **Rank 2 (`RandomForestClassifier`, `9a40f8f5e35a48ddb39a97f9f052dcad`):** `187,236,750` bytes (harness) — **exceeds 50,000,000 bytes** (`>50 MB`; passed over for registration under D15-03).
- **Rank 3 (`ExtraTreesClassifier`, `8af9db9f95e543cfbfebdef82545322c`):** `23,109,033` bytes (harness) / `23,109,193` bytes (artifact) — **under 50,000,000 bytes** (elected `challenger` under D15-03).

Under [D15-03](../decision-log.md#d15-03--register-the-best-two-ranked-models-within-50-mb), the two elected candidates (`champion` rank 1 and `challenger` rank 3) are both strictly under `50,000,000` bytes. By raw validation rank, the top two (`HistGradientBoostingClassifier` and `RandomForestClassifier`) do **not** both satisfy `< 50 MB`. Registry writes were not performed during this read-only run; historical `fraud-candidate` version 2 remains registered until an analyst-approved `--register` operation is executed.
