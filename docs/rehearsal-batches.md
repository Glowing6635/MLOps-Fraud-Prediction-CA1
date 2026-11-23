# Synthetic rehearsal batches (PBI #24, practice-batch item)

`mlops_fraud_prediction_ca1.rehearsal` generates fully synthetic practice
batches in the base file format. Values come from seeded random generators, so
no real order, customer or label is read or copied. Nothing is committed: the
CSVs are written on demand to a directory you choose.

```bash
uv run python -m mlops_fraud_prediction_ca1.rehearsal --out output/rehearsal
```

Each batch is a directory whose three files share the suffix that #20 uses as
`batch_id`, with the base-file columns:

| Batch | Files | Purpose |
| --- | --- | --- |
| `rehearsal_a`, `rehearsal_b` | `customers_<id>.csv`, `orders_<id>.csv`, `labels_<id>.csv` | Valid, distinct (disjoint IDs and values), 40 customers and 120 labelled orders each |
| `rehearsal_edge` | same | Small: one unseen payment method on an otherwise valid row, and rows flagged for rejection |

Edge batch cases (identifiers all start with `SYN-EDGE-`):

- **Unseen category:** the first order has payment method
  `synthetic unseen wallet`. Per #19 it must not stop the batch; it goes to
  cleaning (unknown).
- **Label whose order is missing:** `SYN-EDGE-ORPHAN-ORDER-01` and `-02` have
  labels but no orders row. #20: rejected and counted, no abort.
- **Label whose customer is missing:** order `SYN-EDGE-O-NOCUST` is labelled
  to `SYN-EDGE-C-ABSENT`, which has no customers row. #20 names this case too.

These flags record what #20 says should happen. They are not verified loader
behaviour: the loader does not exist yet, and this change does not define its
rejection policy. Verify against the real command once #20 lands.
