# Reproduce the SGD impact report (PBI #34)

## Synthetic run (no data files)

From the project root, with the repository's locked development environment:

```sh
uv run --locked python -m mlops_fraud_prediction_ca1.impact_report \
  --synthetic --k 2 --capacities 1 2 3 --output-dir /tmp/pbi34-synthetic
```

Choose a new output directory each time. The command writes `report.md`,
`capacity.csv`, `capacity.png` and `receipt.json`. The report labels the run
synthetic. The receipt records K, chart capacities, the input SHA-256 and the
fixed bootstrap seed and draw count; the report includes the seed as well.
Two runs with the same code, locked environment and inputs reproduce the
aggregate report and capacity values. No per-order values go in the outputs.

## Final run: authorized #23 export only

Do not unlock or score the final window to run this command. It does not
train, score, query a database or compute rule or loyalty flags. The
authorized evaluator supplies **one already-scored #23 export** after the
freeze and may run the command in their authorized environment. It must come
from the recorded final champion and the frozen #18 policy. Keep the export
out of Git and do not share it with reviewers.

### Amount check before the #23 claim

The #18 baseline leaves unusable amounts unflagged, while model selection can
still catch fraud on those orders. An unknown amount contributes nothing to
the #18 validation sum, so a zero sum can hide incomplete exposure. #34
requires a finite numeric SGD amount for every aligned order, including
currently unflagged orders that could be selected at another review capacity.
A recorded zero passes.

After the freeze and mode checks, the #23 producer must prepare the authorized
cleaned orders, verify that their unique `(country_code, order_id)` keys
match the scoring cohort one-to-one, and run the amount check **before claiming
the window, fitting, or predicting**. Call it on the same aligned cleaned
frame that will be used for the comparison and export.

```python
from mlops_fraud_prediction_ca1.impact_report import preflight_order_amounts

preflight_order_amounts(orders)
```

On `ValueError`, stop before the claim and use the aggregate diagnostic to
check source amounts and cleaning. Keep the cohort intact and use verified
source amounts when resolving a failure. The check does not validate
alignment, authorization or ledger state. The claim must still check under
its lock that the window is unused.

The #23 draft loads its inputs after claiming the window. The #23 owner must
establish authorized pre-claim input access, wire and rehearse the check, and
retain the same private scored export with durable run provenance and digest
so report retries can use those scores. Until those steps are verified, the
consumer still rejects an invalid export, potentially after the one-shot
window has been claimed.

### CSV and report

Required CSV columns (one row per uniquely keyed order):

- `order_date`, `country_code`, `order_id`: #5 selection date and order key.
- `customer_id`: customer key within the country, blank if unlinked.
- `order_value`: finite SGD amount; `score`: finite champion score.
- `is_fraud`: supplied binary label (0 or 1, never a missing label).
- `loyal`, `rule_flag`: precomputed frozen booleans, before the rule's K cut.
  Use `true`/`false` or `1`/`0`.

The K below is the current 250/day assumption; substitute the frozen K and
record the capacities used. Supply the published #23 MLflow run ID, the
full freeze commit, and the prior formal test-window comparison count
verified in the run records. Do not infer that count from earlier EDA or
treat an unknown count as zero.

```sh
uv run --locked python -m mlops_fraud_prediction_ca1.impact_report \
  --scored-csv "$APPROVED_SCORED_EXPORT" --evaluation-run-id "$FINAL_RUN_ID" \
  --freeze-commit "$FREEZE_COMMIT" --prior-comparisons "$PRIOR_COMPARISONS" \
  --k 250 --capacities 50 100 250 500 --output-dir /tmp/pbi34-final
```

Retain the receipt and aggregate artifacts with the authorized final run
record, including seed 34 and the commit that produced the report. The
command does not upload anything to MLflow. Verify artifact identity,
freeze and comparison history before running: the receipt records supplied
provenance, it does not authenticate it. The final run and its evidence
remain pending.

See [D34-01](decision-log.md#d34-01--criterion-for-reading-the-model-versus-rule-result)
for the paired bootstrap and limits. Unlinked fraud is counted in exposure
and shown separately. The interval holds its contribution fixed and does
not quantify uncertainty from unlinked orders; if there are no linked
customers, there is no customer bootstrap interval.
