# Fixed temporal reads (PBI #6)

`config/time_split.toml` is the versioned date and access configuration.
Default resolution uses the source checkout, independent of the working directory.
Installed consumers outside this checkout must provide `config_path` explicitly;
no missing/malformed configuration falls back to an unlocked read.

```python
from mlops_fraud_prediction_ca1.data import load_base_data, load_category_data

customers, training_orders, labels = load_base_data("data", purpose="train")
_, validation_orders, validation_labels = load_base_data("data", purpose="validation")
# Default purpose="analysis" excludes exactly the frozen test interval.
# Category spellings and blank tokens use the same protected read boundary.
categories = load_category_data("data")
print(training_orders.attrs["split_audit"])
```

| Purpose | Calendar eligibility |
| --- | --- |
| `train` | `[2022-11-12, 2023-03-27)` regardless of unlock |
| `validation` | `[2023-03-27, 2023-04-17)` regardless of unlock |
| `analysis`, locked | On/after 12 Nov, outside `[2023-04-17, 2023-05-11)` |
| `final_evaluation` | Only `[2023-04-17, 2023-05-11)`; requires `final_evaluation_enabled=true` |
| `analysis`, unlocked | All valid dates on/after 12 Nov |

The 2,341 orders on 11 May are excluded from frozen evaluation and reported
separately. Generic analysis includes that day and later batches. It must never
be substituted for `train`. After final evaluation is logged, an explicit
`test_locked=false` change can widen analysis. Train and validation remain fixed.
Release training is outside this API's current scope.

`split_orders(frame, purpose=..., config_path=...)` in `splitting.py` is the
single in-memory split entry point. It returns only allowed rows plus aggregate
membership/exclusion counts. It preserves order/index/values and never silently
deduplicates source keys. Supported date strings are those in `temporal.parse_dates`;
SQL calendar dates and naive Python/pandas datetimes are also supported by day.
Timezone-aware values and missing/unrecognized/invalid dates are withheld.
Rows before the fixed start are always withheld.

CSV parsing is private ingestion; only scoped payloads reach consumers. Customer
and order identifiers are read as strings before scoping, preserving distinct
opaque keys such as `001` and `1` in both ordinary and literal reads. Labels
have no date: they are selected by normalized `(country_code, order_id)` from
allowed orders, customers by `(country_code, customer_id)` referenced in those
labels. Blank/missing keys cannot link. If the same order key appears on both
sides of a boundary, its related payload is withheld rather than guessed.
Other key conflicts remain a downstream cleaning obligation. Customers shared
across windows can remain visible. Customer features still require #4's policy.
All returned frames carry `attrs["split_audit"]`. This contains `purpose`,
`test_locked`, `final_evaluation_enabled`, the four resolved ISO date strings in `cutoffs` (`train_start`,
`validation_start`, `test_start`, `test_end_exclusive`), and aggregate row and
membership counts. The cutoffs come from the same loaded policy that selected
orders, including an explicitly supplied custom configuration. Audits contain
no order IDs, related payloads or withheld rows.

Consumers should record this complete metadata with the consumer's code revision
and the revision of the supplied configuration in their existing run log. Equal
row counts do not establish equal membership; use the resolved cutoffs and lock
state to identify the policy actually applied. Code/configuration revisions are
caller-supplied provenance, not inferred by the loader. For example:

```python
run_metadata = {
    "code_revision": code_revision,  # revision of the running consumer
    "config_revision": config_revision,  # revision of its supplied config
    "split_audit": training_orders.attrs["split_audit"],
}
```

The SQL predicate return signature remains `(predicate, parameters)`; it does
not return a loader audit or establish adapter enforcement.

These reads establish calendar exposure only. Training consumers must validate
labels (zero means not fraud; unmatched stays unknown), exclude conflicting
keys, and enforce feature availability and train-only preprocessing. The retrospective
evaluation does not certify labels existed at historical fit time. D4-04's
assumed 30 days applies to label-history eligibility, not an automatic target
maturity certificate or required gap between these evaluation windows.

Final evaluation requires both `purpose="final_evaluation"` and a versioned
configuration with `final_evaluation_enabled=true`. The setting defaults to
false and is independent of `test_locked`; unlocking analysis does not enable
final evaluation. Enable it only after model, threshold and preprocessing freeze.
Both in-memory selection and SQL-predicate access log the resolved test interval
and ordinary lock state at WARNING level. Loader audits also include the setting.
The caller must retain logs and satisfy the freeze prerequisites. No scoring is
performed by this module. This is a workflow guard, not filesystem access control.
Private ingestion helpers are not consumer APIs.

## SQL integration contract

```python
from mlops_fraud_prediction_ca1.splitting import sql_date_predicate

predicate, params = sql_date_predicate(
    purpose="train", config_path="config/time_split.toml"
)
# Static trusted query and fixed column name; pass params separately to DB-API.
cursor.execute("SELECT * FROM orders WHERE " + predicate, params)
```

The adapter must expose `order_date` as a validated nullable SQL `DATE`, not raw
text. Parse the three known string formats explicitly during ingestion and map
bad dates to NULL; verify #10's actual schema before integration. No identifier
or date is interpolated from caller input. `%s` is the default DB-API placeholder;
SQLite tests translate these placeholders locally and execute the predicates
against independent expected IDs. They do not establish live RDS integration.
Undated labels/customers require the same normalized composite-key scoping as
CSV; downstream/private consumers must adopt and verify that contract themselves.

## Evidence and notebook provenance

The notebook's earlier full-extract EDA is historical and predates the test lock.
It does not establish an untouched holdout. Ordinary notebook reads use the
protected loader. Window counts and rationale are retained in the
[window evidence in PR #84](https://github.com/Glowing6635/MLOps-Fraud-Prediction-CA1/pull/84) and
[decision log](decision-log.md). No duplicate preview implementation is required.
