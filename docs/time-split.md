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
protected loader. The tables below retain the recorded base-extract evidence;
they do not enable ordinary access to test rows. [D6-01](decision-log.md#d6-01--frozen-retrospective-windows-and-protected-exposure)
records the policy and its limitations.

## Frozen windows and country mix

Intervals are half-open. Fraud rates exclude missing labels; country counts
include every safe order. These counts describe the recorded base extract.

| Window | Dates | Orders | Valid labels | Fraud | Missing labels | Fraud rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Train | 12 Nov–before 27 Mar | 721,364 | 721,354 | 75,449 | 10 | 10.4594% |
| Validation | 27 Mar–before 17 Apr | 101,193 | 101,190 | 13,066 | 3 | 12.9123% |
| Test | 17 Apr–before 11 May | 124,991 | 124,989 | 15,584 | 2 | 12.4683% |
| Excluded final day | 11 May | 2,341 | 2,341 | 323 | 0 | 13.7975% |

Country mix as counts (denominator is each row's order total; retain these
counts when producing percentages so the mix is reproducible):

| Window | BD | MY | PH | PK | TH |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 45,216 | 285,680 | 208,007 | 136,393 | 46,068 |
| Validation | 6,567 | 35,422 | 32,835 | 18,196 | 8,173 |
| Test | 7,598 | 38,378 | 39,592 | 30,096 | 9,327 |
| Excluded final day | 179 | 1,089 | 777 | 137 | 159 |


## Rationale and limits

- Counts reconcile to 949,889 orders and 949,874 valid matched labels.
  Fifteen unmatched targets remain unknown; The supplied field definitions define zero as not fraud.
- Before 13 March, training contains 650,711 orders, 650,702 valid labels and
  65,650 fraud (10.0891%). On/after that date it contains 70,653 orders,
  70,652 valid labels and 9,799 fraud (13.8694%). The March movement is
  descriptive; no model metric selected the windows.
- The extract covers 181 observed dates, 12 November–11 May. The final day's
  2,341 orders are excluded conservatively. Date-only data cannot prove the
  day is incomplete. Ordinary analysis can include it outside the test window.
- Prior full-extract EDA was disclosed. The evaluation is retrospective;
  historical label availability and external training chronology are unknown.
  The 30-day assumption governs label history, not historical target maturity.
- Dates were committed on 16 November; the 12 November deadline was missed.
  Local CSV/SQL-contract tests do not establish live RDS or downstream adoption.
- `test_locked=true` and `final_evaluation_enabled=false` are the defaults.
  Explicit final-evaluation access is logged and leaves ordinary reads locked.
  No model or final scoring was run to produce this evidence.
