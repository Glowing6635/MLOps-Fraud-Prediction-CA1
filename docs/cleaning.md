# Deterministic cleaning — PBI #12

`clean_frame` takes one table and returns its cleaned columns plus original
values in `raw_<column>` columns. Every row and its original index are retained.
Rules use a fixed vocabulary and explicit formats. There are no learned
statistics, joins, imputation, scaling or row rejection rules.

```python
from mlops_fraud_prediction_ca1.cleaning import clean_frame

cleaned_orders = clean_frame(orders)
# order_value is numeric; raw_order_value preserves the supplied value.
```

Generated raw-column names must not collide with input columns. A collision
raises rather than overwriting data. IDs remain opaque text: `001` differs
from `1`. Preserve source/batch identity separately when loading multiple tables.
Feature approval and train-only transformations remain consumer responsibilities.

## Rules and numeric types

| Fields | Cleaned representation | Unknown handling |
| --- | --- | --- |
| Country codes | Uppercase, boundary spaces removed | Missing, blank or unseen becomes unknown |
| Collection/payment categories | Lowercase, boundary spaces removed | Missing, blank or unseen becomes unknown |
| Mobile verification | Nullable `boolean` (`True`/`False`/`pd.NA`) | `yes`, blanks and missing remain unknown |
| Order dates | Parsed `datetime64[ns]` | Invalid or unsupported format becomes `NaT` |
| First-order timestamps | Parsed `datetime64[ns]` | Invalid or unsupported format becomes `NaT` |
| Eight numeric amount/count fields | Nullable `Float64` | Invalid, Boolean, nonfinite or unsupported precision becomes unknown |
| `is_fraud` | Nullable `Int8`, only zero/one | Other labels, Boolean and missing values become unknown |
| IDs and other columns | Original values | No inferred coercion |

Numeric fields convert a whole column at a time. Finite negatives, fractions,
outliers and refunds are retained inside the supported range. To avoid silent
rounding of large integers in model inputs, magnitudes at or above `2**53`
become unknown with `unknown_precision` status; their exact originals remain
available. This bound is fixed across batches. Nullable numeric arrays can be
converted to model arrays with an explicit missing-value convention; learned
missing-value handling must still fit on training rows only.

Order formats are `YYYY/MM/DD`, `YYYY-MM-DD` and `YYYY/Mon/DD`, with
case-insensitive English month abbreviations. Customer timestamps require
`YYYY-MM-DD HH:MM:SS`. Padded dates and native database datetime objects are
unsupported: database adapters must serialize to the documented format and
verify their driver boundary.

## Versioned category vocabulary

`CLEANING_POLICY_VERSION = "pbi12-v3"` names the fixed policy for numeric
model dtypes, unsupported-precision handling, binary labels and nullable Boolean
mobile verification. Category normalization reuses `standardise_categories`.

Sources are the supplied field definitions and examples and the historical PBI #3 category
inventory. The [category decisions](category-standardisation.md) describe
unresolved representations. Earlier EDA inspected the full extract, including
the eventual test window; this reuse does not establish a train-only vocabulary
or an untouched holdout. No incoming batch automatically expands the vocabulary.

- Country: `MY`, `PH`, `PK`, `TH`, `BD`.
- Collection: `delivery`, `pickup`, `self-pick-up`. The last two remain distinct.
- Payment: `payment on delivery`, `cybersource`, `online banking`, `credit card`,
  `antfinancial gcash`, `generic creditcard`, `balance`, `antfinancial bkash`,
  `antfinancial truemoney`, `boost wallet`, `jazzcash wallet`, `invoice`,
  `adyen hpp molpay`, `no payment`, `cybersource applepay`, `paypal`,
  `xendit directdebit`.
- Verification: `true`, `false`. `yes` and `on Delivery` remain unknown.

A future vocabulary change requires a version, source and reason. Consumers
should log the policy version. Labels never choose cleaning rules.

## Diagnostics and protected base reads

`clean_table` returns aligned `raw`, `cleaned` and per-cell `status` frames.
Callers needing one DataFrame use `clean_frame` directly. `clean_batch` applies
the same rules independently to customers/orders/labels and records the policy
version. Keep source/batch identity separately. Consumers must validate key
completeness and uniqueness and avoid joining unknown keys.

`load_clean_base_data(data_dir, purpose="train")` calls the protected CSV reader
before cleaning. It cannot expose withheld test rows through ordinary reads.
The existing notebook demonstration uses guarded analysis scope; its reference
counts for the recorded base files are:

| Table | Before | After | Rejected | Unknown cells |
| --- | ---: | ---: | ---: | --- |
| Customers | 554,509 | 554,509 | 0 | 102 mobile verification + 416 first-order dates |
| Orders | 824,898 | 824,898 | 0 | 17,009 payment methods |
| Labels | 824,885 | 824,885 | 0 | 0 |

The reader separately withholds 124,991 test-window orders. Analysis includes
2,341 orders on 11 May outside the frozen evaluation interval. Unknown counts
refer to cells, not distinct rows. Original CSVs remain unchanged.

The local API is available to #13 and #20; their later adoption is separate
integration work. This module does not implement a database loader, choose
model inputs or certify live-driver integration.

Tests live beside package code. Run `uv run pytest` from the repository root.
The suites cover row retention, originals, duplicate indices, numeric dtypes,
precision, determinism, fixed vocabularies, unknowns and protected-reader
integration, including test-only sentinels.
