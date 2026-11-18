# Category standardization

`standardise_categories` in `mlops_fraud_prediction_ca1.categories` returns
original values, proposed values, frequencies and validation status. It leaves
the supplied Series unchanged. Callers inspect the returned report to decide
which proposed values to use; the function never applies changes to its input.

## Formatting supported by observed evidence

Raw CSV inspection found case variants:

| Field            | Evidence                                                                                          | Proposed formatting |
| ---------------- | ------------------------------------------------------------------------------------------------- | ------------------- |
| Customer country | `pk` (225), `my` (165), alongside uppercase codes                                                 | Uppercase           |
| Label country    | `ph` (180), `my` (121), alongside uppercase codes                                                 | Uppercase           |
| Collection type  | `delivery` (85), alongside `Delivery`                                                             | Lowercase           |
| Payment method   | `Online Banking` (103), `payment on delivery` (20), alongside differently capitalized equivalents | Lowercase           |

Counts describe the current local base CSVs; refresh them when inputs change.
Matching after case conversion identifies candidate equivalence, not proof of
business validity. Case standardization does not establish an allowed set.

## Precautions and validation safeguards

- Trimming boundary whitespace is enabled by default as a precaution. The
  preceding inspection did not establish whitespace variants in these fields.
- Missing values remain missing. Empty strings remain visible as `blank`;
  they are not silently replaced or assigned to another category.
- An explicit allowed set marks nonblank proposals `allowed` or `unexpected`.
  Without one, their status is `unchecked`. Unexpected values are retained.
- Original representations and their counts remain separate in the report,
  even when they have the same proposed value. No rows are dropped.
- Alias keys, alias targets and allowed values must follow the selected
  formatting rules; inconsistent configuration raises an error.
- Read category columns as raw strings to preserve CSV spelling. The function
  cannot recover representations already changed by import inference.

## Meaning changes require confirmation

`mobile_verified` contains `yes` (165), `True` and `False`. Mapping `yes` to
`true` is a semantic decision requiring confirmation, rather than a case fix.
Likewise, `on Delivery` (160) must not automatically become
`payment on delivery`. No dataset-specific aliases or allowed sets are enabled
by this function. Alias mappings are explicit and applied once, not chained.

```python
import pandas as pd

from mlops_fraud_prediction_ca1.categories import standardise_categories

raw_country_codes: pd.Series = pd.Series(["MY", "my", " PK "], dtype="string")
preview: pd.DataFrame = standardise_categories(raw_country_codes, case="upper")
```

The report preserves the source representations alongside proposed values.
Callers apply selected proposals explicitly in their preparation pipeline.
