# PBI #3 review evidence — 2026-11-15

## Completion verification — 2026-11-15

The repaired and improved notebook is now executed and complete for the four
PBI #3 criteria. The user explicitly approved the three rendered presentation
previews before inclusion. Final saved notebook/Python inputs agree exactly:
60 cells, 21 code cells executed sequentially in a fresh project `.venv` kernel
using the actual NotebookHelpers magic, zero error outputs, three inspected
figures and 34 HTML outputs. This verifies the saved notebook through nbclient;
no notebook-editor UI interaction was tested.

The recorded baseline findings below match the final saved outputs. The single
semantic inventory contains all 21 columns; expandable evidence retains every
raw category value and all 162 country/aggregate weekly rows. Formatting checks
confirm integral counts, fractional-rate presentation, explicit undefined values
and numeric alignment. All navigation anchors resolve. The three source CSV
SHA-256 values are unchanged and `git diff --check` passes. No split, training,
prepared-data export, staging, commits or issue changes occurred.

Both reproduced temporal-helper edge cases below were fixed and covered by new
regressions. The full suite passes: 59 tests, 9.330 seconds. The earlier review
sections remain historical evidence and no longer describe outstanding defects.

One independent read-only final review found no defects in the final diff and
saved outputs. It checked criterion coverage, complete inventories, weekly
denominators and figures; test execution and source hashing remain lead checks.

## Original review findings — before repairs

The following sections record the original review baseline. Outstanding work
and defects described here were resolved in the completion verification above.

### Original review scope and conclusion

Reviewed the notebook pair, `data.py`, `categories.py`, `numeric.py`, `quality.py`,
`relationships.py`, `logic.py`, `temporal.py`, `scripts/preview_pbi3.py`, tests,
README/setup/sync/category/contributor docs, the CA1 spec and related PBIs.
Two read-only subagents independently checked correctness and narrative/visuals.

All four criteria are present in the Python representation. The saved notebook
is not completion-ready: pasted terminal text causes a SyntaxError and missing
Markdown removes a handling explanation. Accepted improvements are tracked in
[the plan](pbi-3-data-analysis.md). No analytical code was changed by this review.

### Original checks and limitations

- Review baseline HEAD: `735f440d62ff6314c76d5f508aa3897848a629e2`.
- `.venv/bin/python -B -m unittest discover -s tests -q`: 57 passed, 9.536 s.
- All analysis Python sources passed AST parsing.
- Jupytext comparison: 59 source cells versus 58 saved notebook cells. One
  missing Markdown cell and one corrupted code cell; shifts are not dozens of
  independent differences. Missing findings: Python lines 326–337. Broken saved
  cell19 has `),hing` and `GPT-6.1-Sol medium · …`, around JSON lines 1019–1021.
- Saved error: `SyntaxError: invalid character '·' (U+00B7)`; execution count 7.
- All 21 Python-source code cells passed sequentially in fresh in-memory IPython
  in ~146 s, using the actual data and extension. The headless harness supplied
  an `enable_gui` no-op for inline plotting; it was not a notebook editor/kernel
  end-to-end check. Both generated figures were inspected; review temp files
  were removed. The repaired saved notebook still needs its own fresh execution.
- No `to_csv`, `to_parquet`, `to_sql`, prepared-data writer or train/test splitter
  exists in the EDA notebook/helper path. Working-copy joins/parsing serve audits.
- Live GitHub #3/#4/#5/#6/#10/#12 bodies matched local snapshots during review.
  Other issue references below used local `.local/pbis/` snapshots; recheck if
  implementation depends on a changed issue.

### Baseline findings reconciled with final outputs

These are observed base-data findings, not invariants for future batches.

#### Data quality and distributions

| Topic | Evidence / handling implication |
| --- | --- |
| Dimensions | customers 901,312 × 9; orders 949,889 × 8; labels 950,318 × 4. |
| Missingness | Customer first-order timestamp 712; payment method 21,501; item count 1. All other loaded columns zero under pandas inference. Preserve unknowns; train-fit any imputation. |
| Copies and keys | No exact duplicate rows. Customer IDs globally unique. Three order IDs each span two countries; composite order keys unique. |
| Categories | `mobile_verified`: 900,681 True, 466 False, 165 `yes`; keep `yes` unresolved. `on Delivery` 160, not automatically aliased. Payment blanks 21,501. |
| Numeric distributions | Eight right-skewed features, skewness 5.42–55.00; cancelled/refunded counts 83.14%/85.76% zero, zero IQR. Keep values; flags are investigation screens. |
| Refund exceedances | 199,687 / 949,889 (21.0221%); retain/flag, no silent clamp or rejection. |
| Other logic | No cancelled/refunded count above total; no selected numeric negatives. Order-before-first-day 105 / 949,101 evaluable, 788 excluded. |

#### Linkage and coverage

| Topic | Evidence / handling implication |
| --- | --- |
| Safe linkage | orders → labels → customers, using normalized country and validated keys; no direct order/customer ID join. |
| Orders → labels | 949,874 / 949,889 matched; 15 unmatched. |
| Labels → orders | 949,874 / 950,318 matched; 444 unmatched. |
| Labels → customers | 950,304 / 950,318 matched; 14 unmatched. |
| Customers → labels | 622,258 / 901,312 matched; 279,054 unmatched. |
| Country disagreement | Unequal sets fall from 303 to 2 for orders/labels and 569 to 0 for labels/customers after normalization; multicountry IDs remain ambiguous. |
| Safe bridge | All 949,889 safe orders retained; 949,874 labelled, 949,872 also linked to customers. |

#### Labels, dates and fraud patterns

| Topic | Evidence / handling implication |
| --- | --- |
| Raw class balance | 104,504 fraud / 950,318 valid labels = 10.9967%; 845,814 nonfraud, no missing/invalid labels. |
| Order dates | 2022-11-12–2023-05-11; formats `%Y/%m/%d` 949,180, `%Y-%m-%d` 200, `%Y/%b/%d` 509; all parse. |
| Customer dates | 900,600 valid, 712 missing; range 2012-03-25 12:53:43–2023-05-10 20:59:37; `%Y-%m-%d %H:%M:%S`. |
| Country rates | BD 29.2706%, MY 3.8950%, PH 8.4721%, PK 24.5201%, TH 5.9677%; total 104,422 fraud / 949,874 valid matched-order labels. |
| Aggregate movement | Week 2023-03-06: 4,540/38,491 = 11.7950%; week 2023-03-13: 5,295/37,186 = 14.2392%; +2.4443 percentage points. Descriptive candidate only. |

Country case examples: customers `pk` 225 / `my` 165; labels `ph` 180 / `my` 121.
Initial/last weeks are partial; missing weeks must not be compared as adjacent.
Full weekly tables are about 162 rows across aggregate and five country series.
Only numerical distributions currently have plotting code. Full-range boxplots
compress the central bodies; their accompanying P99 histograms disclose excluded
tails and use logarithmic count axes. No new weekly visual was previewed yet.

### Originally reproduced helper edge cases — now fixed

1. `temporal.py:79`, `fraud_rates`: a label Series created with
   `pd.Series([np.int64(1), np.int64(0)], dtype=object)` produces zero valid labels
   and two invalid labels. Accept appropriate numeric scalar types, excluding
   Booleans. Current loaded base-label dtype avoids this case.
2. `temporal.py:131`, `largest_weekly_change`: otherwise valid unique country/week
   rows with index `[0, 0, 0]` raise `TypeError` when label selection returns a
   Series. Reset indices or select positionally. Current notebook concatenates
   with `ignore_index=True`, avoiding the problem.

### Research used to select the original review checks

This was a requirements-driven review, supplemented by published guidance on
notebook evaluation and communication. These sources justify the checks and
recommendations; they do not establish measured benefits for this project.

- [Rule et al., computational notebooks](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1007007): audience-oriented narrative, meaningful cell boundaries, modular code and dependency documentation. Applied to introduction/conclusion, duplicate tables and separation of reusable logic from explanation.
- [ONS line charts](https://service-manual.ons.gov.uk/data-visualisation/chart-types/line-chart): temporal patterns, small multiples, gaps and annotations. Applied to weekly-rate panels rather than six long tables or a crowded multiline chart.
- [Government Analysis Function tables](https://analysisfunction.civilservice.gov.uk/policy-store/data-visualisation-tables/): compact demonstration tables versus detailed reference data, consistent precision and numeric alignment. Applied to consolidated summaries while retaining audit evidence.
- [NIST outlier guidance](https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h.htm): unusual observations need investigation, not automatic deletion. Retained existing IQR caveats; no additional formal outlier test required.
- [Scikit-learn leakage guidance](https://scikit-learn.org/stable/common_pitfalls): split before learned preprocessing and fit only on training data; pipeline can package transformations with models. Applied to the later training workflow, not an obligation to split this EDA notebook.

Method choice: keep the existing analysis and strengthen the actual delivered
notebook plus its visual evidence. Tables alone cover literal weekly reporting,
but hinder pattern inspection; a small chart package adds the missing feedback.
Do not replace this with a full modelling pipeline or formal change-point study
inside #3. Existing requirements and guidance resolve the actionable choices;
no new comparative experiment or higher-level methodology study is warranted.

## Project boundaries and references

- [#3](https://github.com/Glowing6635/MLOps-Fraud-Prediction-CA1/issues/3): four EDA criteria, not a complete training/cleaning implementation.
- [#4](https://github.com/Glowing6635/MLOps-Fraud-Prediction-CA1/issues/4): feature availability audit; date checks do not prove all historical fields are safe.
- [#5](https://github.com/Glowing6635/MLOps-Fraud-Prediction-CA1/issues/5): daily top-K fraud caught is primary; average precision is the model tie-break.
- [#6](https://github.com/Glowing6635/MLOps-Fraud-Prediction-CA1/issues/6): uses #3 temporal evidence; owns fixed split configuration and subsequent test exclusion. Initial exploration precedes it; record the boundary interpretation there.
- [#10](https://github.com/Glowing6635/MLOps-Fraud-Prediction-CA1/issues/10): schema uses #3 key findings and decimal money.
- [#12](https://github.com/Glowing6635/MLOps-Fraud-Prediction-CA1/issues/12): deterministic cleaning, unknown-category policy, observed format tests and before/after demonstration.
- Local #13/#16/#20 snapshots: training harness, train-fitted feature transformations and batch persistence are distinct responsibilities.
- CA1 `spec.txt:218–224,398–405`: overall coursework needs discovery plus automated cleaning/transformation; #3 completion alone does not complete that broader requirement.

The user's accepted outcome is an EDA notebook plus later shared configurable
modules, not a notebook that silently becomes the training or persistence engine.
