# PBI #3 completion plan

## Current state and objective — 2026-11-15

This plan supersedes the old scaffold/first-missingness checklist. The analysis
already covers all four criteria in the Python representation. Finish a coherent,
executed EDA notebook that communicates findings and preparation decisions; leave
it uncommitted for review. Read [the handoff](../../HANDOFF.md) for checkout state,
authorization and verification limits, and [review evidence](pbi-3-review-evidence.md)
for findings, baseline values and source links.

Source: [PBI #3](https://github.com/Glowing6635/MLOps-Fraud-Prediction-CA1/issues/3).
No model training, new split selection, RDS work or MLflow design is part of this
completion unit. Later workflow direction is recorded below, not silently added
to this session's implementation scope.

Progress: implementation and verification complete. The user explicitly approved
all three rendered previews on 2026-11-15; they are integrated. The paired
notebook has 60 matching cells, all 21 code cells executed in a fresh project
kernel with no errors, three inspected saved figures, and complete supporting
inventories/162 weekly rows. All 59 tests pass; source CSV hashes are unchanged.
Changes remain uncommitted and the GitHub issue has not been modified.

## 1. Reconcile the notebook pair

- [x] Recheck current diff/status; preserve existing user changes.
- [x] Compare cells semantically rather than by position alone: the missing
  Markdown shifts subsequent cell numbers.
- [x] Remove terminal contamination from saved notebook cell 19 using the clean
  duplicate-check code in `notebooks/data_analysis.py` around line 349.
- [x] Restore section 3.1.1 missingness findings from Python lines 326–337.
- [x] Reconcile both representations deliberately. The script has 59 cells;
  the saved notebook has 58 at the review baseline. Do not use timestamps alone.
- [x] Keep raw inputs unchanged; do not invoke the pre-commit hook, which stages.

## 2. Complete accepted communication improvements

- [x] Shorten introduction; add a short findings overview before setup, with
  links to supporting sections. No assessment/PBI terminology in reader prose.
- [x] Remove layout/wrapping instructions from sections 3.3/3.4; consolidate
  repeated source-preservation statements in Workflow. Keep local denominator
  exclusions and distinctions between proposals and demonstrated working copies.
- [x] Keep a compact main-path category exception/decision summary. Preserve
  complete raw inventories as supporting detail, including counts and spellings.
- [x] Consolidate duplicated dtype information into a clear semantic-type audit.
- [x] Improve `show_report`: readable labels, integer counts without accidental
  decimal formatting, consistent precision, right-aligned numeric columns and
  explicit undefined values. Retain exact counts and underlying calculations.
- [x] Preview new/consolidated tables in chat before notebook inclusion.
- [x] Render a weekly aggregate fraud-rate chart and five country panels with
  consistent axes; annotate the dated March movement, disclose partial boundary
  weeks and denominators, and leave gaps for missing weeks. Avoid a crowded
  six-line plot. Retain full weekly counts/rates as supporting evidence.
- [x] Show that actual chart preview in chat and obtain inclusion approval.
  All three previews were rendered in chat and explicitly approved before inclusion.
- [x] Refine numerical figures: monetary axes in SGD, integer count ticks, explain
  collapsed zero-IQR boxes. Preserve P99-view exclusions and full-range evidence.
  Additional body-scale insets are optional, not a completion requirement.
- [x] Use bold sparingly for decisive findings/decisions; code formatting for
  column names; no decorative italics or repeated bold number inventories.
- [x] Simplify navigation and finish with a short preparation/limitations handoff.

Preserve: IQR flags are not proven errors; per-feature outlier rates cannot be
summed; raw-label and matched-order denominators differ; the largest weekly
movement is descriptive, not a proven structural change or cause. A formal
change-point algorithm is not required just to complete #3.

Suggested introduction: “This notebook profiles SP-buy’s customer, order and
fraud-label files to identify risks to fraud modelling and decide how to handle
them. It examines data quality, safe joins, business rules and fraud patterns
over time.”

## 3. Address confirmed helper issues proportionately

- [x] `temporal.py:79`: accept valid NumPy numeric 0/1 scalars in object Series;
  keep Boolean labels invalid. Add a meaningful regression fixture.
- [x] `temporal.py:131`: avoid label-based selection assuming unique DataFrame
  indices in `largest_weekly_change`; test duplicate indices with unique weeks.

These accepted low-priority fixes are independent of current results. If deferred
from the completion unit, state that explicitly rather than treating them as
criteria failures. Do not invent a new country-category domain or require a
complete malformed-batch pipeline before closing the initial EDA.

## 4. Verify the finished deliverable

- [x] Fresh kernel, execute all saved notebook cells top to bottom; save outputs.
  Use `.venv` and the actual NotebookHelpers magic. In-memory script verification
  already passed but does not satisfy this remaining check.
- [x] Inspect rendered plots and tables for clipping, overload, units, precision,
  denominator visibility and narrative agreement. Check navigation links.
- [x] Confirm notebook/Python cell agreement after saving, including magics.
- [x] Verify key findings against executed outputs; update static prose if any
  input or result differs. Reconcile counts and source exclusions.
- [x] Run focused regression tests for changed helpers, then the existing suite:
  `.venv/bin/python -B -m unittest discover -s tests -q`.
- [x] Confirm source CSVs unchanged and no accidental prepared-data exports,
  index changes or unrelated edits. Clean only task-created temporary artifacts.
- [x] Report all four acceptance criteria as demonstrated only when integrated
  outputs support them. Leave issue status and commits unchanged.

## Later configurable workflow: accepted direction

The EDA notebook presently has no split and saves no prepared dataset. Keep it an
explanatory consumer of reusable reporting functions. A later entry point calls
package modules directly; it must not depend on notebook execution/kernel state.

| Owner PBI | Responsibility and dependency |
| --- | --- |
| #3 | Initial historical EDA, preparation decisions and linkage/date evidence. |
| #4 | Per-column availability/leakage audit; date sanity alone does not validate every customer-history field. |
| #6 | Shared chronological split with fixed configurable cutoffs, using #3 evidence; no automatic cutoffs at the largest descriptive movement. |
| #12 | Tested deterministic cleaning using #3 findings; explicit fix/unknown/reject decisions. |
| #13 | Shared training harness using #6 split, #5 metrics and #12 cleaning. |
| #16 | Learned feature steps fitted only on training rows and packaged with the model. |
| #20 | Batch cleaning/persistence using #12; original values preserved; learned transforms stay with #16. |

This is an ownership map, not a fabricated strict dependency chain: #20 does not
require every training step to be implemented first. Consult exact issue bodies.

Conceptual training route: inputs → deterministic cleaning and safe linkage →
fixed chronological split → train-fitted preprocessing/model → validation →
withheld final test evaluation. Deterministic date parsing needed for splitting
may occur first; data-learned imputation, scaling, feature selection and similar
operations fit only on training rows. Artifacts/intermediate persistence are
explicit future choices, not outputs this notebook currently produces.

After #6 freezes the split, development analysis must obey its test exclusion.
Clarify in its decision record how initial #3 discovery precedes that boundary;
this is a dependency-based interpretation, not an approved rewrite of #6. Initial
full-period label exploration should be disclosed; do not call the future test
period never examined. A fresh later holdout could strengthen evaluation but is
not an extra #3 acceptance criterion. Align imbalance discussion with #5's daily
top-K fraud-caught objective; precision, recall and average precision support it.

## Subagent briefs and execution arrangement

Use the smallest useful team. One sequential writer owns notebook changes;
lead integrates results. No parallel writers, staging, commits or issue updates.
Use `fork_turns: "none"`. Do not use `step_worker` if its mandatory commit workflow
would conflict with the no-commit rule; a `worker` can perform this bounded unit.

**Notebook worker brief:** Complete accepted #3 presentation improvements and
pair reconciliation in `notebooks/data_analysis.py` and its `.ipynb`, preserving
existing edits. You are not alone; do not revert others' changes. Current baseline:
59 versus 58 cells; saved cell19 contains terminal text/SyntaxError; missingness
findings absent. Use existing package report APIs. No split or dataset export.
New weekly charts and consolidated tables must be rendered in chat and approved
before inclusion. Existing review agreement remains valid; ask only for the
unseen preview, not generic reconfirmation. Preserve denominators, IQR caveats
and descriptive-change wording. Verify fresh execution and pairing; run the
57-test unittest suite. No commits. Return at most 15 lines with changes, checks,
remaining approval/verification gaps and exact touched paths. Lead supplies
approved previews and relevant evidence before dependent insertion.

**Optional independent final review brief:** Read-only check of the finished
notebook pair against all four #3 criteria, accepted communication changes and
source-preservation rules. Check actual saved outputs, pair agreement, exclusions,
weekly visualization and static claims. Do not design requirement4 or expand
scope into the later pipeline. No edits or commits. Review working-tree diff
against the baseline supplied by lead (no new commit range is available).
Return verified defects with path/cell evidence in at most 15 lines. Use one
review pass; lead fixes confirmed findings, without a reviewer chain.
