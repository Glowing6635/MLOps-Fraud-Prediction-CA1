# PR draft: PBI #18 Model vs current rule

Status: **all six acceptance criteria have evidence**, with the open items below.
`Resolves #18` is included on that basis; if the reviewer does not accept the
D18-04 proxy and D5-01 K assumptions, change it to `Refs #18` before merging.

## Summary

Compares the hard-coded rule (`order_value > X` and account age 0 days) with the
registered #15 champion on the validation window at matched flag counts, and
reports fraud caught, fraud exposure, false flags and false flags on loyal
customers. Result: the model caught 102 fraud orders against the rule's 73 at
132 flags each.

## Criterion evidence

| # | Criterion | Evidence |
|---|---|---|
| 1 | Rule stated with X and source; unconfirmed X tuned on validation, labelled proxy, candidates recorded | `docs/baseline-rule.md` (Rule function, Threshold X); D18-01, D18-03, D18-04. X = 13.36 SGD, validation-tuned proxy; candidates 13.36, 17.72, 31.14, 38.78, 61.32, 119.41 in the report and result section. The first coarse grid (X = 50, 5-5 tie) is kept. |
| 2 | Account age definition, use despite #4, missing/negative handling stated and tested | `rules.account_age_days`; baseline-rule.md (Account age, Exception, Missing/negative table); D18-01(b); `test_rules.py`, `test_rule_report.py`. Real run: 101,109 valid, 71 unusable, 10 negative, none imputed; the report prints the #4 caveat. |
| 3 | Loyal defined in numbers before results | >= 2 distinct earlier-date orders; baseline-rule.md (Loyal customer); D18-01(c); frozen at 466f080 before results; `test_loyalty.py`. |
| 4 | #5 evaluator; rule and champion at same flags; rule cut at K by order_value; model cut to rule count and shown at K | `rule_comparison.py` (D18-02), `test_rule_comparison.py`; champion scores from `champion-scores` (D18-06, run 9c3a5977…, v1). Matched 132 and K = 250/day (5,250). |
| 5 | Fraud caught, exposure, false flags, false flags on loyal | Result table in baseline-rule.md: rule 73 / 1,646.24 / 59 / 0; matched model 102 / 625.36 / 30 / 6; model at K 2,893 / 33,650.06 / 2,357 / 663. |
| 6 | Report states whether the model beat the rule on fraud caught | "The model beat the rule on fraud caught at the same number of flags": 102 against 73. D18-07. |

## Checks

Run 2026-11-26 on the branch head: `uv run pytest` 308 passed, 14 skipped;
`uv run ruff check` all checks passed; `uv run ruff format --check` 82 files
already formatted. Independent review result: s180
read-only independent review of `origin/user/spiralsky...6990d81` found no code
defects, pytest 308 passed/14 skipped, ruff clean, and 3 doc defects (fixed in
this commit). Open item, still stated: the re-score is not row-compared with the
registered champion artifact (needs team-server access).

## Decisions

D18-01 to D18-07 in `docs/decision-log.md`.

## Limits and open items

- The scores are an offline re-score reproducing the champion's record (2,893 at
  K, AP 0.3716), not the original registered artifact (D18-06). Deciding owner:
  #15 owner or coordinator.
- X is a proxy fitted on the same validation window; K = 250/day is a Product
  Owner assumption (D5-01); loyalty history is left-censored at 2022-11-12.
- The rule identified more fraud exposure than the model at matched flags.
- `output/champion-validation-scores.csv` is untracked data-derived output; the
  PR carries the code, tests and docs only.
