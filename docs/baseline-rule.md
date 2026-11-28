# Baseline rule and model comparison — PBI #18

Today's rule, as the spec describes it (spec.txt:134-137), is "flag if order
value > $X and account is less than 1 day old". Account age is criterion 2 and the loyal-customer
definition is criterion 3 (see [Loyal customer](#loyal-customer-criterion-3)). X
for the deployed rule is unknown. The comparison uses a validation-tuned
proxy. See [Threshold X](#threshold-x-candidate-grid-and-selection-protocol).

## Rule function

`rules.baseline_rule_flags(order_value, account_age, threshold)` takes the
cleaned `order_value` (SGD, spec.txt:521), the `account_age_days` column from
`account_age_days(...)` and one explicit threshold, which must be finite and
>= 0 (SGD). It flags a row only when the amount is **strictly greater than** the
threshold and the age is **less than one day**, which for whole-day ages means
age 0. An amount equal to X is not flagged; age 1 is not flagged. It returns
`rule_flag` and `rule_status` on the input index and row order.

Unusable inputs never flag: an NA, non-numeric or non-finite amount gives
`unusable_amount`, and an NA age (missing, unusable, unavailable or negative
under the table below) gives `unusable_age`; amount takes precedence when both
fail. Those rows count as unflagged, so reports must give the per-status counts.

## Account age

`rules.account_age_days(order_date, first_order_datetime)` returns, per row,
`account_age_days` (nullable integer) and `account_age_status`. Age is
`order_date` minus the **date part** of `first_order_datetime`, in whole
calendar days. Clock times are ignored: an order on the same day has age 0
whether the first order was at 00:00:00 or 23:59:59, and 23:59:59 the day
before is age 1. Input may be raw text (the explicit formats of
`temporal.parse_dates`) or the datetimes produced by `clean_frame`; both give
identical results for valid dates. Ages use calendar day numbers, so the whole
parser range (1677-09-21 to 2262-04-11) is safe, including a first order at
12:00:00 on the earliest day. A well-formed date beyond that range (for example
9999-12-31) cannot be represented, so that row alone is `unusable_order_date` or
`unusable_first_order` (an impossible date) and the other rows are calculated
as usual. Cleaning and model-scoring eligibility are unchanged.

## Exception to the timing decisions

Provenance: the age-only restriction is anchored at definition freeze 466f080
(`docs(rules): freeze the loyal-customer definition before results`, #18, #123).
That commit fixed the loyalty definition to exclude `first_order_datetime`, so
the field is read for account age alone; later edits to this page correct
wording and checks and do not widen its use.

`first_order_datetime` is unconfirmed in #4 and withheld from champion inputs
(D4-02). The baseline rule uses it anyway because today's rule already does, so
the comparison is with the rule as it actually runs. This is a property of the
rule only; it does not make the field eligible for any model. Reports must note
it.

## Missing, unusable and negative ages

| Status | Meaning | `account_age_days` |
|---|---|---|
| `valid` | both dates usable, age >= 0 | the whole-day age |
| `missing_order_date`, `missing_first_order` | value absent | NA |
| `unusable_order_date`, `unusable_first_order` | unrecognised format or impossible date | NA |
| `unavailable_order_date`, `unavailable_first_order` | cleaned datetime is NaT and the reason is unknown | NA |
| `negative` | first order dated after the order | NA |

Policy (judgment, following D4-03, which withholds inconsistent customer dates
rather than repairing them): an age is never imputed, set to 0, or clipped.
Negative ages are withheld and counted under their own status. Missing order
date takes precedence over missing first order when both fail.

What this means for the rule: an NA age cannot satisfy "younger than N days", so
**the rule does not flag those rows**. Rows the rule cannot judge therefore count
as unflagged, and could include fraud it would have caught had the data been
complete. The comparison report should state the count of each status so this
limit stays visible. 105 orders precede their customer's first-order day and
788 are unevaluable in the full data (decision log D4-02), so such rows exist.

### Cleaned input loses the reason

`clean_frame` turns missing, unrecognised and impossible dates into the same
NaT, so a cleaned datetime alone cannot say which it was. Without more
information a NaT gets `unavailable_order_date` / `unavailable_first_order`: no
age, and no claim that the value was truly missing. `clean_frame` also keeps the
exact input in `raw_order_date` and `raw_first_order_datetime`; passing them as
`raw_order_date=` / `raw_first_order_datetime=` restores `missing_*` or
`unusable_*` per row, identical to calling the function on raw text. A NaT that
the raw value says should have parsed also stays `unavailable`. Status counts
from cleaned input without the raw columns therefore must not be read as missing
counts.

## Loyal customer (criterion 3)

Status: **project assumption**, written before any rule-versus-model comparison
result was produced or looked at (recorded 2026-11-23). It is not a definition
from SP-buy: `spec.txt` only says the rule "flags loyal customers for no good
reason" and defines no threshold, and #18, #123, #124, #125 and the decision log
(D4-02 to D4-04, D6-01) contain none. Operations or the owner can replace it;
change it only by a new dated entry, before results are examined.

**Definition.** An order is from a loyal customer if the same customer has at
least **2 distinct orders in the observed history whose `order_date` falls on a
strictly earlier calendar date** than this order's `order_date`. Otherwise it is
not loyal: either `not_loyal` (fewer than 2 such orders, identity known) or
`unknown` (see below).

Reason: the spec gives no number, so the cut is a project assumption: the
smallest count that, by judgment, shows repeat purchasing across separate days
(one earlier order could be a failed first attempt; two on earlier days is a
pattern). How many customers or orders it covers is **unknown** and is not
inferred here: D4-02 counts customers with more than one order (154,284 of
622,097 linked), which says nothing about how many have two strictly earlier
orders. It is not fitted or tuned, and the count is not chosen from outcome
distributions.

**Which history is observable.** Only orders in the supplied extract, dated on
or after the frozen training start 2022-11-12 (D6-01). Earlier activity is not
observed, so a long-standing customer whose history began before that date can
be undercounted, most of all for orders early in the window. This
left-censoring is a known limit and the report must state it. History is read
only through the protected loaders, so it never includes the locked test
interval for an order outside it, and it uses only dates strictly before the
order, so no later order can make an earlier order loyal. It uses no fraud
labels, so the D4-04 30-day label delay does not apply, and a prior order counts
whether it was later labelled fraud or not.

**Identity linkage.** A customer is the safe (`country_code`, `customer_id`)
pair, reached from the order through the label bridge of
`relationships.link_orders_customers` (D4-03), called with
`normalize_country=True` so country formatting is the same on every side of the
linkage; a bare `customer_id` is not enough. This holds for historical analysis only, since the scoring-time
mapping is unconfirmed (D4-02). Prior orders are counted from the same linkage.
`first_order_datetime` and `num_orders_last_50days` are **not** used: both are
unconfirmed and withheld (D4-02), and the 50-day field is not a lifetime count.

**Duplicates.** Orders are counted by distinct (`country_code`, `order_id`),
using the safe records the helper (`relationships.unique_records`, via
`link_orders_customers`) returns. Identical records, including rows identical
after country normalization, collapse to one safe distinct order, which counts
once. Only a key shared by *differing* records (a conflict) is excluded: all of
its rows are dropped and none is picked, so that key neither counts as a prior
order nor can itself receive `loyal`. The same collapse and exclusion apply to
the label and customer records of the bridge, so a conflicting label gives no
customer link and the order is `unknown`.

**Same-day orders.** Orders on the same calendar date as the current order are
never prior, and intraday order is not used (it is unknown, D4-03). Two orders
placed on one earlier day count as two.

**Unavailable evidence.** The order's own `order_date` missing or unusable, or
its customer unlinked, gives `unknown`. A prior order whose date is missing,
unusable or on/after the current date, or that has no safe key, is not counted.
`unknown` orders are **not loyal**: they are never counted as loyal customers
and the report states their count and the not-loyal count separately, because
an unlinked loyal customer would look new.

| Case (current order on 2023-04-01) | Prior orders, same customer | Result |
|---|---|---|
| Exactly the threshold | one on 2023-03-10 and one on 2023-03-31 | `loyal` (2) |
| One short | one on 2023-03-31 | `not_loyal` (1) |
| Same-day only | two others on 2023-04-01 | `not_loyal` (0) |
| One earlier day, two orders | two on 2023-03-31 | `loyal` (2) |
| Exact copy plus another order | the same record (same country, order id and fields) twice on 2023-03-30, plus a different order on 2023-03-31 | `loyal` (2; the copies collapse to one order) |
| Exact copy only | the same record twice on 2023-03-30 | `not_loyal` (1) |
| Conflicting key | two differing records sharing one (country, order id), dated 2023-03-30, plus a different order on 2023-03-31 | `not_loyal` (1; the conflicting key is excluded) |
| Unknown identity | `customer_id` missing or no safe link | `unknown`, not loyal |
| Unusable prior date | one on 2023-03-31, one with an unparseable date | `not_loyal` (1) |

**Loyalty is not a false positive.** Loyal is a property of the customer's
history, set before and independent of any rule or model output. A *false flag*
is a flagged order with `is_fraud = 0`; a *false flag on a loyal customer* is a
false flag whose order is `loyal`. Loyal customers can be fraudulent and a
loyal flag can be correct, so loyalty is never used to decide the label or to
call a flag wrong. Neither the rule nor any model receives it as an input.

**Evidence on timing.** At definition freeze 466f080, the repository and issue
history had no recorded comparison of a model or rule against loyal customers.
D6-01 records earlier full-extract exploratory analysis of customer fields.
That analysis did not examine loyalty.

**Calculation.** `loyalty.loyalty_status(history, comparison, labels, customers)`
applies this definition to already-scoped frames and returns one row per
comparison row (same index and order) with `customer_id`, `prior_order_count`,
`loyalty_status` and, for `unknown`, a `reason` (`unsafe_order_key`,
`no_label_link`, `no_customer_link`, `order_date_unavailable`). Prior orders
start at the frozen training start from `config/time_split.toml`. The threshold
of 2 is fixed and is not a parameter. An order key whose records differ, within
either input (compared on all of that input's columns, even when the other input
lacks one) or between the two inputs (compared on the shared columns), is unsafe
across both: it counts as no order's prior and its comparison rows are `unknown`
(`unsafe_order_key`); identical overlaps stay valid. Country spelling is
normalised first and `clean_frame`'s `raw_country_code` is ignored, so copies
that differ only in country format collapse to one order before and after
cleaning. A date outside the parser range is unusable for that row only. It is a
report input used by the comparison to count false flags on loyal customers.

## Threshold X: candidate grid and selection protocol

The original grid and objective were fixed before order values were examined.
D18-03 records the first validation comparison and the later training-quantile
grid. D18-04 adopts the training-quantile result as the #18 baseline. The
deployed threshold is unknown.

**Units.** `order_value` is in SGD (spec.txt:521). The rule applies a strict
`>` comparison to cleaned order values.

**Candidate grids.** D18-01 predeclared the coarse grid of 50, 100, 200, 500,
1000 and 2000 SGD before order values were examined. After that first run,
D18-03 recorded a second grid before its rerun. It uses the 0.90, 0.95, 0.99,
0.995, 0.999 and 0.9999 quantiles of cleaned training order values, rounded
to two decimal places and deduplicated.

**Fitting data.** Both grids use the frozen #6 validation window
`[2023-03-27, 2023-04-17)` to select X (D6-01, `config/time_split.toml`).
The training window supplies only the quantile candidates. The protected test
window `[2023-04-17, 2023-05-11)` is excluded.

**Selection objective (identical to the #5 measure, so the rule and the model
are compared on the same footing).** Using the shared #5 evaluator and the #5
selection group (one order_date, or order_date and country if K is set per
country), with the #5 K fixed before results are seen:

1. For each candidate X, compute the rule's flags on the validation window.
   Where a group has more than K flags, rank them by `order_value` descending,
   ties by (country_code, order_id), and cut at K (#18 criterion 4).
2. Primary: total labelled fraud caught across all groups after the cut.
   Higher is better.
3. Tie-break, applied in order: fewer total flags after the cut (lower review
   volume at equal labelled fraud caught; this does not claim fewer false
   flags, because an unlabelled flagged order is not known to be non-fraud,
   D6-01); then the larger X. The final rule picks exactly one X.

The comparison reports performance at the matched count and at K. Rows with unusable
amounts or ages are unflagged and stay in the denominators; their counts are
reported.

**Active #18 validation-tuned proxy.** D18-03 selected X = 13.36 SGD from
the post-hoc training-quantile grid at K = 250 orders per day (D5-01, a Product
Owner assumption). The rule caught 73 fraud orders at 132 flags. A fresh fit
of the [#152 boosted-trees candidate](https://github.com/Glowing6635/MLOps-Fraud-Prediction-CA1/pull/152)
caught 102 at the same flag count. D18-04 adopts the proxy as the #18 baseline.
Selection and comparison used the same validation window. No final-test
evaluation was run for #18.

**Earlier predeclared coarse-grid result.** D18-01's grid selected X = 50 SGD.
The rule and the same #152 candidate each caught 5 fraud orders at 7 matched
flags. The rule flagged zero orders at X = 100 SGD or higher.

**Acceptance status.** Criteria 4–6 are evidenced by an offline re-score of the
[#15 champion](https://github.com/Glowing6635/MLOps-Fraud-Prediction-CA1/pull/161),
`fraud-candidate` v1, run `9c3a59771cf8499d92972f5c7873e280` (D18-06, D18-07):
the rule caught 73 fraud orders in 132 flags, the model caught 102 at the same
132 flags and 2,893 at K. A protected read verified parity with historical
champion v1. Parity for the accepted team binding remains open. See the
[validation result](#result-rule-versus-the-15-champion-re-score-validation-window).

## Comparison with a model (criteria 4 to 6)

`rule_comparison.py` applies the protocol above against the #5 evaluator's groups
(`top_k`: one `order_date`, or `order_date` and `country_code` when K is per
country), with K = 250 a day (D5-01, a Product Owner assumption, not a Trust &
Safety figure).

- `select_threshold(orders, account_age, k)` fits X on the supplied validation
  window over the supplied candidate grid and returns the chosen X and the candidate table.
  The result is a **validation-tuned proxy**.
- `compare(...)` cuts the rule's flags to K per group by `order_value`, cuts the
  model to the rule's count per group, and also reports the model at K. It gives
  flags, fraud caught, fraud exposure (sum of `order_value` of flagged fraud),
  false flags and false flags on loyal orders, the rule status counts, the
  loyalty counts and whether the model caught strictly more fraud.
- Account age comes from `account_age_days` and loyalty from `loyalty_status`.
  The report prints the uncertainty in `first_order_datetime`, the loyalty
  history left-censoring, and the candidate table for a fitted threshold.

The public #148 evaluator supplies candidate flag and fraud counts and the model
counts at K. The comparison currently builds row masks locally for the rule cut,
matched model cut, exposure, and loyal false flags.

Run the report with the base CSVs and a score for every labelled validation
order. The scores file has `country_code`, `order_id` and `score` columns.
Order IDs are matched as literal text, including `NA` and leading zeroes.
Each matched score must be a finite number.

```sh
uv run rule-report --data-dir data --scores scores.csv --k 250 --grid training-quantiles
```

Use `--grid original` for the earlier coarse-grid comparison. The report
reads training history and validation via the #6 protected loader. A linked
first-order timestamp outside the parser's range is counted as an unusable
age and the order remains in the comparison. A validation window with zero
labelled orders exits with an error before fitting a threshold or issuing a
verdict, including when an explicit threshold is supplied.

## Result: rule versus the #15 champion re-score (validation window)

The test-window result, scored once at the frozen X, is logged by
[the final test-window scoring](final-test.md).

Run on 2026-11-26 with `uv run champion-scores` output
(`output/champion-validation-scores.csv`, sha256 `caa580e6…33d2c095`, D18-06) and
`rule-report --scores … --k 250 --grid training-quantiles`. Rule: `order_value >
13.36 SGD` and account age 0 days, a **validation-tuned proxy** (D18-03, D18-04).
Candidate X values tested on validation: 13.36 (132 flags, 73 fraud), 17.72
(64, 34), 31.14 (19, 11), 38.78 (11, 7), 61.32 (2, 2), 119.41 (0, 0).

| Flags | Count | Fraud caught | Fraud exposure (SGD) | False flags | False flags, loyal |
|---|---:|---:|---:|---:|---:|
| Rule | 132 | 73 | 1,646.24 | 59 | 0 |
| Champion, matched to the rule's count | 132 | 102 | 625.36 | 30 | 6 |
| Champion at K (250 a day) | 5,250 | 2,893 | 33,650.06 | 2,357 | 663 |

A later offline replay reproduced all 15 comparison values on the same
101,190 labelled validation orders. It used the D18-06 re-score with
`--k 250 --threshold 13.36 --threshold-source 'D18-04 validation-tuned proxy (frozen)'`.
On 2026-11-28, a separate protected read resolved historical champion v1
(run `9c3a59771cf8499d92972f5c7873e280`) to logged artifact
`m-34712ed5add845fe838835bd2107ec47`. All 101,190 ordered scores matched
with a maximum delta of 0, and all 15 aggregates matched. The later shared-trial
lookup could not resolve the required binding. Registered artifact parity for
the accepted team binding remains open (D18-07, PR #171).

Population: 101,190 labelled validation orders; account-age status `valid`
101,109, `unusable_first_order` 71, `negative` 10; rule status `flagged` 132,
`not_flagged` 100,977, `unusable_age` 81; loyalty `not_loyal` 74,686, `loyal`
26,504.

**Verdict.** The model beat the rule on fraud caught at the same number of
flags (102 against 73). Two limits matter. First, on **fraud exposure** the rule
identified more value (1,646.24 against 625.36 SGD): the rule targets large new
accounts, the model catches more but smaller orders, so the model's win is on
count, which is the #18 target, not on value. Second, the model's matched flags
include 6 loyal customers where the rule's include none. The scores are an
offline re-score of the registered champion (D18-06), not its original
artifact, and X is fitted on the same validation window, which favours the
rule.
