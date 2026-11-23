# Baseline rule inputs — PBI #18 criterion 2 (#123)

Today's rule, as the spec describes it (spec.txt:134-137), is "flag if order
value > $X and account is less than 1 day old". The loyal definition belongs to
another #123 criterion and is not set here. X is not yet known; see
[Threshold X](#threshold-x-candidate-grid-and-selection-protocol).

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
12:00:00 on the earliest day. Cleaning and model-scoring eligibility are unchanged.

## Exception to the timing decisions

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

## Threshold X: candidate grid and selection protocol

Status: **protocol only, written before any order value was examined.** No real
data was read for this section, no X has been selected, and none is fitted here.

**Source status.** The actual deployed threshold is unconfirmed. The spec gives
only the symbol X (spec.txt:134-137); no value has been received from Trust &
Safety. Until it is, any X used in a comparison is a **validation-tuned proxy**
(#18 criterion 1), labelled as such. If Trust & Safety later confirms the real
value, report that value and its source and keep the proxy only as a labelled
sensitivity row.

**Units.** `order_value` is in SGD (spec.txt:521). Candidates are SGD amounts on
cleaned order values; the strict `>` comparison is applied after cleaning.

**Candidate grid (SGD), fixed now:** 50, 100, 200, 500, 1000, 2000. The scale of
order values has not been inspected, so the grid is round amounts a rule author
would plausibly write, spaced roughly geometrically over 1.5 orders of
magnitude so that it can neither be fine-tuned to the validation window nor
fall entirely on one side of it. Any change to the grid, including adding
values because every candidate looks poor, is made before the next evaluation
and recorded in the decision log (#30); it is never made after seeing a result.

**Fitting data.** Only the frozen #6 validation window `[2023-03-27, 2023-04-17)`
(D6-01, `config/time_split.toml`). Training rows are not used for selection and
the protected test window `[2023-04-17, 2023-05-11)` is never read for it.

**Selection objective (identical to the #5 measure, so the rule and the model
are compared on the same footing).** Using the shared #5 evaluator and the #5
selection group (one order_date, or order_date and country if K is set per
country), with the #5 K fixed before results are seen:

1. For each candidate X, compute the rule's flags on the validation window.
   Where a group has more than K flags, rank them by `order_value` descending,
   ties by (country_code, order_id), and cut at K (#18 criterion 4).
2. Primary: total labelled fraud caught across all groups after the cut.
   Higher is better.
3. Tie-break, applied in order: fewer total flags after the cut (fewer false
   flags); then the larger X. The final rule picks exactly one X.

Results at 2K are reported but are not used for selection. Rows with unusable
amounts or ages are unflagged and stay in the denominators; their counts are
reported.

**Proxy versus deployed threshold.** The selected X is a proxy that makes the
baseline as competitive as possible on validation under this capacity measure.
It is not an estimate of the threshold actually deployed, and it is not
evidence about it. A model that beats this proxy has beaten a rule tuned on the
same window; one that does not should be checked against the confirmed X once
Trust & Safety provides it.

**Deferred.** Fitting waits for (a) the shared #5 evaluator, (b) the recorded K
and grouping, and (c) the #6 validation reader. The loyal definition (criterion
3) must also be written in numbers before results are seen.
