# Baseline rule inputs — PBI #18 criteria 2 and 3 (#123)

Today's rule, as the spec describes it, is "flag if order value > $X and account
is less than 1 day old". X and its source belong to criterion 1 and are not set
here. Account age is criterion 2 and the loyal-customer definition is criterion
3 (see [Loyal customer](#loyal-customer-criterion-3)).

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

Reason: the spec gives no number, so the cut is the smallest one that shows
repeat purchasing across separate days (one earlier order could be a failed
first attempt; two on earlier days is a pattern) while leaving the bulk of
multi-order customers eligible (D4-02: 154,284 of 622,097 linked customers have
more than one order). It is a judgment, not a fitted or tuned value, and the
count is not chosen from outcome distributions.

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
`relationships.link_orders_customers` (D4-03); a bare `customer_id` is not
enough. This holds for historical analysis only, since the scoring-time
mapping is unconfirmed (D4-02). Prior orders are counted from the same linkage.
`first_order_datetime` and `num_orders_last_50days` are **not** used: both are
unconfirmed and withheld (D4-02), and the 50-day field is not a lifetime count.

**Duplicates.** Orders are counted by distinct (`country_code`, `order_id`).
Source rows with a duplicated or conflicting key are not "safe" under D4-03
and neither count as prior orders nor receive a loyalty value of `loyal`.

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
| Duplicate key | the same (country, order id) twice on 2023-03-30 plus one on 2023-03-31 | `not_loyal` (1; the duplicated key is excluded) |
| Unknown identity | `customer_id` missing or no safe link | `unknown`, not loyal |
| Unusable prior date | one on 2023-03-31, one with an unparseable date | `not_loyal` (1) |

**Loyalty is not a false positive.** Loyal is a property of the customer's
history, set before and independent of any rule or model output. A *false flag*
is a flagged order with `is_fraud = 0`; a *false flag on a loyal customer* is a
false flag whose order is `loyal`. Loyal customers can be fraudulent and a
loyal flag can be correct, so loyalty is never used to decide the label or to
call a flag wrong. Neither the rule nor any model receives it as an input.

**Evidence on timing.** The repository history was searched for earlier
comparison runs of the rule or a model against loyal customers (commit log,
`notebooks`, `output`, `config`, `src`, issue comments). None exists; the only
code touching the rule is `rules.account_age_days`, which reads dates only. No
result was therefore available to influence this definition. The decision log
(D6-01) records earlier full-extract exploratory analysis of customer fields
generally, but none examined loyalty, so this is not claimed as an untouched
holdout. The evaluator and the report that applies this definition are
separate work (#124, #125).
