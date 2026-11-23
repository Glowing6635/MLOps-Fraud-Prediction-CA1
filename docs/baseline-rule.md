# Baseline rule inputs — PBI #18 criterion 2 (#123)

Today's rule, as the spec describes it, is "flag if order value > $X and account
is less than 1 day old". X, its source and the loyal definition belong to the
other #123 criteria and are not set here.

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
