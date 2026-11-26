# SpiralSky candidate: v1 diagnosis and v2 proposal (PBI #22)

Status: **proposal, not yet logged.** This page gives the weakness found in
SpiralSky's v1 candidate and the change proposed for v2, with the offline
evidence behind both. No run was logged to the shared MLflow server and no
registry entry changed. The v2 run and its description still have to be
produced, which covers the other #22 acceptance criteria.

## What was scored

- **Candidate (v1):** `HistGradientBoostingClassifier` from
  `models.boosted_trees_model()`: the four order columns of D4-01
  (`collect_type`, `payment_method`, `order_value`, `num_items_ordered`), seed 0,
  no class weighting. It is the registered `fraud-candidate` version 1 and the
  champion in [champion.md](champion.md) (validation run `9c3a59771cf8499d92972f5c7873e280`).
- **How it was scored:** the tracking server and its credentials are not
  available to this session, so the existing candidate was refit offline from the
  local CSVs with the same harness code (`training._load_window`,
  `top_k.evaluate_top_k`) and the #6 train and validation windows. The refit
  reproduces the registered result exactly: 2893 fraud caught at K = 250 a day,
  average precision 0.3716, over 101,190 labelled validation orders (13,066
  fraud) from 2023-03-27 to 2023-04-16. The slices below come from that refit,
  not from a logged run. The harness now logs slice tables for every run
  ([model-v2-run.md](model-v2-run.md)).
- **K:** 250 orders a day in total, as D5-01 sets. Slices use the orders the model
  selects under that rule, so country shares add up to the 5,250 selected orders.

Order-value bands (the order value is in the data's currency unit, median 5.45,
99th percentile 30.7): `[0,3)`, `[3,5)`, `[5,8)`, `[8,13)`, `[13,30)`, `[30,∞)`.
The bands put roughly 10 to 28% of orders in each band except the top one (1%).

## Where v1 loses fraud

"Caught" is fraud inside the daily top 250. "Missed" is fraud in the window
that is not selected (13,066 fraud in all; 5,250 selections can catch at most
5,250, so recall at K is bounded at 40%).

By country:

| Country | Orders | Fraud rate | Selected | Caught | Precision at K | AP | AUC |
|---|---|---|---|---|---|---|---|
| PK | 18,196 | 38.7% | 3,059 | 2,080 | 0.68 | 0.553 | 0.652 |
| BD | 6,567 | 33.6% | 892 | 514 | 0.58 | 0.497 | 0.627 |
| PH | 32,834 | 5.9% | 649 | 83 | 0.13 | 0.079 | 0.568 |
| MY | 35,421 | 3.9% | 329 | 105 | 0.32 | 0.105 | 0.644 |
| TH | 8,172 | 6.4% | 321 | 111 | 0.35 | 0.196 | 0.678 |

By payment method (those with at least 1,000 orders):

| Payment method | Orders | Fraud rate | Selected | Caught | AUC |
|---|---|---|---|---|---|
| payment on delivery | 31,167 | 17.8% | 1,475 | 669 | 0.621 |
| generic creditcard | 8,181 | 32.0% | 1,256 | 803 | 0.687 |
| jazzcash wallet | 1,572 | 55.3% | 1,416 | 840 | 0.711 |
| cybersource | 13,004 | 3.2% | 114 | 44 | 0.688 |
| credit card | 12,174 | 8.1% | 194 | 148 | 0.665 |
| antfinancial gcash | 12,000 | 6.0% | 20 | 4 | 0.521 |
| online banking | 11,673 | 3.8% | 78 | 45 | 0.636 |
| antfinancial bkash | 2,032 | 30.0% | 517 | 258 | 0.660 |
| balance | 3,070 | 16.6% | 93 | 50 | 0.662 |

By order-value band:

| Band | Orders | Fraud rate | Selected | Caught | Recall at K | AUC |
|---|---|---|---|---|---|---|
| [0,3) | 22,843 | 17.6% | 1,025 | 561 | 0.14 | 0.714 |
| [3,5) | 22,165 | 12.9% | 913 | 522 | 0.18 | 0.772 |
| [5,8) | 27,971 | 9.4% | 881 | 528 | 0.20 | 0.749 |
| [8,13) | 17,165 | 10.5% | 890 | 502 | 0.28 | 0.760 |
| [13,30) | 9,950 | 13.9% | 1,037 | 534 | 0.39 | 0.770 |
| [30,∞) | 1,096 | 33.4% | 504 | 246 | 0.67 | 0.699 |

Precision-recall and the confusion matrix at K follow from these counts: at
K the model has 5,250 predicted positives, 2,893 true positives, 2,357 false
positives, 10,173 false negatives and 5.2% of the 101,190 orders flagged.

### Reading the slices

1. **Most fraud is missed in low-value orders in two countries.** PK and BD hold
   54% + 17% of the fraud and 58% + 17% of the selections. Inside PK the
   `[0,3)` band alone has 2,774 fraud, of which 326 are caught. The ranking
   works better as value rises (PK precision 0.49 at `[0,3)`, 0.89 at `[13,30)`).
2. **Within a country the model separates fraud only weakly.** Country is not a
   model input (D4-02), so the model reaches country mostly through the payment
   method. Within-country AUC is 0.57 to 0.68, against 0.76 overall, so much of
   the apparent skill is between-country and between-payment priors, not
   order-level signal. PH is the clearest case: 649 selections, 83 caught.
3. **Payment on delivery is the largest single leak.** It carries 43% of the
   fraud and 5,550 fraud orders, yet only 669 are caught (AUC 0.62).
4. **Order columns have little left to give.** Offline refits on the validation
   window showed nothing beyond noise. Adding order value per item gave 2858
   caught, adding day of week gave 2856, and day of month gave 2846. Four
   capacity or regularisation variants gave 2853 to 2879. Every result is
   below v1's 2893 and AP stays at 0.367 to 0.370. Fit time was 2 to 4 seconds.
   The weakness is missing information, not model capacity or tuning.

## Weakness stated for the run description

v1 ranks fraud mainly by country and payment prior, because the four allowed
order columns carry little within-country signal. Order-only variants (value per
item, weekday, deeper or more regularised trees) do not help, so low-value
payment-on-delivery orders in PK, BD and PH go uncaught.

## v2 change proposed

**Add three customer-history counts, built from order dates and delayed
labels, to the same boosted-trees model.** For each scored order, with the
customer identified by `country_code` and `customer_id`:

- `n_prior_orders`: labelled orders by this customer with `order_date` strictly
  before this order's date;
- `n_prior_labelled_30d`: the customer's labelled orders dated more than 30
  days before (the D4-04 label delay, passed as `ASSUMED_LABEL_DELAY_DAYS`);
- `n_prior_fraud_30d`: how many of those were fraud.

Rows with no customer ID get zeros, and history stays inside the labelled orders
the harness already loads. The history is computed per order from earlier rows
only, so the training window sees history from its own past and validation
orders see history from training and from earlier validation days. The code
applies the date rule with a per-customer cumulative count and `pd.merge_asof`
(strictly earlier dates), not `historical_order_mask`; a test compares it with a
brute-force reference and with `historical_order_mask`.

Offline evidence (same refit, same seed, same windows, validation only):

| Variant | Fraud caught at K | Average precision |
|---|---|---|
| v1 | 2893 | 0.3716 |
| v1 + `n_prior_orders` | 2867 | 0.3805 |
| v1 + `n_prior_labelled_30d`, `n_prior_fraud_30d` | 3042 | 0.4186 |
| v1 + all three | 3042 | 0.4239 |

So the primary measure rises by 149 caught (+5.2%) and AP by 0.047 to 0.052.
Only 6% of validation orders have a customer with matured prior fraud and 44%
have any prior order, so the gain is concentrated; the repeat-fraud signal
works on the part of the data where it exists.

### How the primary measure change is read

The primary measure is fraud caught at K = 250 a day over the whole validation
window, then AP, then run ID (#5; D5-01). v2 must be compared with v1 at the
same K, windows, cleaning policy and row counts, as [champion.md](champion.md)
describes. Order-only variants moved the count by up to ±50 around v1 (2846 to
2893), so read a change of under about 50 as noise, treat +100 or more as a clear
win, and put the run description's number next to v1's 2893 and 0.3716.
The per-country, per-payment and per-band tables above are for v1 only; v2's slices are in
[model-v2-run.md](model-v2-run.md). They do not match the prediction that the gain would
come from PK/BD low-value and payment-on-delivery orders with no loss in TH, MY or the
`[30,∞)` band: the gain is in MY, PH and BD, and PK, TH and the `[8,13)`, `[13,30)` and
`[30,∞)` bands lose fraud caught.

## Open questions and risks

These need a decision before the v2 run is logged on the team server.

- **Decision recorded; acceptance open (owner: SpiralSky, with Glowing6635 review).**
  D4-02 keeps `customer_id` out as a direct feature, and D4-01 limits the
  model to four order columns. History counts are derived features of a customer
  link, which D4-04 allows only as a disclosed, explicit-delay experiment.
  D22-01 in [decision-log.md](decision-log.md) records the new input rule and the
  30-day assumption (still unconfirmed with Operations); the team has yet to
  accept v2 as a challenger.
- **Scoring-time availability.** The batch and RDS readers (#20, #25) must supply
  the same history as of the scoring date. The offline history uses only the
  labelled orders in the CSVs, so the production query must reproduce it, and the
  packaged model (#16) must carry the history step or its inputs.
- **Label delay.** The 30-day delay is an assumption (D4-04, D4-05). A shorter
  real delay would only raise the signal; a longer one would shrink it.
- **Validation window only.** The result tunes nothing but the feature set, was
  measured once on validation, and must not be checked against the locked test
  window (`test_locked = true`).
- **Alternative not pursued:** class weighting (#17) is a separate change already
  owned there, and per-country calibration cannot fix a lack of within-country
  signal.
- **Training budget.** The v2 fit adds a groupby over about 820,000 rows. The
  offline prototype ran in well under a minute, but it has to be measured against
  the 85 s train-and-log allocation for #25 before v2 joins the roster.

## Not covered

- The logged v2 run, its MLflow description and artifacts (#22 criteria 1 to 3).
- The Sprint 3 repeat of the #15 review and the #25 roster entry by 25 Nov
  (criteria 4 and 5).
- Glowing6635's candidate; this page covers SpiralSky's only.
- Notebook tables or charts (they need the owner's approval first).
