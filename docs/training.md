# Training harness

`src/mlops_fraud_prediction_ca1/training.py` fits a model on the training window,
scores the validation window with the top-K evaluator and logs the run to MLflow. Every
candidate model uses it, so all runs record the same facts and can be re-run.

## Running it

The run goes to the MLflow server named by `MLFLOW_TRACKING_URI`, in the experiment
`MLFLOW_EXPERIMENT_NAME`, both read from the environment or `.env`. The three base CSV
files must be in `data/`, or in the folder given by `--data-dir`.

```
uv run train baseline --seed 0
uv run train boosted-trees
uv run train random-forest --seed 0
uv run train extra-trees --seed 0
uv run train compare-feature-groups
uv run train boosted-trees-country
uv run train rerun <run ID>
```

`baseline` trains the baseline model, a logistic regression on the order columns, and
prints the top-K report and the run ID. `--k` is the daily review capacity, in orders
per day. It defaults to `top_k.K`, 250 orders a day, the product owner assumption
recorded as D5-01 in [the decision log](decision-log.md), which holds until Trust &
Safety gives a capacity. Every candidate must use the same K, or the runs can't be
compared. Add
`--exploratory` to include the seven unconfirmed customer columns.

The candidate commands are `boosted-trees` (histogram gradient-boosted trees),
`random-forest` (random forest) and `extra-trees` (extremely randomized trees). Each
trains twice, without and with the seven unconfirmed customer columns, as two child
runs under one parent run (see "Comparing the customer columns"). Each command prints
both children's run IDs. Random forest caps its 100 trees at 2,400 leaves per tree to
fit within the retraining model-size limit (#25). Extra trees uses 200 trees. Both
require at least 50 training samples per leaf. The models are in
`src/mlops_fraud_prediction_ca1/models.py`.

`--k-status` says where K comes from, and is tagged `k_status` on all three runs. It
defaults to `D5-01: product owner assumption, K = 250`; with another `--k`, the command
stops unless `--k-status` says where that K comes from, such as
`provisional: <reason>`.

`compare-feature-groups` fits and logs the baseline separately for each
[order feature group](feature-pipeline.md) and prints the validation results and run IDs.

`boosted-trees-country` logs the boosted-trees candidate's version 2 ablation for #22 (see
"Version 2: country input and recency weighting").

A candidate model calls the harness from Python with any scikit-learn classifier or
pipeline that has `predict_proba`:

```python
from mlops_fraud_prediction_ca1.top_k import K
from mlops_fraud_prediction_ca1.training import compare_customer_columns, train

run_id, result = train(my_pipeline, "data", k=K, seed=0)
parent_id, (without_id, without), (with_id, with_columns) = compare_customer_columns(
    my_pipeline, "data", k=K, seed=0, tags={"k_status": "<where K comes from>"}
)
```

`train` adds any `tags` it is given to the run's tags. A run started while another is
active is logged as its child.

The model receives a DataFrame of the feature columns. Numbers and true/false values
are floats, text is a Python string, and every missing value is NaN. Missing values must
be filled by a step fitted inside the pipeline, so only the training window is used.

## Comparing training class weights

Both commands default to the estimator's unweighted fit. Add
`--weighting balanced` to opt in, or pass `class_weight="balanced"` to
`baseline_model()` or `boosted_trees_model()` before calling `train()`.
For `train compare-feature-groups`, the option applies to both feature groups.
To compare the two methods without exploratory customer columns, call `train()`
with each candidate twice, omitting `class_weight` on the first call and setting
it to `"balanced"` on the second. Use the frozen split, `k=250`, and `seed=0`
for both calls.

The classifier derives weights from training labels during fit. The validation
report uses the original orders and fraud rate. MLflow records the final
classifier's `class_weight` parameter, and rerun restores it from the logged
model. See [D17-01](decision-log.md#d17-01--training-only-class-weighting).

## What each run records

| Kind | Recorded |
| --- | --- |
| Parameters | The model's simple parameters and class name, `k`, `seed`, `exploratory`, `country_input`, `recency_half_life_days`, `label_delay_days`, `cleaning_policy_version`, the four cut-off dates from `config/time_split.toml`, and `dataset_sha256_customers`, `dataset_sha256_orders` and `dataset_sha256_labels` |
| Metrics at K and 2K | Fraud caught, fraud catchable, orders selected, total fraud, precision and recall |
| Other metrics | Average precision, `validation_customer_overlap`, `fit_seconds`, `model_size_bytes`, `train_orders` and `validation_orders` |
| Tags | `git_commit`, `git_dirty` and `author`, which is git's `user.name`, or `unknown` when none is set, and any `tags` the caller gives |
| Diagnostics | Tables under `diagnostics/` (PBI #22, [model-v2-run.md](model-v2-run.md)): `precision_recall_curve.json` (thinned to 500 points), `confusion_matrix_at_k<K>.json`, and `errors_by_country.json`, `errors_by_payment_method.json`, `errors_by_order_value_band.json`; the tag `order_value_bands` states the bands |
| Model | The fitted model, under the name `model` |

CSV runs record a SHA-256 for each input file. [Per-batch retraining](retraining.md)
uses the labelled database snapshot and logs the batch ID and validation digest.

`validation_customer_overlap` is the share of validation orders whose customer also
has an order in the training window. A high share means the validation score partly
measures customers the model has already seen.

`model_size_bytes` is the size of the saved model folder. A precision or recall with
nothing to divide by is not logged.

A run from a working tree with uncommitted changes is still logged, with `git_dirty`
set to true and a warning, because the commit alone does not hold its code.
The harness ignores caller-supplied values for `git_commit`, `git_dirty` and
`author`.

## Comparing the customer columns

Seven customer columns are unconfirmed: nobody has shown their values were known when
the order was placed (decision D4-02 in [the decision log](decision-log.md)). Until they
are confirmed, each candidate model is logged with `compare_customer_columns`. The
parent run, named after the model's class, such as
`HistGradientBoostingClassifier: customer columns compared`, carries the `author` and
`git_commit` tags. It has two children: `without customer columns` and
`with customer columns (exploratory)`. Only a run without the columns can be a
champion, so each child has the tag `champion_eligible`, `True` or `False`. The exploratory
run shows what leaving the columns out costs.

The with-columns run adds seven columns, one of them the customer's first order date
as `days_since_first_order`, and their values are NaN for orders whose customer dates
fail the D4-03 checks. A pipeline that picks columns by type, as the models in
`models.py` do, handles them; one that names its columns must name these too.

The parent logs the with-columns result minus the without result as
`fraud_caught_at_k_difference`, `fraud_caught_at_2k_difference` and
`average_precision_difference`. The parent holds no model, so re-run a child by its own
run ID.

## Version 2: country input and recency weighting

`uv run train boosted-trees-country` logs the boosted-trees candidate's ablation ladder under
one parent run, `HistGradientBoostingClassifier: v2 country input and recency weighting compared`:
`v1 order columns` on the four order columns, `v2 country input` adding the order's
`country_code`, and `v2 country input + 14d recency` adding training-only 14-day exponential
recency decay weights ([D22-03](decision-log.md#d22-03--the-orders-country-and-training-only-recency-weighting)).
Pass `--recency-half-life <days>` to change the half-life, or `--no-recency-weighting` to log
only `v1 order columns` and `v2 country input`. Every child is tagged `champion_eligible`
`True` and logs the diagnostics tables. The parent's description (`mlflow.note.content`) states
the weaknesses, the changes, the ablation in fraud caught at K and average precision, and how
many places the final version gives the country with v1's lowest precision at K. A country v1
gave no place has no precision, so it is never the one named. The parent also logs
`fraud_caught_at_k_difference` and `average_precision_difference`, plus
`country_only_fraud_caught_at_k_difference` and `recency_fraud_caught_at_k_difference`
when recency weighting is enabled.

From Python, pass `country_input=True` and optionally `recency_half_life_days=14` to `train`
or `fit_and_score`. Recency weights decay from the latest training `order_date` and apply to
training rows only; validation rows are never weighted. Both options work for rows passed in
too, since every row carries its `country_code` and `order_date`. A model fitted with the
country input needs the column whenever it scores. `score_labelled_rows` adds it for any
model that was fitted with it, so per-batch retraining can score version 2 as the champion.
Training version 2 there as a candidate needs `retrain.py` to pass `country_input=True` and
`recency_half_life_days` for it, which `ModelSpec` can't ask for yet.

## Re-running a run

`uv run train rerun <run ID>` builds an unfitted copy of the logged model, with the same
parameters, and fits it again with the logged seed, K and column choice. It stops if an
input file's SHA-256 differs from the logged one. It prints both results and
`reproduced: True` when fraud caught at K is within 1% of the logged count and average
precision within 0.005. `same_commit` shows whether the code is at the logged commit.

## How the data is kept safe

- Data is read only through the protected reader in `data.load_base_data`, once for
  the training window and once for the validation window, and each table is cleaned
  with `cleaning.clean_frame`. The test window is never loaded.
- Labels join orders by country and order ID. Orders without a valid label are left
  out of both fitting and scoring.
- The features are the four order columns that `leakage.select_feature_inputs` allows
  (decision D4-01 in [the decision log](decision-log.md)). A run with the country input
  also gets the order's `country_code` (D22-03). An exploratory run adds the
  seven customer columns, and withholds them for orders whose customer dates are
  inconsistent or cannot be checked (D4-02 and D4-03). The customer's first order date
  becomes `days_since_first_order`, the whole days from it to the order, because the
  models drop date columns. An exploratory model is never a champion.
- Before fitting, `check_feature_columns` stops the run if a column's name contains
  `is_fraud`, `refund_value`, `order_id`, `customer_id` or `batch_id`.
- A feature built from a customer's earlier labels must pass `check_label_history`.
  A label is usable only when its order date plus the 30-day label delay (D4-04) is
  before the scored order's date. No such feature exists yet.
