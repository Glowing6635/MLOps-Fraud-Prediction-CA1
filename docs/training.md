# Training harness

`src/mlops_fraud_prediction_ca1/training.py` fits a model on the training window,
scores the validation window with the top-K evaluator and logs the run to MLflow. Every
candidate model uses it, so all runs record the same facts and can be re-run.

## Running it

The run goes to the MLflow server named by `MLFLOW_TRACKING_URI`, in the experiment
`MLFLOW_EXPERIMENT_NAME`, both read from the environment or `.env`. The three base CSV
files must be in `data/`, or in the folder given by `--data-dir`.

```
uv run train baseline --k 50 --seed 0
uv run train rerun <run ID>
```

`baseline` trains the baseline model, a logistic regression on the order columns, and
prints the top-K report and the run ID. `--k` is the daily review capacity, in orders
per day. It has no default, because K must be fixed before any result is seen. Add
`--exploratory` to include the seven unconfirmed customer columns.

A candidate model calls the harness from Python with any scikit-learn classifier or
pipeline that has `predict_proba`:

```python
from mlops_fraud_prediction_ca1.training import train

run_id, result = train(my_pipeline, "data", k=50, seed=0)
```

The model receives a DataFrame of the feature columns. Numbers and true/false values
are floats, text is a Python string, and every missing value is NaN. Missing values must
be filled by a step fitted inside the pipeline, so only the training window is used.

## What each run records

| Kind | Recorded |
| --- | --- |
| Parameters | The model's simple parameters and class name, `k`, `seed`, `exploratory`, `label_delay_days`, `cleaning_policy_version`, the four cut-off dates from `config/time_split.toml`, and `dataset_sha256_customers`, `dataset_sha256_orders` and `dataset_sha256_labels` |
| Metrics at K and 2K | Fraud caught, fraud catchable, orders selected, total fraud, precision and recall |
| Other metrics | Average precision, `validation_customer_overlap`, `fit_seconds`, `model_size_bytes`, `train_orders` and `validation_orders` |
| Tags | `git_commit`, `git_dirty` and `author`, which is git's `user.name` |
| Model | The fitted model, under the name `model` |

The dataset version is the SHA-256 of each input file. Once incoming batches are loaded,
it will be their batch IDs instead.

`validation_customer_overlap` is the share of validation orders whose customer also
has an order in the training window. A high share means the validation score partly
measures customers the model has already seen.

`model_size_bytes` is the size of the saved model folder. A precision or recall with
nothing to divide by is not logged.

A run from a working tree with uncommitted changes is still logged, with `git_dirty`
set to true and a warning, because the commit alone does not hold its code.

## Re-running a run

`uv run train rerun <run ID>` builds an unfitted copy of the logged model, with the same
parameters, and fits it again with the logged seed, K and column choice. It stops if an
input file's SHA-256 differs from the logged one. It prints both results and
`reproduced: True` when fraud caught at K is within 1% of the logged count and average
precision within 0.005. `same_commit` shows whether the code is at the logged commit.

## How the data is kept safe

- Data is read only through the protected reader in `cleaning.load_clean_base_data`,
  once for the training window and once for the validation window. The test window is
  never loaded.
- Labels join orders by country and order ID. Orders without a valid label are left
  out of both fitting and scoring.
- The features are the four order columns that `leakage.select_feature_inputs` allows
  (decision D4-01 in [the decision log](decision-log.md)). An exploratory run adds the
  seven customer columns, and withholds them for orders whose customer dates are
  inconsistent or cannot be checked (D4-02 and D4-03). An exploratory model is never a
  champion.
- Before fitting, `check_feature_columns` stops the run if a column's name contains
  `is_fraud`, `refund_value`, `order_id`, `customer_id` or `batch_id`.
- A feature built from a customer's earlier labels must pass `check_label_history`.
  A label is usable only when its order date plus the 30-day label delay (D4-04) is
  before the scored order's date. No such feature exists yet.
