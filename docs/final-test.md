# One-shot final test harness (PBI #23)

`uv run final-test` scores the final champion and today's rule once on the frozen
test window ([D6-01](decision-log.md#d6-01--frozen-retrospective-windows-and-protected-exposure),
`2023-04-17` to `2023-05-11` exclusive). This page describes the command. **It has
only been exercised on made-up data.** No real test row has been read, and the shipped
`config/time_split.toml` still has `final_evaluation_enabled = false`.

## What the command enforces

The checks run in this order, and each refusal happens before any test row is read.

| # | Check | Refusal |
|---|---|---|
| 1 | `--freeze-commit` names a commit that exists, is an ancestor of `HEAD`, and holds `config/final_test_freeze.toml` with exactly the `--k`, `--x` and `--loyal-min-prior-orders` passed. The loyal definition must also equal `loyalty.LOYAL_MIN_PRIOR_ORDERS`. | No commit named, unknown or non-ancestor commit, any value differs from the frozen one. |
| 2 | `--champion-mode` is `as-trained` or `refit-train-validation`. There is no default. | Missing or unknown mode. |
| 3 | The ledger has no entry for this window. | Any earlier entry: scored, failed or interrupted. |
| 4 | Tracked files are committed (real run only). | Uncommitted changes. |

`--dry-run` runs checks 1 to 3, reads no data, writes nothing and scores nothing.

Then a real run:

1. appends a `mode_recorded` ledger line under a file lock (this is the claim; it
   re-checks the window under the lock, so two runs cannot both claim it);
2. reads train, validation and final windows through the protected reader;
3. for `refit-train-validation`, clones the champion (stored hyperparameters, no fitted
   state) and fits it on train plus validation rows, which all precede the test window;
   for `as-trained`, scores the champion unchanged;
4. scores the champion and the rule on the same orders;
5. appends a `scored` line, and with `--mlflow` logs one MLflow run.

If anything fails after the claim, a `failed` line is appended and the window stays
used. A frozen window is scored once.

## Logged measures

Primary measure: fraud caught at K (the #5 definition in `top_k.py`). For the champion
and for the rule (cut to K per day by order value) the entry holds fraud caught, flags,
precision at K (caught / flags) and recall at K (caught / all fraud in the window).
The champion also logs fraud catchable at K and average precision. The matched-count
model comparison from #18 is logged as one extra number.

## Procedure on the run day

```sh
# 1. Dry run. Must pass before anything else.
uv run final-test --freeze-commit SHA --k K --x X --loyal-min-prior-orders N \
    --champion-mode as-trained --champion-uri models:/fraud-candidate@champion --dry-run

# 2. Owner flips final_evaluation_enabled = true in config/time_split.toml (with a
#    decision-log entry), commits, then runs once without --dry-run (add --mlflow).
```

The freeze file looks like:

```toml
k = 250
x = 13.36
loyal_min_prior_orders = 2
```

The values above are examples. The owner chooses them in the freezing commit.

## Ledger

Default `output/final_test_ledger.jsonl`, which is git-ignored. One JSON object per
line. A local file only stops a second run in the same checkout; see the open items in
the PR for where the authoritative copy should live.

## Tests

`uv run pytest src/mlops_fraud_prediction_ca1/test_final_test.py` uses a throwaway git
repository, a made-up split configuration, a made-up ledger and made-up CSVs. The
shipped configuration is only used to show that the protected reader stays closed.
