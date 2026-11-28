# Project Documentation

- [Contributor guide](contributors/README.md) — Git workflow, pull requests,
  reviews, branch synchronisation and releases.
- [Decision log](decision-log.md) — major decisions, with their dates, owners
  and reasons.
- [Feature pipeline](feature-pipeline.md) — feature groups and how to load a
  logged or registered pipeline.
- [MLflow server design](mlflow-design.md) — why the server keeps its data in
  RDS and S3, and the limits of the first setup.
- [Runbook](runbook.md) — creating and setting up the cloud services.
- [Data dictionary](data-dictionary.md) — every table and column in the team
  database.
- [Training harness](training.md) — how to train and re-run a model, and what
  each MLflow run records.
- [Per-batch retraining](retraining.md) — evaluation-mode entry point, inputs,
  counts and MLflow evidence.
- [Champion selection and rollback](champion.md) — compare validation runs,
  register the top two and manage aliases.


- [Baseline rule inputs](baseline-rule.md) — account age for the hard-coded
  rule, and how missing or negative ages are handled.

- [Baseline rule and model comparison](baseline-rule.md) — account age, historical
  loyalty, and the active #18 validation-tuned proxy at X = 13.36 SGD (D18-03).
  At K = 250 per day and 132 matched flags, the rule caught 73 fraud orders
  and a fresh fit of the #152 boosted-trees candidate caught 102. The earlier
  D18-01 coarse grid selected X = 50 SGD and produced a 5–5 tie with that
  candidate at 7 matched flags. The protected team registry replay of the
  pinned #15 champion reproduced the 102–73 result and matched every
  validation score. [D18-08](decision-log.md#d18-08--verify-the-registered-score-pairs-before-finalizing-18)
  records the protected replay and the 15:50 UTC authenticated team metadata
  readback confirming the same historical champion binding on 2026-11-28.
  The Product Owner accepted the validation-tuned proxy and assumed K in
  [D18-07](decision-log.md#d18-07--18-outcome-criteria-status-and-score-file-key-check).
