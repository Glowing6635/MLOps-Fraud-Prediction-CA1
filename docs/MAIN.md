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
