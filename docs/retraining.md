# Per-batch retraining (PBI #25)

`retrain.retrain(models, ...)` reads the training rows once, keeps rows dated before the
#6 validation window (the run logs how many later rows it excluded), fits each model in
the training list with stored hyperparameters, compares each with the current champion
and exits non-zero if any model failed. Run names are `<model>-<batch_id>-<etl_run_id>`;
`etl_run_id` is new for every invocation. It never promotes a model.

Not done here: release mode (waits for #23), the real `fit_and_log` and champion
comparison (the #13 harness and #15 registry; pass them in), and the live rehearsal.
