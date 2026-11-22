# Failure log

This log records every failure the team injected or met, what the system was expected to do, what it did, and how long it took to recover. It has three parts: restart drills and rehearsals, automated failure tests, and real failures found and fixed during development.

Times are written as minutes and seconds, so 1:04 is one minute and four seconds.

## Restart drills and rehearsals

A restart drill ends and restarts the AWS sandbox, then times how long the services take to work again. `uv run restart-drill` measures it, as the runbook's "Restart drill" section describes, and logs an MLflow run named `restart-drill` with the times. Each #29 rehearsal adds a row here.

| Date | Source | Fault | Expected result | Observed result | Recovery time |
| --- | --- | --- | --- | --- | --- |
| 17 Nov 2026 | #11 restart drill, `docs/runbook.md` | The sandbox session ended and was started again. RDS was not stopped. | RDS, the MLflow server and the recorded model work again, with every run and the model still there. | Ending the session terminated the MLflow server's Beanstalk instance, and Beanstalk launched a new one when the sandbox started, about a minute before the drill. RDS kept running and logged no events. The runs and the model were still there, since they live in RDS and S3, and the model reproduced its recorded predictions. | 1:04 in total |
| Not run yet | #29 release rehearsals | Not run yet | Not run yet | Not run yet | Not run yet |

Each stage's time, from the start of the drill:

| Date | RDS answers | MLflow answers | Model check passes | Total |
| --- | --- | --- | --- | --- |
| 17 Nov 2026 | 0:06 | 0:47 | 0:58 | 1:04 |

So on 17 Nov RDS took 0:06, the MLflow server 0:41 more, and the model check 0:11 more, and the drill ended 0:06 after the model check. These times don't include the minute Beanstalk took to replace its instance before the drill started, or an RDS start-up wait, since RDS was not stopped. A drill after RDS was stopped overnight would take several minutes longer.

## Failure tests

Each test injects a fault and checks the result, and runs in `uv run pytest`. The database tests run only when `TEST_MYSQL_URL` points at a throwaway MySQL database, as they do in CI. A test-only failure has no recovery time, since nothing needs recovering.

| Source | Fault | Expected result | Observed result | Recovery time | Test |
| --- | --- | --- | --- | --- | --- |
| #19 | A file's download fails: the server answers 404, keeps answering 503, stops answering, or can't be reached. | The batch stops with `DownloadFailed`, naming the file. A 404 isn't retried, and other failures are retried at most 3 times. | As expected | Not applicable: automated test | `test_a_failed_download_stops_after_at_most_three_retries`, `test_an_unreachable_host_fails_after_three_retries` in `test_batch_files.py` |
| #19 | A file is not CSV text: an HTML page, binary data, JSON, or text that can't be parsed. | The batch stops with `NotCsv`, naming the file. | As expected | Not applicable: automated test | `test_a_bad_file_stops_the_batch_with_a_named_error` in `test_batch_files.py` |
| #19 | A required column is missing or renamed. | The batch stops with `MissingColumns`, naming the file. | As expected | Not applicable: automated test | `test_a_bad_file_stops_the_batch_with_a_named_error` in `test_batch_files.py` |
| #19 | A file has only a header, or is empty. | The batch stops with `NoRows`, naming the file. | As expected | Not applicable: automated test | `test_a_bad_file_stops_the_batch_with_a_named_error` in `test_batch_files.py` |
| #19 | The three file names don't share one batch suffix. | The batch stops with `BadFileNames`. | As expected | Not applicable: automated test | `test_files_must_share_one_batch_suffix` in `test_batch_files.py` |
| #19 | A file has an extra column. | A warning names the column, the column is dropped, and the batch goes on. | As expected | Not applicable: automated test | `test_an_extra_column_is_a_warning_and_is_dropped` in `test_batch_files.py` |
| #20 | Inserting into the third table, labels, fails partway through a load. | The load stops with the database's error, all three tables keep their earlier rows, and the error text shows no data rows. | As expected | Not applicable: automated test | `test_a_failure_on_the_third_table_leaves_all_three_unchanged_and_shows_no_rows` in `test_loading.py` |
| #20 | The same batch is loaded twice. | The second load adds nothing, and reports every row as already stored. | As expected | Not applicable: automated test | `test_loading_the_same_batch_twice_leaves_row_counts_unchanged` in `test_loading.py` |
| #20 | A batch with a #19 fault reaches the load command: a file that can't be read, content that is not CSV, a renamed column, a file with no rows, or file names that don't share a suffix. | The load stops before any database change, so all four tables keep their row counts. | As expected | Not applicable: automated test | `test_a_bad_batch_stops_before_any_database_change` in `test_loading.py` |
| #20 | Rows that can't be stored: blank or unknown keys, an ID over 64 characters, a value over 255 characters, copies, rows that clash on a key, and labels with no matching order or customer. | Each row is rejected and counted by reason, and the rest of the batch loads. | As expected | Not applicable: automated test | `test_rows_that_cannot_be_stored_are_rejected_and_counted_without_stopping` in `test_loading.py` |
| #25 | Not run yet | Not run yet | Not run yet | Not run yet | Not run yet |

## Real failures

These failures showed up while the features were built. Each links to the commit that fixed it. None was measured for recovery time, since each was found and fixed during development, before any release.

| Source | Fault | Expected result | Observed result | Fix |
| --- | --- | --- | --- | --- |
| #20 | A database error during a load. | The error text names the failure without showing data. | The error text printed the data rows the failed statement sent. | [2656580](https://github.com/Glowing6635/MLOps-Fraud-Prediction-CA1/commit/265658072ec27d822cbb5c29b7e0cbb6b5b11dcc): `config.create_db_engine` sets `hide_parameters`, and `test_a_failure_on_the_third_table_leaves_all_three_unchanged_and_shows_no_rows` checks it. |
| #20 | A later batch has an ID that differs from a stored one only by trailing spaces, such as `o1 ` for `o1`. MySQL compares IDs ignoring trailing spaces. | The row counts as already stored. | The whole load stopped. | [2656580](https://github.com/Glowing6635/MLOps-Fraud-Prediction-CA1/commit/265658072ec27d822cbb5c29b7e0cbb6b5b11dcc): IDs are compared without trailing spaces, and `test_a_later_batch_keeps_stored_orders_and_labels_and_adds_customer_rows` checks it. |
| #20 | A row has an ID longer than the database's 64-character limit. | The row is rejected and counted, and the rest of the batch loads. | The whole load stopped. | [2656580](https://github.com/Glowing6635/MLOps-Fraud-Prediction-CA1/commit/265658072ec27d822cbb5c29b7e0cbb6b5b11dcc): such rows are rejected as "order_id longer than 64 characters", and `test_rows_that_cannot_be_stored_are_rejected_and_counted_without_stopping` checks it. |
| #13 | The training harness branch was rebased onto `dev`, whose final cleaning code replaced `load_clean_base_data` with `clean_frame`. | The harness reads and cleans each time window. | The harness broke, since it still called `load_clean_base_data`. | [aafcdd6](https://github.com/Glowing6635/MLOps-Fraud-Prediction-CA1/commit/aafcdd62dc03e3fe43079063c3c357c180f94486): the harness reads each window with `data.load_base_data` and cleans each table with `clean_frame`. |
