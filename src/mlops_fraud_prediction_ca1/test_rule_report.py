"""Regression checks through the baseline rule command and protected CSV loader."""

import pytest

from mlops_fraud_prediction_ca1.rule_report import main


@pytest.fixture
def report_csvs(tmp_path):
    (tmp_path / "customers_base.csv").write_text(
        "customer_id,country_code,first_order_datetime\n"
        "a,MY,2023-04-01 00:00:00\n"
        "b,MY,9999-12-31 00:00:00\n"
    )
    (tmp_path / "orders_base.csv").write_text(
        "order_id,country_code,order_date,order_value\n"
        "t,MY,2023-03-20,20\n"
        "NA,MY,2023-04-01,51\n"
        "001,MY,2023-04-01,60\n"
    )
    labels = tmp_path / "labels_base.csv"
    labels.write_text(
        "country_code,order_id,customer_id,is_fraud\nMY,t,a,0\nMY,NA,a,1\nMY,001,b,1\n"
    )
    scores = tmp_path / "scores.csv"
    scores.write_text("country_code,order_id,score\nMY,NA,0.9\nMY,001,0.8\n")
    return (
        labels,
        scores,
        ["--data-dir", str(tmp_path), "--scores", str(scores), "--k", "250"],
    )


def test_preserves_opaque_ids_and_isolates_out_of_range_age(report_csvs, capsys):
    _, _, args = report_csvs
    main(args)
    report = capsys.readouterr().out
    assert "'valid': 1" in report
    assert "'unusable_first_order': 1" in report
    assert (
        "K = 250 orders/day across all countries (D5-01, Product Owner assumption)"
        in report
    )
    assert "validation-tuned proxy, predeclared D18-01 original grid" in report
    metric_rows = [
        line.split() for line in report.splitlines() if line.startswith("Model at K")
    ]
    assert metric_rows == [["Model", "at", "K", "2", "2", "111.00", "0", "0"]]


def test_rejects_customer_raw_timestamp_column_collision(report_csvs, capsys):
    labels, _, args = report_csvs
    (labels.parent / "customers_base.csv").write_text(
        "customer_id,country_code,first_order_datetime,raw_first_order_datetime\n"
        "a,MY,2023-04-01 00:00:00,2022-01-01 00:00:00\n"
    )

    with pytest.raises(
        ValueError, match="Input columns conflict with generated raw_ columns"
    ):
        main(args)
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("bad_score", ["", "NA", "bad", "inf", "-inf", "1e309"])
def test_rejects_invalid_scores_separately_from_keys(report_csvs, bad_score):
    _, scores, args = report_csvs
    scores.write_text(f"country_code,order_id,score\nMY,NA,{bad_score}\nMY,001,0.8\n")
    with pytest.raises(SystemExit, match="1 orders have an invalid score"):
        main(args)


@pytest.mark.parametrize("extra_row", ["", "MY,unmatched,0.2\n"])
def test_rejects_boolean_scores_independently_of_other_rows(report_csvs, extra_row):
    _, scores, args = report_csvs
    scores.write_text(
        f"country_code,order_id,score\nMY,NA,True\nMY,001,False\n{extra_row}"
    )
    with pytest.raises(SystemExit, match="2 orders have an invalid score"):
        main(args)


def test_reports_an_unmatched_order_separately_from_an_invalid_score(report_csvs):
    _, scores, args = report_csvs
    scores.write_text("country_code,order_id,score\nMY,001,0.8\n")
    with pytest.raises(SystemExit, match="1 orders have no score"):
        main(args)


@pytest.mark.parametrize(
    "threshold", [[], ["--threshold", "50", "--threshold-source", "test"]]
)
def test_empty_labelled_cohort_has_no_threshold_or_verdict(
    report_csvs, threshold, capsys
):
    labels, scores, args = report_csvs
    labels.write_text(
        "country_code,order_id,customer_id,is_fraud\nMY,t,a,0\nMY,NA,a,\nMY,001,b,\n"
    )
    scores.unlink()
    with pytest.raises(SystemExit, match="No labelled validation orders"):
        main([*args, *threshold])
    assert capsys.readouterr().out == ""


def test_duplicate_score_keys_stop_with_a_clean_message(report_csvs):
    _, scores, args = report_csvs
    scores.write_text(
        "country_code,order_id,score\nMY,NA,0.9\nMY,001,0.8\nMY,001,0.7\n"
    )
    with pytest.raises(SystemExit) as stop:
        main(args)
    assert "1 duplicate (country_code, order_id) keys" in str(stop.value)
