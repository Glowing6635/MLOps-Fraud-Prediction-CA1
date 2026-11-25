import json
import subprocess
import sys

import pytest

from mlops_fraud_prediction_ca1.impact_report import _fixture, _parse_export


def _command(output, *args):
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "mlops_fraud_prediction_ca1.impact_report",
            "--output-dir",
            str(output),
            *args,
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def test_synthetic_command_reproduces_aggregate_outputs(tmp_path):
    paths = [tmp_path / "first", tmp_path / "second"]
    for output in paths:
        result = _command(
            output, "--synthetic", "--k", "2", "--capacities", "1", "2", "3"
        )
        assert result.returncode == 0, result.stderr
        assert (output / "capacity.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    for name in ("report.md", "capacity.csv", "receipt.json"):
        assert (paths[0] / name).read_bytes() == (paths[1] / name).read_bytes()

    report = (paths[0] / "report.md").read_text()
    assert "Synthetic dry run only" in report
    assert "1000 customer-level bootstrap resamples" in report
    assert "seed 34" in report and "without a linked customer" in report
    assert "PENDING (count from MLflow run records)" in report
    receipt = json.loads((paths[0] / "receipt.json").read_text())
    assert receipt["bootstrap_draws_used"] == 1000 and receipt["seed"] == 34
    assert receipt["evaluation_run_id"] is None
    assert len(receipt["input_sha256"]) == 64


def test_export_mode_rejects_missing_provenance_before_read(tmp_path):
    output = tmp_path / "result"
    result = _command(
        output,
        "--scored-csv",
        str(tmp_path / "nonexistent.csv"),
        "--k",
        "2",
        "--capacities",
        "1",
        "2",
    )
    assert (
        result.returncode == 2
        and "requires a 32-character MLflow run ID" in result.stderr
    )
    assert not output.exists()


def test_export_preserves_leading_zeros_and_unlinked_customer():
    parsed = _parse_export(_fixture().to_csv(index=False).encode())
    assert parsed["order_id"].iloc[0] == "001"
    assert parsed["customer_id"].isna().sum() == 1
    assert parsed["rule_flag"].dtype == bool


@pytest.mark.parametrize(
    ("column", "bad", "message"),
    [
        ("is_fraud", "", "unknown labels"),
        ("loyal", "unknown", "true/false"),
        ("order_value", "NaN", "finite and numeric"),
        ("score", "inf", "finite and numeric"),
    ],
)
def test_export_rejects_invalid_values(column, bad, message):
    rows = _fixture().astype({column: str})
    rows.loc[0, column] = bad
    with pytest.raises(ValueError, match=message):
        _parse_export(rows.to_csv(index=False).encode())


def test_export_mode_records_supplied_provenance_with_synthetic_csv(tmp_path):
    export = tmp_path / "fixture.csv"
    _fixture().to_csv(export, index=False)
    output = tmp_path / "report"
    result = _command(
        output,
        "--scored-csv",
        str(export),
        "--k",
        "2",
        "--capacities",
        "1",
        "2",
        "--evaluation-run-id",
        "a" * 32,
        "--freeze-commit",
        "b" * 40,
        "--prior-comparisons",
        "3",
    )
    assert result.returncode == 0, result.stderr
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["evaluation_run_id"] == "a" * 32
    assert receipt["freeze_commit"] == "b" * 40
    assert receipt["prior_test_window_comparisons"] == 3
    assert (
        "Test-window comparisons run before the final one: 3."
        in (output / "report.md").read_text()
    )


def test_export_rejects_extra_field_without_silently_shifting_columns():
    lines = _fixture().to_csv(index=False).splitlines()
    raw = "\n".join([lines[0], *[f"{line},extra" for line in lines[1:]]])
    with pytest.raises(ValueError, match="extra fields"):
        _parse_export(raw.encode())
