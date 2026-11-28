import hashlib

import pytest

from mlops_fraud_prediction_ca1.verify_registered_scores import main

HEADER = "country_code,order_id,score\n"
ROWS = ("SG,001,0.25\n", "SG,NA,0.5\n", "MY,001,0.75\n")


def _csv_pair(tmp_path, registered_rows, reference_rows=ROWS):
    reference, registered = (tmp_path / f"{name}.csv" for name in ("ref", "reg"))
    for path, rows in ((reference, reference_rows), (registered, registered_rows)):
        path.write_text(HEADER + "".join(rows), encoding="utf-8")
    return reference, registered


def test_matching_literal_keys_and_scores_inside_absolute_tolerance(tmp_path, capsys):
    reference, registered = _csv_pair(
        tmp_path, (ROWS[0], "SG,NA,0.5000000000005\n", ROWS[2])
    )

    main([str(reference), str(registered)])

    output = capsys.readouterr()
    assert output.err == ""
    assert "key=country_code,order_id" in output.out
    for label, path in (("reference", reference), ("registered", registered)):
        assert (
            f"{label} rows=3 sha256={hashlib.sha256(path.read_bytes()).hexdigest()}"
            in (output.out)
        )
    assert "all scores match (rtol=0, atol=1e-12)" in output.out
    assert "Exactly equal scores: 2/3" in output.out


def test_matching_scores_are_all_exact_when_probabilities_are_equal(tmp_path, capsys):
    paths = _csv_pair(tmp_path, ROWS)

    main([str(path) for path in paths])

    assert (
        "Exactly equal scores: 3/3; max absolute delta: 0." in capsys.readouterr().out
    )


@pytest.mark.parametrize(
    ("registered_rows", "reason"),
    [
        ((ROWS[1], ROWS[0], ROWS[2]), "row order differs"),
        ((ROWS[0], ROWS[1], ROWS[0]), "duplicate (country_code, order_id) key"),
        (ROWS[:2], "row key mismatch (missing=1, extra=0)"),
        ((*ROWS, "SG,002,0.75\n"), "row key mismatch (missing=0, extra=1)"),
        ((ROWS[0], ROWS[1], "MY,1,0.75\n"), "row key mismatch (missing=1, extra=1)"),
        (("SG,001,0.5\n", "SG,NA,0.25\n", ROWS[2]), "score mismatch on 2 rows"),
        ((ROWS[0], "SG,NA,0.500000000002\n", ROWS[2]), "score mismatch on 1 rows"),
        ((ROWS[0], "SG,NA,inf\n", ROWS[2]), "score must be a finite number"),
        ((ROWS[0], "SG,,0.5\n", ROWS[2]), "blank key component"),
        ((), "no scored rows"),
    ],
)
def test_comparison_reports_mismatches_without_publishing_rows(
    tmp_path, capsys, registered_rows, reason
):
    paths = _csv_pair(tmp_path, registered_rows)

    with pytest.raises(SystemExit) as error:
        main([str(path) for path in paths])

    assert error.value.code == 1
    output = capsys.readouterr()
    assert reason in output.err
    assert "PASS" not in output.out
    for row in (*ROWS, *registered_rows):
        assert row.strip() not in output.out + output.err


def test_collisions_in_reference_are_rejected(tmp_path, capsys):
    paths = _csv_pair(tmp_path, ROWS, reference_rows=(*ROWS, ROWS[0]))

    with pytest.raises(SystemExit) as error:
        main([str(path) for path in paths])

    assert error.value.code == 1
    assert (
        "reference: duplicate (country_code, order_id) key" in capsys.readouterr().err
    )
