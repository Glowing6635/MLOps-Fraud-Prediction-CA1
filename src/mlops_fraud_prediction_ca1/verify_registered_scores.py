"""Compare two private validation score CSVs and report aggregate results."""

import argparse
import csv
import hashlib
import io
import math
from pathlib import Path

import numpy as np

KEY = ("country_code", "order_id")
# Use the absolute probability tolerance from redeploy_check.verify.
SCORE_ATOL = 1e-12


def _read_scores(path: Path, label: str) -> tuple[dict[tuple[str, str], float], str]:
    scores = {}
    try:
        contents = path.read_bytes()
        rows = csv.reader(
            io.StringIO(contents.decode("utf-8"), newline=""), strict=True
        )
        header = next(rows, [])
        columns = (*KEY, "score")
        if any(header.count(column) != 1 for column in columns):
            raise ValueError(f"{label}: expected one column for each of {columns}.")
        country, order, score = (header.index(column) for column in columns)
        for row in rows:
            if len(row) != len(header):
                raise ValueError(f"{label}: row has the wrong number of columns.")
            key = (row[country], row[order])
            if any(not part.strip() for part in key):
                raise ValueError(f"{label}: blank key component.")
            if key in scores:
                raise ValueError(f"{label}: duplicate (country_code, order_id) key.")
            try:
                value = float(row[score])
            except ValueError:
                value = math.nan
            if not math.isfinite(value):
                raise ValueError(f"{label}: score must be a finite number.")
            scores[key] = value
    except (OSError, UnicodeError, csv.Error):
        raise ValueError(f"{label}: could not read a UTF-8 score CSV.") from None
    if not scores:
        raise ValueError(f"{label}: no scored rows.")
    return scores, hashlib.sha256(contents).hexdigest()


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path, help="Original D18-06 re-score CSV")
    parser.add_argument(
        "registered", type=Path, help="Exact registered artifact's scores"
    )
    args = parser.parse_args(argv)

    try:
        reference, reference_digest = _read_scores(args.reference, "reference")
        registered, registered_digest = _read_scores(args.registered, "registered")
    except ValueError as error:
        parser.exit(1, f"FAIL: {error}\n")

    print("key=country_code,order_id")  # noqa: T201
    print(f"reference rows={len(reference)} sha256={reference_digest}")  # noqa: T201
    print(f"registered rows={len(registered)} sha256={registered_digest}")  # noqa: T201

    if reference.keys() != registered.keys():
        missing = len(reference.keys() - registered.keys())
        extra = len(registered.keys() - reference.keys())
        parser.exit(1, f"FAIL: row key mismatch (missing={missing}, extra={extra}).\n")
    if list(reference) != list(registered):
        parser.exit(1, "FAIL: row order differs despite matching keys.\n")

    registered_values = np.asarray(list(registered.values()))
    reference_values = np.asarray(list(reference.values()))
    matches = np.isclose(registered_values, reference_values, rtol=0, atol=SCORE_ATOL)
    if not matches.all():
        parser.exit(1, f"FAIL: score mismatch on {int((~matches).sum())} rows.\n")
    exact = np.count_nonzero(registered_values == reference_values)
    max_delta = np.abs(registered_values - reference_values).max()
    print(  # noqa: T201
        f"Exactly equal scores: {exact}/{len(reference)}; max absolute delta: {max_delta:g}."
    )
    print(  # noqa: T201
        f"PASS: keys and order match; all scores match (rtol=0, atol={SCORE_ATOL:g})."
    )


if __name__ == "__main__":
    main()
