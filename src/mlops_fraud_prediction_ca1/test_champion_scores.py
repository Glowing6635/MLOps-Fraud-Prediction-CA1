"""The champion re-score writes one provenance-tagged row per validation order."""

import numpy as np
import pandas as pd
import pytest

from mlops_fraud_prediction_ca1 import champion_scores


class _Model:
    def predict_proba(self, features):
        p = features["x"].to_numpy(dtype=float)
        return np.column_stack([1 - p, p])


@pytest.fixture
def fake_window(monkeypatch):
    keys = pd.DataFrame(
        {
            "order_date": ["2023-04-02", "2023-04-01"],
            "country_code": ["MY", "BD"],
            "order_id": ["NA", "001"],
            "customer_id": ["a", "b"],
            "is_fraud": [1, 0],
        }
    )
    features = pd.DataFrame({"x": [0.9, 0.1]})
    report = type(
        "R",
        (),
        {"at_k": type("T", (), {"fraud_caught": 1})(), "average_precision": 1.0},
    )()
    result = type("F", (), {"model": _Model(), "report": report})()
    monkeypatch.setattr(champion_scores, "fit_and_score", lambda *a, **k: result)
    monkeypatch.setattr(champion_scores, "load_window", lambda *a: (keys, features))
    monkeypatch.setattr(
        champion_scores, "dataset_hashes", lambda d: {"dataset_sha256_orders": "h"}
    )
    monkeypatch.setattr(champion_scores, "_git", lambda *a: "abc123")


def test_rescore_rows_are_keyed_sorted_and_labelled(fake_window):
    rows = champion_scores.rescore("unused")
    assert list(rows.order_id) == ["001", "NA"]  # sorted by country then id
    assert list(rows.score) == [0.1, 0.9]
    assert set(rows.artifact_kind) == {"re-score"}
    assert set(rows.source_run_id) == {champion_scores.CHAMPION_RUN_ID}
    assert set(rows.model_version) == {"1"}
    assert set(rows.code_commit) == {"abc123"}
    assert set(rows.dataset_sha256_orders) == {"h"}


def test_valid_cli_output_is_deterministic(fake_window, monkeypatch, tmp_path):
    monkeypatch.setattr(champion_scores, "EXPECTED_FRAUD_CAUGHT", 1)
    monkeypatch.setattr(champion_scores, "EXPECTED_AVERAGE_PRECISION", 1.0)
    first, second = tmp_path / "a.csv", tmp_path / "b.csv"
    for out in (first, second):
        champion_scores.main(["--out", str(out)])
    assert first.read_bytes() == second.read_bytes()


def test_mismatched_champion_does_not_create_export(fake_window, tmp_path):
    out = tmp_path / "new-directory" / "scores.csv"
    with pytest.raises(SystemExit, match="does not reproduce the champion aggregate"):
        champion_scores.main(["--out", str(out)])
    assert not out.parent.exists()


def test_mismatched_champion_preserves_previous_export(fake_window, tmp_path):
    out = tmp_path / "scores.csv"
    out.write_bytes(b"previous export\n")
    with pytest.raises(SystemExit, match="does not reproduce the champion aggregate"):
        champion_scores.main(["--out", str(out)])
    assert out.read_bytes() == b"previous export\n"


def test_custom_seed_allows_exploratory_export(fake_window, tmp_path):
    out = tmp_path / "scores.csv"
    champion_scores.main(["--out", str(out), "--seed", "7"])
    scores = pd.read_csv(out, keep_default_na=False)
    assert scores["seed"].tolist() == [7, 7]
    assert scores["artifact_kind"].tolist() == ["re-score", "re-score"]
