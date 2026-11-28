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


def test_cli_output_is_deterministic(fake_window, tmp_path):
    first, second = tmp_path / "a.csv", tmp_path / "b.csv"
    for out in (first, second):
        with pytest.raises(SystemExit):  # the fake report differs from the champion
            champion_scores.main(["--out", str(out)])
    assert first.read_bytes() == second.read_bytes()
