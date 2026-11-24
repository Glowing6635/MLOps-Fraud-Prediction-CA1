import pandas as pd
import pytest

from .retrain import ModelSpec, eligible_rows, retrain, run_name, validate_manifest


def _rows():
    return pd.DataFrame(
        {
            "order_date": ["2023-01-01", "2023-03-26", "2023-03-27", "2023-05-01"],
            "batch_id": ["base", "new", "new", "new"],
        }
    )


def _models():
    return [
        ModelSpec("a", "x", object),
        ModelSpec("b", "y", object),
        ModelSpec("c", "y", object),
    ]


def test_excludes_validation_and_later_rows_but_not_a_whole_batch():
    kept, excluded = eligible_rows(_rows(), "2023-03-27")
    assert kept.batch_id.tolist() == ["base", "new"] and excluded == 2


def test_reads_once_names_runs_and_continues_after_failure():
    reads, seen = [], []

    def read():
        reads.append(1)
        return _rows()

    def fit(spec, rows, name):
        seen.append((spec.name, len(rows), name))
        if spec.name == "b":
            raise RuntimeError("boom")
        return {"model_size_bytes": 10, "fit_seconds": 1.0}

    report = retrain(
        _models(),
        batch_id="B1",
        read_rows=read,
        validation_start="2023-03-27",
        fit_and_log=fit,
        champion_comparison=lambda m: {"beats": False},
        etl_run_id="E1",
    )
    assert len(reads) == 1
    assert [s[:2] for s in seen] == [("a", 2), ("b", 2), ("c", 2)]
    assert seen[0][2] == run_name("E1", "B1", "a") == "a-B1-E1"
    assert set(report.results) == {"a", "c"} and "b" in report.failures
    assert report.exit_code == 1 and (report.train_rows, report.excluded_rows) == (2, 2)
    assert report.results["a"]["champion_comparison"] == {"beats": False}


def test_oversize_artifact_is_a_failure():
    report = retrain(
        _models(),
        batch_id="B",
        read_rows=_rows,
        validation_start="2023-03-27",
        fit_and_log=lambda s, r, n: {"model_size_bytes": 60 * 1024 * 1024},
        champion_comparison=lambda m: {},
    )
    assert len(report.failures) == 3


def test_unique_ids_per_invocation_and_clean_exit():
    kwargs = {
        "batch_id": "B",
        "read_rows": _rows,
        "validation_start": "2023-03-27",
        "fit_and_log": lambda s, r, n: {},
        "champion_comparison": lambda m: {},
    }
    a, b = retrain(_models(), **kwargs), retrain(_models(), **kwargs)
    assert a.etl_run_id != b.etl_run_id and a.exit_code == 0


@pytest.mark.parametrize(
    "models",
    [
        [],
        [ModelSpec("a", "x", object)] * 2,
        [ModelSpec("a", "x", object), ModelSpec("b", "x", object)],
    ],
)
def test_manifest_rules(models):
    with pytest.raises(ValueError):
        validate_manifest(models)
