import pandas as pd
import pytest

from . import batch
from .batch_files import Batch, DownloadFailed
from .config import load_settings
from .stages import (
    STAGE_TARGETS,
    FakeCount,
    FakeDatabase,
    FakeFetch,
    FakeLoad,
    FakeRead,
    FakeTrain,
    Stages,
    mlflow_run_url,
)

URLS = [f"https://example.test/{name}?sig=private" for name in ("c", "o", "l")]
NAMES = [name for name, _ in STAGE_TARGETS]


@pytest.fixture(autouse=True)
def settings(monkeypatch):
    loaded = load_settings(
        {
            "DB_HOST": "db.example",
            "DB_NAME": "practice",
            "DB_USER": "private-db-user",
            "DB_PASSWORD": "private-db-password",
            "MLFLOW_TRACKING_URI": "https://user:pw@mlflow.example:5000/prefix/",
            "MLFLOW_EXPERIMENT_NAME": "rehearsal",
            "MLFLOW_TRACKING_USERNAME": "private-mlflow-user",
            "MLFLOW_TRACKING_PASSWORD": "private-mlflow-password",
        }
    )
    monkeypatch.setattr(batch, "load_settings", lambda: loaded)


def fake_batch():
    frame = pd.DataFrame({"x": ["1", "2", "3"]})
    return Batch("b1", frame, frame.head(2), frame.head(1))


class Spy:
    """Wraps a fake stage and records when it ran, and can raise instead."""

    def __init__(self, name, inner, order, error=None):
        self.name, self.inner, self.order, self.error = name, inner, order, error

    def run(self, *args):
        self.order.append(self.name)
        if self.error:
            raise self.error
        return self.inner.run(*args)


def build(order, errors=None, train=None):
    errors = errors or {}
    database = FakeDatabase(customers=10, orders=20, labels=5)
    parts = {
        "fetch": FakeFetch(fake_batch()),
        "load": FakeLoad(database),
        "read": FakeRead(),
        "train": train or FakeTrain(),
        "count": FakeCount(database),
    }
    spies = {k: Spy(k, v, order, errors.get(k)) for k, v in parts.items()}
    return Stages(**spies), parts


def ticking(step):
    """A clock that advances `step` seconds per call, or per-call values from a list."""
    values = iter(step if isinstance(step, list) else [])
    counter = {"now": 0.0}

    def clock():
        if isinstance(step, list):
            return next(values)
        counter["now"] += step
        return counter["now"]

    return clock


def test_stages_run_in_d27_01_order_with_urls_passed_on(capsys):
    order = []
    stages, parts = build(order)
    assert batch.main(URLS, stages=stages, clock=ticking(1.0)) == 0
    assert order == ["fetch", "load", "read", "train", "count"]
    assert parts["fetch"].calls == [tuple(URLS)]
    assert parts["train"].calls == [(2, "b1", parts["train"].calls[0][2])]
    assert [t for _, t in STAGE_TARGETS] == [25, 30, 15, 85, 15]
    assert "private" not in capsys.readouterr().out


def test_timing_output_against_targets_and_total(capsys):
    order = []
    stages, _ = build(order)
    # started, then a (start, end) pair per stage, then the final read.
    times = [0.0]
    now = 0.0
    for seconds in (10.0, 31.5, 4.0, 60.0, 2.5):
        times += [now, now + seconds]
        now += seconds
    times.append(now)
    assert batch.main(URLS, stages=stages, clock=ticking(times)) == 0
    lines = capsys.readouterr().out.splitlines()
    assert "Stage 1/5 fetch and validate: 10.00 s (target 25 s)" in lines
    assert "Stage 2/5 clean and load: 31.50 s (target 30 s) OVER TARGET" in lines
    assert "Stage 3/5 read training data: 4.00 s (target 15 s)" in lines
    assert "Stage 4/5 train and log: 60.00 s (target 85 s)" in lines
    assert "Stage 5/5 report: 2.50 s (target 15 s)" in lines
    assert "Stages total: 108.00 s (target 170 s)" in lines
    assert any(
        line.startswith("Elapsed since start: 108.00 s (command limit 180 s")
        for line in lines
    )


def test_summary_prints_run_id_counts_and_filtered_url(capsys):
    stages, _ = build([])
    assert (
        batch.main(
            URLS, stages=stages, clock=ticking(1.0), new_etl_run_id=lambda: "run42"
        )
        == 0
    )
    lines = capsys.readouterr().out.splitlines()
    assert lines[:4] == [
        "Database host: db.example",
        "Database name: practice",
        "MLflow experiment: rehearsal",
        "etl_run_id: run42",
    ]
    assert "Batch b1 input: customers 3 rows, orders 2 rows, labels 1 rows" in lines
    assert "customers: 10 rows before, 13 after" in lines
    assert "orders: 20 rows before, 22 after" in lines
    assert "labels: 5 rows before, 6 after" in lines
    assert "MLflow runs logged: 2" in lines
    url = next(line for line in lines if line.startswith("MLflow URL: "))
    assert "run42" in url and "pw" not in url and "user" not in url
    assert "https://mlflow.example:5000/prefix/#/experiments/1/runs" in url


def test_mlflow_url_filters_on_the_etl_run_id_tag():
    assert mlflow_run_url("http://localhost:5000", "7", "abc") == (
        "http://localhost:5000/#/experiments/7/runs"
        "?searchFilter=tags.etl_run_id%20%3D%20%22abc%22"
    )


@pytest.mark.parametrize("failing", ["fetch", "load", "read", "train", "count"])
def test_stage_failure_exits_1_and_skips_later_stages(failing, capsys):
    order = []
    error = DownloadFailed("customers_base.csv: no answer")
    stages, _ = build(order, errors={failing: error})
    assert batch.main(URLS, stages=stages, clock=ticking(1.0)) == 1
    names = ["fetch", "load", "read", "train", "count"]
    assert order == names[: names.index(failing) + 1]
    captured = capsys.readouterr()
    assert "FAILED" in captured.out
    stage = NAMES[names.index(failing)]
    assert f"Batch stopped: {stage} stage failed (DownloadFailed" in captured.err
    assert "Stages total" not in captured.out


def test_unexpected_error_exits_1_without_echoing_its_message(capsys):
    stages, _ = build([], errors={"load": RuntimeError("password=private-db-password")})
    assert batch.main(URLS, stages=stages, clock=ticking(1.0)) == 1
    captured = capsys.readouterr()
    assert "RuntimeError" in captured.err
    assert "private" not in captured.out + captured.err


def test_failed_candidates_exit_1_after_training(capsys):
    stages, _ = build([], train=FakeTrain(failures={"rf": "boom", "lr": "boom"}))
    assert batch.main(URLS, stages=stages, clock=ticking(1.0)) == 1
    captured = capsys.readouterr()
    assert "candidates failed: lr, rf" in captured.err
    assert "MLflow URL" not in captured.out


def test_default_stages_are_unwired_and_stop_at_the_first(capsys):
    assert batch.main(URLS) == 1
    captured = capsys.readouterr()
    assert captured.err == (
        "Batch stopped: fetch and validate stage (#19) is unavailable.\n"
    )
    assert "Stage 1/5" not in captured.out
