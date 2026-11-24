import mlflow
import pytest

from mlops_fraud_prediction_ca1.restart_drill import (
    RETRY_SECONDS,
    log_times,
    minutes,
    wait_until,
)


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def failing(times):
    calls = []

    def check():
        calls.append(1)
        if len(calls) <= times:
            raise ConnectionRefusedError("still starting")

    return check, calls


def test_wait_until_retries_until_the_check_passes():
    clock = FakeClock()
    check, calls = failing(times=2)
    wait_until(check, deadline=100 * RETRY_SECONDS, clock=clock, sleep=clock.sleep)
    assert len(calls) == 3
    assert clock.now == 2 * RETRY_SECONDS


def test_wait_until_raises_the_last_error_at_the_deadline():
    clock = FakeClock()
    check, calls = failing(times=1000)
    with pytest.raises(ConnectionRefusedError):
        wait_until(check, deadline=3 * RETRY_SECONDS, clock=clock, sleep=clock.sleep)
    assert clock.now <= 3 * RETRY_SECONDS
    assert len(calls) == 4


def test_minutes():
    assert minutes(0) == "0:00"
    assert minutes(65.9) == "1:05"
    assert minutes(600) == "10:00"


def test_the_logged_run_is_tagged_so_model_queries_can_skip_it(tmp_path):
    previous = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    try:
        run = mlflow.get_run(log_times({"rds": 47.0}))
    finally:
        mlflow.set_tracking_uri(previous)
    assert (run.info.run_name, run.data.tags["purpose"], run.data.metrics) == (
        "restart-drill",
        "restart-drill",
        {"rds_seconds": 47.0},
    )
