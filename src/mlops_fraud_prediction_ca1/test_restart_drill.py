import pytest

from mlops_fraud_prediction_ca1.restart_drill import RETRY_SECONDS, minutes, wait_until


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
