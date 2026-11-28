"""The five timed batch stages of D27-01, behind small interfaces (issue #27).

`batch.main` runs the stages in order and prints each one's elapsed wall-clock
seconds against its target. The interfaces use the stable contracts already on
the branch: `batch_files.Batch` for fetched data and a plain `{table: rows}`
dictionary for counts. Real adapters over #19 (`read_batch`), #20 (`load_batch`),
the #20 reader and #25 (`retrain`) are not wired yet; the in-memory fakes at the
bottom let the wiring, timing output and exit codes be tested without a database.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol
from urllib.parse import quote, urlsplit, urlunsplit

import pandas as pd

from .batch_files import Batch

TABLES = ("customers", "orders", "labels")
# D27-01: (stage, target seconds). The ten seconds left of the 180 s command limit
# cover process startup, configuration and shutdown.
STAGE_TARGETS = (
    ("fetch and validate", 25),
    ("clean and load", 30),
    ("read training data", 15),
    ("train and log", 85),
    ("report", 15),
)
COMMAND_LIMIT_SECONDS = 180


class StageUnavailable(Exception):
    """The real implementation of a stage is not wired in yet."""


class StageFailed(Exception):
    """A stage ran but did not succeed; the message is safe to print."""


@dataclass(frozen=True)
class TrainOutcome:
    """What training and logging produced: MLflow runs logged and failed candidates."""

    runs_logged: int
    experiment_id: str
    failures: Mapping[str, str] = field(default_factory=dict)


class FetchStage(Protocol):
    def run(self, customers: str, orders: str, labels: str) -> Batch: ...


class LoadStage(Protocol):
    def run(self, batch: Batch) -> dict[str, int]:
        """Clean and commit the batch; return each table's row count from before the load."""


class ReadStage(Protocol):
    def run(self) -> pd.DataFrame: ...


class TrainStage(Protocol):
    def run(
        self, rows: pd.DataFrame, batch_id: str, etl_run_id: str
    ) -> TrainOutcome: ...


class CountStage(Protocol):
    def run(self) -> dict[str, int]:
        """Each table's row count now."""


@dataclass(frozen=True)
class Stages:
    fetch: FetchStage
    load: LoadStage
    read: ReadStage
    train: TrainStage
    count: CountStage


class _Unavailable:
    def __init__(self, issue: str, stage: str):
        self._message = f"{stage} stage ({issue}) is unavailable."

    def run(self, *args, **kwargs):
        raise StageUnavailable(self._message)


def unwired_stages() -> Stages:
    """Every stage raises `StageUnavailable`: the default until the real adapters land."""
    return Stages(
        fetch=_Unavailable("#19", "fetch and validate"),
        load=_Unavailable("#20", "clean and load"),
        read=_Unavailable("#20", "read training data"),
        train=_Unavailable("#25", "train and log"),
        count=_Unavailable("#20", "report"),
    )


def mlflow_run_url(tracking_uri: str, experiment_id: str, etl_run_id: str) -> str:
    """The MLflow runs page for one experiment, filtered to this run's `etl_run_id` tag.

    Any credentials embedded in `tracking_uri` are dropped.
    """
    parts = urlsplit(tracking_uri)
    host = parts.hostname or ""
    if parts.port:
        host = f"{host}:{parts.port}"
    base = urlunsplit((parts.scheme, host, parts.path.rstrip("/"), "", ""))
    search = quote(f'tags.etl_run_id = "{etl_run_id}"', safe="")
    return f"{base}/#/experiments/{experiment_id}/runs?searchFilter={search}"


# In-memory stages, for tests and rehearsing the wiring. They touch no database,
# network or file, and raise nothing unless a test asks them to.


class FakeFetch:
    def __init__(self, batch: Batch):
        self.batch = batch
        self.calls: list[tuple[str, str, str]] = []

    def run(self, customers: str, orders: str, labels: str) -> Batch:
        self.calls.append((customers, orders, labels))
        return self.batch


class FakeDatabase:
    """Row counts per table that a fake load grows and a fake count reads."""

    def __init__(self, **counts: int):
        self.counts = {table: counts.get(table, 0) for table in TABLES}


class FakeLoad:
    def __init__(self, database: FakeDatabase):
        self.database = database

    def run(self, batch: Batch) -> dict[str, int]:
        before = dict(self.database.counts)
        for table in TABLES:
            self.database.counts[table] += len(getattr(batch, table))
        return before


class FakeRead:
    def __init__(self, rows: pd.DataFrame | None = None):
        self.rows = pd.DataFrame({"is_fraud": [0, 1]}) if rows is None else rows

    def run(self) -> pd.DataFrame:
        return self.rows


class FakeTrain:
    def __init__(self, runs_logged: int = 2, failures: Mapping[str, str] | None = None):
        self.outcome = TrainOutcome(runs_logged, "1", dict(failures or {}))
        self.calls: list[tuple[int, str, str]] = []

    def run(self, rows: pd.DataFrame, batch_id: str, etl_run_id: str) -> TrainOutcome:
        self.calls.append((len(rows), batch_id, etl_run_id))
        return self.outcome


class FakeCount:
    def __init__(self, database: FakeDatabase):
        self.database = database

    def run(self) -> dict[str, int]:
        return dict(self.database.counts)
