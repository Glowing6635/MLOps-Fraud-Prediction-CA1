"""Fetch a batch's three CSV files and check their shape before anything is loaded (issue #19).

A batch is three files named `customers_<batch>.csv`, `orders_<batch>.csv` and
`labels_<batch>.csv`; the shared suffix is the batch ID, `base` for the base files.
Each file is a URL or a local path, so the base files go through the same steps.

`read_batch` stops with a `BatchError` subclass, before any database work, when a
download fails, a file is not CSV, a required column is missing or renamed, or a
file has no rows. It does not judge values: an unseen category or a bad value
passes through to cleaning, and the loader counts the rows it rejects. An extra
column is logged as a warning and dropped, because the database schema is frozen.

Every value is read as literal text (blank stays blank), the form cleaning expects.
"""

import http.client
import io
import logging
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse

import pandas as pd

_LOGGER = logging.getLogger(__name__)

REQUIRED_COLUMNS = {
    "customers": (
        "country_code",
        "customer_id",
        "mobile_verified",
        "num_orders_last_50days",
        "num_cancelled_orders_last_50days",
        "num_refund_orders_last_50days",
        "total_payment_last_50days",
        "num_associated_customers",
        "first_order_datetime",
    ),
    "orders": (
        "country_code",
        "order_id",
        "collect_type",
        "payment_method",
        "order_value",
        "num_items_ordered",
        "refund_value",
        "order_date",
    ),
    "labels": ("country_code", "order_id", "customer_id", "is_fraud"),
}
# Seconds to wait for a connection, and then for each read from it.
TIMEOUT = 30
# Waits before the first, second and third retry; a server's Retry-After wins,
# up to the longest of these.
RETRY_DELAYS = (2, 5, 10)
_TEMPORARY_HTTP = {408, 429, 500, 502, 503, 504}
# The batch ID becomes a database key of at most 64 characters.
_BATCH_ID = re.compile(r"[A-Za-z0-9_-]{1,64}")


class BatchError(Exception):
    """The batch cannot be loaded; nothing has been written to the database."""


class DownloadFailed(BatchError):
    """A file could not be downloaded or read, even after retrying."""


class NotCsv(BatchError):
    """A file's content is not CSV text."""


class MissingColumns(BatchError):
    """A file lacks a required column, or has one under another name."""


class NoRows(BatchError):
    """A file has no data rows."""


class BadFileNames(BatchError):
    """The three file names do not share one batch suffix."""


@dataclass(frozen=True)
class Batch:
    """A batch's ID and its three files as text-only frames with exactly the required columns."""

    batch_id: str
    customers: pd.DataFrame
    orders: pd.DataFrame
    labels: pd.DataFrame


def read_batch(
    customers: str,
    orders: str,
    labels: str,
    *,
    timeout: float = TIMEOUT,
    sleep: Callable[[float], None] = time.sleep,
) -> Batch:
    """Fetch and check the three files, each a URL or a local path.

    Each download gets four attempts, each waiting up to `timeout` seconds to
    connect and then for each read, with at most 30 s of retry waits between
    them. So a server that never answers costs at worst about 4 * `timeout`
    + 30 s per file (150 s at the default) before `DownloadFailed`; callers
    with a time budget pass a shorter `timeout`. A server that keeps sending a
    little at a time is not cut off. Tests pass `sleep` so they need not wait.
    Raises a `BatchError` subclass naming the file and the problem.
    """
    sources = {"customers": customers, "orders": orders, "labels": labels}
    batch_id = _batch_id(sources)
    frames = {}
    for table, source in sources.items():
        name = _file_name(source)
        content = _fetch(source, name, timeout, sleep)
        frames[table] = _check(table, name, _parse(name, content))
    _LOGGER.info(
        "Batch %s checked | %s",
        batch_id,
        " | ".join(f"{table} {len(frame)} rows" for table, frame in frames.items()),
    )
    return Batch(batch_id, **frames)


def _file_name(source: str) -> str:
    if urlparse(source).scheme in {"http", "https"}:
        return Path(unquote(urlparse(source).path)).name
    return Path(source).name


def _batch_id(sources: dict[str, str]) -> str:
    suffixes = set()
    for table, source in sources.items():
        name = _file_name(source)
        match = re.fullmatch(rf"{table}_(.+)\.csv", name)
        if not match or not _BATCH_ID.fullmatch(match.group(1)):
            raise BadFileNames(
                f"{name or source} should be named {table}_<batch>.csv, where <batch>"
                " is up to 64 letters, digits, _ or -."
            )
        suffixes.add(match.group(1))
    if len(suffixes) != 1:
        raise BadFileNames(
            "The three files must share one batch suffix, but they have "
            + ", ".join(sorted(suffixes))
            + "."
        )
    return suffixes.pop()


def _fetch(
    source: str, name: str, timeout: float, sleep: Callable[[float], None]
) -> bytes:
    if urlparse(source).scheme not in {"http", "https"}:
        try:
            return Path(source).expanduser().read_bytes()
        except OSError as error:
            raise DownloadFailed(f"Could not read {source}: {error}") from error
    for attempt, delay in enumerate((*RETRY_DELAYS, None)):
        try:
            # The timeout applies to the connection and to each read after it, so a
            # server that keeps sending a little at a time is not cut off.
            with urllib.request.urlopen(source, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            if error.code not in _TEMPORARY_HTTP or delay is None:
                raise DownloadFailed(
                    f"Downloading {name} failed with HTTP {error.code}"
                    f" (attempts: {attempt + 1})."
                ) from error
            delay = _retry_after(error, delay)
            problem = f"HTTP {error.code}"
        # OSError covers refused and reset connections, timeouts, DNS and TLS errors.
        # Lasting ones, such as a bad certificate, are retried too: at most 17 s lost.
        except (OSError, http.client.HTTPException) as error:
            if delay is None:
                raise DownloadFailed(
                    f"Downloading {name} failed (attempts: {attempt + 1}): {error}"
                ) from error
            problem = str(error)
        _LOGGER.warning(
            "Download of %s failed (%s); retrying in %s s", name, problem, delay
        )
        sleep(delay)
    raise AssertionError("unreachable: the last attempt returns or raises")


def _retry_after(error: urllib.error.HTTPError, delay: float) -> float:
    value = error.headers.get("Retry-After", "") if error.headers else ""
    if value.strip().isdigit():
        return min(float(value), max(RETRY_DELAYS))
    return delay


def _parse(name: str, content: bytes) -> pd.DataFrame:
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise NotCsv(f"{name} is not UTF-8 text, so it is not a CSV file.") from error
    # An error page or a JSON reply can still parse as one odd column, so look first.
    start = text.lstrip()[:1]
    if "\x00" in text or start in {"<", "{", "["}:
        raise NotCsv(f"{name} looks like a web page or binary file, not CSV.")
    if not text.strip():
        raise NoRows(f"{name} is empty.")
    try:
        return pd.read_csv(io.StringIO(text), dtype="string", keep_default_na=False)
    except pd.errors.ParserError as error:
        raise NotCsv(f"{name} could not be read as CSV: {error}") from error


def _check(table: str, name: str, frame: pd.DataFrame) -> pd.DataFrame:
    required = REQUIRED_COLUMNS[table]
    missing = [column for column in required if column not in frame.columns]
    extra = [column for column in frame.columns if column not in required]
    if missing:
        found = f" It has instead: {', '.join(extra)}." if extra else ""
        raise MissingColumns(
            f"{name} is missing the column(s) {', '.join(missing)}.{found}"
        )
    if frame.empty:
        raise NoRows(f"{name} has a header but no rows.")
    if extra:
        _LOGGER.warning(
            "%s has extra column(s) %s; they are not loaded.", name, ", ".join(extra)
        )
    return frame[list(required)]
