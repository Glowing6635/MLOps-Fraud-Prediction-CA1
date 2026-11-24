import http.server
import logging
import threading
import time

import pytest

from mlops_fraud_prediction_ca1.batch_files import (
    RETRY_DELAYS,
    BadFileNames,
    DownloadFailed,
    MissingColumns,
    NoRows,
    NotCsv,
    read_batch,
)

# Made-up rows: one customer, one order and its label.
GOOD = {
    "customers": (
        "country_code,customer_id,mobile_verified,num_orders_last_50days,"
        "num_cancelled_orders_last_50days,num_refund_orders_last_50days,"
        "total_payment_last_50days,num_associated_customers,first_order_datetime\n"
        "MY,c1,true,3.0,0.0,1.0,45.5,1.0,2022-10-01 10:00:00\n"
    ),
    "orders": (
        "country_code,order_id,collect_type,payment_method,order_value,"
        "num_items_ordered,refund_value,order_date\n"
        "MY,o1,delivery,credit card,20.5,2.0,0.0,2023/01/05\n"
    ),
    "labels": "country_code,order_id,customer_id,is_fraud\nMY,o1,c1,0\n",
}


def write_batch(directory, batch="7", **replace):
    paths = []
    for table, text in GOOD.items():
        path = directory / f"{table}_{batch}.csv"
        content = replace.get(table, text)
        path.write_bytes(content if isinstance(content, bytes) else content.encode())
        paths.append(str(path))
    return paths


def test_a_good_batch_keeps_every_value_as_literal_text(tmp_path):
    batch = read_batch(*write_batch(tmp_path, batch="base"))
    assert batch.batch_id == "base"
    assert batch.labels.to_dict("records") == [
        {"country_code": "MY", "order_id": "o1", "customer_id": "c1", "is_fraud": "0"}
    ]
    assert batch.orders.dtypes.eq("string").all()


@pytest.mark.parametrize(
    ("table", "content", "error"),
    [
        ("orders", "<html><body>Not found</body></html>", NotCsv),
        ("orders", b"\xff\xfe\x00binary", NotCsv),
        ("labels", '{"error": "denied"}', NotCsv),
        ("labels", 'country_code,order_id\n"MY,o1\n', NotCsv),
        ("labels", "country_code,order_id,customer_id\nMY,o1,c1\n", MissingColumns),
        (
            "labels",
            "country_code,order_id,cust_id,is_fraud\nMY,o1,c1,0\n",
            MissingColumns,
        ),
        ("customers", GOOD["customers"].splitlines()[0] + "\n", NoRows),
        ("customers", "", NoRows),
    ],
    ids=[
        "html",
        "binary",
        "json",
        "unparseable",
        "missing",
        "renamed",
        "header",
        "empty",
    ],
)
def test_a_bad_file_stops_the_batch_with_a_named_error(tmp_path, table, content, error):
    with pytest.raises(error, match=f"{table}_7.csv"):
        read_batch(*write_batch(tmp_path, **{table: content}))


@pytest.mark.parametrize(
    "names",
    [
        ("customers_1.csv", "orders_1.csv", "labels_2.csv"),
        ("customers_1.csv", "orders_1.csv", "fraud_1.csv"),
        ("customers_1.csv", "orders_1.csv", "labels_1 2.csv"),
    ],
)
def test_files_must_share_one_batch_suffix(names):
    with pytest.raises(BadFileNames):
        read_batch(*names)


def test_an_extra_column_is_a_warning_and_is_dropped(tmp_path, caplog):
    labels = "country_code,order_id,customer_id,is_fraud,note\nMY,o1,c1,0,x\n"
    with caplog.at_level(logging.WARNING):
        batch = read_batch(*write_batch(tmp_path, labels=labels))
    assert list(batch.labels.columns) == [
        "country_code",
        "order_id",
        "customer_id",
        "is_fraud",
    ]
    assert "note" in caplog.text


def test_unseen_categories_and_bad_values_pass_through_unchanged(tmp_path):
    orders = (
        GOOD["orders"].splitlines()[0]
        + "\nzz,o2,drone,barter,not a number,,-1,31/31/2023\n"
    )
    batch = read_batch(*write_batch(tmp_path, orders=orders))
    assert batch.orders.iloc[0].tolist() == [
        "zz",
        "o2",
        "drone",
        "barter",
        "not a number",
        "",
        "-1",
        "31/31/2023",
    ]


class Server:
    """A local HTTP server that answers each request with the next scripted reply."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = 0
        server = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                server.requests += 1
                status, body, headers, pause = server.replies.pop(0)
                time.sleep(pause)
                try:
                    self.send_response(status)
                    for name, value in headers.items():
                        self.send_header(name, value)
                    self.end_headers()
                    self.wfile.write(body.encode())
                except ConnectionError:
                    pass  # The client gave up waiting, which is what the test wants.

            def log_message(self, *args):
                pass

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(
            target=self.httpd.serve_forever, args=(0.05,), daemon=True
        )
        self.thread.start()

    def urls(self, batch="7"):
        port = self.httpd.server_address[1]
        return [f"http://127.0.0.1:{port}/files/{t}_{batch}.csv" for t in GOOD]

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def serve():
    servers = []

    def start(replies):
        servers.append(Server(replies))
        return servers[-1]

    yield start
    for server in servers:
        server.close()


def ok(table):
    return (200, GOOD[table], {}, 0)


def test_downloads_retry_temporary_failures_and_honour_retry_after(serve):
    server = serve(
        [(503, "", {}, 0), (429, "", {"Retry-After": "1"}, 0), ok("customers")]
        + [ok("orders"), ok("labels")]
    )
    waits = []
    batch = read_batch(*server.urls(), sleep=waits.append)
    assert (batch.batch_id, server.requests, waits) == ("7", 5, [RETRY_DELAYS[0], 1])


@pytest.mark.parametrize(
    ("reply", "requests"),
    [((404, "", {}, 0), 1), ((503, "", {}, 0), 4), ((200, "", {}, 0.5), 4)],
    ids=["not-found-is-not-retried", "three-retries-at-most", "read-timeout"],
)
def test_a_failed_download_stops_after_at_most_three_retries(serve, reply, requests):
    server = serve([reply] * 4)
    with pytest.raises(DownloadFailed, match="customers_7.csv"):
        read_batch(*server.urls(), timeout=0.1, sleep=lambda _: None)
    assert server.requests == requests


def test_an_unreachable_host_fails_after_three_retries(serve):
    server = serve([])
    urls = server.urls()
    server.close()  # Nothing listens on the port now, so every connection is refused.
    waits = []
    with pytest.raises(DownloadFailed):
        read_batch(*urls, timeout=0.1, sleep=waits.append)
    assert waits == list(RETRY_DELAYS)
