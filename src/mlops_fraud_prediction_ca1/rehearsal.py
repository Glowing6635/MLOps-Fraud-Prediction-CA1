"""Generate fully synthetic rehearsal batches in the base CSV format (PBI #24).

Every value comes from a seeded generator; no real order, customer or label is
read or copied. Each batch is a directory holding ``customers_<batch_id>.csv``,
``orders_<batch_id>.csv`` and ``labels_<batch_id>.csv``: the three files share
the suffix that #20 uses as the batch_id. Output is generated on demand and is
never committed (``*.csv`` is ignored).

Expected rejections in the edge batch follow #20 only: a label whose order or
customer is missing is rejected and counted. The unseen category and malformed
values are not rejection rules (#19: they go to cleaning), so the edge batch
marks them as "must pass", not "must reject". Whether the loader really rejects
the flagged rows is for the #20 integration to verify, not asserted here.
"""

import argparse
import csv
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

CUSTOMER_COLUMNS = (
    "country_code",
    "customer_id",
    "mobile_verified",
    "num_orders_last_50days",
    "num_cancelled_orders_last_50days",
    "num_refund_orders_last_50days",
    "total_payment_last_50days",
    "num_associated_customers",
    "first_order_datetime",
)
ORDER_COLUMNS = (
    "country_code",
    "order_id",
    "collect_type",
    "payment_method",
    "order_value",
    "num_items_ordered",
    "refund_value",
    "order_date",
)
LABEL_COLUMNS = ("country_code", "order_id", "customer_id", "is_fraud")
TABLE_COLUMNS = {
    "customers": CUSTOMER_COLUMNS,
    "orders": ORDER_COLUMNS,
    "labels": LABEL_COLUMNS,
}

# Documented vocabulary (docs/cleaning.md); only category values already seen.
COUNTRIES = ("MY", "PH", "PK", "TH", "BD")
COLLECT_TYPES = ("delivery", "pickup", "self-pick-up")
PAYMENT_METHODS = ("payment on delivery", "online banking", "credit card", "balance")
NOVEL_PAYMENT_METHOD = "synthetic unseen wallet"

VALID_BATCHES = {"rehearsal_a": (1, "2025-01-06"), "rehearsal_b": (2, "2025-02-03")}
EDGE_BATCH = "rehearsal_edge"
EDGE_SEED = 3


@dataclass
class Batch:
    """Rows for one batch plus the identifiers an integration test can check."""

    batch_id: str
    customers: list[dict] = field(default_factory=list)
    orders: list[dict] = field(default_factory=list)
    labels: list[dict] = field(default_factory=list)
    novel_category_order_ids: list[str] = field(default_factory=list)
    missing_order_label_ids: list[str] = field(default_factory=list)
    missing_customer_label_ids: list[str] = field(default_factory=list)

    def tables(self) -> dict[str, list[dict]]:
        return {
            "customers": self.customers,
            "orders": self.orders,
            "labels": self.labels,
        }


def _add_linked(
    batch: Batch,
    rng: random.Random,
    *,
    tag: str,
    customer_count: int,
    order_count: int,
    start: date,
) -> None:
    """Append customers and orders, each order with a label to a known customer."""
    countries = {}
    for index in range(1, customer_count + 1):
        customer_id = f"SYN-{tag}-C{index:04d}"
        country = rng.choice(COUNTRIES)
        countries[customer_id] = country
        first = datetime.combine(start, datetime.min.time()) - timedelta(
            days=rng.randint(30, 400), seconds=rng.randint(0, 86399)
        )
        batch.customers.append(
            {
                "country_code": country,
                "customer_id": customer_id,
                "mobile_verified": rng.choice(("True", "False")),
                "num_orders_last_50days": 0,
                "num_cancelled_orders_last_50days": 0,
                "num_refund_orders_last_50days": 0,
                "total_payment_last_50days": "0.00",
                "num_associated_customers": rng.randint(0, 3),
                "first_order_datetime": first.strftime("%Y-%m-%d %H:%M:%S"),
            }
        )
    customer_ids = list(countries)
    batch_orders: dict[str, list[float]] = {cid: [] for cid in customer_ids}
    batch_refunds: dict[str, int] = dict.fromkeys(customer_ids, 0)
    for index in range(1, order_count + 1):
        order_id = f"SYN-{tag}-O{index:05d}"
        customer_id = rng.choice(customer_ids)
        country = countries[customer_id]
        value = rng.uniform(5, 400)
        refunded = rng.random() < 0.05
        batch_orders[customer_id].append(value)
        batch_refunds[customer_id] += refunded
        batch.orders.append(
            {
                "country_code": country,
                "order_id": order_id,
                "collect_type": rng.choice(COLLECT_TYPES),
                "payment_method": rng.choice(PAYMENT_METHODS),
                "order_value": f"{value:.2f}",
                "num_items_ordered": rng.randint(1, 6),
                "refund_value": f"{(value if refunded else 0):.2f}",
                "order_date": (start + timedelta(days=rng.randint(0, 20))).isoformat(),
            }
        )
        batch.labels.append(
            {
                "country_code": country,
                "order_id": order_id,
                "customer_id": customer_id,
                "is_fraud": int(rng.random() < 0.08),
            }
        )
    # The 50-day window is read as of the end of the batch's order dates
    # (``start`` + 20 days), so every batch order falls inside it. Each count
    # is the batch's own orders plus earlier history in the same window;
    # cancelled and refunded orders are distinct members of the order count.
    for customer in batch.customers:
        customer_id = customer["customer_id"]
        history = rng.randint(0, 4)
        cancelled = rng.randint(0, history)
        history_refunds = rng.randint(0, history - cancelled)
        customer["num_orders_last_50days"] = len(batch_orders[customer_id]) + history
        customer["num_cancelled_orders_last_50days"] = cancelled
        customer["num_refund_orders_last_50days"] = (
            batch_refunds[customer_id] + history_refunds
        )
        paid = sum(batch_orders[customer_id]) + sum(
            rng.uniform(5, 400) for _ in range(history - cancelled)
        )
        customer["total_payment_last_50days"] = f"{paid:.2f}"


def build_valid_batch(batch_id: str) -> Batch:
    """A valid batch: unique linked keys, in-vocabulary categories, 0/1 labels."""
    seed, start = VALID_BATCHES[batch_id]
    tag = batch_id.rsplit("_", 1)[-1].upper()
    batch = Batch(batch_id)
    _add_linked(
        batch,
        random.Random(seed),
        tag=tag,
        customer_count=40,
        order_count=120,
        start=date.fromisoformat(start),
    )
    return batch


def build_edge_batch() -> Batch:
    """A small batch: one unseen category on a valid row, plus flagged orphans."""
    batch = Batch(EDGE_BATCH)
    rng = random.Random(EDGE_SEED)
    _add_linked(
        batch,
        rng,
        tag="EDGE",
        customer_count=4,
        order_count=10,
        start=date(2025, 3, 3),
    )
    novel = batch.orders[0]
    novel["payment_method"] = NOVEL_PAYMENT_METHOD
    batch.novel_category_order_ids.append(novel["order_id"])
    known_customer = batch.customers[0]
    # Labels whose order has no orders row (#20: rejected and counted).
    for index in (1, 2):
        order_id = f"SYN-EDGE-ORPHAN-ORDER-{index:02d}"
        batch.labels.append(
            {
                "country_code": known_customer["country_code"],
                "order_id": order_id,
                "customer_id": known_customer["customer_id"],
                "is_fraud": 0,
            }
        )
        batch.missing_order_label_ids.append(order_id)
    # A label whose order exists but whose customer has no customers row.
    target = batch.orders[1]
    missing_customer = {
        "country_code": target["country_code"],
        "order_id": "SYN-EDGE-O-NOCUST",
        "customer_id": "SYN-EDGE-C-ABSENT",
        "is_fraud": 1,
    }
    batch.orders.append({**target, "order_id": "SYN-EDGE-O-NOCUST"})
    batch.labels.append(missing_customer)
    batch.missing_customer_label_ids.append("SYN-EDGE-O-NOCUST")
    return batch


def build_batches() -> list[Batch]:
    """Two distinct valid batches followed by the edge batch."""
    return [*(build_valid_batch(name) for name in VALID_BATCHES), build_edge_batch()]


def write_batch(batch: Batch, out_dir: Path) -> Path:
    """Write <out_dir>/<batch_id>/<table>_<batch_id>.csv for the three tables."""
    directory = out_dir / batch.batch_id
    directory.mkdir(parents=True, exist_ok=True)
    for name, rows in batch.tables().items():
        path = directory / f"{name}_{batch.batch_id}.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=TABLE_COLUMNS[name])
            writer.writeheader()
            writer.writerows(rows)
    return directory


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, type=Path, help="output directory")
    args = parser.parse_args(argv)
    for batch in build_batches():
        directory = write_batch(batch, args.out)
        print(  # noqa: T201 - the command's output
            f"{batch.batch_id}: {len(batch.customers)} customers, "
            f"{len(batch.orders)} orders, {len(batch.labels)} labels -> {directory}"
        )


if __name__ == "__main__":
    main()
