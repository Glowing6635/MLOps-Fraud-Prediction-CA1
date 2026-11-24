# Data dictionary

The team database in RDS MySQL holds the three source files, one row per source row, plus a table of loaded batches. `src/mlops_fraud_prediction_ca1/schema.sql` creates it, and `docs/runbook.md` gives the steps.

## Diagram

```mermaid
erDiagram
    batches ||--o{ customers : "loaded in"
    batches ||--o{ orders : "loaded in"
    batches ||--o{ labels : "loaded in"
    orders ||--o| labels : "labelled by"
    customers ||--o{ labels : "placed (customer_id)"

    batches {
        varchar batch_id PK
        int batch_seq UK
        datetime loaded_at
    }
    customers {
        varchar customer_id PK
        varchar batch_id PK, FK
        char country_code
        boolean mobile_verified
        decimal total_payment_last_50days
        datetime first_order_datetime
    }
    orders {
        char country_code PK
        varchar order_id PK
        varchar batch_id FK
        decimal order_value
        decimal refund_value
        date order_date
    }
    labels {
        char country_code PK, FK
        varchar order_id PK, FK
        varchar batch_id FK
        varchar customer_id
        tinyint is_fraud
    }
```

The diagram shows only the key and money columns; the tables below list every column.

## Keys

The keys follow the kickoff data analysis (issue #3), section 4.4 "Linkage findings and handling" in `notebooks/data_analysis.py`:

- **Orders and labels** are keyed by `country_code` and `order_id` together, because three order IDs each appear in two countries.
- **Customers** are keyed by `customer_id` and `batch_id`. Customer IDs are unique across countries, so `country_code` is not part of the key, and a customer can have one row per batch. `docs/time-split.md` still links customers by `country_code` and `customer_id` together. On the base files both find the same customers, because the same section finds that labels and customers agree on every shared customer's country once country codes are normalised.
- **Labels to customers.** A label names its order's customer. This is not a foreign key, because a customer has a row for each batch and two labelled orders in the base files have no customer row.
- **IDs are opaque text** and compare exactly, so `ab12` and `AB12` are different keys, and so are `001` and `1`.

A row whose key is blank or whose country is not recognised cannot be stored, so the loader rejects and counts it.

## Cleaned and original values

Each source column the cleaning module (issue #12) can change is stored twice: the cleaned value under the source column's name, and the original text exactly as it arrived under the same name with `raw_` in front. The cleaned value is empty (NULL) when cleaning could not read the original. IDs are never changed by cleaning, so they have no `raw_` column.

Money uses an exact decimal type with 12 decimal places. Some source amounts have 13 to 16 digits after the point, such as `5.387818896776103`, but those extra digits are rounding noise from how the source stored its numbers, so 12 places keep every real digit. The `raw_` column keeps the exact text. Counts that the source writes with a decimal point use `DOUBLE`.

## Tables

### batches

One row per loaded batch.

| Column | Type | Meaning |
| --- | --- | --- |
| `batch_id` | text, key | The suffix the batch's three file names share, such as `base` for `customers_base.csv`. |
| `batch_seq` | whole number | Loading order: 1 for the first batch loaded, then 2, and so on. |
| `loaded_at` | date and time | When the batch row was added, in the server's time zone. |

### customers

One row per customer per batch.

| Column | Type | Meaning |
| --- | --- | --- |
| `customer_id` | text, key | The customer. |
| `batch_id` | text, key | The batch the row came in. |
| `country_code` | 2 letters | The customer's country: `MY`, `PH`, `PK`, `TH` or `BD`. |
| `mobile_verified` | true or false | Whether the customer's mobile number is verified. |
| `num_orders_last_50days` | number | Orders in the last 50 days. |
| `num_cancelled_orders_last_50days` | number | Cancelled orders in the last 50 days. |
| `num_refund_orders_last_50days` | number | Orders with a refund request in the last 50 days. |
| `total_payment_last_50days` | money | Total paid in the last 50 days. |
| `num_associated_customers` | number | Customers linked to this one. |
| `first_order_datetime` | date and time | The customer's first order. |
| `raw_...` | text | The original value of each column above except `customer_id` and `batch_id`. |

Issue #4 records when each customer value becomes known. None is confirmed yet, so none of them is a model input by default.

### orders

One row per order.

| Column | Type | Meaning |
| --- | --- | --- |
| `country_code` | 2 letters, key | The order's country. |
| `order_id` | text, key | The order, unique within its country. |
| `batch_id` | text | The batch the order came in. |
| `collect_type` | text | How the order is collected: `delivery`, `pickup` or `self-pick-up`. |
| `payment_method` | text | How the order was paid, in lowercase, such as `credit card`. |
| `order_value` | money | The order's value when it was placed. |
| `num_items_ordered` | number | Items in the order. |
| `refund_value` | money | The amount refunded. It is known only after the order, so it is used to evaluate models and never as a model input. |
| `order_date` | date | The day the order was placed. |
| `raw_...` | text | The original value of each column above except `order_id` and `batch_id`. |

### labels

One row per labelled order. Every label's order must already be in `orders`.

| Column | Type | Meaning |
| --- | --- | --- |
| `country_code` | 2 letters, key | The order's country. |
| `order_id` | text, key | The order. |
| `batch_id` | text | The batch the label came in. |
| `customer_id` | text | The customer who placed the order. |
| `is_fraud` | 0 or 1 | 1 if the order was fraud. Empty when the source value is not 0 or 1. |
| `raw_country_code`, `raw_is_fraud` | text | The original values. |

## Who can read it

The admin user creates the tables and users. Each engineer has their own user, named `<name>_reader`, that can only read. `uv run rds-schema reader <name>` creates it, and the password is never in git.
