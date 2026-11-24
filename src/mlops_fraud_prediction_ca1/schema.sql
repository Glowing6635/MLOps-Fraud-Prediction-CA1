-- The team database's tables (issue #10). `uv run rds-schema tables` runs this file;
-- every statement leaves an existing table alone, so it is safe to re-run.
-- docs/data-dictionary.md explains every column.
-- rds_schema.statements() drops comment lines and splits the rest on semicolons,
-- so never put a semicolon in a string or in a comment that follows code on a line.
-- IDs use a binary collation because they are opaque: MySQL's default collation
-- would treat "ab12" and "AB12" as the same key.

CREATE TABLE IF NOT EXISTS batches (
    batch_id VARCHAR(64) COLLATE utf8mb4_bin NOT NULL,
    batch_seq INT UNSIGNED NOT NULL AUTO_INCREMENT,
    loaded_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (batch_id),
    UNIQUE KEY batch_seq (batch_seq)
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4;

CREATE TABLE IF NOT EXISTS customers (
    customer_id VARCHAR(64) COLLATE utf8mb4_bin NOT NULL,
    batch_id VARCHAR(64) COLLATE utf8mb4_bin NOT NULL,
    country_code CHAR(2) NULL,
    mobile_verified BOOLEAN NULL,
    num_orders_last_50days DOUBLE NULL,
    num_cancelled_orders_last_50days DOUBLE NULL,
    num_refund_orders_last_50days DOUBLE NULL,
    total_payment_last_50days DECIMAL(30, 12) NULL,
    num_associated_customers DOUBLE NULL,
    first_order_datetime DATETIME NULL,
    raw_country_code VARCHAR(255) NULL,
    raw_mobile_verified VARCHAR(255) NULL,
    raw_num_orders_last_50days VARCHAR(255) NULL,
    raw_num_cancelled_orders_last_50days VARCHAR(255) NULL,
    raw_num_refund_orders_last_50days VARCHAR(255) NULL,
    raw_total_payment_last_50days VARCHAR(255) NULL,
    raw_num_associated_customers VARCHAR(255) NULL,
    raw_first_order_datetime VARCHAR(255) NULL,
    PRIMARY KEY (customer_id, batch_id),
    KEY customers_batch (batch_id),
    CONSTRAINT customers_batch FOREIGN KEY (batch_id) REFERENCES batches (batch_id)
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4;

CREATE TABLE IF NOT EXISTS orders (
    country_code CHAR(2) NOT NULL,
    order_id VARCHAR(64) COLLATE utf8mb4_bin NOT NULL,
    batch_id VARCHAR(64) COLLATE utf8mb4_bin NOT NULL,
    collect_type VARCHAR(64) NULL,
    payment_method VARCHAR(64) NULL,
    order_value DECIMAL(30, 12) NULL,
    num_items_ordered DOUBLE NULL,
    refund_value DECIMAL(30, 12) NULL,
    order_date DATE NULL,
    raw_country_code VARCHAR(255) NULL,
    raw_collect_type VARCHAR(255) NULL,
    raw_payment_method VARCHAR(255) NULL,
    raw_order_value VARCHAR(255) NULL,
    raw_num_items_ordered VARCHAR(255) NULL,
    raw_refund_value VARCHAR(255) NULL,
    raw_order_date VARCHAR(255) NULL,
    PRIMARY KEY (country_code, order_id),
    KEY orders_batch (batch_id),
    KEY orders_date (order_date),
    CONSTRAINT orders_batch FOREIGN KEY (batch_id) REFERENCES batches (batch_id)
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4;

CREATE TABLE IF NOT EXISTS labels (
    country_code CHAR(2) NOT NULL,
    order_id VARCHAR(64) COLLATE utf8mb4_bin NOT NULL,
    batch_id VARCHAR(64) COLLATE utf8mb4_bin NOT NULL,
    customer_id VARCHAR(64) COLLATE utf8mb4_bin NULL,
    is_fraud TINYINT NULL,
    raw_country_code VARCHAR(255) NULL,
    raw_is_fraud VARCHAR(255) NULL,
    PRIMARY KEY (country_code, order_id),
    KEY labels_batch (batch_id),
    KEY labels_customer (customer_id),
    CONSTRAINT labels_is_fraud CHECK (is_fraud IN (0, 1)),
    CONSTRAINT labels_batch FOREIGN KEY (batch_id) REFERENCES batches (batch_id),
    CONSTRAINT labels_order FOREIGN KEY (country_code, order_id)
        REFERENCES orders (country_code, order_id)
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4;
