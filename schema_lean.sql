-- KartNest support agent: LEAN schema
-- Part 1: Olist tables (loaded from the CSVs)
-- Part 2: refund tables (written by the chatbot and human agents)
-- Part 3: safe views + restricted role for the LLM's SQL tool

-- =====================================================================
-- PART 1: OLIST TABLES
-- =====================================================================

-- Olist creates a NEW customer_id for every order.
-- customer_unique_id is the real person (the token's `sub`).
CREATE TABLE customers (
    customer_id              TEXT PRIMARY KEY,
    customer_unique_id       TEXT NOT NULL,
    customer_zip_code_prefix TEXT,
    customer_city            TEXT,
    customer_state           TEXT
);
CREATE INDEX idx_customers_unique ON customers (customer_unique_id);

CREATE TABLE orders (
    order_id                      TEXT PRIMARY KEY,
    customer_id                   TEXT NOT NULL REFERENCES customers (customer_id),
    order_status                  TEXT NOT NULL,
    order_purchase_timestamp      TIMESTAMP,
    order_approved_at             TIMESTAMP,
    order_delivered_carrier_date  TIMESTAMP,
    order_delivered_customer_date TIMESTAMP,
    order_estimated_delivery_date TIMESTAMP
);
CREATE INDEX idx_orders_customer ON orders (customer_id);

CREATE TABLE products (
    product_id            TEXT PRIMARY KEY,
    product_category_name TEXT
);

CREATE TABLE category_translation (
    product_category_name         TEXT PRIMARY KEY,
    product_category_name_english TEXT
);

CREATE TABLE order_items (
    order_id            TEXT NOT NULL REFERENCES orders (order_id),
    order_item_id       INT  NOT NULL,
    product_id          TEXT NOT NULL,
    seller_id           TEXT,
    shipping_limit_date TIMESTAMP,
    price               NUMERIC(10, 2) NOT NULL,
    freight_value       NUMERIC(10, 2) NOT NULL,
    PRIMARY KEY (order_id, order_item_id)
);
CREATE INDEX idx_items_product ON order_items (product_id);

CREATE TABLE order_payments (
    order_id             TEXT NOT NULL REFERENCES orders (order_id),
    payment_sequential   INT  NOT NULL,
    payment_type         TEXT,        -- credit_card, boleto, debit_card, voucher
    payment_installments INT,
    payment_value        NUMERIC(10, 2) NOT NULL,
    PRIMARY KEY (order_id, payment_sequential)
);

-- Olist has no product names or list prices, so we build a public catalog.
-- display_name: generate it. list_price: e.g. median price from order_items.
CREATE TABLE product_catalog (
    product_id   TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    category_en  TEXT,
    list_price   NUMERIC(10, 2) NOT NULL
);

-- =====================================================================
-- PART 2: REFUND TABLES
-- =====================================================================

-- One row per refund request.
CREATE TABLE refund_requests (
    refund_id          BIGSERIAL PRIMARY KEY,
    customer_unique_id TEXT NOT NULL,            -- from the token, never from the LLM
    order_id           TEXT NOT NULL,
    order_item_id      INT  NOT NULL,
    product_id         TEXT NOT NULL,
    amount             NUMERIC(10, 2) NOT NULL CHECK (amount > 0),
    reason             TEXT NOT NULL,
    customer_query     TEXT,                     -- what the customer wrote
    bot_response       TEXT,                     -- what the chatbot replied
    status             TEXT NOT NULL CHECK (status IN
                         ('auto_approved', 'pending_human', 'rejected')),
    approval_key       TEXT UNIQUE,              -- only set when sent to a human
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    FOREIGN KEY (order_id, order_item_id)
        REFERENCES order_items (order_id, order_item_id)
);

-- Policy rule: one active refund per order item
CREATE UNIQUE INDEX one_active_refund_per_item
    ON refund_requests (order_id, order_item_id)
    WHERE status IN ('auto_approved', 'pending_human');

CREATE INDEX idx_refund_customer_time
    ON refund_requests (customer_unique_id, created_at);

-- One row per human decision.
CREATE TABLE refund_approvals (
    approval_key TEXT PRIMARY KEY REFERENCES refund_requests (approval_key),
    decision     TEXT NOT NULL CHECK (decision IN ('approved', 'rejected')),
    approved_by  TEXT NOT NULL,
    reason       TEXT NOT NULL,
    decided_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- =====================================================================
-- PART 3: SAFE QUERY LAYER FOR THE LLM'S SQL TOOL
-- The LLM writes SELECT queries but never sees or passes the customer id.
-- The APP sets it per request, on a connection that logs in as llm_readonly:
--     SELECT set_config('app.customer_id', '<customer_unique_id from token>', false);
-- If it is not set, the personal views return 0 rows.
-- The app-side validator must allow only a single SELECT and block
-- set_config, pg_* functions and multiple statements.
-- =====================================================================

CREATE SCHEMA llm;

CREATE VIEW llm.my_orders WITH (security_barrier = true) AS
SELECT o.order_id, o.order_status, o.order_purchase_timestamp, o.order_approved_at,
       o.order_delivered_carrier_date, o.order_delivered_customer_date,
       o.order_estimated_delivery_date
FROM public.orders o
JOIN public.customers c ON c.customer_id = o.customer_id
WHERE c.customer_unique_id = current_setting('app.customer_id', true);

CREATE VIEW llm.my_order_items WITH (security_barrier = true) AS
SELECT i.order_id, i.order_item_id, i.product_id, i.price, i.freight_value
FROM public.order_items i
JOIN llm.my_orders o ON o.order_id = i.order_id;

CREATE VIEW llm.my_payments WITH (security_barrier = true) AS
SELECT p.order_id, p.payment_sequential, p.payment_type,
       p.payment_installments, p.payment_value
FROM public.order_payments p
JOIN llm.my_orders o ON o.order_id = p.order_id;

CREATE VIEW llm.my_refunds WITH (security_barrier = true) AS
SELECT r.refund_id, r.order_id, r.order_item_id, r.product_id,
       r.amount, r.reason, r.status, r.created_at
FROM public.refund_requests r
WHERE r.customer_unique_id = current_setting('app.customer_id', true);

-- Same for everyone: product names and prices
CREATE VIEW llm.catalog AS
SELECT product_id, display_name, category_en, list_price
FROM public.product_catalog;

-- Restricted login role used ONLY by the SQL tool.
-- It can read the llm.* views and nothing else. CHANGE THE PASSWORD.
CREATE ROLE llm_readonly LOGIN PASSWORD 'change-me';
GRANT USAGE ON SCHEMA llm TO llm_readonly;
GRANT SELECT ON llm.my_orders, llm.my_order_items, llm.my_payments,
                llm.my_refunds, llm.catalog TO llm_readonly;
ALTER ROLE llm_readonly SET search_path = llm;
ALTER ROLE llm_readonly SET statement_timeout = '3s';
ALTER ROLE llm_readonly SET default_transaction_read_only = on;
