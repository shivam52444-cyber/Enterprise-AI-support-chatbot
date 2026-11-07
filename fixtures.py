"""
fixtures.py - finds real test data in YOUR database for the agent evaluation.

The Olist data is real and random, so the test cases cannot hard-code order ids.
Instead the golden file uses placeholders like {a_small_order_id}, and this file looks up
real values (and the correct answers, computed by plain SQL) to put in their place.

To use one of your demo customers, set EVAL_CUSTOMER_ID=<customer_unique_id> in .env.
"""

import datetime as dt
import os

# Same list as NON_RETURNABLE_CATEGORIES in refunds.py
NON_RETURNABLE = ("food", "food_drink", "drinks", "fashion_underwear_beach")

KNOWN_KEYS = {
    "customer_id",
    "a_latest_order_id", "a_latest_status", "a_latest_items", "a_latest_freight", "a_latest_total",
    "a_small_order_id", "a_small_product_id", "a_small_price", "a_small_freight",
    "a_small_amount_defective", "a_delivered_days", "a_delivered_timing", "a_voucher_expected",
    "other_customer_id", "other_order_id", "other_order_total",
    "catalog_name", "catalog_price",
}

# A delivered order with exactly ONE item (so the bot never has to ask "which item?"), a price
# up to R$ 150 (so a change-of-mind refund is auto-approved), a returnable category, and no
# refund request on it yet.
SMALL_ORDER_SQL = """
SELECT c.customer_unique_id, o.order_id, i.product_id, i.price, i.freight_value,
       o.order_purchase_timestamp, o.order_delivered_customer_date, o.order_estimated_delivery_date
FROM orders o
JOIN customers c   ON c.customer_id = o.customer_id
JOIN order_items i ON i.order_id = o.order_id
LEFT JOIN product_catalog pc ON pc.product_id = i.product_id
WHERE o.order_status = 'delivered'
  AND o.order_delivered_customer_date IS NOT NULL
  AND o.order_estimated_delivery_date IS NOT NULL
  AND i.price BETWEEN 20 AND 150
  AND NOT (COALESCE(pc.category_en, '') = ANY(%s))
  AND (SELECT COUNT(*) FROM order_items x WHERE x.order_id = o.order_id) = 1
  AND NOT EXISTS (SELECT 1 FROM refund_requests r WHERE r.order_id = o.order_id)
  AND c.customer_unique_id IN (SELECT customer_unique_id FROM customers GROUP BY 1 HAVING COUNT(*) >= 2)
  {customer_filter}
ORDER BY c.customer_unique_id, o.order_id
LIMIT 1
"""

LATEST_ORDER_SQL = """
SELECT o.order_id, o.order_status, COALESCE(SUM(i.price), 0), COALESCE(SUM(i.freight_value), 0)
FROM orders o
JOIN customers c ON c.customer_id = o.customer_id
LEFT JOIN order_items i ON i.order_id = o.order_id
WHERE c.customer_unique_id = %s
GROUP BY o.order_id, o.order_status, o.order_purchase_timestamp
ORDER BY o.order_purchase_timestamp DESC LIMIT 1
"""

OTHER_ORDER_SQL = """
SELECT c.customer_unique_id, o.order_id, SUM(i.price + i.freight_value)
FROM orders o
JOIN customers c   ON c.customer_id = o.customer_id
JOIN order_items i ON i.order_id = o.order_id
WHERE c.customer_unique_id <> %s
GROUP BY c.customer_unique_id, o.order_id
HAVING SUM(i.price + i.freight_value) > 30
ORDER BY c.customer_unique_id, o.order_id LIMIT 1
"""

CATALOG_SQL = "SELECT display_name, list_price FROM product_catalog ORDER BY product_id LIMIT 1 OFFSET 100"


def _money(x):
    return f"{float(x):.2f}"


def _timing(days):
    if days > 0:
        return f"{days} days AFTER the estimated delivery date"
    if days < 0:
        return f"{-days} days BEFORE the estimated delivery date (early)"
    return "exactly on the estimated delivery date"


def build_fixtures(pool, customer_id=None):
    """Returns (fixtures, delivered_date). Missing pieces are simply left out, and the test
    cases that need them are skipped with a clear message."""
    fx, delivered_date = {}, None
    customer_id = customer_id or os.getenv("EVAL_CUSTOMER_ID") or None
    with pool.connection() as conn, conn.cursor() as cur:
        sql = SMALL_ORDER_SQL.format(
            customer_filter="AND c.customer_unique_id = %s" if customer_id else "")
        blocked = list(NON_RETURNABLE)   # a list, because psycopg 3 needs ANY(list), not IN (tuple)
        cur.execute(sql, (blocked, customer_id) if customer_id else (blocked,))
        row = cur.fetchone()
        if row is None:
            return fx, None
        (cust, order_id, product_id, price, freight, purchased, delivered, estimated) = row
        late_days = (delivered.date() - estimated.date()).days
        fx.update({
            "customer_id": cust, "a_small_order_id": order_id, "a_small_product_id": product_id,
            "a_small_price": _money(price), "a_small_freight": _money(freight),
            "a_small_amount_defective": _money(float(price) + float(freight)),
            "a_delivered_days": str((delivered - purchased).days),
            "a_delivered_timing": _timing(late_days),
            "a_voucher_expected": "yes" if late_days > 7 else "no",
        })
        delivered_date = delivered.date()

        cur.execute(LATEST_ORDER_SQL, (cust,))
        row = cur.fetchone()
        if row:
            oid, status, items, freight_sum = row
            fx.update({"a_latest_order_id": oid, "a_latest_status": status,
                       "a_latest_items": _money(items), "a_latest_freight": _money(freight_sum),
                       "a_latest_total": _money(float(items) + float(freight_sum))})

        cur.execute(OTHER_ORDER_SQL, (cust,))
        row = cur.fetchone()
        if row:
            fx.update({"other_customer_id": row[0], "other_order_id": row[1],
                       "other_order_total": _money(row[2])})

        cur.execute(CATALOG_SQL)
        row = cur.fetchone()
        if row:
            fx.update({"catalog_name": row[0], "catalog_price": _money(row[1])})
    return fx, delivered_date


def as_of_date(delivered_date, days_after):
    return delivered_date + dt.timedelta(days=days_after)
