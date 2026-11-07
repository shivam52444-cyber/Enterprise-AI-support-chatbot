"""
refunds.py - the KartNest refund rules written as plain Python.

This is NOT AI-written SQL. The AI only says WHICH order, WHICH item and WHY.
The server checks ownership, applies the policy rules, decides who must approve,
and writes the row. The AI cannot skip a rule.

Rules come from the policy files: 04 returns, 05 refund policy, 07 approval limits,
08 damaged/wrong/missing items, 03 delivery delays.

"Today" is AS_OF_DATE (from .env). The Olist data is from 2016-2018, so for demos set
AS_OF_DATE to a date near your test order's delivery date.
"""

import datetime as dt
import logging
import secrets

log = logging.getLogger("kartnest.refunds")

REASONS = ("change_of_mind", "defective", "wrong_item", "missing_items", "not_delivered")

RETURN_WINDOW_DAYS = 30            # change of mind: days after delivery
CLAIM_WINDOW_DAYS = 7              # defective / wrong / missing: days after delivery
NOT_RECEIVED_WAIT_BUSINESS_DAYS = 2
NOT_RECEIVED_CLAIM_DAYS = 15       # "marked delivered but not received": days after delivery
DELAY_CASE_BUSINESS_DAYS = 2       # late shipment: wait this long past the estimate
LATE_CANCEL_DAYS = 10              # late shipment: cancel + full refund after this many days
RETURN_LABEL_FEE = 15.0
AUTO_LIMIT = 150.0                 # up to this: approved automatically
AGENT_LIMIT = 800.0                # up to this: human agent; above: senior team
RECENT_REFUND_LIMIT = 3            # this many requests ...
RECENT_REFUND_DAYS = 60            # ... in this many days -> human review
# Hygiene/perishable categories can't be returned for change of mind. Check your own list with:
#   SELECT DISTINCT category_en FROM product_catalog ORDER BY 1;
NON_RETURNABLE_CATEGORIES = {"food", "food_drink", "drinks", "fashion_underwear_beach"}

LEVELS = ("auto", "human_agent", "senior")
ACTIVE_STATUSES = ("auto_approved", "pending_human", "approved")


# ------------------------------------------------------------------ small helpers
def business_days_between(start, end):
    """Weekdays (Mon-Fri) after `start`, up to and including `end`. Holidays are ignored."""
    count, day = 0, start
    while day < end:
        day += dt.timedelta(days=1)
        if day.weekday() < 5:
            count += 1
    return count


def add_business_days(start, n):
    day = start
    while n > 0:
        day += dt.timedelta(days=1)
        if day.weekday() < 5:
            n -= 1
    return day


def _to_date(value):
    if value is None:
        return None
    return value.date() if isinstance(value, dt.datetime) else value


def _level_for_amount(amount):
    if amount <= AUTO_LIMIT:
        return "auto"
    return "human_agent" if amount <= AGENT_LIMIT else "senior"


def _max_level(a, b):
    return LEVELS[max(LEVELS.index(a), LEVELS.index(b))]


def _next_step(approval, reason):
    if approval == "auto":
        if reason == "change_of_mind":
            return ("The request is approved. A return label will be emailed. The refund is sent "
                    "after the returned item is received and inspected (2 to 3 business days) and "
                    "then reaches the original payment method (credit card 5 to 10 business days, "
                    "debit card 3 to 7, boleto 5 to 8 after bank details, voucher immediately).")
        return "The request is approved."
    if approval == "human_agent":
        return ("A human support agent will review the request within 2 business days and contact "
                "the customer (for example to ask for photos). Approval is not guaranteed.")
    return ("The senior support team will review the request within 5 business days. "
            "Approval is not guaranteed.")


def _deny(reasons, next_step, **extra):
    result = {"eligible": False, "amount": None, "approval": None,
              "reasons": reasons, "notes": [], "next_step": next_step}
    result.update(extra)
    return result


# ------------------------------------------------------------------ the rules (pure function)
def evaluate_refund(facts, reason, as_of, recent_refunds=0):
    """Decide eligibility. `facts` has: status, delivered_at, estimated_at, price, freight,
    category, active_refund (None or dict with status). No database access here."""
    status = facts["status"]
    price, freight = float(facts["price"]), float(facts["freight"])
    delivered, estimated = _to_date(facts.get("delivered_at")), _to_date(facts.get("estimated_at"))
    category = (facts.get("category") or "").lower()
    notes = []

    if facts.get("active_refund"):
        active = facts["active_refund"]
        return _deny([f"A refund request for this item already exists (status: {active['status']})."],
                     "No new request is needed. Check the status of the existing request.")

    if status in ("canceled", "unavailable"):
        return _deny([f"The order is {status}. Payments for such orders are refunded "
                      f"automatically, so no refund request is needed."],
                     "Tell the customer the refund is automatic; use the refund timelines.")
    if status in ("created", "approved", "invoiced", "processing"):
        return _deny(["The order has not shipped yet, so it needs a cancellation, not a refund."],
                     "Cancellation is done in My Orders or by a human agent. This assistant cannot cancel.")

    in_transit = status == "shipped" or (delivered is not None and delivered > as_of)
    if status not in ("shipped", "delivered"):
        return _deny([f"Unexpected order status '{status}'."], "Suggest a human support agent.")

    # ---- order still on its way
    if in_transit:
        if reason != "not_delivered":
            return _deny(["The order has not been delivered yet."],
                         "A refund can be requested after delivery.")
        if estimated is None:
            return _deny(["There is no estimated delivery date for this order."],
                         "Suggest a human support agent.")
        late_days = (as_of - estimated).days
        if late_days > LATE_CANCEL_DAYS:
            amount = round(price + freight, 2)
            approval = _max_level(_level_for_amount(amount), "human_agent")
            notes.append("Order is more than 10 days past its estimated delivery date: "
                         "cancel and full refund, including shipping.")
            return _finish(amount, approval, reason, notes, recent_refunds)
        if business_days_between(estimated, as_of) > DELAY_CASE_BUSINESS_DAYS:
            return _deny(["The order is late, but not yet 10 days past the estimate."],
                         f"A human agent can open a delay case. A full refund becomes possible after "
                         f"{estimated + dt.timedelta(days=LATE_CANCEL_DAYS)} if it is still not delivered.")
        return _deny(["The order is within 2 business days of its estimated delivery date."],
                     "Ask the customer to wait a little longer.")

    # ---- delivered orders
    if delivered is None:
        return _deny(["The order is marked delivered but has no delivery date."],
                     "Suggest a human support agent.")
    days_since = (as_of - delivered).days

    if reason == "change_of_mind":
        if category in NON_RETURNABLE_CATEGORIES:
            return _deny([f"Items in the '{category}' category cannot be returned for change of mind."],
                         "A refund is still possible if the item is defective or wrong.")
        if days_since > RETURN_WINDOW_DAYS:
            return _deny([f"The 30-day return window ended ({days_since} days since delivery)."],
                         "Suggest a human agent only if the customer has a special case.")
        amount = round(price, 2)
        notes.append(f"Shipping fee is not refunded. If the KartNest return label is used, "
                     f"R$ {RETURN_LABEL_FEE:.0f} is deducted.")
        return _finish(amount, _level_for_amount(amount), reason, notes, recent_refunds)

    if reason in ("defective", "wrong_item", "missing_items"):
        if days_since > CLAIM_WINDOW_DAYS:
            return _deny([f"Claims must be reported within 7 days of delivery ({days_since} days passed)."],
                         "A human agent can review special cases after the deadline.")
        amount = round(price if reason == "missing_items" else price + freight, 2)
        notes.append("Photos of the item, label and packaging are needed, so a human agent reviews it.")
        approval = _max_level(_level_for_amount(amount), "human_agent")
        return _finish(amount, approval, reason, notes, recent_refunds)

    # reason == "not_delivered": marked delivered, customer did not receive it
    waited = business_days_between(delivered, as_of)
    if waited < NOT_RECEIVED_WAIT_BUSINESS_DAYS:
        ready = add_business_days(delivered, NOT_RECEIVED_WAIT_BUSINESS_DAYS)
        return _deny(["The customer should wait 2 business days after the delivered date "
                      "(neighbors, building reception)."],
                     f"A claim can be opened from {ready}.")
    if days_since > NOT_RECEIVED_CLAIM_DAYS:
        return _deny([f"Claims must be opened within 15 days of the delivered date ({days_since} days passed)."],
                     "A human agent can review special cases after the deadline.")
    amount = round(price + freight, 2)
    notes.append("The carrier investigation takes up to 7 business days. If the package is "
                 "confirmed lost, the refund includes shipping.")
    return _finish(amount, _max_level(_level_for_amount(amount), "human_agent"),
                   reason, notes, recent_refunds)


def _finish(amount, approval, reason, notes, recent_refunds):
    if recent_refunds >= RECENT_REFUND_LIMIT:
        approval = _max_level(approval, "human_agent")
        notes.append(f"{recent_refunds} refund requests in the last {RECENT_REFUND_DAYS} days, "
                     f"so a human reviews this one.")
    return {"eligible": True, "amount": amount, "approval": approval, "reasons": [],
            "notes": notes, "next_step": _next_step(approval, reason)}


# ------------------------------------------------------------------ database part
def _clean(value):
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "as_tuple"):          # Decimal
        return float(value)
    return value


FACTS_SQL = """
SELECT i.order_item_id, i.product_id, i.price, i.freight_value,
       o.order_status, o.order_delivered_customer_date, o.order_estimated_delivery_date,
       pc.category_en, pc.display_name
FROM orders o
JOIN customers c    ON c.customer_id = o.customer_id
JOIN order_items i  ON i.order_id = o.order_id
LEFT JOIN product_catalog pc ON pc.product_id = i.product_id
WHERE o.order_id = %s AND c.customer_unique_id = %s
"""
ACTIVE_SQL = ("SELECT refund_id, status FROM refund_requests "
              "WHERE order_id = %s AND order_item_id = %s "
              "AND status IN ('auto_approved', 'pending_human', 'approved') LIMIT 1")
RECENT_SQL = (f"SELECT COUNT(*) FROM refund_requests WHERE customer_unique_id = %s "
              f"AND status <> 'rejected' AND created_at > now() - interval '{RECENT_REFUND_DAYS} days'")
INSERT_SQL = ("INSERT INTO refund_requests (customer_unique_id, order_id, order_item_id, product_id, "
              "amount, reason, customer_query, status, approval_key) "
              "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING refund_id")


class RefundService:
    def __init__(self, pool, as_of):
        self.pool = pool          # the app's own pool (can write refund_requests)
        self.as_of = as_of        # the date treated as "today"

    # -- shared: find the item, run the rules
    def _evaluate(self, cur, customer_id, order_id, reason, product_id, order_item_id):
        """Returns (result_dict, item_row_or_None). result_dict is JSON-ready."""
        if reason not in REASONS:
            return {"error": f"reason must be one of: {', '.join(REASONS)}"}, None
        sql, params = FACTS_SQL, [order_id, customer_id]
        if product_id:
            sql += " AND i.product_id = %s"
            params.append(product_id)
        if order_item_id is not None:
            sql += " AND i.order_item_id = %s"
            params.append(order_item_id)
        cur.execute(sql + " ORDER BY i.order_item_id", params)
        rows = cur.fetchall()
        # Same answer whether the order does not exist or belongs to someone else.
        if not rows:
            return {"found": False,
                    "message": "No such order (or item) was found in this customer's account."}, None
        if len(rows) > 1:
            return {"found": True, "needs_item_choice": True,
                    "message": "The order has several matching items. Ask the customer which one.",
                    "items": [{"order_item_id": r[0], "product_id": r[1], "name": r[8],
                               "category": r[7], "price": _clean(r[2])} for r in rows]}, None
        item_id, product, price, freight, status, delivered, estimated, category, name = rows[0]
        cur.execute(ACTIVE_SQL, (order_id, item_id))
        active_row = cur.fetchone()
        active = {"refund_id": active_row[0], "status": active_row[1]} if active_row else None
        cur.execute(RECENT_SQL, (customer_id,))
        recent = cur.fetchone()[0]
        facts = {"status": status, "delivered_at": delivered, "estimated_at": estimated,
                 "price": price, "freight": freight, "category": category, "active_refund": active}
        decision = evaluate_refund(facts, reason, self.as_of, recent)
        decision.update({"found": True, "order_id": order_id, "order_item_id": item_id,
                         "product_id": product, "product_name": name, "category": category,
                         "order_status": status, "reason_code": reason,
                         "delivered_date": _clean(_to_date(delivered)),
                         "as_of_date": self.as_of.isoformat()})
        return decision, {"item_id": item_id, "product_id": product}

    def check(self, customer_id, order_id, reason, product_id=None, order_item_id=None):
        with self.pool.connection() as conn, conn.cursor() as cur:
            result, _ = self._evaluate(cur, customer_id, order_id, reason, product_id, order_item_id)
        return result

    def request(self, customer_id, order_id, reason, product_id=None, order_item_id=None,
                details="", customer_query=""):
        """Re-checks everything, then writes the request. Returns a JSON-ready dict."""
        try:
            with self.pool.connection() as conn, conn.cursor() as cur:
                result, item = self._evaluate(cur, customer_id, order_id, reason,
                                              product_id, order_item_id)
                if item is None or not result.get("eligible"):
                    result["submitted"] = False
                    return result
                approval = result["approval"]
                status = "auto_approved" if approval == "auto" else "pending_human"
                approval_key = None if approval == "auto" else "APR-" + secrets.token_hex(5).upper()
                reason_text = reason + (f": {details.strip()[:300]}" if details and details.strip() else "")
                cur.execute(INSERT_SQL, (customer_id, order_id, item["item_id"], item["product_id"],
                                         result["amount"], reason_text, (customer_query or "")[:1000],
                                         status, approval_key))
                refund_id = cur.fetchone()[0]
        except Exception as e:
            if getattr(e, "sqlstate", None) == "23505":      # unique violation: already requested
                return {"submitted": False, "eligible": False,
                        "reasons": ["A refund request for this item already exists."],
                        "next_step": "No new request is needed."}
            raise
        result.update({"submitted": True, "refund_id": refund_id, "status": status,
                       "reference": approval_key})
        return result

    def attach_response(self, customer_id, refund_ids, text):
        """Save what the bot finally told the customer next to the refund request."""
        if not refund_ids:
            return
        with self.pool.connection() as conn, conn.cursor() as cur:
            cur.execute("UPDATE refund_requests SET bot_response = %s "
                        "WHERE refund_id = ANY(%s) AND customer_unique_id = %s",
                        ((text or "")[:2000], list(refund_ids), customer_id))
