"""
approve_refund.py - for the HUMAN support agent.

Refunds that the chatbot cannot approve by itself wait here with status 'pending_human'
and a reference like APR-1A2B3C4D5E. A person looks at them and decides.

    python approve_refund.py list
    python approve_refund.py decide APR-1A2B3C4D5E approved --by "Ravi" --reason "Photos show the damage"
    python approve_refund.py decide APR-1A2B3C4D5E rejected --by "Ravi" --reason "Item was used"

Each decision is saved in the refund_approvals table (who, when, why) and the refund
request's status changes to approved or rejected, both in ONE transaction. A request can
only be decided once.

Uses the same DB_HOST / DB_NAME / DB_USER / DB_PASSWORD lines from your .env file.
"""

import argparse
import os
import sys

PENDING_SQL = ("SELECT approval_key, refund_id, left(customer_unique_id, 8), order_id, product_id, "
               "amount, reason, created_at FROM refund_requests "
               "WHERE status = 'pending_human' ORDER BY created_at")


def decide_in_db(cur, approval_key, decision, approved_by, reason):
    """Record the decision. Raises ValueError with a plain message if it is not allowed."""
    if decision not in ("approved", "rejected"):
        raise ValueError("decision must be 'approved' or 'rejected'")
    if not approved_by.strip() or not reason.strip():
        raise ValueError("--by and --reason are both required")
    cur.execute("SELECT refund_id, status FROM refund_requests WHERE approval_key = %s FOR UPDATE",
                (approval_key,))
    row = cur.fetchone()
    if row is None:
        raise ValueError(f"No refund request has the reference {approval_key}")
    refund_id, status = row
    if status != "pending_human":
        raise ValueError(f"Refund {refund_id} is '{status}', not waiting for approval")
    cur.execute("INSERT INTO refund_approvals (approval_key, decision, approved_by, reason) "
                "VALUES (%s, %s, %s, %s)", (approval_key, decision, approved_by.strip(), reason.strip()))
    cur.execute("UPDATE refund_requests SET status = %s WHERE approval_key = %s",
                (decision, approval_key))
    return refund_id


def connect():
    from dotenv import load_dotenv
    import psycopg
    load_dotenv()
    host, password = os.getenv("DB_HOST"), os.getenv("DB_PASSWORD")
    if not host or not password:
        sys.exit("DB_HOST and DB_PASSWORD must be in your .env file")
    try:
        return psycopg.connect(host=host, port=5432, dbname=os.getenv("DB_NAME", "postgres"),
                               user=os.getenv("DB_USER", "postgres"), password=password,
                               sslmode="require")
    except psycopg.OperationalError as e:
        sys.exit("Could not connect to the database. Check .env and that your CURRENT IP is "
                 "allowed on port 5432.\n" + str(e))


def main():
    parser = argparse.ArgumentParser(description="Approve or reject chatbot refund requests")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="show requests waiting for a decision")
    d = sub.add_parser("decide", help="approve or reject one request")
    d.add_argument("approval_key")
    d.add_argument("decision", choices=["approved", "rejected"])
    d.add_argument("--by", required=True, help="your name")
    d.add_argument("--reason", required=True, help="why you decided this")
    args = parser.parse_args()

    conn = connect()
    try:
        with conn:   # one transaction: commits on success, rolls back on any error
            with conn.cursor() as cur:
                if args.command == "list":
                    cur.execute(PENDING_SQL)
                    rows = cur.fetchall()
                    if not rows:
                        print("No refund requests are waiting for approval.")
                    for key, rid, cust, order, product, amount, reason, created in rows:
                        print(f"{key}  refund #{rid}  R$ {amount}  customer {cust}...  order {order}\n"
                              f"    product {product}\n    {reason}\n    requested {created:%Y-%m-%d %H:%M}\n")
                else:
                    refund_id = decide_in_db(cur, args.approval_key, args.decision, args.by, args.reason)
                    print(f"Refund #{refund_id} ({args.approval_key}) marked {args.decision} by {args.by}.")
    except ValueError as e:
        sys.exit(f"Not done: {e}")
    except Exception as e:
        sys.exit(f"Database error: {type(e).__name__}: {e}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
