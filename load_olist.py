"""
Load the Olist CSV files into the KartNest Postgres database on AWS RDS.

Run from your laptop:
    pip install "psycopg[binary]"
    python load_olist.py

It asks for your RDS endpoint, username and password (the password is hidden
and never saved). Everything loads in ONE transaction: if anything fails,
nothing is saved, so you can fix the problem and run it again.
"""

import csv
import getpass
import sys
from pathlib import Path

import psycopg

DEFAULT_DIR = r"D:\DATA science projects\E-commerce_ Chatbot\Dataset"
DATA_DIR = Path(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DIR)

# (csv file name, table name, columns to copy). CSV headers and table columns
# use the same names. Order matters: parents first, children after.
JOBS = [
    ("olist_customers_dataset.csv", "customers",
     ["customer_id", "customer_unique_id", "customer_zip_code_prefix",
      "customer_city", "customer_state"]),
    ("olist_products_dataset.csv", "products",
     ["product_id", "product_category_name"]),
    ("product_category_name_translation.csv", "category_translation",
     ["product_category_name", "product_category_name_english"]),
    ("olist_orders_dataset.csv", "orders",
     ["order_id", "customer_id", "order_status", "order_purchase_timestamp",
      "order_approved_at", "order_delivered_carrier_date",
      "order_delivered_customer_date", "order_estimated_delivery_date"]),
    ("olist_order_items_dataset.csv", "order_items",
     ["order_id", "order_item_id", "product_id", "seller_id",
      "shipping_limit_date", "price", "freight_value"]),
    ("olist_order_payments_dataset.csv", "order_payments",
     ["order_id", "payment_sequential", "payment_type",
      "payment_installments", "payment_value"]),
]

# Olist has no product names or list prices, so build a simple public catalog.
# The names are PLACEHOLDERS ("Health Beauty #a1b2c3"). Replace them later with
# nicer generated names. list_price = median price the product sold for.
CATALOG_SQL = """
INSERT INTO product_catalog (product_id, display_name, category_en, list_price)
SELECT p.product_id,
       initcap(replace(coalesce(t.product_category_name_english,
                                p.product_category_name, 'general'), '_', ' '))
           || ' #' || left(p.product_id, 6),
       t.product_category_name_english,
       round((percentile_cont(0.5) WITHIN GROUP (ORDER BY i.price::float8))::numeric, 2)
FROM products p
JOIN order_items i ON i.product_id = p.product_id
LEFT JOIN category_translation t ON t.product_category_name = p.product_category_name
GROUP BY p.product_id, p.product_category_name, t.product_category_name_english
"""


def find_files():
    """Check all CSV files exist (case-insensitive) and return their paths."""
    if not DATA_DIR.is_dir():
        sys.exit(f"Folder not found: {DATA_DIR}")
    on_disk = {p.name.lower(): p for p in DATA_DIR.iterdir() if p.is_file()}
    found, missing = {}, []
    for csv_name, _, _ in JOBS:
        path = on_disk.get(csv_name.lower())
        if path:
            found[csv_name] = path
        else:
            missing.append(csv_name)
    if missing:
        print("These files were not found in", DATA_DIR)
        for m in missing:
            print("  -", m)
        print("\nFiles that ARE in the folder:")
        for name in sorted(on_disk):
            print("  -", name)
        sys.exit(1)
    return found


def load_one(cur, path, table, cols):
    # utf-8-sig removes the hidden BOM that some Olist files start with
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        missing = [c for c in cols if c not in (reader.fieldnames or [])]
        if missing:
            sys.exit(f"{path.name}: columns not found in CSV: {missing}")
        count = 0
        with cur.copy(f"COPY {table} ({', '.join(cols)}) FROM STDIN") as copy:
            for row in reader:
                # empty cell in the CSV -> NULL in the database
                copy.write_row([row[c] if row[c] != "" else None for c in cols])
                count += 1
    return count


def main():
    files = find_files()
    print(f"All 6 CSV files found in {DATA_DIR}\n")

    host = input("RDS endpoint (e.g. kartnest-db.xxxx.ap-south-2.rds.amazonaws.com): ").strip()
    dbname = input("Database name [kartnest]: ").strip() or "kartnest"
    user = input("Master username [postgres]: ").strip() or "postgres"
    password = getpass.getpass("Password (hidden): ")

    # "with connect()" commits if everything works, rolls back on any error
    with psycopg.connect(host=host, port=5432, dbname=dbname,
                         user=user, password=password, sslmode="require") as conn:
        with conn.cursor() as cur:
            for csv_name, table, cols in JOBS:
                n = load_one(cur, files[csv_name], table, cols)
                print(f"  {table:<22} {n:>8,} rows")

            cur.execute(CATALOG_SQL)
            print(f"  {'product_catalog':<22} {cur.rowcount:>8,} rows (placeholder names)")

            print("\nOrders by status:")
            cur.execute("SELECT order_status, COUNT(*) FROM orders GROUP BY 1 ORDER BY 2 DESC")
            for status, n in cur.fetchall():
                print(f"  {status:<14} {n:>8,}")

    print("\nDone. Everything was saved.")


if __name__ == "__main__":
    try:
        main()
    except psycopg.errors.UniqueViolation as e:
        sys.exit("\nData already exists in these tables (nothing was saved this run). "
                 "Drop and recreate the tables with schema_lean.sql, then run again.\n" + str(e))
    except psycopg.OperationalError as e:
        sys.exit("\nCould not connect. Check: endpoint, password, Public access = Yes, "
                 "and your CURRENT IP allowed on port 5432 in the security group.\n" + str(e))
