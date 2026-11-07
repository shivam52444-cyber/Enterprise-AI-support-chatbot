"""
tools.py - the tools the AI is allowed to call.

  search_policy(question)  -> finds the right KartNest policy text (RAG)
  query_db(sql)            -> the AI writes ONE read-only SQL query; we check it, then run it
                              as the restricted "llm_readonly" database user on views that
                              already show only the logged-in customer's rows.

HOW THE CUSTOMER STAYS SAFE (4 locks, so one mistake is not enough):
  1. The customer id is NOT a tool argument. The AI never sees it, never types it.
     The server puts it into the database session (set_config) itself.
  2. validate_sql(): deny-by-default. Only SELECT, only the 5 views, only listed
     columns and functions. set_config, pg_*, comments, ';' tricks ... all rejected.
  3. The database user can only read those views (no base tables, read-only, 3s timeout).
  4. The views themselves filter by the customer id in the session.
"""

import datetime
import decimal
import json
import logging
import re

from refunds import REASONS

log = logging.getLogger("kartnest.tools")

MAX_ROWS = 50
MAX_SQL_CHARS = 1500
MAX_TOOL_RESULT_CHARS = 6000

# ------------------------------------------------------------------ what SQL may touch
VIEW_COLUMNS = {
    "my_orders": ["order_id", "order_status", "order_purchase_timestamp", "order_approved_at",
                  "order_delivered_carrier_date", "order_delivered_customer_date",
                  "order_estimated_delivery_date"],
    "my_order_items": ["order_id", "order_item_id", "product_id", "price", "freight_value"],
    "my_payments": ["order_id", "payment_sequential", "payment_type", "payment_installments",
                    "payment_value"],
    "my_refunds": ["refund_id", "order_id", "order_item_id", "product_id", "amount", "reason",
                   "status", "created_at"],
    "catalog": ["product_id", "display_name", "category_en", "list_price"],
}
ALLOWED_TABLES = set(VIEW_COLUMNS)
ALLOWED_COLUMNS = {c for cols in VIEW_COLUMNS.values() for c in cols}

ALLOWED_KEYWORDS = {
    "select", "from", "where", "and", "or", "not", "in", "is", "null", "like", "ilike",
    "between", "order", "by", "group", "having", "limit", "offset", "asc", "desc", "as",
    "join", "left", "inner", "on", "distinct", "case", "when", "then", "else", "end",
    "true", "false", "interval", "date", "timestamp", "nulls", "first", "last",
    # type names (for CAST) and EXTRACT fields
    "integer", "int", "bigint", "numeric", "text", "float",
    "day", "month", "year", "hour", "epoch",
}
ALLOWED_FUNCS = {"count", "sum", "avg", "min", "max", "coalesce", "round", "lower", "upper",
                 "abs", "length", "cast", "extract", "date_trunc", "date_part"}
# words that may be followed by "(" without being a function call
KEYWORDS_BEFORE_PAREN = {"select", "from", "where", "and", "or", "not", "in", "on", "when",
                         "then", "else", "by", "having", "join", "between"}

_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_NUM = re.compile(r"\d+(?:\.\d+)?")
_OPS = set("=<>!+-*/%(),.")


class SqlRejected(Exception):
    """The query broke a rule. The message is shown to the AI so it can fix the query."""


class SqlExecutionError(Exception):
    """The database refused or failed to run a (valid) query."""


def _tokenize(sql):
    tokens, i, n = [], 0, len(sql)
    while i < n:
        ch = sql[i]
        if ch in " \t\r\n":
            i += 1
        elif ch == "'":                                   # text value: skip its content
            j, buf = i + 1, []
            while True:
                if j >= n:
                    raise SqlRejected("a text value ('...') is not closed")
                if sql[j] == "'":
                    if j + 1 < n and sql[j + 1] == "'":
                        buf.append("'")
                        j += 2
                        continue
                    break
                buf.append(sql[j])
                j += 1
            text = "".join(buf)
            if len(text) > 200 or "\x00" in text:
                raise SqlRejected("text value too long or invalid")
            tokens.append(("str", text))
            i = j + 1
        elif ch.isascii() and ch.isdigit():
            m = _NUM.match(sql, i)
            tokens.append(("num", m.group()))
            i = m.end()
            if i < n and (sql[i].isalpha() or sql[i] == "_"):
                raise SqlRejected("invalid number")
        elif ch.isascii() and (ch.isalpha() or ch == "_"):
            m = _WORD.match(sql, i)
            tokens.append(("word", m.group().lower()))
            i = m.end()
        elif ch == "-" and sql[i + 1:i + 2] == "-":
            raise SqlRejected("comments are not allowed")
        elif ch == "/" and sql[i + 1:i + 2] == "*":
            raise SqlRejected("comments are not allowed")
        elif ch in _OPS:
            tokens.append(("op", ch))
            i += 1
        elif ch == ";":
            if sql[i:].strip(" \t\r\n;") == "":
                break                                      # trailing ';' is fine
            raise SqlRejected("only ONE statement is allowed (no ';' in the middle)")
        else:
            raise SqlRejected(f"the character {ch!r} is not allowed")
    return tokens


def validate_sql(sql):
    """Deny-by-default check. Returns the cleaned SQL, or raises SqlRejected."""
    if not isinstance(sql, str) or not sql.strip():
        raise SqlRejected("the query is empty")
    if len(sql) > MAX_SQL_CHARS:
        raise SqlRejected(f"the query is too long (max {MAX_SQL_CHARS} characters)")
    tokens = _tokenize(sql)
    if not tokens or tokens[0] != ("word", "select"):
        raise SqlRejected("only SELECT queries are allowed; the query must start with SELECT")

    # Names the query gives itself with AS (the AI must always write AS for aliases)
    aliases = set()
    for idx, (kind, value) in enumerate(tokens):
        if kind == "word" and value == "as":
            if idx + 1 >= len(tokens) or tokens[idx + 1][0] != "word":
                raise SqlRejected("AS must be followed by a name")
            name = tokens[idx + 1][1]
            if name not in ALLOWED_KEYWORDS:               # 'integer' in CAST(x AS integer) is a type
                aliases.add(name)

    for idx, (kind, value) in enumerate(tokens):
        if kind != "word":
            continue
        prev = tokens[idx - 1] if idx else None
        nxt = tokens[idx + 1] if idx + 1 < len(tokens) else None
        followed_by_paren = nxt == ("op", "(")

        if prev == ("op", "."):                            # alias.column or llm.view
            if followed_by_paren or (value not in ALLOWED_COLUMNS and value not in ALLOWED_TABLES):
                raise SqlRejected(f"'{value}' is not an allowed column or table")
            continue
        if followed_by_paren:                              # anything followed by "(" is a call
            if value in ALLOWED_FUNCS or value in KEYWORDS_BEFORE_PAREN:
                continue
            raise SqlRejected(f"'{value}(...)' is not allowed. Allowed functions: "
                              f"{', '.join(sorted(ALLOWED_FUNCS))}")
        if value in ALLOWED_KEYWORDS or value in ALLOWED_TABLES or value in ALLOWED_COLUMNS \
                or value in aliases or (value == "llm" and nxt == ("op", ".")):
            continue
        raise SqlRejected(
            f"unknown name '{value}'. Use only these views and columns: {describe_views()}. "
            f"Write aliases with AS (for example: FROM my_orders AS o).")
    return sql.strip().rstrip("; \t\r\n")


def prepare_sql(sql):
    """Validate, then wrap so that at most MAX_ROWS + 1 rows can ever come back."""
    cleaned = validate_sql(sql)
    return f"SELECT * FROM ({cleaned}) AS q LIMIT {MAX_ROWS + 1}"


def describe_views():
    return "; ".join(f"{view}({', '.join(cols)})" for view, cols in VIEW_COLUMNS.items())


# ------------------------------------------------------------------ running + formatting
def run_scoped_sql(pool, customer_id, prepared_sql):
    """Run the checked query on the restricted pool, for THIS customer only.
    set_config(..., true) lasts only for this transaction, so the setting can never
    leak to the next person who borrows the same database connection."""
    try:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT set_config('app.customer_id', %s, true)", (customer_id,))
                cur.execute(prepared_sql)
                columns = [d.name for d in cur.description]
                rows = cur.fetchmany(MAX_ROWS + 1)
        return columns, rows
    except Exception as e:
        raise SqlExecutionError(str(e).strip().splitlines()[0][:200] if str(e).strip() else type(e).__name__) from e


def _convert(value):
    if isinstance(value, decimal.Decimal):
        return float(value)
    if isinstance(value, (datetime.datetime, datetime.date)):
        return value.isoformat(sep=" ") if isinstance(value, datetime.datetime) else value.isoformat()
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return str(value)


def format_rows(columns, rows):
    truncated = len(rows) > MAX_ROWS
    rows = rows[:MAX_ROWS]
    text = json.dumps({"columns": columns,
                       "rows": [[_convert(v) for v in row] for row in rows],
                       "row_count": len(rows), "truncated": truncated}, ensure_ascii=False)
    if len(text) > MAX_TOOL_RESULT_CHARS:
        return _error("the result is too large; select fewer columns or add filters")
    return text


def _error(message):
    return json.dumps({"error": message})


# ------------------------------------------------------------------ what the AI is shown
QUERY_DB_DESCRIPTION = (
    "Look up the CURRENT customer's own data (it is already filtered to them; you never need "
    "or may use a customer id) and the public product catalog. Write ONE PostgreSQL SELECT.\n"
    "Views and columns:\n"
    "- my_orders(order_id, order_status, order_purchase_timestamp, order_approved_at, "
    "order_delivered_carrier_date, order_delivered_customer_date, order_estimated_delivery_date). "
    "order_status is one of: created, approved, invoiced, processing, shipped, delivered, "
    "canceled, unavailable.\n"
    "- my_order_items(order_id, order_item_id, product_id, price, freight_value)\n"
    "- my_payments(order_id, payment_sequential, payment_type, payment_installments, payment_value)\n"
    "- my_refunds(refund_id, order_id, order_item_id, product_id, amount, reason, status, created_at)\n"
    "- catalog(product_id, display_name, category_en, list_price)  (public prices)\n"
    "Rules: SELECT only; always write aliases with AS (FROM my_orders AS o); no comments, no ';' "
    "in the middle, no :: casts (use CAST(x AS type)); allowed functions: count, sum, avg, min, "
    "max, coalesce, round, lower, upper, abs, length, cast, extract, date_trunc, date_part. "
    "Days between two timestamps: EXTRACT(DAY FROM (end_ts - start_ts)). Keep results small "
    "(add WHERE and LIMIT)."
)

TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "search_policy",
        "description": ("Search KartNest's policy documents: returns, refunds, refund approval "
                        "limits, delivery times and delays, cancellations, payments, privacy and "
                        "when a human agent takes over. Use it for ANY question about rules, "
                        "deadlines, timelines or eligibility."),
        "parameters": {"type": "object",
                       "properties": {"question": {"type": "string",
                                                   "description": "A short, self-contained search query in English"}},
                       "required": ["question"]}}},
    {"type": "function", "function": {
        "name": "query_db",
        "description": QUERY_DB_DESCRIPTION,
        "parameters": {"type": "object",
                       "properties": {"sql": {"type": "string",
                                              "description": "One PostgreSQL SELECT statement"}},
                       "required": ["sql"]}}},
    {"type": "function", "function": {
        "name": "check_refund_eligibility",
        "description": ("Check whether an item in the customer's order can be refunded under "
                        "KartNest policy, how much, and what happens next. The tool knows today's "
                        "date and every rule, so use its answer for any eligibility question and "
                        "never decide it yourself. If the order has several items and you do not "
                        "know which one, leave product_id and order_item_id empty: the tool lists "
                        "the items so you can ask the customer."),
        "parameters": {"type": "object", "properties": {
            "order_id": {"type": "string"},
            "reason": {"type": "string", "enum": list(REASONS),
                       "description": ("change_of_mind = customer no longer wants it; defective = "
                                       "broken or not working; wrong_item = different from what "
                                       "was ordered; missing_items = something missing from the "
                                       "package; not_delivered = late, or marked delivered but "
                                       "not received")},
            "product_id": {"type": "string"},
            "order_item_id": {"type": "integer"}},
            "required": ["order_id", "reason"]}}},
    {"type": "function", "function": {
        "name": "request_refund",
        "description": ("Submit a refund request for the customer. Call it ONLY when the customer "
                        "clearly asks you to submit or request the refund (not when they are just "
                        "asking whether they could get one). Call check_refund_eligibility first "
                        "and only submit if it says eligible. The server checks everything again. "
                        "Large or special cases go to a human agent for approval."),
        "parameters": {"type": "object", "properties": {
            "order_id": {"type": "string"},
            "reason": {"type": "string", "enum": list(REASONS)},
            "product_id": {"type": "string"},
            "order_item_id": {"type": "integer"},
            "details": {"type": "string",
                        "description": "One short sentence: the customer's problem in their own words"}},
            "required": ["order_id", "reason"]}}},
]


class ToolBox:
    """Runs the tools for ONE customer. The customer id lives here, outside the AI's reach."""

    schemas = TOOL_SCHEMAS

    def __init__(self, customer_id, policy_search, sql_runner, refunds=None, customer_message=""):
        self._customer_id = customer_id        # never in a schema, never read from tool arguments
        self._policy_search = policy_search    # question -> list of {doc_id, section, text, score}
        self._sql_runner = sql_runner          # (customer_id, prepared_sql) -> (columns, rows)
        self._refunds = refunds                # RefundService (or None: refund tools switched off)
        self._customer_message = customer_message   # what the customer typed (stored with a refund)
        self.sources = set()                   # policy docs used, for the API response
        self.refund_ids = []                   # refund requests created in this conversation turn

    def execute(self, name, raw_args):
        try:
            if isinstance(raw_args, str):
                args = json.loads(raw_args) if raw_args.strip() else {}
            else:
                args = raw_args or {}
            if not isinstance(args, dict):
                raise ValueError("arguments must be an object")
        except (ValueError, TypeError):
            return _error("the tool arguments were not valid JSON")
        try:
            if name == "search_policy":
                return self._search_policy(args)
            if name == "query_db":
                return self._query_db(args)
            if name in ("check_refund_eligibility", "request_refund"):
                return self._refund_tool(name, args)
            return _error(f"unknown tool '{name}'")
        except Exception:
            log.exception("tool %s crashed", name)
            return _error("the tool failed. Try once more, or tell the customer you cannot "
                          "look this up right now")

    def _search_policy(self, args):
        question = args.get("question")
        if not isinstance(question, str) or not question.strip() or len(question) > 300:
            return _error("question must be a short, non-empty text")
        chunks = self._policy_search(question.strip())
        if not chunks:
            return _error("no policy text found")
        self.sources.update(c["doc_id"] for c in chunks)
        text = "\n\n".join(f"[{c['doc_id']} | {c['section']}]\n{c['text']}" for c in chunks)
        return text[:MAX_TOOL_RESULT_CHARS]

    def _query_db(self, args):
        try:
            prepared = prepare_sql(args.get("sql"))
        except SqlRejected as e:
            return _error(f"query rejected: {e}")
        try:
            columns, rows = self._sql_runner(self._customer_id, prepared)
        except SqlExecutionError as e:
            return _error(f"the database could not run the query: {e}")
        return format_rows(columns, rows)

    def _refund_tool(self, name, args):
        if self._refunds is None:
            return _error("refund tools are not available")
        order_id = args.get("order_id")
        reason = args.get("reason")
        product_id = args.get("product_id") or None
        order_item_id = args.get("order_item_id")
        if not isinstance(order_id, str) or not order_id.strip() or len(order_id) > 64:
            return _error("order_id must be a short, non-empty text")
        if reason not in REASONS:
            return _error(f"reason must be one of: {', '.join(REASONS)}")
        if product_id is not None and (not isinstance(product_id, str) or len(product_id) > 64):
            return _error("product_id must be a short text")
        if order_item_id is not None and (isinstance(order_item_id, bool) or not isinstance(order_item_id, int)):
            return _error("order_item_id must be a whole number")

        if name == "check_refund_eligibility":
            result = self._refunds.check(self._customer_id, order_id.strip(), reason,
                                         product_id, order_item_id)
        else:
            details = args.get("details")
            details = details.strip()[:300] if isinstance(details, str) else ""
            result = self._refunds.request(self._customer_id, order_id.strip(), reason,
                                           product_id, order_item_id, details, self._customer_message)
            if result.get("submitted") and result.get("refund_id") is not None:
                self.refund_ids.append(result["refund_id"])
        return json.dumps(result, ensure_ascii=False)
