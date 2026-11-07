"""
main.py - the KartNest chatbot API (FastAPI).

POST /chat   header:  Authorization: Bearer <token from the website>
             body:    {"message": "..."}          <- NO customer id in the body
GET  /health

Setup (once):
    pip install fastapi "uvicorn[standard]" pyjwt "psycopg[binary,pool]"
    (plus everything from policy_bot.py and guardrails.py)

.env must also contain a long random secret (shared with the website's backend):
    HOST_SHARED_SECRET=<64 random characters>
    make one with:  python -c "import secrets; print(secrets.token_hex(32))"

Also add the restricted database user's password (the one you set for llm_readonly):
    LLM_DB_USER=llm_readonly
    LLM_DB_PASSWORD=<that password>

Optional: the date the refund rules treat as 'today' (the Olist data is from 2016-2018):
    AS_OF_DATE=2018-09-10

Run:
    uvicorn main:app --port 8000
Test: open http://127.0.0.1:8000/docs, click Authorize, paste a token made with
    python fake_host/make_token.py <customer_unique_id>
"""

import asyncio
import datetime
import logging
import os
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, Field

from agent import run_agent
from auth import MIN_SECRET_LENGTH, AuthError, verify_token
from guardrails import PII_NOTICE, Guardrails
from policy_bot import DEFAULT_LLM_MODEL, EMBED_MODEL, REPLY_LLM_ERROR, search_policy
from refunds import RefundService
from tools import ToolBox, run_scoped_sql

log = logging.getLogger("kartnest.api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


class BotState:
    """Everything that is expensive to create, built once at start-up."""


def build_state():
    """Create the guardrails, models, Groq client and DB pool. Any failure stops the server
    (fail closed: the API never runs without its guardrails or its secret)."""
    from dotenv import load_dotenv
    load_dotenv()

    secret = os.getenv("HOST_SHARED_SECRET", "")
    if len(secret) < MIN_SECRET_LENGTH:
        raise RuntimeError(f"HOST_SHARED_SECRET must be at least {MIN_SECRET_LENGTH} characters. "
                           f'Make one: python -c "import secrets; print(secrets.token_hex(32))"')
    api_key, host, password = os.getenv("GROQ_API_KEY"), os.getenv("DB_HOST"), os.getenv("DB_PASSWORD")
    llm_db_password = os.getenv("LLM_DB_PASSWORD")
    missing = [n for n, v in [("GROQ_API_KEY", api_key), ("DB_HOST", host), ("DB_PASSWORD", password),
                              ("LLM_DB_PASSWORD", llm_db_password)] if not v]
    if missing:
        raise RuntimeError(f"Missing in .env: {', '.join(missing)}")

    from groq import Groq
    from psycopg.conninfo import make_conninfo
    from psycopg_pool import ConnectionPool
    from sentence_transformers import SentenceTransformer

    state = BotState()
    state.secret = secret
    state.guards = Guardrails.build()
    state.embedder = SentenceTransformer(EMBED_MODEL)
    state.groq = Groq(api_key=api_key)
    state.pool = ConnectionPool(
        conninfo=make_conninfo(host=host, port=5432, dbname=os.getenv("DB_NAME", "postgres"),
                               user=os.getenv("DB_USER", "postgres"), password=password,
                               sslmode="require"),
        min_size=1, max_size=5, open=False)
    state.pool.open(wait=True, timeout=20)   # fails fast if the database is unreachable
    # A SECOND pool for the AI's SQL tool: it logs in as llm_readonly, which can only
    # read the customer-scoped views. The admin pool above is never used for AI-written SQL.
    state.llm_pool = ConnectionPool(
        conninfo=make_conninfo(host=host, port=5432, dbname=os.getenv("DB_NAME", "postgres"),
                               user=os.getenv("LLM_DB_USER", "llm_readonly"),
                               password=llm_db_password, sslmode="require"),
        min_size=1, max_size=3, open=False)
    state.llm_pool.open(wait=True, timeout=20)

    # Refund rules need a "today". Olist orders are from 2016-2018, so for demos set
    # AS_OF_DATE in .env (for example 2018-09-10). Without it, the real date is used.
    as_of_text = os.getenv("AS_OF_DATE", "").strip()
    try:
        state.as_of = datetime.date.fromisoformat(as_of_text) if as_of_text else datetime.date.today()
    except ValueError:
        raise RuntimeError("AS_OF_DATE must look like 2018-09-10 (year-month-day)")
    state.refunds = RefundService(state.pool, state.as_of)
    return state


@asynccontextmanager
async def lifespan(app):
    # Loading models is slow and blocking, so do it in a thread instead of freezing the loop.
    app.state.bot = await asyncio.to_thread(build_state)
    log.info("KartNest API ready. Refund rules treat %s as today", app.state.bot.as_of)
    yield
    app.state.bot.pool.close()
    app.state.bot.llm_pool.close()


app = FastAPI(title="KartNest support chatbot", lifespan=lifespan)
bearer = HTTPBearer(auto_error=True)


def get_customer_id(request: Request,
                    creds: HTTPAuthorizationCredentials = Depends(bearer)) -> str:
    """The ONLY place a customer id comes from: the verified token."""
    try:
        return verify_token(creds.credentials, request.app.state.bot.secret)
    except AuthError as e:
        raise HTTPException(status_code=401, detail=str(e),
                            headers={"WWW-Authenticate": "Bearer"})


class ChatRequest(BaseModel):
    # extra="forbid": if anyone sends a customer_id (or anything else) in the body, it is
    # rejected with 422 instead of being silently ignored.
    model_config = ConfigDict(extra="forbid")
    message: str = Field(..., max_length=5000)


class ChatResponse(BaseModel):
    answer: str
    reason: str = "ok"          # "ok" or the name of the guardrail that acted
    sources: list[str] = []     # policy documents used for the answer
    tools_used: list[str] = []  # which tools the AI called (handy for debugging and tests)


def _policy_search(bot, question):
    with bot.pool.connection() as conn:
        return search_policy(conn, bot.embedder, question)


def _agent_blocking(bot, customer_id, question):
    """Runs the AI + its tools. Blocking (database, Groq), so call it in a worker thread.
    The customer id goes into the ToolBox here: the AI never receives it."""
    toolbox = ToolBox(
        customer_id=customer_id,
        policy_search=lambda q: _policy_search(bot, q),
        sql_runner=lambda cid, sql: run_scoped_sql(bot.llm_pool, cid, sql),
        refunds=bot.refunds,
        customer_message=question)
    answer, tools_used = run_agent(bot.groq, os.getenv("LLM_MODEL", DEFAULT_LLM_MODEL),
                                   question, toolbox)
    return answer, tools_used, sorted(toolbox.sources), list(toolbox.refund_ids)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest, request: Request,
               customer_id: str = Depends(get_customer_id)):
    bot = request.app.state.bot
    try:
        verdict = await bot.guards.check_input_async(body.message)
        if not verdict.allowed:
            log.info("chat customer=%s blocked_by=%s", customer_id[:8], verdict.reason)
            return ChatResponse(answer=verdict.reply, reason=verdict.reason)

        try:
            answer, tools_used, sources, refund_ids = await asyncio.to_thread(
                _agent_blocking, bot, customer_id, verdict.text)
        except Exception:
            log.exception("search/LLM failed")
            return ChatResponse(answer=REPLY_LLM_ERROR, reason="llm_error")

        out = await bot.guards.check_output_async(answer)
        if not out.allowed:
            log.info("chat customer=%s blocked_by=%s", customer_id[:8], out.reason)
            return ChatResponse(answer=out.reply, reason=out.reason)

        final = out.text + (PII_NOTICE if verdict.pii_found else "")
        if refund_ids:   # keep what the bot told the customer next to the refund request
            try:
                await asyncio.to_thread(bot.refunds.attach_response, customer_id, refund_ids, final)
            except Exception:
                log.exception("could not save the bot response for refunds %s", refund_ids)
        log.info("chat customer=%s ok tools=%s", customer_id[:8], tools_used)
        return ChatResponse(answer=final, reason="ok", sources=sources, tools_used=tools_used)
    except HTTPException:
        raise
    except Exception:
        log.exception("unhandled error in /chat")
        raise HTTPException(status_code=500, detail="Something went wrong. Please try again.")
