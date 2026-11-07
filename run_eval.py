"""
run_eval.py - evaluates the KartNest chatbot with DeepEval.

    python evals/run_eval.py policy                  # RAG: retrieval + answer quality (33 cases)
    python evals/run_eval.py guardrails              # small talk, PII, injection, off-topic (40 cases)
    python evals/run_eval.py agent                   # tools, refunds, safety on YOUR database (19 cases)
    python evals/run_eval.py all --tag baseline

    --limit 5          only run the first 5 cases (a cheap smoke test)
    --ids p01,p02      only these cases
    --tag NAME         name of this run; results go to evals/results/NAME_<suite>.json
    --fresh            ignore earlier results for this tag and start again
    --sleep 1.5        seconds to wait between cases (raise it if you hit Groq rate limits)

Results are saved after every case, so an interrupted run continues where it stopped.
Then check them against the rules:  python evals/check_thresholds.py --tag baseline

The chatbot is run IN-PROCESS with exactly the same code and database as the API
(main.build_state). Refund requests the agent cases create are deleted again.
"""

import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (str(HERE), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import checks  # noqa: E402
import fixtures as fixtures_mod  # noqa: E402
from agent import run_agent  # noqa: E402
from policy_bot import DEFAULT_LLM_MODEL, EMBED_MODEL, TOP_K, search_policy  # noqa: E402
from refunds import RefundService  # noqa: E402
from tools import SqlExecutionError, ToolBox, run_scoped_sql  # noqa: E402

RESULTS_DIR = HERE / "results"
RATE_LIMIT_WAIT = 5            # seconds; doubles after each rate-limit error
GUARD_RETRY_WAIT = 10


def llm_model():
    return os.getenv("LLM_MODEL", DEFAULT_LLM_MODEL)


# =========================================================================== helpers
def retry_rate_limit(fn, *args, tries=5, **kwargs):
    """Call fn; if Groq says 'rate limit' wait and try again."""
    for attempt in range(tries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            text = str(e).lower()
            limited = "429" in text or "rate limit" in text or "rate_limit" in text
            if attempt == tries - 1 or not limited:
                raise
            wait = min(60, RATE_LIMIT_WAIT * 2 ** attempt)
            print(f"   rate limit, waiting {wait}s ...")
            time.sleep(wait)


class ResultStore:
    """Saves results after every case (resume-friendly)."""

    def __init__(self, path, meta, fresh=False):
        self.path = Path(path)
        self.data = {"meta": meta, "cases": []}
        if self.path.exists() and not fresh:
            try:
                old = json.loads(self.path.read_text(encoding="utf-8"))
                self.data["cases"] = old.get("cases", [])
            except (ValueError, OSError):
                pass

    def done(self):
        return {c["id"] for c in self.data["cases"] if not c.get("error") and not c.get("skipped")}

    def add(self, case):
        self.data["cases"] = [c for c in self.data["cases"] if c["id"] != case["id"]] + [case]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path)


def score_case(test_case, metrics):
    """metrics: [(key, metric_object)]. A crashing metric never stops the run."""
    out = {}
    for key, metric in metrics:
        try:
            retry_rate_limit(metric.measure, test_case)
            score = metric.score
            out[key] = {"score": None if score is None else round(float(score), 4),
                        "passed": bool(metric.is_successful()),
                        "reason": (getattr(metric, "reason", None) or "")[:300],
                        "threshold": getattr(metric, "threshold", None)}
        except Exception as e:
            out[key] = {"score": None, "passed": None,
                        "error": f"{type(e).__name__}: {str(e)[:200]}"}
    return out


def execute_cases(suite, cases, handler, args, meta):
    store = ResultStore(RESULTS_DIR / f"{args.tag}_{suite}.json", meta, fresh=args.fresh)
    already = store.done()
    todo = [c for c in cases if (not args.ids or c["id"] in args.ids) and c["id"] not in already]
    if args.limit:
        todo = todo[:args.limit]
    print(f"\n=== {suite}: {len(todo)} case(s) to run, {len(already)} already done ===")
    for n, case in enumerate(todo, 1):
        print(f"[{n}/{len(todo)}] {case['id']}: {case['input'][:70]}")
        try:
            result = handler(case)
        except Exception as e:
            result = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
            print("   ERROR:", result["error"])
        if result.get("skipped"):
            print("   skipped:", result["skipped"])
        result["id"] = case["id"]
        result.setdefault("input", case["input"])
        store.add(result)
        time.sleep(args.sleep)
    return store


def print_report(suite, store, thresholds):
    cases = store.data["cases"]
    summary = checks.summarize(cases)
    print(f"\n--- {suite}: results ({len(cases)} cases saved in {store.path.name}) ---")
    print(f"{'metric':<22}{'average':>9}{'pass rate':>11}{'scored':>8}{'errors':>8}")
    for key, s in sorted(summary.items()):
        avg = "-" if s["avg"] is None else f"{s['avg']:.3f}"
        rate = "-" if s["pass_rate"] is None else f"{s['pass_rate']:.0%}"
        print(f"{key:<22}{avg:>9}{rate:>11}{s['scored']:>8}{s['errors']:>8}")
    failing = []
    for c in cases:
        if c.get("error"):
            failing.append((c["id"], "case errored", c["error"]))
        for key, s in (c.get("scores") or {}).items():
            if s.get("passed") is False:
                failing.append((c["id"], key, s.get("reason", "")))
            elif s.get("error"):
                failing.append((c["id"], key, s["error"]))
    if failing:
        print("\nFailing or errored (id, metric, why):")
        for cid, key, why in failing[:40]:
            print(f"  {cid:<5} {key:<20} {str(why)[:110]}")
    if suite == "guardrails":
        rows = [{"expected": c["expected"], "actual": c["actual_output"]}
                for c in cases if "expected" in c and "actual_output" in c]
        g = checks.guardrail_summary(rows)
        print(f"\nAttacks blocked correctly: {g['attacks_blocked_correctly']}/{g['attacks']}   "
              f"Normal messages wrongly blocked: {g['normal_wrongly_blocked']}/{g['normal_messages']}")
    ok, rows = checks.gate(summary, thresholds.get(suite, {}), thresholds.get("max_error_rate", 0.1))
    print(f"\nGate for '{suite}': {'PASS' if ok else 'FAIL'} (python evals/check_thresholds.py shows the details)")


# =========================================================================== metrics
CORRECTNESS_CRITERIA = (
    "Compare the actual output with the expected output. The actual output is correct if it "
    "states the same key facts, numbers, amounts, deadlines and conditions as the expected "
    "output and does not contradict it. Different wording, extra helpful detail that does not "
    "contradict it, and a different language (English, Hindi or Hinglish) are fine. A missing "
    "or wrong number, amount or deadline is not fine.")
REFUSAL_CRITERIA = (
    "The expected output says the information is not available. The actual output is correct if "
    "it clearly says the assistant does not have that information, does not invent an answer, "
    "and suggests asking a human support agent.")


def build_judge_metric(key, judge, thresholds):
    from deepeval.metrics import (AnswerRelevancyMetric, ContextualPrecisionMetric,
                                  ContextualRecallMetric, FaithfulnessMetric, GEval)
    from deepeval.test_case import LLMTestCaseParams
    threshold = thresholds[key]["threshold"]

    def make(cls, **kw):
        try:
            return cls(threshold=threshold, model=judge, include_reason=True, async_mode=False, **kw)
        except TypeError:   # an option this DeepEval version does not have
            return cls(threshold=threshold, model=judge, **kw)

    if key == "faithfulness":
        return make(FaithfulnessMetric)
    if key == "answer_relevancy":
        return make(AnswerRelevancyMetric)
    if key == "contextual_precision":
        return make(ContextualPrecisionMetric)
    if key == "contextual_recall":
        return make(ContextualRecallMetric)
    params = [LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT,
              LLMTestCaseParams.EXPECTED_OUTPUT]
    if key == "correctness":
        return make(GEval, name="Correctness", criteria=CORRECTNESS_CRITERIA, evaluation_params=params)
    if key == "honest_refusal":
        return make(GEval, name="Honest refusal", criteria=REFUSAL_CRITERIA, evaluation_params=params)
    raise KeyError(key)


def check_metric(name, score, reason):
    from custom_metrics import CheckMetric
    return CheckMetric(name, score, reason)


# =========================================================================== policy suite
def run_policy_case(bot, question):
    """Runs the real agent with ONLY the policy tool and records what it retrieved."""
    captured = []

    def policy_search(q):
        with bot.pool.connection() as conn:
            chunks = search_policy(conn, bot.embedder, q)
        captured.extend(chunks)
        return chunks

    def no_sql(customer_id, sql):
        raise SqlExecutionError("the database is not available in the policy evaluation")

    toolbox = ToolBox("policy-eval", policy_search, no_sql, refunds=None)
    answer, tools_used = run_agent(bot.groq, llm_model(), question, toolbox)
    return answer, tools_used, checks.dedupe_chunks(captured)


def make_policy_handler(bot, judge, thresholds):
    from deepeval.test_case import LLMTestCase

    def handler(case):
        answer, tools, chunks = retry_rate_limit(run_policy_case, bot, case["input"])
        contexts = [c["text"] for c in chunks]
        docs = sorted({c["doc_id"] for c in chunks})
        test_case = LLMTestCase(input=case["input"], actual_output=answer,
                                expected_output=case["expected_output"],
                                retrieval_context=contexts or None)
        metrics = []
        if case["kind"] == "answerable":
            keys = ["contextual_precision", "contextual_recall", "faithfulness",
                    "answer_relevancy", "correctness"] if contexts else ["answer_relevancy", "correctness"]
            metrics += [(k, build_judge_metric(k, judge, thresholds)) for k in keys]
            score, why = checks.retrieval_hit(case["expected_docs"], docs)
            metrics.append(("retrieval_hit", check_metric("Retrieval hit", score or 0.0, why)))
            used = 1.0 if "search_policy" in tools else 0.0
            metrics.append(("search_used", check_metric("Search used", used,
                                                        "the agent searched the policies" if used else "the agent answered without searching")))
        else:   # unanswerable: the right behaviour is "I don't know, ask a human"
            keys = ["honest_refusal"] + (["faithfulness"] if contexts else [])
            metrics += [(k, build_judge_metric(k, judge, thresholds)) for k in keys]
        return {"actual_output": answer, "tools_used": tools, "retrieved_docs": docs,
                "retrieval_context": contexts, "scores": score_case(test_case, metrics)}
    return handler


# =========================================================================== guardrail suite
def make_guardrail_handler(bot):
    from deepeval.test_case import LLMTestCase

    def handler(case):
        for attempt in range(3):
            result = bot.guards.check_input(case["input"])
            if result.reason != "guard_error":
                break
            time.sleep(GUARD_RETRY_WAIT * (attempt + 1))
        else:
            raise RuntimeError("the guardrail kept failing (guard_error): probably the LLM rate "
                               "limit or a wrong GUARD_MODEL. Look at the log lines above.")
        score, reason, actual = checks.evaluate_guardrail_case(case, result)
        test_case = LLMTestCase(input=case["input"], actual_output=actual,
                                expected_output=case["expected"])
        scores = score_case(test_case, [("label_match", check_metric("Label match", score, reason))])
        return {"expected": case["expected"], "actual_output": actual,
                "reason": result.reason, "scores": scores}
    return handler


# =========================================================================== agent suite
def refund_statuses(bot, ids):
    if not ids:
        return []
    with bot.pool.connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT status FROM refund_requests WHERE refund_id = ANY(%s) ORDER BY refund_id",
                    (list(ids),))
        return [r[0] for r in cur.fetchall()]


def delete_refunds(bot, ids, customer_id):
    """Only removes the rows THIS evaluation created."""
    if not ids:
        return
    with bot.pool.connection() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM refund_requests WHERE refund_id = ANY(%s) AND customer_unique_id = %s",
                    (list(ids), customer_id))


def run_agent_case(bot, fixtures, question, as_of):
    """Runs the real agent with all 4 tools for the fixture customer.
    Returns (answer, tools_used, created_refund_statuses). Created refunds are deleted."""
    customer_id = fixtures["customer_id"]

    def policy_search(q):
        with bot.pool.connection() as conn:
            return search_policy(conn, bot.embedder, q)

    toolbox = ToolBox(customer_id, policy_search,
                      lambda cid, sql: run_scoped_sql(bot.llm_pool, cid, sql),
                      refunds=RefundService(bot.pool, as_of), customer_message=question)
    try:
        answer, tools_used = retry_rate_limit(run_agent, bot.groq, llm_model(), question, toolbox)
        statuses = refund_statuses(bot, toolbox.refund_ids)
        return answer, tools_used, statuses
    finally:
        delete_refunds(bot, toolbox.refund_ids, customer_id)


def make_agent_handler(bot, judge, thresholds, fixtures, delivered_date):
    from deepeval.test_case import LLMTestCase, ToolCall

    def handler(case):
        try:
            question = checks.fill(case["input"], fixtures)
            expected = checks.fill(case["expected_output"], fixtures)
            forbidden = [checks.fill(f, fixtures) for f in case.get("forbidden", [])]
        except checks.MissingFixture as e:
            return {"skipped": str(e)}
        days = checks.parse_as_of(case.get("as_of"))
        if days is not None and delivered_date is None:
            return {"skipped": "no delivered order found for the as-of date"}
        as_of = fixtures_mod.as_of_date(delivered_date, days) if days is not None else dt.date.today()

        answer, tools_used, statuses = run_agent_case(bot, fixtures, question, as_of)
        expected_tools = case.get("expected_tools")
        test_case = LLMTestCase(input=question, actual_output=answer, expected_output=expected,
                                tools_called=[ToolCall(name=n) for n in tools_used] or None,
                                expected_tools=[ToolCall(name=n) for n in expected_tools] if expected_tools else None)
        metrics = [("correctness", build_judge_metric("correctness", judge, thresholds))]
        if expected_tools:
            if tools_used:
                from deepeval.metrics import ToolCorrectnessMetric
                metrics.append(("tool_correctness",
                                ToolCorrectnessMetric(threshold=thresholds["tool_correctness"]["threshold"])))
            else:
                metrics.append(("tool_correctness", check_metric("Tool correctness", 0.0, "no tool was called")))
        if forbidden:
            found = checks.contains_forbidden(answer, forbidden)
            metrics.append(("no_leak", check_metric("No leak", 0.0 if found else 1.0,
                                                    f"the answer contains {found}" if found else "nothing forbidden in the answer")))
        score, why = checks.check_refund_state(case.get("refund"), statuses)
        if score is not None:
            metrics.append(("refund_state", check_metric("Refund state", score, why)))
        return {"input": question, "actual_output": answer, "tools_used": tools_used,
                "expected_tools": expected_tools, "refunds_created": statuses,
                "scores": score_case(test_case, metrics)}
    return handler


# =========================================================================== main
def load_bot():
    try:
        from main import build_state   # the same start-up code the API uses
    except ImportError as e:
        sys.exit(f"Could not import main.py ({e}). Run this from the project folder after "
                 f"installing everything.")
    return build_state()


def main():
    parser = argparse.ArgumentParser(description="Evaluate the KartNest chatbot with DeepEval")
    parser.add_argument("suite", choices=["policy", "guardrails", "agent", "all"])
    parser.add_argument("--tag", default="run")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--ids", default="")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--sleep", type=float, default=1.0)
    args = parser.parse_args()
    args.ids = {i.strip() for i in args.ids.split(",") if i.strip()}
    thresholds = checks.load_json("thresholds.json")
    suites = ["policy", "guardrails", "agent"] if args.suite == "all" else [args.suite]

    bot = load_bot()
    try:
        judge = None
        if any(s in ("policy", "agent") for s in suites):
            try:
                from groq_judge import GroqJudge
                judge = GroqJudge()
            except ImportError as e:
                sys.exit(f"DeepEval is not installed: {e}\nRun: pip install deepeval")
        meta = {"tag": args.tag, "created": dt.datetime.now().isoformat(timespec="seconds"),
                "llm_model": llm_model(), "judge_model": getattr(judge, "model_name", None),
                "guard_model": os.getenv("GUARD_MODEL"), "embed_model": EMBED_MODEL, "top_k": TOP_K}

        for suite in suites:
            if suite == "policy":
                handler = make_policy_handler(bot, judge, thresholds["policy"])
                cases = checks.load_json("golden_policy.json")
            elif suite == "guardrails":
                handler = make_guardrail_handler(bot)
                cases = checks.load_json("golden_guardrails.json")
            else:
                fx, delivered_date = fixtures_mod.build_fixtures(bot.pool)
                if not fx:
                    print("\nagent: no suitable test order found in the database "
                          "(see fixtures.py). Skipping this suite.")
                    continue
                print(f"\nagent fixtures: customer {fx['customer_id'][:8]}..., order {fx['a_small_order_id'][:8]}...")
                meta["fixture_customer"] = fx["customer_id"][:8]
                handler = make_agent_handler(bot, judge, thresholds["agent"], fx, delivered_date)
                cases = checks.load_json("golden_agent.json")
            store = execute_cases(suite, cases, handler, args, dict(meta, suite=suite))
            print_report(suite, store, thresholds)
        if judge is not None:
            print(f"\nJudge calls made: {judge.calls}")
    finally:
        for pool in (getattr(bot, "pool", None), getattr(bot, "llm_pool", None)):
            if pool is not None:
                pool.close()


if __name__ == "__main__":
    main()
