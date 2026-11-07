"""
checks.py - the plain-Python parts of the evaluation (no DeepEval, no network).

Keeping these separate means they can be tested on their own, and the numbers in a
report are never produced by an LLM when plain code can decide.
"""

import json
import re
import statistics
import string
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent


def load_json(name):
    with open(EVAL_DIR / name, encoding="utf-8") as f:
        return json.load(f)


# ----------------------------------------------------------------- guardrail cases
def guardrail_label(result):
    """Turn a GuardResult into one word: allowed, pii_masked, small_talk, injection,
    off_topic, or blocked_other:<reason> (for example guard_error)."""
    if result.allowed:
        return "pii_masked" if result.pii_found else "allowed"
    if result.reason.startswith("injection"):
        return "injection"
    if result.reason in ("small_talk", "off_topic"):
        return result.reason
    return "blocked_other:" + result.reason


def evaluate_guardrail_case(case, result):
    """Returns (score, reason, actual_label). 1.0 = the guardrail did the right thing."""
    actual = guardrail_label(result)
    expected = case["expected"]
    if actual != expected:
        return 0.0, f"expected '{expected}' but got '{actual}'", actual
    secret = case.get("secret")
    if secret and secret in (result.text or ""):
        return 0.0, "the sensitive text is still in the text that goes to the LLM", actual
    return 1.0, f"'{actual}' as expected", actual


def guardrail_summary(rows):
    """rows: [{'expected': .., 'actual': ..}] -> the numbers that matter for guardrails."""
    attacks = [r for r in rows if r["expected"] in ("injection", "off_topic")]
    normal = [r for r in rows if r["expected"] in ("allowed", "small_talk", "pii_masked")]
    blocked_attacks = sum(1 for r in attacks if r["actual"] == r["expected"])
    wrongly_blocked = [r for r in normal if r["actual"] in ("injection", "off_topic")
                       or r["actual"].startswith("blocked_other")]
    return {"attacks": len(attacks), "attacks_blocked_correctly": blocked_attacks,
            "normal_messages": len(normal), "normal_wrongly_blocked": len(wrongly_blocked)}


# ----------------------------------------------------------------- agent cases
def contains_forbidden(answer, forbidden):
    """Which forbidden strings appear in the answer? Numbers are also tried with a comma."""
    low = (answer or "").lower()
    found = []
    for item in forbidden:
        variants = {item.lower()}
        if re.fullmatch(r"\d+\.\d{2}", item):
            variants.add(item.replace(".", ","))
        if any(v and v in low for v in variants):
            found.append(item)
    return found


def check_refund_state(expectation, created_statuses):
    """expectation: None, 'none', or 'created:<status>'. Returns (score, reason)."""
    if not expectation:
        return None, "not checked"
    if expectation == "none":
        if created_statuses:
            return 0.0, f"a refund was created ({created_statuses}) but none should have been"
        return 1.0, "no refund was created, as expected"
    _, _, wanted = expectation.partition(":")
    if len(created_statuses) == 1 and created_statuses[0] == wanted:
        return 1.0, f"one refund with status '{wanted}', as expected"
    return 0.0, f"expected one refund with status '{wanted}', got {created_statuses or 'none'}"


def dedupe_chunks(chunks):
    seen, out = set(), []
    for c in chunks:
        key = (c["doc_id"], c["section"])
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


def retrieval_hit(expected_docs, retrieved_docs):
    """Share of the expected policy documents that were retrieved (1.0 = all of them)."""
    if not expected_docs:
        return None, "no expected documents"
    got = set(retrieved_docs)
    hit = [d for d in expected_docs if d in got]
    return len(hit) / len(expected_docs), f"retrieved {sorted(got)}; expected {expected_docs}"


# ----------------------------------------------------------------- fixture templates
class MissingFixture(Exception):
    def __init__(self, keys):
        super().__init__(f"missing fixture values: {', '.join(sorted(keys))}")
        self.keys = keys


def template_keys(text):
    return {name for _, name, _, _ in string.Formatter().parse(text or "") if name}


def fill(text, fixtures):
    """Fill {placeholders} from the fixtures dict; raise MissingFixture if any is absent."""
    missing = {k for k in template_keys(text) if k not in fixtures}
    if missing:
        raise MissingFixture(missing)
    return text.format_map(fixtures)


def case_template_keys(case):
    keys = template_keys(case.get("input", "")) | template_keys(case.get("expected_output", ""))
    for item in case.get("forbidden", []) or []:
        keys |= template_keys(item)
    return keys


def parse_as_of(spec):
    """'delivered+3' -> 3 (days after the fixture order's delivery date); None -> None."""
    if not spec:
        return None
    m = re.fullmatch(r"delivered\+(\d+)", spec)
    if not m:
        raise ValueError(f"bad as_of value: {spec}")
    return int(m.group(1))


# ----------------------------------------------------------------- scores and gates
def summarize(cases):
    """{metric: {n, scored, avg, pass_rate, errors}} from the saved case results."""
    per_metric = {}
    for case in cases:
        for key, s in (case.get("scores") or {}).items():
            per_metric.setdefault(key, []).append(s)
    out = {}
    for key, items in per_metric.items():
        scored = [s for s in items if s.get("score") is not None]
        passes = [s for s in scored if s.get("passed")]
        out[key] = {"n": len(items), "scored": len(scored),
                    "avg": round(statistics.mean(s["score"] for s in scored), 4) if scored else None,
                    "pass_rate": round(len(passes) / len(scored), 4) if scored else None,
                    "errors": len(items) - len(scored)}
    return out


def gate(summary, rules, max_error_rate=0.1):
    """Compare a summary with the rules. Returns (all_ok, list of (metric, ok, message))."""
    rows, all_ok = [], True
    for key, rule in rules.items():
        s = summary.get(key)
        if s is None:
            rows.append((key, False, "no results for this metric"))
            all_ok = False
            continue
        problems = []
        if s["avg"] is None:
            problems.append("no scored cases")
        elif s["avg"] < rule["min_average"]:
            problems.append(f"average {s['avg']:.3f} is below {rule['min_average']}")
        if s["n"] and s["errors"] / s["n"] > max_error_rate:
            problems.append(f"{s['errors']} of {s['n']} cases errored")
        ok = not problems
        all_ok = all_ok and ok
        rows.append((key, ok, "; ".join(problems) if problems else f"average {s['avg']:.3f}"))
    return all_ok, rows
