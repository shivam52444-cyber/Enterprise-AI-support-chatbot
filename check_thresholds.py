"""
check_thresholds.py - did the evaluation meet the quality rules in thresholds.json?

    python evals/check_thresholds.py --tag baseline
    python evals/check_thresholds.py --tag baseline --suite agent

Exits with code 1 if any rule is broken, so it can stop a deployment or a CI job.
The first time, run the evaluation, look at the real averages, then set min_average
in thresholds.json a little BELOW what you measured and tighten it as the bot improves.
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import checks  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Check evaluation results against thresholds.json")
    parser.add_argument("--tag", default="run")
    parser.add_argument("--suite", choices=["policy", "guardrails", "agent"])
    args = parser.parse_args()
    thresholds = checks.load_json("thresholds.json")
    suites = [args.suite] if args.suite else ["policy", "guardrails", "agent"]

    overall_ok, found_any = True, False
    for suite in suites:
        path = HERE / "results" / f"{args.tag}_{suite}.json"
        if not path.exists():
            if args.suite:
                sys.exit(f"No results file: {path}")
            continue
        found_any = True
        cases = json.loads(path.read_text(encoding="utf-8")).get("cases", [])
        summary = checks.summarize([c for c in cases if not c.get("skipped")])
        ok, rows = checks.gate(summary, thresholds.get(suite, {}), thresholds.get("max_error_rate", 0.1))
        overall_ok = overall_ok and ok
        print(f"\n{suite}: {'PASS' if ok else 'FAIL'}")
        for key, row_ok, message in rows:
            print(f"  {'ok  ' if row_ok else 'FAIL'} {key:<22} {message}")
    if not found_any:
        sys.exit(f"No results for tag '{args.tag}' in {HERE / 'results'}")
    print("\nOVERALL:", "PASS" if overall_ok else "FAIL")
    sys.exit(0 if overall_ok else 1)


if __name__ == "__main__":
    main()
