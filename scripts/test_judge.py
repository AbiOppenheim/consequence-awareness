#!/usr/bin/env python
"""Regression test for the API judge. No network, no key, no cost.

    python scripts/test_judge.py

Stubs the OpenAI client so the parts that are ours — request shape, concurrency, per-row error
handling, and the aggregation — can be checked without spending anything. What this cannot
check is whether the live API accepts the request; run one small file first for that.

The check that matters most is the last one: a row whose judgment FAILED must not be counted as
a refusal. Failures that quietly read as "not a bypass" would deflate every bypass rate, which
is exactly the direction that flatters a defense claim.
"""

import json
import os
import sys
import types

import _bootstrap  # noqa: F401
from consequence import judge as J

VALID = '{"label": "bypass", "strongreject": 0.8, "coherent": true}'


def install_stub_openai(behaviour):
    """Put a fake `openai` module in sys.modules. behaviour(i) -> reply text, or raises."""
    seen = []

    class Message:
        def __init__(self, content):
            self.content = content

    class Choice:
        def __init__(self, content):
            self.message = Message(content)

    class Completions:
        def create(self, **kw):
            seen.append(kw)
            return types.SimpleNamespace(choices=[Choice(behaviour(len(seen) - 1))])

    class Client:
        def __init__(self, *a, **kw):
            self.chat = types.SimpleNamespace(completions=Completions())

    sys.modules["openai"] = types.SimpleNamespace(OpenAI=Client)
    return seen


def check(name, ok):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    return ok


def main() -> int:
    os.environ.setdefault("OPENAI_API_KEY", "test-key-not-used")
    rows = [{"prompt": f"p{i}", "completion": f"c{i}", "condition": "steer_vc", "alpha": 1.0}
            for i in range(5)]
    ok = True

    # ---- happy path -------------------------------------------------------------------
    seen = install_stub_openai(lambda i: VALID)
    scored = J.score_batch(rows, model="gpt-4.1-mini", provider="openai", max_workers=4)
    ok &= check("every row scored", all(r["label"] == "bypass" for r in scored))
    ok &= check("input rows not mutated", all("label" not in r for r in rows))
    ok &= check("one request per row", len(seen) == len(rows))

    body = seen[0]
    ok &= check("request pins the JSON schema",
                body["response_format"]["json_schema"]["schema"] == J.VERDICT_SCHEMA
                and body["response_format"]["json_schema"]["strict"] is True)
    ok &= check("no temperature/seed sent (newer models reject them)",
                "temperature" not in body and "seed" not in body)
    ok &= check("system prompt + one user turn",
                [m["role"] for m in body["messages"]] == ["system", "user"])
    ok &= check("prompt and completion both reach the judge",
                "p0" in body["messages"][1]["content"] and "c0" in body["messages"][1]["content"])

    # ---- a failing row must not sink the run, or become a refusal ----------------------
    def flaky(i):
        if i == 2:
            raise RuntimeError("rate limit")
        return VALID

    install_stub_openai(flaky)
    scored = J.score_batch(rows, model="gpt-4.1-mini", provider="openai", max_workers=1)
    failed = [r for r in scored if r.get("label") is None]
    ok &= check("the failing row is recorded, not dropped", len(scored) == len(rows))
    ok &= check("failure carries a judge_error", len(failed) == 1 and "judge_error" in failed[0])

    s = J.summarize(scored)
    ok &= check("summary counts the failure separately",
                s["n_total"] == 5 and s["n_judged"] == 4 and s["n_failed"] == 1)
    ok &= check("rates are over JUDGED rows, so a failure is not a free refusal",
                s["bypass_rate"] == 1.0 and s["refusal_rate"] == 0.0)

    # ---- malformed replies are caught by the parser -------------------------------------
    install_stub_openai(lambda i: "sorry, I can't help with that")
    scored = J.score_batch(rows, model="gpt-4.1-mini", provider="openai", max_workers=1)
    ok &= check("prose instead of JSON -> all rows flagged failed",
                all(r.get("label") is None for r in scored))

    install_stub_openai(lambda i: '{"label": "refused", "strongreject": 0.1, "coherent": true}')
    scored = J.score_batch(rows, model="gpt-4.1-mini", provider="openai", max_workers=1)
    ok &= check("out-of-vocabulary label rejected",
                all(r.get("label") is None for r in scored))

    # ---- configuration mistakes fail loudly ---------------------------------------------
    try:
        J.score_batch(rows, model="x", provider="gemini")
        ok &= check("unknown provider rejected", False)
    except ValueError:
        ok &= check("unknown provider rejected", True)

    saved = os.environ.pop("OPENAI_API_KEY")
    try:
        J.score_batch(rows, model="x", provider="openai")
        ok &= check("missing API key rejected", False)
    except RuntimeError as e:
        ok &= check("missing API key names the right env var", "OPENAI_API_KEY" in str(e))
    finally:
        os.environ["OPENAI_API_KEY"] = saved

    # ---- the batch path builds the JSONL the API expects --------------------------------
    line = json.dumps({"custom_id": "gen-0", "method": "POST", "url": "/v1/chat/completions",
                       "body": J._openai_params("gpt-4.1-mini", rows[0])})
    rec = json.loads(line)
    ok &= check("batch line has custom_id / method / url / body",
                {"custom_id", "method", "url", "body"} == set(rec)
                and rec["url"] == "/v1/chat/completions")

    print("\n" + ("ALL PASS — judge plumbing is provider-correct; run one small file against "
                  "the live API to confirm the request is accepted"
                  if ok else "FAILURES ABOVE"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
