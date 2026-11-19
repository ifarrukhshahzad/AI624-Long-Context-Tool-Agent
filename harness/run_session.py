#!/usr/bin/env python3
"""
run_session.py - Runs one BFCL task turn by turn and logs everything.

For every model call it logs (project spec, Task 2 "client harness"):
  tokens sent and reprocessed, prefill time, decode time, tool time,
  and call validity of every tool call the model emits.
After every user turn it runs BFCL's ground-truth calls in a separate
environment and compares states -> turn pass/fail; session success means
every turn passed.

Modes:
  --oracle   replay BFCL's ground-truth calls instead of the model. Checks that
             the environment + checker work (oracle must score 100%) and
             measures how long the context grows, without spending Pi time.
  default    call llama-server (start it with scripts/start_server.sh).

Examples (from the repo root):
  python3 -m harness.run_session --task multi_turn_long_context_100 --oracle
  python3 -m harness.run_session --task multi_turn_long_context_100
  python3 -m harness.run_session --split dev --limit 3 --oracle
  python3 -m harness.run_session --split custom --oracle   (our long tasks)
"""

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime

from harness.bfcl_env import Env, load_custom_tasks, load_task, parse_call_string, states_match
from harness.llm_client import LlamaServerClient
from harness.prompt import parse_tool_calls, render
from harness.validity import check_call
from tasks.checks import run_check

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def count_tokens(server, text):
    """Exact token count from llama-server /tokenize; None if no server."""
    if not server:
        return None
    try:
        req = urllib.request.Request(server.rstrip("/") + "/tokenize",
                                     data=json.dumps({"content": text}).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return len(json.loads(r.read())["tokens"])
    except Exception:
        return None


def run_task(task_id, schemas, args, log):
    task, gt_turns = load_task(task_id)
    by_name = {s["name"]: s for s in schemas}
    model_env, gt_env = Env(task), Env(task)
    client = None if args.oracle else LlamaServerClient(args.server, cache_prompt=False,
                                                         n_predict=args.n_predict)
    messages = []
    turn_results = []
    overflow = False
    totals = {"calls": 0, "valid": 0, "prompt_tokens": 0, "prefill_ms": 0.0,
              "decode_ms": 0.0, "tool_ms": 0.0, "llm_calls": 0}

    for turn, user_msgs in enumerate(task["question"]):
        for m in user_msgs:
            messages.append({"role": m["role"], "content": m["content"]})
        oracle_queue = [parse_call_string(c) for c in gt_turns[turn]] if args.oracle else []

        for step in range(args.max_steps):
            prompt = render(messages, schemas)
            rec = {"task": task_id, "turn": turn, "step": step,
                   "prompt_chars": len(prompt)}
            if args.oracle:
                if oracle_queue:
                    name, a = oracle_queue.pop(0)
                    text = "<tool_call>\n" + json.dumps({"name": name, "arguments": a}) + "\n</tool_call>"
                else:
                    text = "Done."
                rec["prompt_tokens"] = count_tokens(args.server, prompt)
            else:
                try:
                    out = client.complete(prompt)
                except urllib.error.HTTPError as e:
                    # llama-server refuses prompts longer than its context (-c 16384)
                    rec["error"] = f"HTTP {e.code}: {e.read()[:300].decode(errors='replace')}"
                    log.write(json.dumps(rec) + "\n")
                    log.flush()
                    overflow = True
                    print(f"  ! server error at turn {turn} step {step}: {rec['error'][:120]}")
                    break
                text = out.pop("text")
                rec.update(out)
                totals["llm_calls"] += 1
                totals["prompt_tokens"] += out["prompt_tokens"] or 0
                totals["prefill_ms"] += out["prefill_ms"] or 0
                totals["decode_ms"] += out["decode_ms"] or 0

            calls = parse_tool_calls(text)
            messages.append({"role": "assistant", "content": text})
            rec["n_calls"] = len(calls)
            rec["calls"] = []
            t0 = time.time()
            for raw, parsed in calls:
                ok, why = check_call(parsed, by_name)
                totals["calls"] += 1
                totals["valid"] += ok
                if ok:
                    result = model_env.execute(parsed["name"], parsed.get("arguments", {}))
                else:
                    result = json.dumps({"error": f"invalid call: {why}"})
                messages.append({"role": "tool", "content": result})
                rec["calls"].append({"call": raw[:300], "valid": ok, "reason": why,
                                     "result_chars": len(result)})
            rec["tool_ms"] = round((time.time() - t0) * 1000, 2)
            totals["tool_ms"] += rec["tool_ms"]
            log.write(json.dumps(rec) + "\n")
            log.flush()
            if not calls:
                break

        for c in gt_turns[turn]:
            name, a = parse_call_string(c)
            gt_env.execute(name, a)
        ok, diffs = states_match(model_env, gt_env)
        turn_results.append(ok)
        print(f"  turn {turn}: {'PASS' if ok else 'FAIL'}"
              + ("" if ok else f" (differs: {', '.join(diffs[:4])})"), flush=True)
        if overflow:
            break

    final_prompt = render(messages, schemas)
    check_ok, check_msg = run_check(task_id, model_env)
    n_turns = len(task["question"])
    success = (len(turn_results) == n_turns and all(turn_results)
               and check_ok is not False)
    if check_ok is not None:
        print(f"  functional check: {'PASS' if check_ok else 'FAIL'} - {check_msg}")
    summary = {
        "task": task_id,
        "mode": "oracle" if args.oracle else "model",
        "turns": n_turns,
        "turns_passed": sum(turn_results),
        "session_success": success,
        "functional_check": check_msg,
        "context_overflow": overflow,
        "call_validity": round(totals["valid"] / totals["calls"], 3) if totals["calls"] else None,
        "final_context_chars": len(final_prompt),
        "final_context_tokens": count_tokens(args.server, final_prompt),
        **totals,
    }
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", help="one BFCL task id")
    ap.add_argument("--split", choices=["dev", "test", "all", "custom"],
                    help="dev/test/all = BFCL tasks from configs/bfcl_tasks.json; "
                         "custom = our long tasks from tasks/long_tasks.json")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--oracle", action="store_true")
    ap.add_argument("--server", default="http://127.0.0.1:8080")
    ap.add_argument("--no-server", action="store_true", help="oracle mode without token counting")
    ap.add_argument("--max-steps", type=int, default=6, help="model calls per user turn")
    ap.add_argument("--n-predict", type=int, default=512)
    ap.add_argument("--outdir", default=os.path.join(ROOT, "results", "sessions"))
    args = ap.parse_args()
    if args.no_server:
        args.server = None

    with open(os.path.join(ROOT, "configs", "tool_schemas.json")) as f:
        schemas = json.load(f)
    if args.task:
        ids = [args.task]
    elif args.split == "custom":
        ids = [t["id"] for t in load_custom_tasks()]
    else:
        with open(os.path.join(ROOT, "configs", "bfcl_tasks.json")) as f:
            tasks = json.load(f)["tasks"]
        ids = [t["id"] for t in tasks if args.split in (None, "all") or t["split"] == args.split]
        if args.limit:
            ids = ids[:args.limit]

    os.makedirs(args.outdir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode = "oracle" if args.oracle else "model"
    log_path = os.path.join(args.outdir, f"{mode}_{stamp}.jsonl")
    sum_path = os.path.join(args.outdir, f"{mode}_{stamp}_summary.json")
    summaries = []
    with open(log_path, "w") as log:
        for tid in ids:
            print(f"[{tid}]", flush=True)
            s = run_task(tid, schemas, args, log)
            summaries.append(s)
            print(f"  -> session {'SUCCESS' if s['session_success'] else 'FAIL'}, "
                  f"turns {s['turns_passed']}/{s['turns']}, validity {s['call_validity']}, "
                  f"context {s['final_context_tokens'] or s['final_context_chars']} "
                  f"{'tokens' if s['final_context_tokens'] else 'chars'}", flush=True)
    with open(sum_path, "w") as f:
        json.dump(summaries, f, indent=2)
    n = len(summaries)
    print(f"\n{n} tasks: session success {sum(s['session_success'] for s in summaries)}/{n}, "
          f"turns {sum(s['turns_passed'] for s in summaries)}/{sum(s['turns'] for s in summaries)}")
    print(f"log: {log_path}\nsummary: {sum_path}")


if __name__ == "__main__":
    main()
