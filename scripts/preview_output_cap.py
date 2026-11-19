#!/usr/bin/env python3
"""
preview_output_cap.py - How many tasks would fit in the 16k cache if long tool
outputs were capped? (A preview of the planned context-management step.)

Method: replay every task with BFCL's ground-truth calls (oracle, no model),
then measure the final conversation length with the exact Qwen3 tokenizer,
once with full tool outputs and once per cap, where each tool output is cut to
its first N tokens.

What this shows: the SIZE saving only. Whether the agent still succeeds with
capped outputs (it may need data that was cut) is tested later on the Pi.

Run on the Pi (needs the tokenizers package, installed with the converter):
  source ~/venv/bin/activate
  python3 scripts/preview_output_cap.py \
      --tokenizer ~/models/Qwen3-1.7B/tokenizer.json
"""

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tokenizers import Tokenizer  # noqa: E402

from harness.bfcl_env import Env, load_custom_tasks, load_task, parse_call_string  # noqa: E402
from harness.prompt import render  # noqa: E402

CACHE = 16384


def oracle_messages(task_id):
    task, gt = load_task(task_id)
    env = Env(task)
    msgs = []
    for turn, user_msgs in enumerate(task["question"]):
        msgs += [{"role": m["role"], "content": m["content"]} for m in user_msgs]
        for call in gt[turn]:
            name, args = parse_call_string(call)
            msgs.append({"role": "assistant", "content": "<tool_call>\n"
                         + json.dumps({"name": name, "arguments": args}) + "\n</tool_call>"})
            msgs.append({"role": "tool", "content": env.execute(name, args)})
        msgs.append({"role": "assistant", "content": "Done."})
    return msgs


def capped(msgs, tok, cap):
    if cap is None:
        return msgs
    out = []
    for m in msgs:
        if m["role"] == "tool":
            enc = tok.encode(m["content"])
            if len(enc.ids) > cap:
                cut = tok.decode(enc.ids[:cap])
                m = {"role": "tool", "content": cut + f" ...[truncated, {len(enc.ids)} tokens total]"}
        out.append(m)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default=os.path.expanduser("~/models/Qwen3-1.7B/tokenizer.json"))
    ap.add_argument("--caps", default="none,4096,2048,1024,512")
    args = ap.parse_args()

    tok = Tokenizer.from_file(args.tokenizer)
    with open(os.path.join(ROOT, "configs", "tool_schemas.json")) as f:
        schemas = json.load(f)
    with open(os.path.join(ROOT, "configs", "bfcl_tasks.json")) as f:
        ids = [t["id"] for t in json.load(f)["tasks"]]
    ids += [t["id"] for t in load_custom_tasks()]
    caps = [None if c == "none" else int(c) for c in args.caps.split(",")]

    rows = []
    for tid in ids:
        msgs = oracle_messages(tid)
        rows.append([len(tok.encode(render(capped(msgs, tok, c), schemas)).ids) for c in caps])

    print(f"{len(ids)} tasks ({len(ids) - len(load_custom_tasks())} BFCL + "
          f"{len(load_custom_tasks())} ours), final context with perfect answers:\n")
    print(f"{'tool-output cap':>16} | {'median tokens':>13} | {'max tokens':>10} | fit in 16k")
    print("-" * 60)
    for i, c in enumerate(caps):
        vals = sorted(r[i] for r in rows)
        fit = sum(v <= CACHE for v in vals)
        label = "none (baseline)" if c is None else f"{c} tokens"
        print(f"{label:>16} | {vals[len(vals) // 2]:>13,} | {vals[-1]:>10,} | {fit}/{len(vals)}")

    out = os.path.join(ROOT, "results", "preview_output_cap.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"caps": args.caps.split(","), "tasks": ids, "tokens": rows}, f, indent=1)
    print(f"\nper-task numbers: {out}")


if __name__ == "__main__":
    main()
