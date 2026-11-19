#!/usr/bin/env python3
"""
build_tool_subset.py - Builds our BFCL tool subset and task list.

Choice (data-driven): TradingBot (20 functions, required by the project spec)
+ MessageAPI (10 functions) = 30 functions, the top of the allowed 20-30 range.
In BFCL's multi_turn_long_context set, 33 tasks use ONLY these two backends
(20 TradingBot-only + 13 MessageAPI+TradingBot), so session success can be
measured on real BFCL tasks without needing any other backend.

Outputs (in --out, default configs/):
  tool_schemas.json   - the 30 function schemas, exactly as BFCL defines them
  bfcl_tasks.json     - the 33 task ids, their backends, and a dev/test split
                        (every 4th task is held out as test)

Usage (on any machine with a clone of the gorilla repo):
  python3 scripts/build_tool_subset.py --bfcl /path/to/gorilla/berkeley-function-call-leaderboard
"""

import argparse
import json
import os

BACKENDS = {
    "TradingBot": "trading_bot.json",
    "MessageAPI": "message_api.json",
}
TASK_FILE = "BFCL_v4_multi_turn_long_context.json"


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bfcl", required=True, help="path to berkeley-function-call-leaderboard")
    ap.add_argument("--out", default="configs")
    args = ap.parse_args()

    data_dir = os.path.join(args.bfcl, "bfcl_eval", "data")
    doc_dir = os.path.join(data_dir, "multi_turn_func_doc")
    os.makedirs(args.out, exist_ok=True)

    # 1. Tool schemas
    schemas = []
    for backend, fname in BACKENDS.items():
        for fn in read_jsonl(os.path.join(doc_dir, fname)):
            fn["backend"] = backend
            schemas.append(fn)
    names = [s["name"] for s in schemas]
    assert len(names) == len(set(names)), "duplicate function names"
    with open(os.path.join(args.out, "tool_schemas.json"), "w", encoding="utf-8") as f:
        json.dump(schemas, f, indent=2)

    # 2. Tasks that use only our backends
    allowed = set(BACKENDS)
    tasks = [t for t in read_jsonl(os.path.join(data_dir, TASK_FILE))
             if set(t["involved_classes"]) <= allowed]
    task_list = []
    for i, t in enumerate(tasks):
        task_list.append({
            "id": t["id"],
            "backends": t["involved_classes"],
            "turns": len(t["question"]),
            "split": "test" if i % 4 == 3 else "dev",
        })
    with open(os.path.join(args.out, "bfcl_tasks.json"), "w", encoding="utf-8") as f:
        json.dump({"source": TASK_FILE, "backends": list(BACKENDS),
                   "tasks": task_list}, f, indent=2)

    # 3. Summary
    print(f"Functions : {len(schemas)} "
          + ", ".join(f"{b}={sum(s['backend'] == b for s in schemas)}" for b in BACKENDS))
    print(f"Tasks     : {len(task_list)} "
          f"(dev={sum(t['split'] == 'dev' for t in task_list)}, "
          f"test={sum(t['split'] == 'test' for t in task_list)})")
    turns = [t["turns"] for t in task_list]
    print(f"Turns/task: min {min(turns)}, max {max(turns)}, avg {sum(turns) / len(turns):.1f}")
    schema_chars = len(json.dumps(schemas))
    print(f"Schemas   : {schema_chars} characters (~{schema_chars // 4} tokens) "
          "sent in the system prompt every turn in the baseline")


if __name__ == "__main__":
    main()
