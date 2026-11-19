"""
bfcl_env.py - Runs BFCL's own TradingBot / MessageAPI backends for one task.

We import BFCL's original backend classes (from a clone of the gorilla repo),
load the task's initial state in LONG-CONTEXT mode (verbose tool outputs, as
the project spec requires), and execute tool calls against them.

Two environments are created per task:
  * model env        - receives the calls the model makes
  * ground-truth env - receives BFCL's reference calls for the same turn
After each turn we compare their states (BFCL's state check: all public
attributes of every backend instance must match).

Set BFCL_ROOT to the berkeley-function-call-leaderboard folder if it is not at
~/gorilla/berkeley-function-call-leaderboard.
"""

import ast
import copy
import json
import os
import sys

BFCL_ROOT = os.path.expanduser(os.environ.get(
    "BFCL_ROOT", "~/gorilla/berkeley-function-call-leaderboard"))
if BFCL_ROOT not in sys.path:
    sys.path.insert(0, BFCL_ROOT)

DATA_DIR = os.path.join(BFCL_ROOT, "bfcl_eval", "data")
TASK_FILE = os.path.join(DATA_DIR, "BFCL_v4_multi_turn_long_context.json")
ANSWER_FILE = os.path.join(DATA_DIR, "possible_answer", "BFCL_v4_multi_turn_long_context.json")


def _backend_classes():
    from bfcl_eval.eval_checker.multi_turn_eval.func_source_code.message_api import MessageAPI
    from bfcl_eval.eval_checker.multi_turn_eval.func_source_code.trading_bot import TradingBot
    return {"TradingBot": TradingBot, "MessageAPI": MessageAPI}


def _read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_task(task_id):
    """Return (task, ground_truth_turns) for one BFCL long-context task id."""
    task = next(t for t in _read_jsonl(TASK_FILE) if t["id"] == task_id)
    answer = next(a for a in _read_jsonl(ANSWER_FILE) if a["id"] == task_id)
    return task, answer["ground_truth"]


def parse_call_string(call):
    """"get_stock_info(symbol='NVDA')" -> ("get_stock_info", {"symbol": "NVDA"})"""
    node = ast.parse(call, mode="eval").body
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
        raise ValueError(f"not a simple call: {call}")
    args = {kw.arg: ast.literal_eval(kw.value) for kw in node.keywords}
    return node.func.id, args


class Env:
    def __init__(self, task):
        classes = _backend_classes()
        self.instances = {}
        self.method_owner = {}
        for name in task["involved_classes"]:
            inst = classes[name]()
            config = copy.deepcopy(task.get("initial_config", {}).get(name, {}))
            inst._load_scenario(config, long_context=True)
            self.instances[name] = inst
            for attr in dir(inst):
                if not attr.startswith("_") and callable(getattr(inst, attr)):
                    self.method_owner[attr] = inst

    def execute(self, name, args):
        """Run one tool call; always returns a string (result or error)."""
        if name not in self.method_owner:
            return json.dumps({"error": f"Unknown function: {name}"})
        try:
            result = getattr(self.method_owner[name], name)(**args)
        except Exception as e:  # wrong args etc. -> the model sees the error
            return json.dumps({"error": f"{type(e).__name__}: {e}"})
        try:
            return json.dumps(result, default=str)
        except TypeError:
            return str(result)

    def state(self):
        """Public attributes of every backend instance (BFCL's state check)."""
        snap = {}
        for name, inst in self.instances.items():
            snap[name] = {k: v for k, v in vars(inst).items() if not k.startswith("_")}
        return snap


def states_match(model_env, gt_env):
    a, b = model_env.state(), gt_env.state()
    diffs = []
    for backend in b:
        for key, value in b[backend].items():
            if a.get(backend, {}).get(key) != value:
                diffs.append(f"{backend}.{key}")
    return len(diffs) == 0, diffs
