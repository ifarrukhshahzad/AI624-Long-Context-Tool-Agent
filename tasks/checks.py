"""
checks.py - Functional success checks for our long tasks.

The harness already compares the full backend state with the ground truth after
every turn. These checks state, in plain terms, what "the task got done" means,
and report a readable reason when it did not.
"""


def _orders(env):
    return env.instances["TradingBot"].orders


def _new_orders(env, symbol):
    # Note: BFCL's place_order reuses order id 12446 (it overwrites the existing
    # GOOG order), so we find the new order by symbol and type, not by id.
    return [o for o in _orders(env).values()
            if o.get("symbol") == symbol and o.get("order_type") == "Buy"]


def check_long_A_budget_recall(env):
    buys = _new_orders(env, "NVDA")
    if not buys:
        return False, "no new NVDA buy order (did it forget 'Nvidia' from turn 0?)"
    o = buys[-1]
    if o["amount"] != 6:
        return False, f"bought {o['amount']} shares; budget $1,500 at 220.34 allows 6"
    return True, "6 NVDA shares bought within the $1,500 budget"


def check_long_B_colleague_request(env):
    buys = _new_orders(env, "MSFT")
    if not buys or buys[-1]["amount"] != 3:
        return False, "no 3-share MSFT buy (MSFT 310.23 is cheaper than TSLA 667.92)"
    inbox = env.instances["MessageAPI"].inbox
    want = {"USR003": "Done - bought 3 shares as you asked."}
    if want not in inbox:
        return False, "exact message to Catherine (USR003) not found"
    return True, "3 MSFT shares bought and Catherine messaged"


CHECKS = {
    "long_A_budget_recall": check_long_A_budget_recall,
    "long_B_colleague_request": check_long_B_colleague_request,
}


def run_check(task_id, env):
    fn = CHECKS.get(task_id)
    return fn(env) if fn else (None, "no functional check (BFCL state check only)")
