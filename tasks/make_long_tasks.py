#!/usr/bin/env python3
"""
make_long_tasks.py - Writes our own long-session tasks to tasks/long_tasks.json.

Why: even in BFCL's long-context mode most sessions are shorter than the project
targets, so the spec asks for at least 5 extra tasks over the same backends that
reach 16k-32k tokens through many calls and verbose outputs, each with a
functional success check, and at least two needing a detail from early in the
session. This file holds three: A and B are early-detail tasks, C is a long session
whose final instruction is self-contained (tests long-context state tracking).

Design note: in long-context mode get_stock_info returns ~31.7k tokens and
get_transaction_history ~49.3k tokens (exact Qwen3 tokenizer counts), so a
single such call overflows the baseline's 16k cache. These tasks therefore give
price quotes in the first turn and grow the context with the 1.6k-4k-token
outputs instead (watchlist, sector lists, order details), landing in the
16k-32k range the spec asks for.

Each task uses BFCL's format (involved_classes, initial_config, question turns)
plus our ground-truth calls per turn, so harness/run_session.py runs them like
any BFCL task. The functional check for each task is in tasks/checks.py.

  python3 tasks/make_long_tasks.py      (rewrites tasks/long_tasks.json)
"""

import json
import os


def user(text):
    return [{"role": "user", "content": text}]


TASKS = [
    {
        "id": "long_A_budget_recall",
        "involved_classes": ["TradingBot"],
        "initial_config": {},
        "split": "dev",
        "early_detail": "Budget ($1,500), favourite company (Nvidia) and its price quote ($220.34) "
                        "are only stated in turn 0; the final turn needs all three.",
        "question": [
            user("Hi. Three things to keep in mind for later today: my budget for a single "
                 "purchase is $1,500, the company I'm most interested in is Nvidia, and my broker "
                 "quoted it at $220.34 this morning, so use that quote and don't look the price "
                 "up. For now, please log me into my trading account (username 'farrukh_s', "
                 "password 'edge624') and tell me the current time."),
            user("Show me my watchlist."),
            user("Give me the full list of available stocks in the Technology sector."),
            user("Now the full list of available stocks in the Automobile sector."),
            user("What are the full details of my order 12345?"),
            user("And the full details of order 12446?"),
            user("Add Tesla (TSLA) to my watchlist, then show me the updated watchlist."),
            user("Now buy as many whole shares as my budget allows of the company I said I was "
                 "most interested in, at the price I gave you."),
        ],
        "ground_truth": [
            ["trading_login(username='farrukh_s', password='edge624')", "get_current_time()"],
            ["get_watchlist()"],
            ["get_available_stocks(sector='Technology')"],
            ["get_available_stocks(sector='Automobile')"],
            ["get_order_details(order_id=12345)"],
            ["get_order_details(order_id=12446)"],
            ["add_to_watchlist(stock='TSLA')", "get_watchlist()"],
            ["place_order(order_type='Buy', symbol='NVDA', price=220.34, amount=6)"],
        ],
    },
    {
        "id": "long_B_colleague_request",
        "involved_classes": ["TradingBot", "MessageAPI"],
        "initial_config": {},
        "split": "dev",
        "early_detail": "Who asked (Catherine), what to buy (3 shares of the cheaper of TSLA/MSFT), "
                        "the two price quotes, the user's own message ID (USR001) and the exact "
                        "reply text are only stated in turn 0; the final turn just says 'do what "
                        "Catherine asked'.",
        "question": [
            user("Hi, I'm Alice and my messaging user ID is USR001. This morning Catherine asked me "
                 "to buy 3 shares of whichever is cheaper, Tesla (TSLA) or Microsoft (MSFT); my "
                 "broker's quotes were TSLA $667.92 and MSFT $310.23, so use those and don't look "
                 "prices up. Afterwards I must message her exactly this text: 'Done - bought 3 "
                 "shares as you asked.' I'll ask you to do all that at the end. First, log me into "
                 "my trading account with username 'alice_w' and password 'edge624'."),
            user("Show me all available stocks in the Automobile sector."),
            user("And all available stocks in the Technology sector."),
            user("Show me my watchlist."),
            user("Get the full details of order 12446."),
            user("And the full details of order 12345."),
            user("Show my watchlist once more, I lost track."),
            user("Now please do what Catherine asked this morning."),
        ],
        "ground_truth": [
            ["trading_login(username='alice_w', password='edge624')"],
            ["get_available_stocks(sector='Automobile')"],
            ["get_available_stocks(sector='Technology')"],
            ["get_watchlist()"],
            ["get_order_details(order_id=12446)"],
            ["get_order_details(order_id=12345)"],
            ["get_watchlist()"],
            ["place_order(order_type='Buy', symbol='MSFT', price=310.23, amount=3)",
             "message_login(user_id='USR001')",
             "get_user_id(user='Catherine')",
             "send_message(receiver_id='USR003', message='Done - bought 3 shares as you asked.')"],
        ],
    },
    {
        "id": "long_C_watchlist_update",
        "involved_classes": ["TradingBot", "MessageAPI"],
        "initial_config": {},
        "split": "dev",
        "early_detail": "None (control task): every instruction is complete in its own turn; the "
                        "difficulty is staying correct after ~20k tokens of verbose outputs.",
        "question": [
            user("Please log me into my trading account: username 'daniel_k', password 'edge624'."),
            user("Show me all available stocks in the Technology sector."),
            user("And all available stocks in the Automobile sector."),
            user("Show me my current watchlist."),
            user("What are the full details of my order 12345?"),
            user("Remove NVDA from my watchlist, then add AAPL and MSFT to it."),
            user("Show me the updated watchlist."),
            user("Finally, log me into messaging with user ID USR004 and send Bob (user ID USR002) "
                 "exactly this text: 'Watchlist updated: AAPL and MSFT added, NVDA removed.'"),
        ],
        "ground_truth": [
            ["trading_login(username='daniel_k', password='edge624')"],
            ["get_available_stocks(sector='Technology')"],
            ["get_available_stocks(sector='Automobile')"],
            ["get_watchlist()"],
            ["get_order_details(order_id=12345)"],
            ["remove_stock_from_watchlist(symbol='NVDA')",
             "add_to_watchlist(stock='AAPL')",
             "add_to_watchlist(stock='MSFT')"],
            ["get_watchlist()"],
            ["message_login(user_id='USR004')",
             "send_message(receiver_id='USR002', message='Watchlist updated: AAPL and MSFT "
             "added, NVDA removed.')"],
        ],
    },
]

if __name__ == "__main__":
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "long_tasks.json")
    for t in TASKS:
        assert len(t["question"]) == len(t["ground_truth"]), t["id"]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(TASKS, f, indent=2)
    print(f"wrote {len(TASKS)} tasks to {path}")
