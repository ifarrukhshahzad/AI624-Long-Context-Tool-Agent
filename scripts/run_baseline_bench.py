#!/usr/bin/env python3
"""
run_baseline_bench.py - Baseline throughput + energy sweep on the Pi.

What it measures (project spec, Task 3 "baseline characterization"):
  * Prefill throughput at several prompt lengths   (llama-bench -p N -n 0)
  * Decode  throughput at several context depths   (llama-bench -p 0 -n G -d D)
For every single test it also:
  * waits until the Pi has cooled below --cooldown C (fair start for each test)
  * runs scripts/power_logger.py alongside (power, temperature, throttle flags)
  * computes average power and energy per token over the MEASURED part only
    (the last reps * tokens / speed seconds of the log, so model loading and
    filling the context are excluded)
  * flags the test as SUSPECT if any throttle flag other than 0x0 appeared

Outputs in --outdir (default results/):
  <name>_summary.csv      one row per test (the numbers for the slides)
  <name>_meta.json        llama.cpp commit, model file, settings, timestamps
  <name>_<test>.json      raw llama-bench output per test
  <name>_<test>_power.csv raw power log per test

Example (BF16 baseline, overnight):
  python3 scripts/run_baseline_bench.py --model ~/models/qwen3-1.7b-bf16.gguf \
      --name bf16 --prompts 512,1024,2048,4096 --depths 0,1024,2048,4096
Quick check without running anything:
  python3 scripts/run_baseline_bench.py --model X --name test --dry-run
"""

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
LLAMA_DIR = os.path.expanduser("~/llama.cpp")
BENCH = os.path.join(LLAMA_DIR, "build", "bin", "llama-bench")


def sh(cmd):
    return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()


def temp_c():
    out = sh(["vcgencmd", "measure_temp"])  # temp=49.4'C
    try:
        return float(out.split("=")[1].split("'")[0])
    except (IndexError, ValueError):
        return float("nan")


def wait_cool(limit, max_wait=900):
    t0 = time.time()
    while True:
        t = temp_c()
        if not t > limit:  # also exits on nan
            return t
        if time.time() - t0 > max_wait:
            print(f"  ! still {t:.1f} C after {max_wait}s, starting anyway")
            return t
        print(f"  cooling: {t:.1f} C > {limit} C, waiting...", flush=True)
        time.sleep(10)


def read_power(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append((float(r["t_s"]), float(r["power_w"]),
                         float(r["temp_c"]), r["throttled"]))
    return rows


def window_stats(rows, seconds):
    """Average power over the last `seconds` of the log (the measured phase)."""
    if not rows:
        return float("nan"), float("nan"), set()
    t_end = rows[-1][0]
    win = [r for r in rows if r[0] >= t_end - seconds] or rows[-2:]
    avg_p = sum(r[1] for r in win) / len(win)
    max_t = max(r[2] for r in rows)
    flags = {r[3] for r in rows}
    return avg_p, max_t, flags


def run_test(args, label, bench_args):
    tag = f"{args.name}_{label}"
    json_path = os.path.join(args.outdir, f"{tag}.json")
    power_path = os.path.join(args.outdir, f"{tag}_power.csv")
    cmd = [BENCH, "-m", args.model, "-t", str(args.threads),
           "-r", str(args.reps), "-o", "json"] + bench_args
    print(f"\n[{label}] {' '.join(cmd)}", flush=True)
    if args.dry_run:
        return None

    start_temp = wait_cool(args.cooldown)
    logger = subprocess.Popen([sys.executable, os.path.join(HERE, "power_logger.py"),
                               "--out", power_path, "--interval", "0.5"],
                              stdout=subprocess.DEVNULL)
    time.sleep(1.0)
    t0 = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.time() - t0
    time.sleep(1.0)
    logger.send_signal(signal.SIGINT)
    logger.wait(timeout=30)

    if res.returncode != 0:
        print("  ! llama-bench failed:\n" + res.stderr[-2000:])
        return None
    with open(json_path, "w") as f:
        f.write(res.stdout)
    data = json.loads(res.stdout)[0]

    n_tok = data.get("n_gen") or data.get("n_prompt")
    ts = data["avg_ts"]
    measured_s = args.reps * n_tok / ts
    avg_p, max_t, flags = window_stats(read_power(power_path), measured_s)
    suspect = any(f != "0x0" for f in flags)
    row = {
        "test": "decode" if data.get("n_gen") else "prefill",
        "tokens": n_tok,
        "depth": data.get("n_depth", 0),
        "tok_per_s": round(ts, 3),
        "stddev": round(data.get("stddev_ts", 0.0), 3),
        "avg_power_w": round(avg_p, 3),
        "joule_per_token": round(avg_p / ts, 3),
        "start_temp_c": start_temp,
        "max_temp_c": max_t,
        "throttle_flags": " ".join(sorted(flags)),
        "suspect": suspect,
        "wall_s": round(wall, 1),
    }
    print(f"  -> {row['tok_per_s']} tok/s, {row['avg_power_w']} W, "
          f"{row['joule_per_token']} J/token, max {max_t:.1f} C"
          + ("  SUSPECT (throttled)" if suspect else ""), flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True, help="short label, e.g. bf16")
    ap.add_argument("--prompts", default="512,1024,2048,4096",
                    help="prefill prompt lengths (comma list)")
    ap.add_argument("--depths", default="0,1024,2048,4096",
                    help="decode context depths (comma list)")
    ap.add_argument("--gen", type=int, default=32, help="tokens generated per decode test")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--cooldown", type=float, default=50.0, help="start each test below this C")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    args.model = os.path.expanduser(args.model)
    os.makedirs(args.outdir, exist_ok=True)

    meta = {
        "name": args.name,
        "model": args.model,
        "model_bytes": os.path.getsize(args.model) if os.path.exists(args.model) else None,
        "llama_cpp_commit": sh(["git", "-C", LLAMA_DIR, "rev-parse", "--short", "HEAD"]),
        "settings": vars(args),
        "started": datetime.now().isoformat(timespec="seconds"),
    }

    jobs = [(f"pp{p}", ["-p", p, "-n", "0"]) for p in args.prompts.split(",") if p]
    jobs += [(f"tg{args.gen}_d{d}", ["-p", "0", "-n", str(args.gen), "-d", d])
             for d in args.depths.split(",") if d]

    rows = []
    for label, bench_args in jobs:
        row = run_test(args, label, bench_args)
        if row:
            rows.append(row)
            with open(os.path.join(args.outdir, f"{args.name}_summary.csv"), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(row.keys()))
                w.writeheader()
                w.writerows(rows)

    meta["finished"] = datetime.now().isoformat(timespec="seconds")
    with open(os.path.join(args.outdir, f"{args.name}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nDone. {len(rows)} tests -> {args.outdir}/{args.name}_summary.csv")


if __name__ == "__main__":
    main()
