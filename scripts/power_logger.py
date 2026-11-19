#!/usr/bin/env python3
"""
power_logger.py - Logs Raspberry Pi 5 power, temperature and throttle state.

Every INTERVAL seconds it reads:
  - vcgencmd pmic_read_adc  -> voltage (V) and current (A) for each power rail
  - vcgencmd measure_temp   -> SoC temperature (C)
  - vcgencmd get_throttled  -> throttle flags (0x0 = all good)

Power per sample = sum over rails of V * I (the method the project spec asks for).
Energy = power integrated over time (trapezoid rule).

Usage (on the Pi):
  python3 scripts/power_logger.py --out results/power_idle.csv
  python3 scripts/power_logger.py --out results/power_run.csv --interval 0.5 --duration 60
Stop any time with Ctrl+C; a summary is printed at the end.
"""

import argparse
import csv
import re
import subprocess
import sys
import time
from datetime import datetime

RAIL_RE = re.compile(r"^\s*(\S+)_([AV])\s+\w+\(\d+\)=([0-9.]+)[AV]\s*$")


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout


def read_power():
    """Return (total_watts, number_of_rails_used)."""
    volts, amps = {}, {}
    for line in run(["vcgencmd", "pmic_read_adc"]).splitlines():
        m = RAIL_RE.match(line)
        if not m:
            continue
        rail, kind, value = m.group(1), m.group(2), float(m.group(3))
        (amps if kind == "A" else volts)[rail] = value
    rails = [r for r in amps if r in volts]
    total = sum(volts[r] * amps[r] for r in rails)
    return total, len(rails)


def read_temp():
    out = run(["vcgencmd", "measure_temp"])  # e.g. temp=49.4'C
    return float(re.search(r"([0-9.]+)", out).group(1))


def read_throttled():
    out = run(["vcgencmd", "get_throttled"])  # e.g. throttled=0x0
    return out.strip().split("=")[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="CSV file to write")
    ap.add_argument("--interval", type=float, default=0.5, help="seconds between samples")
    ap.add_argument("--duration", type=float, default=0, help="seconds to run (0 = until Ctrl+C)")
    args = ap.parse_args()

    start = time.monotonic()
    samples = []  # (t, watts)
    max_temp = 0.0
    throttle_seen = set()

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["wall_time", "t_s", "power_w", "rails", "temp_c", "throttled"])
        print(f"Logging to {args.out} every {args.interval}s ... (Ctrl+C to stop)")
        try:
            while True:
                t = time.monotonic() - start
                watts, n = read_power()
                temp = read_temp()
                thr = read_throttled()
                w.writerow([datetime.now().isoformat(timespec="milliseconds"),
                            f"{t:.3f}", f"{watts:.4f}", n, f"{temp:.1f}", thr])
                f.flush()
                samples.append((t, watts))
                max_temp = max(max_temp, temp)
                throttle_seen.add(thr)
                if args.duration and t >= args.duration:
                    break
                time.sleep(max(0.0, args.interval - ((time.monotonic() - start) - t)))
        except KeyboardInterrupt:
            pass

    if len(samples) < 2:
        print("Not enough samples for a summary.")
        return
    energy = sum((t2 - t1) * (p1 + p2) / 2 for (t1, p1), (t2, p2) in zip(samples, samples[1:]))
    dur = samples[-1][0] - samples[0][0]
    print("\n--- Summary ---")
    print(f"Samples        : {len(samples)}")
    print(f"Duration       : {dur:.1f} s")
    print(f"Average power  : {energy / dur:.3f} W")
    print(f"Energy         : {energy:.1f} J")
    print(f"Max temp       : {max_temp:.1f} C")
    print(f"Throttle flags : {', '.join(sorted(throttle_seen))}")
    if any(x not in ("0x0",) for x in throttle_seen):
        print("WARNING: throttling flags seen - treat this run as suspect.")


if __name__ == "__main__":
    sys.exit(main())
