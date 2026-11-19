#!/usr/bin/env python3
"""
analyze_baseline.py - Turns the benchmark CSVs into the slide figures.

Inputs (from run_baseline_bench.py): results/<name>_summary.csv for each name
given with --runs (default: bf16 f16).

Outputs (results/figures/):
  decode_roofline.png    measured decode speed vs context depth, with the
                         bandwidth-bound ceiling  tok/s <= BW / (weights + KV(depth))
  prefill_vs_length.png  prefill speed vs prompt length
  energy_per_token.png   joules per token (prefill and decode)
  baseline_table.md      all numbers in one markdown table

Model constants for Qwen3-1.7B (from its config.json): 28 layers, 8 KV heads,
head_dim 128; GGUF weights 4.06 GB in 16-bit. KV cache in llama.cpp's default
F16: 2 (K and V) x 28 x 8 x 128 x 2 bytes = 114,688 bytes per token.

Run on the laptop after copying the CSVs back:
  pip install matplotlib
  python scripts/analyze_baseline.py
"""

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LAYERS, KV_HEADS, HEAD_DIM, BYTES_PER_VALUE = 28, 8, 128, 2
KV_BYTES_PER_TOKEN = 2 * LAYERS * KV_HEADS * HEAD_DIM * BYTES_PER_VALUE  # 114,688
WEIGHT_BYTES = 4.06e9

BANDWIDTHS = {  # GB/s, all measured on our Pi 5 except the theoretical peak
    "LPDDR4X peak (theoretical)": 17.07,
    "pure-read test (bw_read)": 11.80,
    "STREAM Triad": 8.13,
}
COLORS = {"bf16": "#c0392b", "f16": "#2471a3"}


def load(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("tokens", "depth"):
            r[k] = int(float(r[k]))
        for k in ("tok_per_s", "avg_power_w", "joule_per_token", "max_temp_c"):
            r[k] = float(r[k])
        r["suspect"] = r["suspect"].strip().lower() == "true"
    return rows


def ceiling(bw_gbs, depth):
    return bw_gbs * 1e9 / (WEIGHT_BYTES + depth * KV_BYTES_PER_TOKEN)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="bf16,f16")
    ap.add_argument("--resdir", default="results")
    args = ap.parse_args()
    out = os.path.join(args.resdir, "figures")
    os.makedirs(out, exist_ok=True)

    data = {}
    for name in args.runs.split(","):
        p = os.path.join(args.resdir, f"{name}_summary.csv")
        if os.path.exists(p):
            data[name] = load(p)
        else:
            print(f"(skipping {name}: {p} not found)")
    if not data:
        raise SystemExit("no summary CSVs found")

    # 1. Decode roofline
    max_depth = max([r["depth"] for rows in data.values() for r in rows] + [16384])
    xs = list(range(0, max_depth + 1, max(1, max_depth // 200)))
    fig, ax = plt.subplots(figsize=(8, 5))
    for (label, bw), ls in zip(BANDWIDTHS.items(), ["-", "--", ":"]):
        ax.plot(xs, [ceiling(bw, x) for x in xs], ls, color="gray", lw=1.2,
                label=f"ceiling @ {bw} GB/s ({label})")
    for name, rows in data.items():
        d = sorted((r for r in rows if r["test"] == "decode"), key=lambda r: r["depth"])
        if not d:
            continue
        ax.plot([r["depth"] for r in d], [r["tok_per_s"] for r in d], "o-",
                color=COLORS.get(name), lw=2, ms=7, label=f"measured {name.upper()}")
        for r in d:
            if r["suspect"]:
                ax.annotate("throttled", (r["depth"], r["tok_per_s"]), fontsize=8,
                            xytext=(4, 6), textcoords="offset points")
    ax.set_xlabel("context depth (tokens already in the KV cache)")
    ax.set_ylabel("decode speed (tokens/s)")
    ax.set_title("Qwen3-1.7B decode on Raspberry Pi 5: measured vs bandwidth ceiling")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "decode_roofline.png"), dpi=160)

    # 2. Prefill vs prompt length
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, rows in data.items():
        p = sorted((r for r in rows if r["test"] == "prefill"), key=lambda r: r["tokens"])
        if p:
            ax.plot([r["tokens"] for r in p], [r["tok_per_s"] for r in p], "o-",
                    color=COLORS.get(name), lw=2, ms=7, label=name.upper())
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("prompt length (tokens)")
    ax.set_ylabel("prefill speed (tokens/s, log scale)")
    ax.set_title("Prefill speed vs prompt length")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out, "prefill_vs_length.png"), dpi=160)

    # 3. Energy per token
    fig, ax = plt.subplots(figsize=(8, 5))
    labels, vals, cols = [], [], []
    for name, rows in data.items():
        for r in sorted(rows, key=lambda r: (r["test"], r["tokens"], r["depth"])):
            tag = f"pp{r['tokens']}" if r["test"] == "prefill" else f"tg@{r['depth']}"
            labels.append(f"{name.upper()}\n{tag}")
            vals.append(r["joule_per_token"])
            cols.append(COLORS.get(name, "gray"))
    ax.bar(range(len(vals)), vals, color=cols)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, fontsize=7, rotation=90)
    ax.set_yscale("log")
    ax.set_ylabel("energy per token (J, log scale)")
    ax.set_title("Energy per token (PMIC rails, measured window only)")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "energy_per_token.png"), dpi=160)

    # 4. Table
    lines = ["| run | test | tokens | depth | tok/s | bytes read GB/s | % of 17.07 GB/s peak | "
             "power W | J/token | max C | throttled |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, rows in data.items():
        for r in rows:
            if r["test"] == "decode":
                bw = r["tok_per_s"] * (WEIGHT_BYTES + r["depth"] * KV_BYTES_PER_TOKEN) / 1e9
                cs, pct = f"{bw:.2f}", f"{100 * bw / 17.07:.0f}%"
            else:
                cs, pct = "-", "-"
            lines.append(f"| {name.upper()} | {r['test']} | {r['tokens']} | {r['depth']} | "
                         f"{r['tok_per_s']:.2f} | {cs} | {pct} | {r['avg_power_w']:.2f} | "
                         f"{r['joule_per_token']:.3f} | {r['max_temp_c']:.1f} | "
                         f"{'YES' if r['suspect'] else 'no'} |")
    lines += ["", f"KV cache per token: {KV_BYTES_PER_TOKEN:,} bytes; "
              f"at 16,384 tokens: {16384 * KV_BYTES_PER_TOKEN / 1e9:.2f} GB; weights: "
              f"{WEIGHT_BYTES / 1e9:.2f} GB"]
    with open(os.path.join(out, "baseline_table.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nfigures written to {out}/")


if __name__ == "__main__":
    main()
