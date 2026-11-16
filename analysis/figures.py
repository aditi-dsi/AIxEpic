#!/usr/bin/env python3
"""Paper figures from runs/<run>/results.json (made by analysis/metrics.py).

    python3 analysis/figures.py --run main   # needs matplotlib

Saves PNG and PDF files in runs/<run>/figures/, sized for a 6.5-inch text width.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e6e5e1"
NULL = "#c3c2bd"
WIDTH = 6.5
ORDER = ["gpt-4.1-mini", "qwen3-235b", "llama-3.3-70b", "gemini-2.5-flash-lite", "gpt-oss-120b-low",
         "gpt-oss-120b-high", "claude-haiku-4.5"]
NAMES = {"gpt-4.1-mini": "GPT-4.1 mini", "qwen3-235b": "Qwen3-235B", "llama-3.3-70b": "Llama 3.3 70B",
         "gemini-2.5-flash-lite": "Gemini 2.5 Flash-Lite", "gpt-oss-120b-low": "gpt-oss-120b low effort",
         "gpt-oss-120b-high": "gpt-oss-120b high effort", "claude-haiku-4.5": "Claude Haiku 4.5"}
TICKS = {"gpt-4.1-mini": "GPT-4.1\nmini", "qwen3-235b": "Qwen3\n235B", "llama-3.3-70b": "Llama 3.3\n70B",
         "gemini-2.5-flash-lite": "Gemini 2.5\nFlash-Lite", "gpt-oss-120b-low": "gpt-oss\n120b low",
         "gpt-oss-120b-high": "gpt-oss\n120b high", "claude-haiku-4.5": "Claude\nHaiku 4.5"}

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 8.5, "axes.titlesize": 9, "figure.titlesize": 9.5,
    "legend.fontsize": 8, "text.color": INK, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "legend.frameon": False, "figure.facecolor": "white", "pdf.fonttype": 42,
})


def models(res, *keys):
    return [m for m in ORDER if m in res["models"]
            and all(isinstance(res["models"][m]["est"].get(k), (int, float)) for k in keys)]


def label(res, m):
    return NAMES[m] + "\n(40 questions)" if res["models"][m]["on_subset"] else NAMES[m]


def est(res, m, key, scale=100):
    v = res["models"][m]["est"][key] * scale
    ci = res["models"][m]["ci"].get(key) or [None, None]
    if not all(isinstance(x, (int, float)) for x in ci):
        return v, v, v
    return v, ci[0] * scale, ci[1] * scale


def hbar(ax, y, v, lo, hi, color, height, name=None):
    ax.barh(y, v, height, color=color, label=name, edgecolor="white", linewidth=1)
    ax.errorbar(v, y, xerr=[[max(0, v - lo)], [max(0, hi - v)]], fmt="none", ecolor=INK2, elinewidth=1, capsize=2)


def title(fig, text):
    fig.suptitle(text, x=0.01, ha="left")


def title_with_legend(fig, text, ax, ncol):
    fig.legend(*ax.get_legend_handles_labels(), loc="outside upper left", ncol=ncol,
               title=text, alignment="left", title_fontsize=9.5)


def save(fig, out, name):
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def pressure_ladder(res, out):
    ms = models(res, "D_A0", "D_A", "D_A2")
    series = [("D_A0", "Request only ($D_0$)", BLUE), ("D_A", "Request + personal bonus ($D_A$)", ORANGE),
              ("D_A2", "After one round of insistence ($D_2$)", AQUA)]
    fig, ax = plt.subplots(figsize=(WIDTH, 2.6), layout="constrained")
    w = 0.27
    for i, (key, name, color) in enumerate(series):
        vals = [est(res, m, key) for m in ms]
        xs = [j + (i - 1) * w for j in range(len(ms))]
        ys = [v for v, _, _ in vals]
        ax.bar(xs, ys, w, color=color, label=name, edgecolor="white", linewidth=1)
        ax.errorbar(xs, ys, yerr=[[max(0, v - lo) for v, lo, _ in vals], [max(0, hi - v) for v, _, hi in vals]],
                    fmt="none", ecolor=INK2, elinewidth=1, capsize=1.5)
    ax.axhline(0, color=MUTED, linewidth=1)
    ax.set_xticks(range(len(ms)), [TICKS[m] for m in ms])
    ax.grid(axis="x", visible=False)
    subset = [j for j, m in enumerate(ms) if res["models"][m]["on_subset"]]
    if subset:
        ax.axvline(subset[0] - 0.5, color=MUTED, linewidth=0.8, linestyle=(0, (3, 3)))
        ax.text(subset[0] - 0.42, ax.get_ylim()[1], "40-question subset", va="top", fontsize=8, color=INK2,
                backgroundcolor="white")
    ax.set_ylabel("Points toward the CFO\n(below 0 = away)")
    title_with_legend(fig, "How far the recommendation moves toward what the CFO asks for", ax, 3)
    save(fig, out, "fig1_pressure_ladder")


def signed_bars(res, out, key, scale, fmt, unit, less, more, text, name):
    ms = sorted(models(res, key), key=lambda m: res["models"][m]["est"][key])
    fig, ax = plt.subplots(figsize=(WIDTH, 0.36 * len(ms) + 1.3), layout="constrained")
    kinds = {"95% CI above 0": ORANGE, "95% CI includes 0": NULL, "95% CI below 0": BLUE}
    used = set()
    for j, m in enumerate(ms):
        v, lo, hi = est(res, m, key, scale)
        meaning = "95% CI above 0" if lo > 0 else "95% CI below 0" if hi < 0 else "95% CI includes 0"
        used.add(meaning)
        hbar(ax, j, v, lo, hi, kinds[meaning], 0.62)
        if v == 0:
            ax.plot(0, j, "o", color=INK2, markersize=3)
        right = v >= 0
        ax.annotate(fmt(v), (hi if right else lo, j), xytext=(4 if right else -4, 0), textcoords="offset points",
                    ha="left" if right else "right", va="center", fontsize=8)
    ax.axvline(0, color=MUTED, linewidth=1)
    ax.margins(x=0.2)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: fmt(x, axis=True)))
    ax.set_yticks(range(len(ms)), [label(res, m) for m in ms])
    ax.grid(axis="y", visible=False)
    for word, dx, ha in ((f"← {less}", -6, "right"), (f"{more} →", 6, "left")):
        ax.annotate(word, (0, 0), xycoords=("data", "axes fraction"), xytext=(dx, -21), textcoords="offset points",
                    ha=ha, va="top", fontsize=8, color=INK)
    ax.set_xlabel(unit, labelpad=17)
    shown = [k for k in kinds if k in used]
    ax.legend([Patch(color=kinds[k]) for k in shown], shown, loc="lower right")
    title(fig, text)
    save(fig, out, name)


def points(v, axis=False):
    if axis:
        return f"{v:.0f}".replace("-", "−")
    return "0.0" if abs(v) < 0.05 else f"{v:+.1f}".replace("-", "−")


def dollars(v, axis=False):
    sign = "−" if v < 0 else "" if axis or v == 0 else "+"
    return sign + (f"${abs(v) / 1000:.0f}k" if axis and v else f"${abs(v):,.0f}")


def against_own_belief(res, out):
    signed_bars(res, out, "AOB_excess", 100, points, "Percentage points, relative to the neutral re-ask (N2)",
                "less often than after N2", "more often than after N2",
                "After insistence: following the CFO against the model's own final P",
                "fig2_against_own_belief")


def cost(res, out):
    signed_bars(res, out, "cost2_usd_per_decision", 1, dollars,
                r"Dollars per decision, priced with the model's prior belief P$_0$ (\$100,000 at stake)",
                "cheaper than after N2", "costlier than after N2",
                "After insistence: extra expected cost per decision, relative to the neutral re-ask (N2)",
                "fig5_cost_after_insistence")


def belief_vs_threshold(res, out):
    ms = [m for m in models(res, "D_bel", "D_thr", "D_bel0", "D_thr0") if res["models"][m]["est"].get("BA_N", 0) >= 0.8]
    if not ms:
        return
    series = [("D_bel", "Belief shift: the stated P moved", BLUE),
              ("D_thr", "Threshold shift: P stayed, the recommendation moved", ORANGE)]
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 0.4 * len(ms) + 1.25), sharex=True, sharey=True, layout="constrained")
    for ax, (tag, panel) in zip(axes, (("0", "Request only"), ("", "Request + personal bonus"))):
        for i, (key, name, color) in enumerate(series):
            for j, m in enumerate(ms):
                hbar(ax, j + (i - 0.5) * 0.36, *est(res, m, key + tag), color, 0.36, name if j == 0 else None)
        ax.axvline(0, color=MUTED, linewidth=1)
        ax.grid(axis="y", visible=False)
        ax.set_title(panel, loc="left")
    axes[0].set_yticks(range(len(ms)), [label(res, m) for m in ms])
    axes[0].invert_yaxis()
    fig.supxlabel("Points of buy rate (positive = toward the CFO, negative = away)", fontsize=8.5, color=INK2)
    title_with_legend(fig, "Belief shift vs. threshold shift (models with ≥ 80% neutral agreement)", axes[0], 2)
    save(fig, out, "fig3_belief_vs_threshold")


def switches(res, out):
    rate = {m: res["models"][m]["est"]["switches_to_cfo"] / res["models"][m]["n_questions"]
            for m in models(res, "share_belief_rewritten")}
    ms = sorted(rate, key=rate.get)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(WIDTH, 0.28 * len(ms) + 1.3), sharey=True,
                                   width_ratios=[1, 1.5], layout="constrained")
    parts = [("Only the recommendation switched, P still disagrees", ORANGE), ("P also moved to match", BLUE)]
    for j, m in enumerate(ms):
        e = res["models"][m]["est"]
        ax1.barh(j, rate[m], 0.62, color=MUTED)
        ax1.annotate(f"n = {e['switches_to_cfo']}", (rate[m], j), xytext=(4, 0), textcoords="offset points",
                     va="center", fontsize=8)
        rewritten = 100 * e["share_belief_rewritten"]
        left = 0
        for (name, color), width in zip(parts, (100 - rewritten, rewritten)):
            ax2.barh(j, width, 0.62, left=left, color=color, edgecolor="white", linewidth=1, label=name if j == 0 else None)
            if width >= 9:
                ax2.text(left + width / 2, j, f"{width:.0f}%", ha="center", va="center", fontsize=8, color=INK)
            left += width
    ax1.set_xlim(0, 1.6 * max(rate.values()))
    ax1.set_yticks(range(len(ms)), [label(res, m) for m in ms])
    ax1.set_xlabel("Switches per question\n(out of 10 pressured re-asks)")
    ax1.set_title("How often", loc="left")
    ax2.set_xlim(0, 100)
    ax2.set_xlabel("Share of those switches (%)")
    ax2.set_title("How", loc="left")
    for ax in (ax1, ax2):
        ax.grid(axis="y", visible=False)
    title_with_legend(fig, "After insistence: answers that switched to the CFO's side", ax2, 2)
    save(fig, out, "fig4_switches_after_insistence")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="main")
    a = ap.parse_args()
    run_dir = ROOT / "runs" / a.run
    res = json.loads((run_dir / "results.json").read_text())
    out = run_dir / "figures"
    out.mkdir(exist_ok=True)
    for make in (pressure_ladder, against_own_belief, belief_vs_threshold, switches, cost):
        make(res, out)
    print(f"Saved figures to {out}")


if __name__ == "__main__":
    main()
