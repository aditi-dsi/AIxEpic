#!/usr/bin/env python3
"""Extra numbers quoted in the paper (not in results.md): the cost priced with real outcomes (E6), the parser
check (E10), second-turn breakdowns, and the example replies in Appendix C.

    python3 analysis/secondary.py --run main

Reads the logs only (no API calls). Writes runs/<run>/secondary.md.
"""
import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import analysis.metrics as M  # noqa: E402
from analysis.gate1 import load_run, sufficient_stats  # noqa: E402
from run.parse import extract_reasoning, parse_belief, parse_decision  # noqa: E402
from run.runner import CONFIGS, read_jsonl  # noqa: E402

MODELS = ["gpt-4.1-mini", "qwen3-235b", "llama-3.3-70b", "gemini-2.5-flash-lite", "gpt-oss-120b-low",
          "gpt-oss-120b-high", "claude-haiku-4.5"]
EXAMPLES = [("gpt-4.1-mini", "fb-73a5b33a1cf6", 0.3, "IB"), ("gpt-oss-120b-low", "fb-9480b3185e12", 0.1, "IN"),
            ("claude-haiku-4.5", "fb-c07e375ce9ab", 0.1, "IB")]
E10_KEYS = ["D_A", "D_A0", "D_A2", "AOB_excess", "D_bel", "D_thr", "D_bel0", "D_thr0", "BA_N", "LDS_raw", "LDS0_raw"]


def cost_vs_n2(data, ref, base_q, boot):
    """Extra expected cost per decision after insistence relative to N2, priced with ref, point and 95% CI."""
    per_q = {q: sufficient_stats(data[q]["dec"], data[q]["t2"], {q: ref[q]})[q] for q in data if q in ref}

    def est(qs):
        qs = [q for q in qs if q in per_q]
        return sum(sum(per_q[q][("cost2", c)][0] for q in qs) / sum(per_q[q][("cost2", c)][1] for q in qs)
                   for c in ("IB", "IN")) / 2
    rng = random.Random(M.SEED)
    return est(base_q), M.ci([est([rng.choice(base_q) for _ in base_q]) for _ in range(boot)])


def old_rule(parse):
    """The earlier parser rule: a probability strictly between 0 and 1 is read as a fraction."""
    def wrapped(content):
        p = parse(content)
        return dict(p, probability=100 * p["probability"]) if p["valid"] and p["below_one"] else p
    return wrapped


def e10(rows, data):
    n_valid = n_below = 0
    for r in rows.values():
        p = (parse_belief if r["kind"] == "belief" else parse_decision)(r.get("content"))
        n_valid += p["valid"]
        n_below += p["valid"] and p["below_one"]
    M.parse_belief, M.parse_decision = old_rule(parse_belief), old_rule(parse_decision)
    old, _ = M.load_model(rows)
    M.parse_belief, M.parse_decision = parse_belief, parse_decision
    new_e, old_e = M.estimate(data, list(data)), M.estimate(old, list(old))
    changes = {k: 100 * abs(old_e[k] - new_e[k]) for k in E10_KEYS
               if isinstance(old_e.get(k), float) and isinstance(new_e.get(k), float)}
    key = max(changes, key=changes.get)
    return n_below, n_valid, key, changes[key]


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def reasoning(content):
    text = (content or "").strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    return extract_reasoning(text)


def describe(r):
    p = parse_decision(r["content"])
    return f"P = {p['probability']:g}%, {'buy' if p['recommendation'] else 'do not buy'}. \"{reasoning(r['content'])}\""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="main")
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    run_dir = ROOT / "runs" / a.run
    ok, _ = load_run(run_dir)
    questions = {q["qid"]: q["question"] for q in read_jsonl(ROOT / "data/questions/main.jsonl")}
    outcome = {h["qid"]: h["outcome"] for h in read_jsonl(ROOT / "data/questions/hidden.jsonl")}
    main_q = list(questions)
    subset_q = [q for q in json.loads((ROOT / "data/questions/disclaimer_subset.json").read_text()) if q in questions]

    L = [f"# Secondary numbers: run `{a.run}`", "",
         "Cost columns: extra expected cost per decision after insistence (IB2/IN2) relative to the neutral re-ask (N2), "
         f"$100,000 at stake, 95% cluster-bootstrap CIs (B = {a.boot}). E6 prices it with the real outcome. "
         "E10 rereads probabilities strictly between 0 and 1 as fractions (the earlier parser rule).", "",
         "| Model | Questions | Cost, priced with P0 | Cost, priced with outcome (E6) | Changed after N2 | Changed after IB2 | "
         "Changed after IN2 | Buy rate IB2 / IN2 | Turn-1 IB: declined although N bought and P > t | "
         "E10: replies with 0 < P < 1 | E10: largest change (points) |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    by_turn = ["", "## Second turn by condition", "",
               "Expected loss: $100,000 x |t - P0| for each answer that differs from the best decision under the "
               "model's prior belief P0 (buy if and only if P0 > t), averaged over answers.", "",
               "| Model | Turn 2 | Answers | Buy rate | Buy while final P <= t | No buy while final P > t | "
               "Expected loss per decision |", "|---|---|---|---|---|---|---|"]
    for m in MODELS:
        if m not in ok:
            continue
        data, _ = M.load_model(ok[m])
        base_q = main_q if CONFIGS[m]["pooled"] else subset_q
        own = {q: data[q]["own"] for q in data}
        c_own, ci_own = cost_vs_n2(data, own, base_q, a.boot)
        c_out, ci_out = cost_vs_n2(data, outcome, base_q, a.boot)
        t2 = [d for q in data for d in data[q]["t2"]]
        changed = {c: mean([d["a1"] != d["a2"] for d in t2 if d["c"] == c]) for c in ("N", "IB", "IN")}
        buy2 = {c: mean([d["a2"] for d in t2 if d["c"] == c]) for c in ("IB", "IN")}
        for c in ("N", "IB", "IN"):
            cells = [d for d in t2 if d["c"] == c]
            loss_best = mean([100_000 * abs(d["t"] - own[d["qid"]]) * ((d["a2"] == 1) != (own[d["qid"]] > d["t"]))
                              for d in cells])
            by_turn.append(f"| {m} | {c}2 | {len(cells)} | {mean([d['a2'] for d in cells]):.1%} | "
                           f"{mean([d['a2'] == 1 and d['p2_pct'] <= 100 * d['t'] for d in cells]):.1%} | "
                           f"{mean([d['a2'] == 0 and d['p2_pct'] > 100 * d['t'] for d in cells]):.1%} | "
                           f"${loss_best:,.0f} |")
        dec =[d for q in data for d in data[q]["dec"]]
        n_buy = {(d["qid"], d["t"]) for d in dec if d["c"] == "N" and d["a"] == 1}
        n_ib = sum(d["c"] == "IB" for d in dec)
        away = sum(d["c"] == "IB" and d["a"] == 0 and d["p"] > d["t"] and (d["qid"], d["t"]) in n_buy for d in dec)
        n_below, n_valid, key, change = e10(ok[m], data)
        L.append(f"| {m} | {len(base_q)} | ${c_own:,.0f} [${ci_own[0]:,.0f}, ${ci_own[1]:,.0f}] | "
                 f"${c_out:,.0f} [${ci_out[0]:,.0f}, ${ci_out[1]:,.0f}] | {changed['N']:.1%} | {changed['IB']:.1%} | "
                 f"{changed['IN']:.1%} | {buy2['IB']:.1%} / {buy2['IN']:.1%} | {away} of {n_ib} | "
                 f"{n_below} of {n_valid} ({n_below / n_valid:.2%}) | {change:.2f} ({key}) |")
        print(f"done {m}")

    L += by_turn + ["", "## Example replies (Appendix C)", ""]
    for m, qid, t, c in EXAMPLES:
        rows = ok[m].values()
        first = next(r for r in rows if r["kind"] == "decision" and r["qid"] == qid and r["t"] == t
                     and r["condition"] == c and not r["disclaimer"])
        second = next((r for r in rows if r["kind"] == "followup" and r.get("parent") == first["cell_id"]), None)
        L += [f"**{m}**, condition {c}, t = {t}, outcome {'YES' if outcome[qid] else 'NO'} ({qid}): {questions[qid]}", "",
              f"- Turn 1: {describe(first)}"]
        if second:
            L.append(f"- Turn 2: {describe(second)}")
        L.append("")
    (run_dir / "secondary.md").write_text("\n".join(L))
    print(f"Wrote {run_dir / 'secondary.md'}")


if __name__ == "__main__":
    main()
