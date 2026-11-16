#!/usr/bin/env python3
"""Summarise a run's logs: progress, valid replies, routes and spend. Makes no API calls.

    python3 run/status.py --run main
"""
import argparse
import collections
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run.runner import read_jsonl  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--runs-dir", default=str(ROOT / "runs"))
    a = ap.parse_args(argv)
    d = Path(a.runs_dir) / a.run
    if not d.exists():
        sys.exit(f"no run folder {d}")

    total = 0.0
    for f in sorted(d.glob("*.jsonl")):
        if f.name == "run_meta.jsonl":
            continue
        rows = read_jsonl(f)
        ok = {}
        for r in rows:
            if r["status"] == "ok":
                ok.setdefault(r["cell_id"], r)
        failed_cells = {r["cell_id"] for r in rows if r["status"] != "ok"} - set(ok)
        valid = sum(r["parse"]["valid"] for r in ok.values())
        errs = collections.Counter(r["parse"]["error"] for r in ok.values() if not r["parse"]["valid"])
        routes = collections.Counter(r["route"] for r in ok.values())
        spend = collections.defaultdict(float)
        for r in rows:
            spend[r["route"]] += r.get("cost_usd") or 0.0
        total += sum(spend.values())
        fallbacks = collections.Counter(r["fallback"] for r in ok.values() if r.get("fallback"))
        providers = collections.Counter(str(r.get("provider")) for r in ok.values() if r["route"] == "openrouter")
        finish = collections.Counter(str(r.get("finish_reason")) for r in ok.values())
        lat = [r["attempts"][-1]["latency_s"] for r in ok.values() if r.get("attempts")]
        out_tok = [(r.get("usage") or {}).get("completion_tokens") or 0 for r in ok.values()]
        print(f"\n== {f.stem}")
        print(f"   cells with a model response: {len(ok)}   cells with no response yet (re-run the same command to retry): {len(failed_cells)}")
        print(f"   valid: {valid}/{len(ok)} ({100 * valid / max(1, len(ok)):.1f}%)   invalid by reason: {dict(errs) or '-'}")
        kinds = collections.Counter(r["kind"] for r in ok.values())
        kvalid = collections.Counter(r["kind"] for r in ok.values() if r["parse"]["valid"])
        print("   by kind: " + ", ".join(f"{k} {kvalid[k]}/{n} valid" for k, n in sorted(kinds.items()))
              + ("   (followup = second turn)" if kinds.get("followup") else ""))
        print(f"   routes: {dict(routes)}   fallbacks: {dict(fallbacks) or '-'}")
        print(f"   spend: " + ", ".join(f"{k} ${v:.4f}" for k, v in spend.items()) + f"   (OpenAI native spend is tokens x list price)")
        print(f"   OpenRouter providers: {dict(providers) or '-'}   finish reasons: {dict(finish)}")
        if lat:
            print(f"   latency median {statistics.median(lat):.1f}s   output tokens (incl. reasoning) mean {statistics.mean(out_tok):.0f}, max {max(out_tok)}")

    meta = d / "run_meta.jsonl"
    if meta.exists():
        print("\n== route switches (from run_meta.jsonl)")
        any_switch = False
        for m in read_jsonl(meta):
            for s in m.get("route_switches", []):
                any_switch = True
                print(f"   {m.get('finished_utc', '?')}  {s['reason']}")
        if not any_switch:
            print("   none recorded (OpenAI models stayed on the native API, or the run has not finished)")
    print(f"\nTotal spend logged in runs/{a.run}: ${total:.4f}")


if __name__ == "__main__":
    main()
